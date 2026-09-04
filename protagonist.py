import os
import re
import json
import shutil
import time
import sys
import logging
import glob
from datetime import datetime
from openai import OpenAI
try:
    from openai import APIStatusError
except Exception:
    APIStatusError = Exception
from tqdm import tqdm
import concurrent.futures
import threading
from api_client import make_chat_completion
from token_tracker import create_default_tracker
from shared_utils import get_base_dir, read_file_safely

BASE_URL = os.environ.get("BASE_URL", "https://tb.api.mkeai.com/v1")
MODEL = os.environ.get("MODEL_NAME", "deepseek-chat")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "6"))

API_KEY_POOL = [
    k.strip() for k in os.environ.get("API_KEY_POOL", os.environ.get("API_KEY", "")).split(",")
    if k.strip()
]
API_KEY = API_KEY_POOL[0] if API_KEY_POOL else ""

CHUNK_SIZE = 10000

DEFAULT_PROGRESS_FLAGS = {
    "scanned": False,
    "male_identified": False,
    "alias_merged": False,
    "heroines_identified": False,
    "report_generated": False,
    "report_files": {}
}

# 以下变量在 main() 中按每本小说重新初始化
NOVEL_FILE_PATH = None
clean_filename = None
OUTPUT_DIR = None
logger = logging.getLogger(__name__)


def find_latest_checkpoint_dir(prefix: str):
    safe_prefix = glob.escape(prefix)
    results_dir = os.path.join(get_base_dir(), "results")
    pattern = os.path.join(results_dir, f"{safe_prefix}_heroine_*", "latest_checkpoint.json")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    return os.path.dirname(latest)


def get_latest_report_files(prefix: str = None):
    target_prefix = (prefix or clean_filename or "").strip()
    if not target_prefix:
        return {}
    checkpoint_dir = find_latest_checkpoint_dir(target_prefix)
    if not checkpoint_dir:
        return {}
    checkpoint_file = os.path.join(checkpoint_dir, "latest_checkpoint.json")
    if not os.path.exists(checkpoint_file):
        return {}
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"读取 protagonist checkpoint 失败: {exc}")
        return {}
    progress = data.get("progress", {}) if isinstance(data, dict) else {}
    report_files = progress.get("report_files", {}) if isinstance(progress, dict) else {}
    return dict(report_files or {})

# ---- API 调用封装：统一收敛到 api_client.py（只需修改 api_client.py 即可全局生效）----
MAX_403_RETRIES = 3
MAX_TIMEOUT_RETRIES = 5  # 连续超时 3 次则标记 key 不可用
REQUEST_TIMEOUT = 150  # 请求超时时间（秒）

def _openai_client_factory(api_key: str, base_url: str, timeout: int):
    """
    创建 OpenAI 客户端，关闭 SDK 暗重试并使用细粒度 timeout。
    
    【关键】max_retries=0 关闭 SDK 自动重试：
    - SDK 默认会重试 2 次，每次都有 timeout
    - 外层 api_client.py 再重试 5 次
    - 不关闭的话，总耗时可能达到 120s * 3 * 5 = 1800s
    
    【关键】使用 httpx.Timeout 细粒度配置：
    - connect: 连接超时（10s）
    - read: 读取超时（根据请求规模动态调整）
    - write: 写入超时（30s）
    - pool: 连接池超时（10s）
    """
    try:
        import httpx
        http_timeout = httpx.Timeout(
            connect=10.0,
            read=float(timeout),
            write=30.0,
            pool=10.0,
        )
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=http_timeout,
            max_retries=0,  # 关闭 SDK 自动重试
        )
    except ImportError:
        # 没有 httpx 时使用简单 timeout
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # 关闭 SDK 自动重试
        )

def _normalize_person_name(name: str) -> str:
    """
    规范化人名字符串：
    - 去首尾空白/引号/全角空格
    - 统一部分标点（全角逗号等）
    - 压缩多余空白
    注意：不做大小写/空格强行移除（避免影响外国人名，如 'Jean Pierre'）。
    """
    if name is None:
        return ""
    s = str(name).strip().strip("\u3000")
    # 去掉常见引号
    s = s.strip("“”\"'`")
    # 统一中文逗号/分号为英文逗号，便于后续拆分判断
    s = s.replace("，", ",").replace("；", ";").replace("：", ":")
    # 压缩多空白
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_multi_names(name: str):
    """
    将可能被模型/作者写在同一个字段里的“多个角色名”拆分出来。

    重点处理：'、' / ',' / '，' 等枚举分隔。
    安全策略：如果两侧都是纯英文(含空格/点/连字符)，不拆分（避免 'Smith, John' 之类外文格式）。
    """
    s = _normalize_person_name(name)
    if not s:
        return []

    # 最强拆分：顿号（中文枚举）
    if "、" in s:
        parts = [p.strip() for p in s.split("、") if p.strip()]
        # 过滤明显无效段
        parts = [_normalize_person_name(p) for p in parts if _normalize_person_name(p)]
        return parts if len(parts) >= 2 else [s]

    # 次强拆分：逗号（仅当包含中日韩字符时才拆）
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) < 2:
            return [s]

        ascii_wordish = re.compile(r"^[A-Za-z][A-Za-z .'\-]*$")
        # 两侧都像英文名 → 不拆
        if all(ascii_wordish.match(p) for p in parts):
            return [s]
        # 有明显中日韩字符 → 拆
        has_cjk = any(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", p) for p in parts)
        if has_cjk:
            parts = [_normalize_person_name(p) for p in parts if _normalize_person_name(p)]
            return parts if len(parts) >= 2 else [s]

    return [s]


def _levenshtein_distance(a: str, b: str, max_dist: int = 2) -> int:
    """小字符串编辑距离（带上限早停），用于错别字/近似名候选判断。"""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    # 早停：长度差过大不算近似
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    # DP（带上限剪枝）
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            rep = prev[j - 1] + (0 if ca == cb else 1)
            v = min(ins, dele, rep)
            cur.append(v)
            if v < row_min:
                row_min = v
        prev = cur
        if row_min > max_dist:
            return max_dist + 1
    return prev[-1]


def _quick_text_similarity(a_texts, b_texts) -> float:
    """
    基于摘要/互动的快速相似度：把多条文本拼接后做 SequenceMatcher。
    仅用于“是否有足够证据合并”的兜底判定。
    """
    try:
        from difflib import SequenceMatcher
        a = " ".join([str(x) for x in (a_texts or []) if x])[:8000]
        b = " ".join([str(x) for x in (b_texts or []) if x])[:8000]
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0


def _extract_indexed_texts(raw_list, limit: int = 50):
    """
    将可能为 [(chunk_idx, text), ...] / [[chunk_idx, text], ...] / [text, ...] 的结构
    转成纯文本列表（尽量保留时间顺序，不做抽样逻辑）。
    """
    if not raw_list:
        return []
    first = raw_list[0]
    if isinstance(first, (tuple, list)) and len(first) == 2 and isinstance(first[0], (int, float)):
        items = sorted(raw_list, key=lambda x: x[0])
        texts = [str(x[1]) for x in items if len(x) == 2 and x[1]]
        return texts[:limit]
    texts = [str(x) for x in raw_list if x]
    return texts[:limit]


def _extract_chunk_indices(stats_entry: dict) -> set:
    """
    从角色的 summaries/interactions/emotion_signals/chunk_scores 中
    提取所有 chunk 索引，返回 set[int]。用于共现分析。
    """
    indices = set()
    for field in ("summaries", "interactions", "emotion_signals", "chunk_scores"):
        raw = stats_entry.get(field, [])
        for item in (raw or []):
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                try:
                    indices.add(int(item[0]))
                except (ValueError, TypeError):
                    pass
    return indices


def _should_accept_merge_pair(stats_dict: dict, name_a: str, name_b: str, conflict_pairs=None, same_name_prefix_pairs=None) -> (bool, str):
    """
    对“是否允许把 name_b 合并进 name_a”做确定性校验，降低误合并：
    - 强证据：别名互含/共享别名、摘要或互动出现完全相同文本
    - 软证据：文本相似度很高（摘要/互动），或轻微错别字(编辑距离<=1) + 文本相似度较高
    """
    if not name_a or not name_b or name_a == name_b:
        return False, "空名或同名"
    if stats_dict is None:
        return False, "无统计数据"
    if name_a not in stats_dict or name_b not in stats_dict:
        return False, "角色不在统计中"

    # 辈分冲突直接否决
    for p in (conflict_pairs or []):
        if name_a in p and name_b in p:
            return False, f"辈分冲突：{name_a} vs {name_b}"

    a = stats_dict[name_a]
    b = stats_dict[name_b]

    # ===== 新增：共现分析 =====
    a_chunks = _extract_chunk_indices(a)
    b_chunks = _extract_chunk_indices(b)
    shared_chunks = a_chunks & b_chunks
    co_occ_ratio = 0.0
    if a_chunks and b_chunks:
        smaller = min(len(a_chunks), len(b_chunks))
        co_occ_ratio = len(shared_chunks) / smaller if smaller > 0 else 0.0
    has_strong_co_occurrence = len(shared_chunks) >= 3 and co_occ_ratio >= 0.15

    # ===== 新增：同前缀可疑对检测 =====
    is_same_prefix_pair = False
    for sp in (same_name_prefix_pairs or []):
        if name_a in sp[:2] and name_b in sp[:2]:
            is_same_prefix_pair = True
            break

    a_others = a.get("other_names", set()) or set()
    b_others = b.get("other_names", set()) or set()
    if isinstance(a_others, list):
        a_others = set(a_others)
    if isinstance(b_others, list):
        b_others = set(b_others)

    # ===== 提前计算文本相似度（供后续所有判断使用） =====
    a_sum = set(_extract_indexed_texts(a.get("summaries", []), limit=80))
    b_sum = set(_extract_indexed_texts(b.get("summaries", []), limit=80))
    a_int = set(_extract_indexed_texts(a.get("interactions", []), limit=80))
    b_int = set(_extract_indexed_texts(b.get("interactions", []), limit=80))
    sim_s = _quick_text_similarity(list(a_sum)[:30], list(b_sum)[:30])
    sim_i = _quick_text_similarity(list(a_int)[:30], list(b_int)[:30])
    sim = max(sim_s, sim_i)

    # ===== 共现硬性否决：大量共现 = 几乎一定是不同角色 =====
    if len(shared_chunks) >= 5 and co_occ_ratio >= 0.25:
        return False, (
            f"高度共现({len(shared_chunks)}个chunk，比率={co_occ_ratio:.1%})，极可能是不同角色"
        )

    # 1) 别名互含 — 不再无条件自动通过
    if name_b in a_others or name_a in b_others:
        both_significant = (a.get('count', 0) >= 3 and b.get('count', 0) >= 3)
        if both_significant and sim < 0.50:
            # 文本差异太大，别名很可能是污染/称号混淆，不是真正的别名
            logger.info(f"别名互含但文本差异大: {name_a} vs {name_b} (sim={sim:.2f})")
            # 不自动通过，继续往下检查
        elif is_same_prefix_pair and has_strong_co_occurrence:
            # 同前缀+共现 = 典型的同姓不同人污染
            logger.info(f"别名互含但同前缀共现: {name_a} vs {name_b}")
            # 不自动通过
        else:
            return True, "别名互含"

    # 共享别名 — 同样加入验证
    shared_aliases = a_others & b_others
    if shared_aliases:
        both_significant = (a.get('count', 0) >= 3 and b.get('count', 0) >= 3)
        if both_significant and sim < 0.50:
            logger.info(f"共享别名但文本差异大: {name_a} vs {name_b}, 共享={list(shared_aliases)[:3]}, sim={sim:.2f}")
        elif is_same_prefix_pair and has_strong_co_occurrence:
            logger.info(f"共享别名但同前缀共现: {name_a} vs {name_b}")
        else:
            return True, f"共享别名：{list(shared_aliases)[:3]}"

    # 2) 摘要/互动出现完全相同条目
    if a_sum and b_sum and (a_sum & b_sum):
        return True, "摘要存在完全相同条目"
    if a_int and b_int and (a_int & b_int):
        return True, "互动存在完全相同条目"

    # 3) 文本相似度
    if is_same_prefix_pair:
        # 同前缀可疑对需要更高阈值
        if sim >= 0.95:
            return True, f"同前缀但文本极度相似(sim={sim:.2f})"
    else:
        if sim >= 0.90:
            return True, f"文本高度相似(sim={sim:.2f})"

    # 4) 轻微错别字 + 证据较高
    if 2 <= len(name_a) <= 8 and 2 <= len(name_b) <= 8:
        dist = _levenshtein_distance(name_a, name_b, max_dist=2)
        if dist <= 1 and sim >= 0.78:
            return True, f"轻微错别字(dist={dist})且文本相似(sim={sim:.2f})"

    return False, f"缺少合并证据(sim={sim:.2f})"


###############################################################################
# 统计学女主提示（HSI: Heroine Separation Index）
# - 目的：在“别称合并后”给逐个判定女主的大模型一个稳定的统计学先验提示
# - 注意：只作为提示，不做程序强行判定
###############################################################################

# HSI 关键词（可按小说类型微调）
_HSI_KEYWORDS = [
    # 明确亲密/性关系
    r"发生关系",
    r"上床",
    r"同床",
    r"睡在一起",
    r"侍寝",
    r"做爱|做\s*爱",
    r"插入|进入",
    r"高潮",
    r"湿(了|润)|潮(湿|红)",
    r"揉|摸|抚|撩|挑逗|调情",
    r"推倒|强上|献身|投怀送抱",
    # 恋爱/关系标签
    r"恋人|情人|女友|妻|未婚妻|定情|表白|我爱你",
    r"暧昧|吃醋|脸红|心跳|心动",
    # 男主后宫常见“认主/专属关系”
    r"认主|效忠|宣誓",
    r"亲吻(脚尖|脚背)|吻(脚尖|脚背)",
    r"洗脚|按摩",
    r"专属女仆|女仆契约|契约",
    # 轻度亲密（容易出现，但不是强证据；用于密度信号）
    r"牵手|拥抱|亲吻|舌吻|接吻|宠溺|抱",
]
_HSI_PATTERNS = [re.compile(p) for p in _HSI_KEYWORDS]


def _hsi_compute(avg_score: float, count: int, density: float) -> float:
    """HSI（0~1）：频次饱和 + 证据密度 + avg_score 弱化项。"""
    try:
        import math

        A = max(0.0, min(1.0, float(avg_score) / 10.0))
        F = 1.0 - math.exp(-max(0, int(count)) / 20.0)
        D = math.tanh(0.8 * max(0.0, float(density)))
        return 0.55 * F + 0.35 * D + 0.10 * A
    except Exception:
        return 0.0


def _hsi_text_from_stats(stats: dict, max_items_each: int = 200) -> str:
    """从 merged_stats 单角色数据拼出用于统计的文本（不做语义理解，仅关键词计数）。"""
    if not stats:
        return ""
    interactions = _extract_indexed_texts(stats.get("interactions", []), limit=max_items_each)
    summaries = _extract_indexed_texts(stats.get("summaries", []), limit=max_items_each)
    emotions = _extract_indexed_texts(stats.get("emotion_signals", []), limit=max_items_each)
    relationships = stats.get("relationships", []) or []
    if isinstance(relationships, (set, tuple)):
        relationships = list(relationships)
    relationships = [str(x) for x in relationships[: max(10, max_items_each // 20)] if x]
    text = " ".join([*(str(x) for x in interactions if x), *(str(x) for x in summaries if x), *(str(x) for x in emotions if x), *relationships])
    return text


def _hsi_hits_and_density(text: str) -> (int, float, float):
    """
    返回：
    - hits: 关键词命中次数（带上限防爆）
    - density: hits / (1 + 文本KB)（每千字密度）
    - text_kb: 文本长度（KB）
    """
    if not text:
        return 0, 0.0, 0.0
    hits = 0
    for pat in _HSI_PATTERNS:
        found = pat.findall(text)
        if found:
            hits += min(len(found), 30)
    text_kb = len(text) / 1000.0
    density = hits / (1.0 + text_kb)
    return int(hits), float(density), float(text_kb)


def _prepare_hsi_hints(merged_stats: dict):
    """
    基于 merged_stats 计算所有角色的 HSI，并给出一个“推荐阈值”（用于提示）。
    返回：
      name_to_hint[name] = {
        "hsi": float,
        "rank": int,  # 1-based
        "total": int,
        "hits": int,
        "density": float,
        "text_kb": float,
        "threshold": float,
        "strong_k": int
      }
      plus: {"threshold":..., "strong_k":..., "total":...}
    """
    # 先计算每个角色的 HSI
    rows = []
    for name, data in (merged_stats or {}).items():
        cnt = int((data or {}).get("count", 0) or 0)
        avg = (data.get("total_score", 0) / cnt) if cnt > 0 else 0.0
        txt = _hsi_text_from_stats(data or {})
        hits, density, text_kb = _hsi_hits_and_density(txt)
        score = _hsi_compute(avg, cnt, density)
        rows.append((name, score, hits, density, text_kb, avg, cnt))

    rows.sort(key=lambda x: x[1], reverse=True)
    total = len(rows)
    if total == 0:
        return {}, {"threshold": 0.0, "strong_k": 0, "total": 0}

    # 推荐阈值：取 Top ~15%（最少12，最多40）对应的分数作为“强候选阈值”
    strong_k = min(max(12, int(round(total * 0.15))), 40, total)
    threshold = rows[strong_k - 1][1] if strong_k >= 1 else rows[-1][1]

    name_to_hint = {}
    for i, (name, score, hits, density, text_kb, avg, cnt) in enumerate(rows, start=1):
        name_to_hint[name] = {
            "hsi": float(score),
            "rank": int(i),
            "total": int(total),
            "hits": int(hits),
            "density": float(density),
            "text_kb": float(text_kb),
            "threshold": float(threshold),
            "strong_k": int(strong_k),
        }
    return name_to_hint, {"threshold": float(threshold), "strong_k": int(strong_k), "total": int(total)}


###############################################################################
# 证据档案计算（Evidence Profile）
# - 目的：为每个候选角色提取关键亲密/恋爱证据、评分分布/趋势、数据量上下文
# - 用于在 AI 判断提示词中前置展示最强证据，避免信号稀释导致漏判
###############################################################################

# 关键词分级（由强到弱）
_PEAK_EVIDENCE_TIERS = [
    # Tier 0: 确认性关系（最强信号）
    re.compile(r"发生关系|肉体关系|上床|做爱|做\s*爱|同床共枕|侍寝|推倒并|强上|献身|投怀送抱|圆房|春宵|共度良宵|怀孕|怀上"),
    # Tier 1: 确认恋爱行为
    re.compile(r"亲吻|接吻|舌吻|主动亲|主动吻|表白|告白|我爱你|我喜欢你|定情|求婚"),
    # Tier 2: 确认双向关系
    re.compile(r"恋人|女友|妻子|未婚妻|情人|正宫|暧昧关系|确认关系|成为.*伴侣"),
    # Tier 3: 强烈感情信号
    re.compile(r"暧昧|吃醋|暗恋|心动|约会|牵手|拥抱|脸红心跳|心跳加速|占有欲"),
]


def _compute_evidence_profile(char_stats, all_merged_stats):
    """
    为单个角色计算证据档案，包含：
    - score_distribution: 评分分布（各分段出场次数）
    - score_trend: 评分趋势（前半 vs 后半平均分）
    - peak_evidence: 关键亲密/恋爱证据（按强度排序的文本摘录）
    - volume_context: 数据量上下文（与全体角色的对比）
    """
    chunk_scores = char_stats.get("chunk_scores", [])
    count = char_stats.get("count", 0) or len(chunk_scores)

    # ─── 评分分布 ───
    score_buckets = {"8-10": 0, "6-7": 0, "4-5": 0, "1-3": 0}
    score_trend = ""
    if chunk_scores:
        sorted_scores = sorted(chunk_scores, key=lambda x: x[0])
        for _, s in sorted_scores:
            s = int(s) if isinstance(s, (int, float)) else 0
            if s >= 8:
                score_buckets["8-10"] += 1
            elif s >= 6:
                score_buckets["6-7"] += 1
            elif s >= 4:
                score_buckets["4-5"] += 1
            else:
                score_buckets["1-3"] += 1

        mid = len(sorted_scores) // 2
        if mid > 0:
            first_half_avg = sum(s for _, s in sorted_scores[:mid]) / mid
            second_half_avg = sum(s for _, s in sorted_scores[mid:]) / (len(sorted_scores) - mid)
            score_trend = f"前半段平均 {first_half_avg:.1f} → 后半段平均 {second_half_avg:.1f}"
            if second_half_avg - first_half_avg >= 2.0:
                score_trend += "（显著上升趋势，暗示感情在发展）"
            elif second_half_avg - first_half_avg >= 1.0:
                score_trend += "（上升趋势）"
        else:
            score_trend = "数据不足"
    else:
        score_trend = "无逐块分数数据"

    # ─── 关键证据提取 ───
    all_texts = []
    for field in ("summaries", "interactions", "emotion_signals"):
        raw = char_stats.get(field, [])
        for item in raw:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                all_texts.append((int(item[0]), str(item[1])))
            elif isinstance(item, str):
                all_texts.append((0, item))

    peak_evidence = []
    seen_texts = set()
    for tier_idx, pattern in enumerate(_PEAK_EVIDENCE_TIERS):
        for chunk_idx, text in all_texts:
            if pattern.search(text):
                text_key = text[:80]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    peak_evidence.append((tier_idx, chunk_idx, text[:300]))

    # 按级别排序（最强优先），同级别按出现位置倒序（后期证据优先）
    peak_evidence.sort(key=lambda x: (x[0], -x[1]))
    peak_evidence_texts = [t for _, _, t in peak_evidence[:15]]

    # ─── 数据量上下文 ───
    all_counts = sorted([
        d.get("count", 0) for d in all_merged_stats.values()
    ])
    n = len(all_counts)
    volume_context = {"count": count}
    if n > 0:
        median_count = all_counts[n // 2]
        higher_count = sum(1 for c in all_counts if c <= count)
        percentile = higher_count * 100.0 / n
        volume_context.update({
            "median": median_count,
            "percentile": round(percentile, 1),
            "ratio_to_median": round(count / max(median_count, 1), 1),
            "total_chars": n,
        })

    return {
        "score_distribution": score_buckets,
        "score_trend": score_trend,
        "peak_evidence": peak_evidence_texts,
        "volume_context": volume_context,
    }


chat_completion = make_chat_completion(
    openai_client_factory=_openai_client_factory,
    api_key_pool=API_KEY_POOL,
    base_url=BASE_URL,
    request_timeout=REQUEST_TIMEOUT,
    max_retries=6,
    max_403_retries=MAX_403_RETRIES,
    max_timeout_retries=MAX_TIMEOUT_RETRIES,
    base_delay=2,
    logger=logger,
)

# ---------------- JSON 容错解析（避免“控制字符”导致反复重试浪费时间） ----------------
def _safe_json_loads(text: str):
    """
    尽最大可能把模型输出解析成 JSON。
    主要应对：Invalid control character / 夹杂不可见控制字符 / ```json``` 包裹等。
    """
    if text is None:
        raise json.JSONDecodeError("empty", "", 0)
    s = str(text).strip()
    # 去掉 code fence
    s = re.sub(r"```json\s*|\s*```", "", s).strip()

    # 1) 直接解析
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) 宽松解析（允许控制字符）
    try:
        return json.JSONDecoder(strict=False).decode(s)
    except Exception:
        pass

    # 3) 清理 ASCII 控制字符（保留 \t \n \r）
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4) 兜底：截取最外层 JSON（从第一个 { 到最后一个 }）
    l = cleaned.find("{")
    r = cleaned.rfind("}")
    if l != -1 and r != -1 and r > l:
        snippet = cleaned[l : r + 1]
        return json.loads(snippet)

    raise json.JSONDecodeError("unable to parse json", cleaned[:200], 0)

token_tracker = None


def init_token_tracker(book_name, run_id=None):
    global token_tracker
    token_tracker = create_default_tracker(
        "protagonist.py",
        book_name=book_name,
        out_path=os.path.join(get_base_dir(), "results", "token_usage.json"),
        run_id=run_id,
    )
    return token_tracker


def record_usage(resp):
    """统计大模型输入/输出 token"""
    try:
        if token_tracker is not None:
            token_tracker.record(resp)
    except Exception:
        pass


def validate_config():
    """验证配置"""
    if not API_KEY or API_KEY == "your_api_key_here":
        raise ValueError("请设置有效的API_KEY")
    if not NOVEL_FILE_PATH or NOVEL_FILE_PATH == "path/to/your/novel.txt":
        raise ValueError("请设置小说文件路径")
    if not os.path.exists(NOVEL_FILE_PATH):
        raise FileNotFoundError(f"文件不存在: {NOVEL_FILE_PATH}")
    return True


def _serialize_character_stats(stats_dict):
    """通用的角色数据序列化，支持 merged_stats/global_stats"""
    serialized = {}
    for name, data in stats_dict.items():
        types_val = data.get("types", [])
        if isinstance(types_val, set):
            types_val = list(types_val)
        other_names_val = data.get("other_names", [])
        if isinstance(other_names_val, set):
            other_names_val = list(other_names_val)
        serialized[name] = {
            "total_score": data.get("total_score", 0),
            "count": data.get("count", 0),
            "chunk_scores": data.get("chunk_scores", []),
            "summaries": data.get("summaries", []),
            "types": types_val,
            "other_names": other_names_val,
            "appearances": data.get("appearances", []),
            "features": data.get("features", []),
            "relationships": data.get("relationships", []),
            "interactions": data.get("interactions", []),
            "emotion_signals": data.get("emotion_signals", []),
        }
    return serialized


def save_checkpoint(global_stats, male_protagonist_stats, last_processed_index, progress_flags=None, merged_stats=None, heroine_result=None, male_protagonist_final=None, completed_chunks=None):
    """
    保存分析进度和结果，用于断点续传（包含男主和女性角色信息）
    progress_flags 用于记录阶段性状态：是否完成各阶段
    completed_chunks: 实际成功完成分析的块索引集合（用于精确追踪）
    """
    progress = dict(DEFAULT_PROGRESS_FLAGS)
    if progress_flags:
        progress.update(progress_flags)
    # report_files 需要确保为全新字典而非 None/共享引用
    progress["report_files"] = dict(progress.get("report_files") or {})
    # 序列化女性角色信息（扫描阶段产物）
    serializable_stats = _serialize_character_stats(global_stats)

    # 序列化 merged_stats（别称合并后产物）
    serializable_merged_stats = _serialize_character_stats(merged_stats) if merged_stats else None

    # 序列化男主信息
    serializable_male_stats = {}
    for name, data in male_protagonist_stats.items():
        other_names_val = data.get("other_names", set())
        if isinstance(other_names_val, set):
            other_names_val = list(other_names_val)
        elif other_names_val is None:
            other_names_val = []
        serializable_male_stats[name] = {
            "count": data["count"],
            "other_names": other_names_val,
            "identities": data.get("identities", []),
            "summaries": data.get("summaries", [])
        }

    checkpoint_data = {
        "last_processed_chunk": last_processed_index,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "global_stats": serializable_stats,
        "male_protagonist_stats": serializable_male_stats,
        "progress": progress,
        "merged_stats": serializable_merged_stats,
        "heroine_result": heroine_result,
        "male_protagonist_final": male_protagonist_final,
        "completed_chunks": list(completed_chunks) if completed_chunks else [],  # 实际成功的块索引列表
    }
    
    checkpoint_file = f"{OUTPUT_DIR}/latest_checkpoint.json"
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        completed_count = len(completed_chunks) if completed_chunks else 0
        logger.info(f"断点已保存 (已完成 {completed_count} 块)")
    except Exception as e:
        logger.error(f"保存断点文件失败: {e}")


def _restore_character_stats(serialized_stats):
    """从序列化数据恢复角色信息结构"""
    restored = {}
    for name, data in serialized_stats.items():
        restored[name] = {
            "total_score": data.get("total_score", 0),
            "count": data.get("count", 0),
            "chunk_scores": data.get("chunk_scores", []),
            "summaries": data.get("summaries", []),
            "types": set(data.get("types", [])),
            "other_names": set(data.get("other_names", [])),
            "appearances": data.get("appearances", []),
            "features": data.get("features", []),
            "relationships": data.get("relationships", []),
            "interactions": data.get("interactions", []),
            "emotion_signals": data.get("emotion_signals", []),
        }
        # 将带 chunk_index 的字段中的 list 转回 tuple（JSON 加载后 tuple 变成 list）
        for field in ['summaries', 'interactions', 'emotion_signals', 'chunk_scores']:
            if field in restored[name]:
                field_data = restored[name][field]
                if field_data and isinstance(field_data[0], list) and len(field_data[0]) == 2:
                    restored[name][field] = [tuple(s) for s in field_data]
    return restored


def load_checkpoint():
    """加载最新的断点文件（包含男主和女性角色信息）"""
    checkpoint_file = f"{OUTPUT_DIR}/latest_checkpoint.json"
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 加载已完成的块索引集合（优先使用新格式）
                completed_chunks = set(data.get('completed_chunks', []))
                last_index = data.get('last_processed_chunk', -1)
                
                # 兼容旧格式：如果没有 completed_chunks，用 last_processed_chunk 推断
                if not completed_chunks and last_index >= 0:
                    completed_chunks = set(range(last_index + 1))
                
                logger.info(f"找到断点文件。已完成 {len(completed_chunks)} 块。")
                progress = dict(DEFAULT_PROGRESS_FLAGS)
                progress.update(data.get('progress', {}))
                progress["report_files"] = dict(progress.get("report_files") or {})
                
                # 加载女性角色信息
                loaded_stats = _restore_character_stats(data.get('global_stats', {}))

                # 加载 merged_stats
                loaded_merged_stats = None
                if data.get("merged_stats"):
                    loaded_merged_stats = _restore_character_stats(data.get("merged_stats", {}))
                
                # 加载男主信息
                loaded_male_stats = data.get('male_protagonist_stats', {})
                for name in loaded_male_stats:
                    if 'other_names' in loaded_male_stats[name] and isinstance(loaded_male_stats[name]['other_names'], list):
                        loaded_male_stats[name]['other_names'] = set(loaded_male_stats[name]['other_names'])

                heroine_result = data.get("heroine_result")
                male_protagonist_final = data.get("male_protagonist_final")

                return loaded_stats, loaded_male_stats, completed_chunks, progress, loaded_merged_stats, heroine_result, male_protagonist_final
        except Exception as e:
            logger.error(f"加载断点文件失败: {e}。将从头开始。")
            default_progress = dict(DEFAULT_PROGRESS_FLAGS)
            default_progress["report_files"] = {}
            return {}, {}, set(), default_progress, None, None, None
    default_progress = dict(DEFAULT_PROGRESS_FLAGS)
    default_progress["report_files"] = {}
    return {}, {}, set(), default_progress, None, None, None


def read_novel(file_path):
    """读取小说，自动处理编码（UTF-8 -> GB18030 兜底）"""
    content = read_file_safely(file_path)
    logger.info(f"成功读取文件: {file_path}")
    return content


def split_text_by_length(text, chunk_size):
    """智能切分文本，尽量在段落边界处断开"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        if end < text_length:
            for lookahead in range(0, min(500, text_length - end)):
                if text[end + lookahead] in ['\n\n', '。', '！', '？', '……']:
                    end = end + lookahead + 1
                    break
        
        chunks.append(text[start:end])
        start = end
    
    logger.info(f"将文本切分为 {len(chunks)} 个块")
    return chunks


def analyze_chunk_for_heroines(text_chunk, chunk_index, total_chunks, max_retries=3):
    """
    分析单个文本块，识别女主角/女性重要角色，同时识别男主
    收集更详细的特征信息用于后续别称合并
    """
    
    system_prompt = """你是一个专业的小说分析师，负责识别小说中的**男主角**和**女性角色**。

## 核心任务：
1. 识别男主角（通常只有一个）
2. 识别所有与男主有互动的女性角色，**重点关注与男主的感情互动**

## 评分标准（核心依据：与男主的感情互动程度）：

### 男主角：
- 10分：绝对男主角（第一人称视角/故事核心）
- 8-9分：可能是男主

### 女性角色（按与男主互动程度打分）：
- 9-10分：核心女主（与男主有明确恋爱关系/深度感情互动/亲密肢体接触）
- 7-8分：重要女性（与男主有暧昧/好感/频繁单独相处/情感交流）
- 5-6分：有互动的女性（与男主有日常交流但无明显感情线）
- 3-4分：少量互动（偶尔与男主交谈）
- 1-2分：几乎无互动的路人女性

## 关键：判断女主的核心依据是【与男主的感情互动】，而非单纯的出场次数！

## 输出格式（JSON）：
{
    "male_protagonist": {
        "name": "男主名字",
        "other_names": ["别称1", "昵称2"],
        "identity": "身份背景",
        "summary": "本段行为概要"
    },
    "female_characters": [
        {
            "name": "角色名字",
            "other_names": ["别称1", "昵称2"],
            "score": 1-10,
            "appearance": "外貌特征（发色、眼睛、身材等）",
            "identity": "身份背景（职业、身份、与男主的社会关系）",
            "interaction_with_male_lead": "【重要】与男主的具体互动描述（对话内容、肢体接触、情感表达、单独相处等细节）",
            "relationship_type": "与男主的关系类型（恋人/暧昧/好感/朋友/同事/陌生人等）",
            "emotion_signals": "感情信号（脸红、心跳加速、吃醋、关心、撒娇等情感表现）",
            "summary": "本段中该角色的行为概要",
            "is_potential_heroine": true/false
        }
    ]
}

如果片段中没有明确的男主，male_protagonist 可以为 null。
如果没有女性角色，female_characters 为 []。
**只有与男主有感情互动迹象的女性才标记 is_potential_heroine 为 true。**"""

    user_prompt = f"""请分析以下小说片段（第 {chunk_index + 1}/{total_chunks} 块），**重点关注女性角色与男主的感情互动**：

--- 小说片段开始 ---
{text_chunk} 
--- 小说片段结束 ---

请识别并详细记录：
1. 男主角信息
2. 所有女性角色，**特别是她们与男主的互动细节**：
   - 有没有单独相处的场景？
   - 有没有亲密的对话或肢体接触？
   - 有没有表现出好感/爱意/暧昧的信号？
   - 男主对她的态度如何？

请以 JSON 格式输出。"""

    for retry in range(max_retries):
        try:
            # 优先强制 JSON 输出（若服务端不支持 response_format，会自动降级）
            try:
                response = chat_completion(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.warning(f"Chunk {chunk_index} response_format 不可用，降级重试: {e}")
                response = chat_completion(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                )
            record_usage(response)
            
            content = response.choices[0].message.content.strip()
            
            try:
                # 容错解析：避免“控制字符”导致 JSON 解析失败从而反复调用 API
                data = _safe_json_loads(content)
                
                result = {
                    "male_protagonist": None,
                    "female_characters": []
                }
                
                # 解析男主信息
                if isinstance(data, dict):
                    male_proto = data.get('male_protagonist')
                    if male_proto and isinstance(male_proto, dict):
                        name = male_proto.get('name', '').strip()
                        if name:
                            result["male_protagonist"] = {
                                "name": name,
                                "other_names": male_proto.get('other_names', []),
                                "identity": male_proto.get('identity', ''),
                                "summary": male_proto.get('summary', ''),
                                "chunk_index": chunk_index
                            }
                    
                    # 解析女性角色
                    females = data.get('female_characters', [])
                    if isinstance(females, list):
                        for char in females:
                            if not isinstance(char, dict):
                                continue
                            
                            name = char.get('name', '').strip()
                            if not name:
                                continue
                            
                            score = char.get('score', 0)
                            try:
                                score = int(score)
                            except:
                                score = 0
                            score = max(1, min(10, score))
                            
                            result["female_characters"].append({
                                "name": name,
                                "other_names": char.get('other_names', []),
                                "score": score,
                                "appearance": char.get('appearance', ''),
                                "identity": char.get('identity', ''),
                                "interaction_with_male_lead": char.get('interaction_with_male_lead', ''),
                                "relationship_type": char.get('relationship_type', ''),
                                "emotion_signals": char.get('emotion_signals', ''),
                                "summary": char.get('summary', ''),
                                "is_potential_heroine": char.get('is_potential_heroine', score >= 7),
                                "chunk_index": chunk_index
                            })
                
                male_info = f", 男主: {result['male_protagonist']['name']}" if result['male_protagonist'] else ""
                logger.info(f"Chunk {chunk_index} 分析完成，发现 {len(result['female_characters'])} 个女性角色{male_info}")
                result["_success"] = True  # 标记为成功
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Chunk {chunk_index} JSON解析失败 (尝试 {retry+1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    # JSON解析失败，标记为失败
                    return {"male_protagonist": None, "female_characters": [], "_success": False, "_error": f"JSON解析失败: {e}"}
                    
        except Exception as e:
            logger.error(f"Chunk {chunk_index} API调用失败 (尝试 {retry+1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
            else:
                # API调用失败，标记为失败
                return {"male_protagonist": None, "female_characters": [], "_success": False, "_error": f"API调用失败: {e}"}
    return {"male_protagonist": None, "female_characters": [], "_success": False, "_error": "所有重试均失败"}


def sample_summaries_by_timeline(summaries, max_count=20):
    """
    按时间线（chunk_index）均匀采样 summaries
    确保前期、中期、后期的剧情都能被保留
    
    Args:
        summaries: list of (chunk_index, summary) tuples/lists 或 list of strings
        max_count: 最多保留多少条
    
    Returns:
        list of summary strings (按时间顺序)
    """
    if not summaries:
        return []
    
    # 检查是否为带索引的格式（tuple 或 list 长度为2且第一个是数字）
    first = summaries[0]
    is_indexed = (isinstance(first, (tuple, list)) and len(first) == 2 and isinstance(first[0], (int, float)))
    
    # 处理旧格式（纯字符串列表）的兼容
    if not is_indexed:
        return summaries[:max_count]
    
    # 按 chunk_index 排序
    sorted_summaries = sorted(summaries, key=lambda x: x[0])
    
    if len(sorted_summaries) <= max_count:
        # 不需要采样，直接返回
        return [s[1] for s in sorted_summaries]
    
    # 均匀采样：确保前中后期都有
    step = len(sorted_summaries) / max_count
    sampled = []
    for i in range(max_count):
        idx = int(i * step)
        sampled.append(sorted_summaries[idx][1])
    
    return sampled


def identify_male_protagonist(male_stats):
    """
    识别男主角（简单逻辑：出现次数最多且合并别称）
    """
    if not male_stats:
        return None
    
    def _male_flatten_summaries(raw_list):
        """
        兼容 summaries 结构：
        - [(chunk_idx, text), ...] / [[chunk_idx, text], ...]
        - [text, ...]
        返回：按 chunk_idx 排序后的纯文本列表（去重）
        """
        texts = _extract_indexed_texts(raw_list, limit=100000)  # 基本不截断
        # 去重但尽量保序
        seen = set()
        out = []
        for t in texts:
            tt = (t or "").strip()
            if not tt:
                continue
            if tt in seen:
                continue
            seen.add(tt)
            out.append(tt)
        return out
    
    # 按出现次数排序
    sorted_males = sorted(male_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # 取出现次数最多的作为男主
    top_name, top_data = sorted_males[0]
    
    # 收集可能的别称（其他出现较少但可能是同一人的名字）
    all_other_names = set(top_data.get('other_names', set()))
    
    # 检查其他候选人是否可能是男主的别称
    for name, data in sorted_males[1:]:
        # 如果这个名字在男主的别称列表中，合并数据
        if name in all_other_names:
            top_data['count'] += data['count']
            all_other_names.update(data.get('other_names', set()))
            top_data['identities'].extend(data.get('identities', []))
            top_data['summaries'].extend(data.get('summaries', []))
        # 或者如果男主的名字在这个候选的别称中
        elif top_name in data.get('other_names', set()):
            top_data['count'] += data['count']
            all_other_names.add(name)
            all_other_names.update(data.get('other_names', set()))
            top_data['identities'].extend(data.get('identities', []))
            top_data['summaries'].extend(data.get('summaries', []))
    
    # 移除主名字本身
    all_other_names.discard(top_name)

    # identities/summaries 做去重保序（summaries 支持带 chunk_index 的结构）
    identities = []
    _seen_id = set()
    for it in (top_data.get("identities", []) or []):
        s = (it or "").strip()
        if not s or s in _seen_id:
            continue
        _seen_id.add(s)
        identities.append(s)
    
    summaries_all = _male_flatten_summaries(top_data.get("summaries", []) or [])
    
    return {
        "name": top_name,
        "other_names": list(all_other_names),
        "count": top_data['count'],
        # 兼容旧字段：identity 保留为字符串（拼接所有身份线索），并新增 identities 全量列表
        "identity": "；".join(identities) if identities else "未知",
        "identities": identities,
        # 全量 summaries（按时间线/去重）
        "summaries": summaries_all,
    }


def _get_generation_conflict_pairs(global_stats):
    """
    检测辈分冲突的角色对
    返回不应该合并的角色对列表，用于二次验证
    """
    # 辈分关键词
    ELDER_KEYWORDS = {'太太', '夫人', '妈妈', '母亲', '婆婆', '奶奶', '外婆', '阿姨', '姑姑', '姨妈'}
    YOUNGER_KEYWORDS = {'小姐', '姐姐', '妹妹', '姐', '妹', '女儿', '孙女', '外孙女', '学妹'}
    
    def get_generation_type(name):
        """返回角色的辈分类型：'elder', 'younger', 或 None"""
        for kw in ELDER_KEYWORDS:
            if kw in name:
                return 'elder'
        for kw in YOUNGER_KEYWORDS:
            if kw in name:
                return 'younger'
        return None
    
    # 提取姓氏（简单启发式：取前1-2个字）
    def extract_surname(name):
        # 去除括号内容
        core = re.sub(r'[（(][^）)]*[）)]', '', name).strip()
        # 去除常见后缀
        for suffix in ['太太', '夫人', '小姐', '妈妈', '母亲', '姐姐', '妹妹', '女儿', '同学', '老师', '前辈', '学姐', '学妹']:
            if core.endswith(suffix):
                core = core[:-len(suffix)]
        if len(core) >= 2:
            # 常见复姓
            if core[:2] in ['欧阳', '司马', '上官', '诸葛', '东方', '西门', '南宫', '北堂']:
                return core[:2]
            return core[0]  # 单姓
        return core
    
    conflict_pairs = []
    names = list(global_stats.keys())
    
    for i, name1 in enumerate(names):
        gen1 = get_generation_type(name1)
        if not gen1:
            continue
        surname1 = extract_surname(name1)
        
        for name2 in names[i+1:]:
            gen2 = get_generation_type(name2)
            if not gen2:
                continue
            
            # 如果辈分不同且姓氏相同，则是冲突对
            if gen1 != gen2:
                surname2 = extract_surname(name2)
                if surname1 and surname2 and surname1 == surname2:
                    conflict_pairs.append((name1, name2))
    
    return conflict_pairs


def _detect_same_name_prefix_pairs(global_stats):
    """
    检测共享姓氏前缀但有不同名字后缀的角色对（如"雪之下雪乃" vs "雪之下阳乃"）。
    这些角色可能是姐妹/母女，需要更强证据才能合并。

    使用最长公共前缀(LCP)方法，不依赖姓氏字典。
    排除一方名字是另一方子串的情况（那是合法别名）。

    同时检测 summaries/relationships 中的家族关系关键词。

    返回：[(name_a, name_b, hint_str), ...]
    """
    FAMILY_KEYWORDS = ['姐姐', '妹妹', '母亲', '女儿', '姐', '妹', '母女', '姐妹',
                       '姑姑', '阿姨', '婆婆', '奶奶', '外婆', '嫂子']

    def get_lcp_length(s1, s2):
        """返回两个字符串的最长公共前缀长度"""
        n = min(len(s1), len(s2))
        for i in range(n):
            if s1[i] != s2[i]:
                return i
        return n

    def strip_brackets(name):
        """去掉括号内容"""
        return re.sub(r'[（(][^）)]*[）)]', '', name).strip()

    def is_substring_relation(name1, name2):
        """检查一方是否是另一方的子串（合法别名的典型特征）"""
        c1 = strip_brackets(name1)
        c2 = strip_brackets(name2)
        if len(c1) >= 2 and len(c2) >= 2:
            if c1 in c2 or c2 in c1:
                return True
        return False

    def check_family_references(name_a, name_b, stats_a, stats_b):
        """检查 summaries/relationships 中是否提到对方为家族成员"""
        # 提取名字的短名部分用于搜索
        core_a = strip_brackets(name_a)
        core_b = strip_brackets(name_b)

        texts_a = []
        for field in ('summaries', 'relationships'):
            raw = stats_a.get(field, [])
            for item in (raw or []):
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    texts_a.append(str(item[1]))
                elif isinstance(item, str):
                    texts_a.append(item)
        combined_a = ' '.join(texts_a)

        texts_b = []
        for field in ('summaries', 'relationships'):
            raw = stats_b.get(field, [])
            for item in (raw or []):
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    texts_b.append(str(item[1]))
                elif isinstance(item, str):
                    texts_b.append(item)
        combined_b = ' '.join(texts_b)

        hints = []
        for kw in FAMILY_KEYWORDS:
            # 检查 A 的文本中是否提及 B + 家族关键词
            if core_b and len(core_b) >= 2:
                if core_b in combined_a and kw in combined_a:
                    hints.append(f"{name_a}的描述中提及'{core_b}'和'{kw}'")
                    break
            if core_a and len(core_a) >= 2:
                if core_a in combined_b and kw in combined_b:
                    hints.append(f"{name_b}的描述中提及'{core_a}'和'{kw}'")
                    break
        return hints

    suspicious_pairs = []
    names = list(global_stats.keys())

    for i, name_a in enumerate(names):
        core_a = strip_brackets(name_a)
        if len(core_a) < 2:
            continue

        for name_b in names[i+1:]:
            core_b = strip_brackets(name_b)
            if len(core_b) < 2:
                continue

            # 排除子串关系（合法别名）
            if is_substring_relation(name_a, name_b):
                continue

            # 计算最长公共前缀
            lcp_len = get_lcp_length(core_a, core_b)
            if lcp_len < 2:
                continue

            # 前缀去掉后，两边都要有不同的后缀
            suffix_a = core_a[lcp_len:]
            suffix_b = core_b[lcp_len:]
            if not suffix_a or not suffix_b:
                continue  # 一方是另一方的前缀，应该由子串检查处理

            prefix = core_a[:lcp_len]
            hint = f"同前缀'{prefix}'但后缀不同('{suffix_a}' vs '{suffix_b}')，可能是姐妹/母女/亲属"

            # 额外检查家族关系关键词
            family_hints = check_family_references(
                name_a, name_b,
                global_stats.get(name_a, {}),
                global_stats.get(name_b, {})
            )
            if family_hints:
                hint += "；" + "；".join(family_hints)

            suspicious_pairs.append((name_a, name_b, hint))

    if suspicious_pairs:
        logger.info(f"检测到 {len(suspicious_pairs)} 对同前缀疑似不同人（需更强证据才合并）:")
        for a, b, h in suspicious_pairs[:10]:
            logger.info(f"  - {a} vs {b}: {h[:80]}...")

    return suspicious_pairs


def _clean_contaminated_other_names(global_stats, same_name_prefix_pairs, conflict_pairs):
    """
    预清理 other_names 中的污染条目。在所有合并逻辑之前调用。

    策略A（同姓/辈分冲突对清理）：
      alias 匹配另一个角色 Y 的名字，(X, Y) 是可疑对或冲突对，且两者共现 → 移除

    策略B（独立角色误入清理）：
      alias 精确等于另一个重要独立角色 Y 的名字，且 X 和 Y 文本相似度极低 → 移除
      （如紫夫人的other_names中的"宫主"因剧情关联被误放入）

    修改 global_stats in place。返回移除的条目数。
    """
    # 构建可疑对/冲突对查找集
    blocked_pairs = set()
    for a, b, *_ in same_name_prefix_pairs:
        blocked_pairs.add((a, b))
        blocked_pairs.add((b, a))
    for pair in conflict_pairs:
        if len(pair) >= 2:
            blocked_pairs.add((pair[0], pair[1]))
            blocked_pairs.add((pair[1], pair[0]))

    # 预计算 chunk 索引
    char_chunks = {}
    for name, data in global_stats.items():
        char_chunks[name] = _extract_chunk_indices(data)

    all_char_names = set(global_stats.keys())
    removed_count = 0

    # 提取给名部分的辅助函数（用于子串匹配）
    def get_suffix_after_lcp(name_a, name_b):
        """获取 name_b 去掉与 name_a 共同前缀后的后缀"""
        core_a = re.sub(r'[（(][^）)]*[）)]', '', name_a).strip()
        core_b = re.sub(r'[（(][^）)]*[）)]', '', name_b).strip()
        lcp = 0
        for i in range(min(len(core_a), len(core_b))):
            if core_a[i] != core_b[i]:
                break
            lcp = i + 1
        if lcp >= 2:
            return core_b[lcp:]
        return None

    for name, data in global_stats.items():
        others = data.get("other_names", set())
        if isinstance(others, list):
            others = set(others)
        if not others:
            continue

        to_remove = set()
        my_chunks = char_chunks.get(name, set())

        for alias in list(others):
            # === 策略A：可疑对/冲突对 + 共现 ===
            for other_name in all_char_names:
                if other_name == name:
                    continue
                if (name, other_name) not in blocked_pairs:
                    continue

                other_chunks = char_chunks.get(other_name, set())
                shared_chunks = my_chunks & other_chunks

                # 精确匹配
                if alias == other_name and len(shared_chunks) >= 1:
                    to_remove.add(alias)
                    logger.info(f"清理污染别名[策略A]: {name} 移除 '{alias}'（与{other_name}是可疑对，共现{len(shared_chunks)}次）")
                    break

                # alias 是 other_name 的子串（如"阳乃"匹配"雪之下阳乃"）
                other_core = re.sub(r'[（(][^）)]*[）)]', '', other_name).strip()
                if len(alias) >= 2 and alias in other_core and len(shared_chunks) >= 1:
                    to_remove.add(alias)
                    logger.info(f"清理污染别名[策略A]: {name} 移除 '{alias}'（匹配{other_name}，共现{len(shared_chunks)}次）")
                    break

                # alias 包含 other_name 的给名部分（如"阳乃姐姐"含"阳乃"）
                given_name = get_suffix_after_lcp(name, other_name)
                if given_name and len(given_name) >= 2 and given_name in alias and len(shared_chunks) >= 1:
                    to_remove.add(alias)
                    logger.info(f"清理污染别名[策略A]: {name} 移除 '{alias}'（含{other_name}的给名'{given_name}'，共现{len(shared_chunks)}次）")
                    break

            if alias in to_remove:
                continue

            # === 策略B：独立角色误入（不依赖可疑对或共现） ===
            if alias in all_char_names and alias != name:
                alias_data = global_stats.get(alias, {})
                alias_count = alias_data.get('count', 0)

                # alias 是重要独立角色（count >= 3），且不是当前名字的子串
                alias_core = re.sub(r'[（(][^）)]*[）)]', '', alias).strip()
                name_core = re.sub(r'[（(][^）)]*[）)]', '', name).strip()
                is_substr = (len(alias_core) >= 2 and len(name_core) >= 2 and
                             (alias_core in name_core or name_core in alias_core))

                if alias_count >= 3 and not is_substr:
                    # 计算文本相似度
                    my_texts = _extract_indexed_texts(data.get("summaries", []), limit=30)
                    alias_texts = _extract_indexed_texts(alias_data.get("summaries", []), limit=30)
                    sim = _quick_text_similarity(my_texts, alias_texts)

                    if sim < 0.30:
                        to_remove.add(alias)
                        logger.info(
                            f"清理污染别名[策略B]: {name} 移除 '{alias}'"
                            f"（独立角色count={alias_count}，文本相似度={sim:.2f}）"
                        )

        if to_remove:
            new_others = others - to_remove
            data["other_names"] = new_others
            removed_count += len(to_remove)

    return removed_count


def _call_merge_ai(characters_batch, conflict_pairs, batch_info="", mutual_pairs=None):
    """
    调用 AI 对一批角色进行别称合并分析
    返回 merge_groups 列表
    
    参数:
        mutual_pairs: 检测到的 other_names 互相包含的角色对列表 [(name_a, name_b, reason), ...]
    """
    batch_names = {c['name'] for c in characters_batch}
    
    # 构建辈分冲突提示
    conflict_prompt = ""
    relevant_conflicts = [(p[0], p[1]) for p in conflict_pairs if p[0] in batch_names or p[1] in batch_names]
    if relevant_conflicts:
        conflict_text = "\n".join([f"- {p[0]} 和 {p[1]}（辈分不同，疑似母女/姐妹）" for p in relevant_conflicts])
        conflict_prompt = f"""
【系统检测到的辈分冲突】（绝对不能合并）：
{conflict_text}
"""
    
    # 构建互引提示（强烈建议合并）
    mutual_prompt = ""
    if mutual_pairs:
        # 只显示双方都在本批次中的互引对
        relevant_mutuals = [(a, b, r) for a, b, r in mutual_pairs if a in batch_names and b in batch_names]
        # 也显示单方在本批次中的互引对（作为参考）
        partial_mutuals = [(a, b, r) for a, b, r in mutual_pairs if (a in batch_names or b in batch_names) and not (a in batch_names and b in batch_names)]
        
        if relevant_mutuals:
            mutual_text = "\n".join([f"- 【必须合并】{a} 和 {b}：{r}" for a, b, r in relevant_mutuals])
            mutual_prompt = f"""
【系统检测到的别名互引 - 强制合并】：
以下角色对的 other_names 字段互相包含对方或共享大量别名，这意味着它们几乎肯定是同一个角色的不同称呼。
**除非你发现明确的辈分冲突（如XX太太vs XX小姐），否则必须合并这些角色对！**
{mutual_text}

注意：这些角色有相同的外貌描写、相同的剧情摘要、相同的与男主互动记录，只是名字略有不同。
如果你不合并它们，必须在 rejected_merges 中给出明确的拒绝理由（辈分冲突/身份完全不同等）。
"""
        
        if partial_mutuals:
            partial_text = "\n".join([f"- {a} 和 {b}：{r}" for a, b, r in partial_mutuals[:5]])
            mutual_prompt += f"""
【参考：跨批次互引】（另一方不在本批次，仅供参考）：
{partial_text}
"""
    
    system_prompt = """你是一个专业的小说角色分析专家，专门负责识别同一角色的不同称呼（别称/别名）。

## 你的任务：
分析提供的角色列表，识别哪些名字实际上指的是**同一个女性角色**。

## 【重要】合并规则：

### 应该合并的情况（需综合判断，不能仅凭名字相似）：
1. **名字包含关系 + 特征一致**：
   - 如"十六夜"和"黑崎十六夜"，需确认外貌/身份/与男主关系一致才能合并
   - 如"十六夜（黑崎十六夜）"明确标注了全名，可以合并
2. **带括号的阶段称呼**：如"雪乃（童年）"和"雪乃"是同一人不同时期
3. **昵称/简称**：如"小雪"和"雪之下雪乃"，必须结合外貌/身份/互动判断
4. **外貌+身份+关系**三者一致时可合并

### 【警惕】名字相似但可能是不同人的情况：
1. **男主对不同女性使用相似昵称**：如男主叫A"小雪"，但B的名字里也有"雪"
2. **同姓不同人**：如"黑崎十六夜"和"黑崎美琴"虽同姓但是不同人
3. **常见名字重复**：如"小雪"可能是多个角色的昵称

### 【绝对不能合并】的情况：
1. **辈分称呼不同**：
   - "黑崎太太"（母亲辈）vs "黑崎小姐"（女儿辈）→ 绝对不同人！
   - "XX妈妈/母亲/夫人/太太" vs "XX姐姐/妹妹/小姐" → 绝对不同人！
2. **姓氏相同但身份明显不同**：母女、姐妹、婆媳 → 不同人
3. **年龄/外貌差距明显**：成熟女性 vs 少女

## 判断依据优先级：
1. 【最高】辈分称呼（太太/小姐/妈妈等）→ 不同则绝对不合并
2. 【高】身份背景（职业、家庭角色）→ 不同则不合并
3. 【中】外貌特征一致性 → 必须一致
4. 【参考】名字相似性 → 仅作参考，不能单独作为合并依据！

## 输出格式（JSON）：
{
    "merge_groups": [
        {
            "main_name": "保留的主要名字（选择最正式/最完整的）",
            "aliases": ["别名1", "别名2"],
            "reason": "合并原因（必须说明外貌/身份/关系的一致性证据）"
        }
    ],
    "rejected_merges": [
        {
            "names": ["名字1", "名字2"],
            "reason": "虽然名字相似但不合并的原因"
        }
    ],
    "standalone": ["不需要合并的角色名"]
}

## 注意：
- 宁可漏合不要错合！名字相似不代表是同一人！
- 辈分不同（太太vs小姐）绝对不合并！
- 必须有外貌/身份/关系的证据支持才能合并
- main_name 应选择最正式、最完整的名字

### 【特别警惕】other_names 污染现象：
当两个不同角色在同一章节中一起出现或有剧情关联时，扫描可能会错误地将一个角色的名字放入另一个角色的 other_names。
例如：雪之下雪乃（妹妹）和雪之下阳乃（姐姐）经常一起出场，导致雪乃的 other_names 中出现了"阳乃"。
**判断方法**：
- 如果两人的 summaries 描述的是不同事件/不同行为 → 不是同一人，other_names 是被污染的
- 如果两人有不同的外貌描写 → 不是同一人
- 同姓但名字不同的角色（如雪之下雪乃 vs 雪之下阳乃、桃沢爱 vs 桃沢咲夜）要特别小心，很可能是姐妹/母女！
- 有剧情关联但完全不同的角色（如紫夫人和宫主因男主有交集）不要因 other_names 就合并"""

    user_prompt = f"""请分析以下女性角色列表{batch_info}，识别并合并同一角色的不同称呼：

{json.dumps(characters_batch, ensure_ascii=False, indent=2)}
{conflict_prompt}{mutual_prompt}
请仔细分析每个角色的 other_names（别名列表）、外貌、身份、与男主关系等信息，判断哪些名字是同一个人。

【判断合并的关键步骤】：
1. **首先检查 other_names**：如果 A 的 other_names 包含 B，说明系统已识别出 A 和 B 可能是同一人
2. **再检查 summaries/features/relationships**：如果内容高度相似或完全相同，基本确定是同一人
3. **最后检查辈分冲突**：只有辈分冲突（太太vs小姐）才能阻止合并

【重要提醒】：
1. 对于【系统检测到的别名互引 - 强制合并】中的角色对，除非有明确辈分冲突，否则必须合并！
2. 同姓但辈分称呼不同的（如XX太太和XX小姐）通常是母女，绝对不要合并！
3. 如果两个角色的 summaries/features 内容几乎相同，必须合并！这意味着它们是同一角色的重复记录
4. 名字相似（如"十六夜"和"黑崎十六夜"）加上 other_names 互相包含，应该合并"""

    response = chat_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=8000
    )
    record_usage(response)
    
    content = response.choices[0].message.content.strip()
    content = re.sub(r'```json\s*|\s*```', '', content)
    
    # 尝试解析 JSON，如果失败则尝试修复
    try:
        merge_result = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}，尝试修复...")
        last_brace = max(content.rfind('}'), content.rfind(']'))
        if last_brace > 0:
            truncated = content[:last_brace + 1]
            open_braces = truncated.count('{') - truncated.count('}')
            open_brackets = truncated.count('[') - truncated.count(']')
            truncated += '}' * open_braces + ']' * open_brackets
            try:
                merge_result = json.loads(truncated)
                logger.info("JSON 修复成功")
            except json.JSONDecodeError:
                logger.error(f"JSON 修复失败，原始内容（前500字符）: {content[:500]}...")
                return [], []
        else:
            logger.error(f"无法修复 JSON，原始内容（前500字符）: {content[:500]}...")
            return [], []
    
    merge_groups = merge_result.get('merge_groups', [])
    rejected_merges = merge_result.get('rejected_merges', [])
    
    return merge_groups, rejected_merges


def _judge_single_pair_merge(char_a_info, char_b_info, conflict_pairs=None):
    """
    专门判断单个角色对是否应该合并
    
    参数:
        char_a_info: 角色 A 的详细信息（包括 name, other_names, summaries, features 等）
        char_b_info: 角色 B 的详细信息
        conflict_pairs: 辈分冲突对列表
    
    返回:
        should_merge (bool): 是否应该合并
        main_name (str): 如果合并，哪个作为主名
        reason (str): 判断理由
    """
    name_a = char_a_info.get('name', '')
    name_b = char_b_info.get('name', '')
    
    # 检查辈分冲突
    ELDER_KEYWORDS = {'太太', '夫人', '妈妈', '母亲', '婆婆', '奶奶', '外婆', '阿姨', '姑姑', '姨妈'}
    YOUNGER_KEYWORDS = {'小姐', '姐姐', '妹妹', '姐', '妹', '女儿', '孙女', '外孙女', '学妹'}
    
    has_elder_a = any(kw in name_a for kw in ELDER_KEYWORDS)
    has_younger_a = any(kw in name_a for kw in YOUNGER_KEYWORDS)
    has_elder_b = any(kw in name_b for kw in ELDER_KEYWORDS)
    has_younger_b = any(kw in name_b for kw in YOUNGER_KEYWORDS)
    
    if (has_elder_a and has_younger_b) or (has_elder_b and has_younger_a):
        return False, "", f"辈分冲突：{name_a} vs {name_b}"
    
    for p in (conflict_pairs or []):
        if name_a in p and name_b in p:
            return False, "", f"已检测辈分冲突：{name_a} vs {name_b}"
    
    system_prompt = """你是专业的小说角色分析专家。你的任务是判断两个名字是否指的是**同一个角色**。

## 判断依据（按重要性排序）：
1. **summaries（剧情摘要）**：内容高度相似或完全相同 → 几乎肯定是同一人（最可靠的证据）
2. **features（外貌特征）**：外貌描写一致 → 强烈支持是同一人
3. **other_names（别名列表）**：如果 A 的 other_names 包含 B 的名字，*可能*暗示是同一人，但**注意 other_names 可能被污染**！
   - 当两个不同角色在同一章节中一起出现或有剧情关联时，扫描可能错误地将一个角色的名字放入另一个角色的 other_names
   - 特别是同姓角色（如姐妹、母女），other_names 互含**不能**作为合并的单独依据
4. **relationships（关系类型）**：与男主的关系一致 → 支持是同一人
5. **interactions（互动记录）**：互动内容相似 → 支持是同一人

## 【关键】判断标准：
- 如果 summaries 内容**几乎相同**或大量重叠 → **必须合并**！这说明是系统重复记录
- 如果 other_names 互相包含 + features 一致 → **应该合并**
- 如果名字相似（如"十六夜"和"黑崎十六夜"）+ summaries/features 高度一致 → **应该合并**
- 如果 summaries 描述的是不同事件、features 描述的是不同外貌 → 即使 other_names 互含也**必须拒绝合并**
- 同姓但名字不同的角色（如XX雪乃 vs XX阳乃）需要**特别强**的 summaries/features 一致性证据

## 输出格式（JSON）：
{
    "should_merge": true/false,
    "main_name": "如果合并，选择更正式/完整的名字作为主名",
    "confidence": "high/medium/low",
    "reason": "判断理由（必须说明是基于哪些证据得出结论）"
}

## 注意：
- **不要仅凭 other_names 互含就合并**：other_names 可能被污染
- 必须结合 summaries 和 features 的**独立证据**来判断
- 如果两人 summaries 描述不同事件、features 描述不同外貌，即使 other_names 互含也拒绝合并"""

    user_prompt = f"""请判断以下两个角色是否是同一人：

【角色 A】：{name_a}
{json.dumps(char_a_info, ensure_ascii=False, indent=2)}

【角色 B】：{name_b}
{json.dumps(char_b_info, ensure_ascii=False, indent=2)}

请仔细对比：
1. other_names 是否互相包含？
2. summaries 内容是否高度相似或相同？
3. features 外貌描写是否一致？
4. relationships 与男主关系是否一致？

如果以上有2项以上一致，除非有明确矛盾，否则应该合并。"""

    try:
        response = chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        record_usage(response)
        
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        
        should_merge = result.get('should_merge', False)
        main_name = result.get('main_name', name_a)
        reason = result.get('reason', '未提供理由')
        
        return should_merge, main_name, reason
        
    except Exception as e:
        logger.warning(f"单对判断失败 ({name_a} vs {name_b}): {e}")
        return False, "", f"API 调用失败: {e}"


def _detect_mutual_other_names(global_stats, conflict_pairs=None):
    """
    检测 other_names 互相包含的角色对
    
    规则（放宽版）：
    1. A 的 other_names 包含 B（精确或模糊），且 B 的 other_names 包含 A（精确或模糊）
    2. A 的 other_names 包含 B（精确或模糊），且两者共享别名
    3. 两者共享大量别名（>=3个）
    4. A 的名称是 B 名称的子串（或反过来），且有共享别名
    
    返回：疑似同一角色的候选对列表 [(name_a, name_b, reason), ...]
    """
    if len(global_stats) < 2:
        return []
    
    conflict_pairs = conflict_pairs or []
    
    # 辈分关键词（用于安全检查）
    ELDER_KEYWORDS = {'太太', '夫人', '妈妈', '母亲', '婆婆', '奶奶', '外婆', '阿姨', '姑姑', '姨妈'}
    YOUNGER_KEYWORDS = {'小姐', '姐姐', '妹妹', '姐', '妹', '女儿', '孙女', '外孙女', '学妹'}
    
    def has_generation_conflict(name1, name2):
        """检查两个名字是否有辈分冲突"""
        for p in conflict_pairs:
            if name1 in p and name2 in p:
                return True
        has_elder1 = any(kw in name1 for kw in ELDER_KEYWORDS)
        has_younger1 = any(kw in name1 for kw in YOUNGER_KEYWORDS)
        has_elder2 = any(kw in name2 for kw in ELDER_KEYWORDS)
        has_younger2 = any(kw in name2 for kw in YOUNGER_KEYWORDS)
        return (has_elder1 and has_younger2) or (has_elder2 and has_younger1)
    
    def fuzzy_contains(name, others_set):
        """模糊匹配：检查 name 是否在 others_set 中（精确匹配或子串匹配）"""
        if name in others_set:
            return True, "精确"
        # 去掉括号内容后匹配
        name_core = re.sub(r'[（(][^）)]*[）)]', '', name).strip()
        if name_core and name_core != name:
            if name_core in others_set:
                return True, f"核心名'{name_core}'"
        # 检查 name 是否是某个别名的子串，或某个别名是 name 的子串
        for alias in others_set:
            alias_core = re.sub(r'[（(][^）)]*[）)]', '', alias).strip()
            # 至少3个字才做子串匹配，避免太短的误匹配
            if len(name_core) >= 3 and name_core in alias_core:
                return True, f"子串'{name_core}'在'{alias}'"
            if len(alias_core) >= 3 and alias_core in name_core:
                return True, f"子串'{alias_core}'在'{name}'"
        return False, ""
    
    def name_similarity(name_a, name_b):
        """检查两个名称是否相似（子串关系或核心名相同）"""
        # 去掉括号内容
        core_a = re.sub(r'[（(][^）)]*[）)]', '', name_a).strip()
        core_b = re.sub(r'[（(][^）)]*[）)]', '', name_b).strip()
        # 核心名相同
        if core_a and core_b and core_a == core_b:
            return True, f"核心名相同：'{core_a}'"
        # 子串关系（至少3字）
        if len(core_a) >= 3 and len(core_b) >= 3:
            if core_a in core_b or core_b in core_a:
                return True, f"名称子串关系：'{core_a}' vs '{core_b}'"
        return False, ""
    
    # 构建 name -> other_names 映射
    name_to_others = {}
    for name, data in global_stats.items():
        others = set(data.get('other_names', set()))
        if isinstance(others, list):
            others = set(others)
        name_to_others[name] = others
    
    # 检测互相引用的角色对
    mutual_pairs = []
    names = list(global_stats.keys())
    for i, name_a in enumerate(names):
        others_a = name_to_others.get(name_a, set())
        for name_b in names[i+1:]:
            others_b = name_to_others.get(name_b, set())
            
            # 检查是否互相包含（支持模糊匹配）
            a_contains_b, a_match_type = fuzzy_contains(name_b, others_a)
            b_contains_a, b_match_type = fuzzy_contains(name_a, others_b)
            
            # 共享别名
            shared_aliases = others_a & others_b
            
            # 名称相似性检查
            names_similar, similarity_reason = name_similarity(name_a, name_b)
            
            # 构建原因说明
            reasons = []
            should_merge = False
            
            # 条件1：双向互引（精确或模糊）
            if a_contains_b and b_contains_a:
                reasons.append(f"双向互引：{name_a}的别名包含{name_b}({a_match_type})，{name_b}的别名包含{name_a}({b_match_type})")
                should_merge = True
            
            # 条件2：单向互引 + 共享别名
            elif a_contains_b and shared_aliases:
                reasons.append(f"{name_a}的别名包含{name_b}({a_match_type})")
                reasons.append(f"共享别名：{list(shared_aliases)[:3]}")
                should_merge = True
            elif b_contains_a and shared_aliases:
                reasons.append(f"{name_b}的别名包含{name_a}({b_match_type})")
                reasons.append(f"共享别名：{list(shared_aliases)[:3]}")
                should_merge = True
            
            # 条件3：大量共享别名（>=3个）
            elif len(shared_aliases) >= 3:
                reasons.append(f"大量共享别名({len(shared_aliases)}个)：{list(shared_aliases)[:5]}")
                should_merge = True
            
            # 条件4：名称相似 + 有共享别名
            elif names_similar and shared_aliases:
                reasons.append(similarity_reason)
                reasons.append(f"共享别名：{list(shared_aliases)[:3]}")
                should_merge = True
            
            # 条件5：单向互引 + 名称相似
            elif (a_contains_b or b_contains_a) and names_similar:
                if a_contains_b:
                    reasons.append(f"{name_a}的别名包含{name_b}({a_match_type})")
                else:
                    reasons.append(f"{name_b}的别名包含{name_a}({b_match_type})")
                reasons.append(similarity_reason)
                should_merge = True
            
            if should_merge:
                # 新增：共现否决 — 频繁共现的角色不太可能是同一人
                a_chunks = _extract_chunk_indices(global_stats.get(name_a, {}))
                b_chunks = _extract_chunk_indices(global_stats.get(name_b, {}))
                shared = a_chunks & b_chunks
                if a_chunks and b_chunks:
                    smaller = min(len(a_chunks), len(b_chunks))
                    ratio = len(shared) / smaller if smaller > 0 else 0
                    if len(shared) >= 3 and ratio >= 0.15:
                        logger.info(f"互引检测跳过: {name_a} 和 {name_b}（高共现{len(shared)}个chunk，比率={ratio:.1%}）")
                        continue

                # 安全检查：辈分冲突
                if has_generation_conflict(name_a, name_b):
                    logger.info(f"互引检测跳过: {name_a} 和 {name_b}（辈分冲突）")
                    continue
                mutual_pairs.append((name_a, name_b, '；'.join(reasons)))
    
    if mutual_pairs:
        logger.info(f"检测到 {len(mutual_pairs)} 对疑似同一角色的候选，将交给大模型判断")
        for a, b, reason in mutual_pairs[:10]:  # 打印前10对
            logger.info(f"  - {a} <-> {b}: {reason[:80]}...")
    
    return mutual_pairs


def merge_aliases(global_stats):
    """
    使用 AI 进行别称合并
    这是一个关键步骤，专门用于识别和合并同一角色的不同称呼
    名字包含关系完全交给大模型判断，避免简单规则导致的误合并
    支持分批处理：当角色数量超过阈值时，分批调用 AI
    """
    if len(global_stats) < 2:
        return global_stats
    
    # 第一步：检测辈分冲突对（用于后续二次验证）
    conflict_pairs = _get_generation_conflict_pairs(global_stats)
    if conflict_pairs:
        logger.info(f"检测到辈分冲突对（不应合并）: {conflict_pairs}")

    # 第一步（新增）：检测同前缀可疑对
    same_name_prefix_pairs = _detect_same_name_prefix_pairs(global_stats)

    # 第一步（新增）：预清理被污染的 other_names
    cleaned_count = _clean_contaminated_other_names(global_stats, same_name_prefix_pairs, conflict_pairs)
    if cleaned_count:
        logger.info(f"预清理完成: 共移除 {cleaned_count} 个疑似污染的别名条目")
    
    # 准备角色信息供 AI 分析（第一、二轮不使用互引提示，留到第三轮单独处理）
    characters_info = []
    for name, data in global_stats.items():
        avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
        char_info = {
            "name": name,
            "avg_score": round(avg_score, 1),
            "count": data['count'],
            "other_names": list(data.get('other_names', set())),
            "appearances": data.get('appearances', [])[:3],
            "features": data.get('features', [])[:3],
            "relationships": data.get('relationships', [])[:3],
            "summaries": data.get('summaries', [])[:3]
        }
        characters_info.append(char_info)
    
    # 按出现次数排序，便于 AI 判断
    characters_info.sort(key=lambda x: x['count'], reverse=True)
    
    # 分批处理阈值
    BATCH_SIZE = 30
    all_merge_groups = []
    
    try:
        if len(characters_info) <= BATCH_SIZE:
            # 角色数量不多，单次调用
            logger.info(f"正在进行别称识别与合并（{len(characters_info)} 个角色）...")
            merge_groups, rejected_merges = _call_merge_ai(characters_info, conflict_pairs, mutual_pairs=None)
            all_merge_groups.extend(merge_groups)
            
            # 记录被拒绝的合并
            if rejected_merges:
                for rej in rejected_merges:
                    logger.info(f"拒绝合并: {rej.get('names')} - {rej.get('reason')}")
        else:
            # 分批处理（多线程并行）
            total_batches = (len(characters_info) + BATCH_SIZE - 1) // BATCH_SIZE
            logger.info(f"角色数量较多（{len(characters_info)}），将分 {total_batches} 批并行处理...")
            
            # 准备所有批次
            batches = []
            for i in range(0, len(characters_info), BATCH_SIZE):
                batch = characters_info[i:i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                batch_info = f"（第 {batch_num}/{total_batches} 批）"
                batches.append((batch, batch_num, batch_info))
            
            # 定义单批次处理函数
            def process_batch(batch_data):
                batch, batch_num, batch_info = batch_data
                try:
                    merge_groups, rejected_merges = _call_merge_ai(batch, conflict_pairs, batch_info, mutual_pairs=None)
                    return batch_num, merge_groups, rejected_merges, None
                except Exception as e:
                    return batch_num, [], [], str(e)
            
            # 第一轮：多线程并行分批内部合并
            merge_results_lock = threading.Lock()
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_batch = {
                    executor.submit(process_batch, batch_data): batch_data[1]
                    for batch_data in batches
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_batch),
                                   total=len(batches), desc="别称合并分批", unit="批"):
                    batch_num = future_to_batch[future]
                    try:
                        result_batch_num, merge_groups, rejected_merges, error = future.result()
                        if error:
                            logger.warning(f"第 {result_batch_num} 批处理失败: {error}")
                        else:
                            with merge_results_lock:
                                all_merge_groups.extend(merge_groups)
                            if rejected_merges:
                                for rej in rejected_merges:
                                    logger.info(f"拒绝合并: {rej.get('names')} - {rej.get('reason')}")
                    except Exception as e:
                        logger.warning(f"第 {batch_num} 批处理异常: {e}")
            
            logger.info(f"第一轮分批处理完成，共收集到 {len(all_merge_groups)} 个合并组")
            
            # 第二轮：跨批次合并检查
            # 将第一轮合并后的主名字汇总，检查是否有跨批次的遗漏合并
            logger.info("开始第二轮跨批次合并检查...")
            
            # 构建主名字到合并组的映射
            main_names_info = []
            for group in all_merge_groups:
                main_name = group.get('main_name', '')
                if not main_name or main_name not in global_stats:
                    continue
                # 获取该角色的特征信息
                data = global_stats[main_name]
                avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
                # 合并别名的特征
                all_features = list(data.get('features', []))[:3]
                all_appearances = list(data.get('appearances', []))[:3]
                all_other_names = list(data.get('other_names', set()))
                for alias in group.get('aliases', []):
                    if alias in global_stats:
                        alias_data = global_stats[alias]
                        all_features.extend(alias_data.get('features', [])[:2])
                        all_appearances.extend(alias_data.get('appearances', [])[:2])
                        all_other_names.extend(list(alias_data.get('other_names', set())))
                
                main_names_info.append({
                    "name": main_name,
                    "aliases": group.get('aliases', []),
                    "avg_score": round(avg_score, 1),
                    "count": data['count'],
                    "other_names": list(set(all_other_names))[:10],  # 添加 other_names
                    "features": all_features[:5],
                    "appearances": all_appearances[:5],
                    "relationships": data.get('relationships', [])[:3]
                })
            
            # 添加未被合并的高频角色
            merged_in_round1 = set()
            for group in all_merge_groups:
                merged_in_round1.add(group.get('main_name', ''))
                for alias in group.get('aliases', []):
                    merged_in_round1.add(alias)
            
            for char_info in characters_info:
                char_name = char_info['name']
                if char_name not in merged_in_round1 and char_info['count'] >= 5:
                    main_names_info.append(char_info)
            
            if len(main_names_info) > 1:
                try:
                    cross_merge_groups, _ = _call_merge_ai(main_names_info, conflict_pairs, "（跨批次检查）", mutual_pairs=None)
                    if cross_merge_groups:
                        logger.info(f"跨批次检查发现 {len(cross_merge_groups)} 个额外合并组")
                        # 合并跨批次结果
                        for cross_group in cross_merge_groups:
                            cross_main = cross_group.get('main_name', '')
                            cross_aliases = cross_group.get('aliases', [])
                            
                            # 查找是否已有包含这个主名的合并组
                            found = False
                            for existing_group in all_merge_groups:
                                if existing_group.get('main_name') == cross_main:
                                    # 添加新别名
                                    for alias in cross_aliases:
                                        if alias not in existing_group.get('aliases', []):
                                            existing_group['aliases'].append(alias)
                                    found = True
                                    break
                                elif cross_main in existing_group.get('aliases', []):
                                    # cross_main 是某个组的别名，需要合并
                                    for alias in cross_aliases:
                                        if alias not in existing_group.get('aliases', []) and alias != existing_group.get('main_name'):
                                            existing_group['aliases'].append(alias)
                                    found = True
                                    break
                            
                            if not found:
                                all_merge_groups.append(cross_group)
                except Exception as e:
                    logger.warning(f"跨批次检查失败: {e}")
            
            logger.info(f"两轮合并完成，最终收集到 {len(all_merge_groups)} 个合并组")
        
        if not all_merge_groups:
            logger.info("没有发现需要合并的别称")
            return global_stats
        
        # 辈分关键词（用于二次验证）
        ELDER_KEYWORDS = {'太太', '夫人', '妈妈', '母亲', '婆婆', '奶奶', '外婆', '阿姨', '姑姑', '姨妈'}
        YOUNGER_KEYWORDS = {'小姐', '姐姐', '妹妹', '姐', '妹', '女儿', '孙女', '外孙女', '学妹'}
        
        def has_generation_conflict(name1, name2):
            """检查两个名字是否有辈分冲突"""
            for p in conflict_pairs:
                if (name1 in p and name2 in p):
                    return True
            has_elder1 = any(kw in name1 for kw in ELDER_KEYWORDS)
            has_younger1 = any(kw in name1 for kw in YOUNGER_KEYWORDS)
            has_elder2 = any(kw in name2 for kw in ELDER_KEYWORDS)
            has_younger2 = any(kw in name2 for kw in YOUNGER_KEYWORDS)
            return (has_elder1 and has_younger2) or (has_elder2 and has_younger1)
        
        # 执行合并
        merged_stats = {}
        merged_names = set()
        
        for group in all_merge_groups:
            main_name = group.get('main_name', '')
            aliases = group.get('aliases', [])
            
            if not main_name:
                continue
            
            # 二次验证：过滤掉有辈分冲突的别名
            valid_aliases = []
            for alias in aliases:
                if has_generation_conflict(main_name, alias):
                    logger.warning(f"安全检查拒绝合并: {main_name} 和 {alias}（辈分冲突）")
                else:
                    valid_aliases.append(alias)
            aliases = valid_aliases

            # 三次验证：缺少证据的合并直接拒绝（降低“明明两个人却被合并”）
            evidence_ok_aliases = []
            for alias in aliases:
                if alias not in global_stats or main_name not in global_stats:
                    continue
                ok, reason = _should_accept_merge_pair(global_stats, main_name, alias, conflict_pairs=conflict_pairs, same_name_prefix_pairs=same_name_prefix_pairs)
                if ok:
                    evidence_ok_aliases.append(alias)
                else:
                    logger.warning(f"证据校验拒绝合并: {main_name} <- {alias}（{reason}）")
            aliases = evidence_ok_aliases
            
            if not aliases:
                continue
            
            logger.info(f"合并角色: {main_name} <- {aliases}")
            
            # 初始化主名字的数据
            if main_name in global_stats:
                merged_stats[main_name] = {
                    "total_score": global_stats[main_name]["total_score"],
                    "count": global_stats[main_name]["count"],
                    "chunk_scores": list(global_stats[main_name].get("chunk_scores", [])),
                    "summaries": list(global_stats[main_name].get("summaries", [])),
                    "types": set(global_stats[main_name].get("types", [])),
                    "other_names": set(global_stats[main_name].get("other_names", set())),
                    "appearances": list(global_stats[main_name].get("appearances", [])),
                    "features": list(global_stats[main_name].get("features", [])),
                    "relationships": list(global_stats[main_name].get("relationships", [])),
                    "interactions": list(global_stats[main_name].get("interactions", [])),
                    "emotion_signals": list(global_stats[main_name].get("emotion_signals", []))
                }
            else:
                merged_stats[main_name] = {
                    "total_score": 0,
                    "count": 0,
                    "chunk_scores": [],
                    "summaries": [],
                    "types": set(),
                    "other_names": set(),
                    "appearances": [],
                    "features": [],
                    "relationships": [],
                    "interactions": [],
                    "emotion_signals": []
                }
            
            merged_names.add(main_name)
            
            # 合并别名数据
            for alias in aliases:
                if alias in global_stats and alias != main_name:
                    alias_data = global_stats[alias]
                    merged_stats[main_name]["total_score"] += alias_data.get("total_score", 0)
                    merged_stats[main_name]["count"] += alias_data.get("count", 0)
                    merged_stats[main_name]["chunk_scores"].extend(alias_data.get("chunk_scores", []))
                    merged_stats[main_name]["summaries"].extend(alias_data.get("summaries", []))
                    merged_stats[main_name]["types"].update(alias_data.get("types", []))
                    merged_stats[main_name]["other_names"].add(alias)
                    merged_stats[main_name]["other_names"].update(alias_data.get("other_names", set()))
                    merged_stats[main_name]["appearances"].extend(alias_data.get("appearances", []))
                    merged_stats[main_name]["features"].extend(alias_data.get("features", []))
                    merged_stats[main_name]["relationships"].extend(alias_data.get("relationships", []))
                    merged_stats[main_name]["interactions"].extend(alias_data.get("interactions", []))
                    merged_stats[main_name]["emotion_signals"].extend(alias_data.get("emotion_signals", []))
                    merged_names.add(alias)
        
        # 添加未被合并的角色
        for name, data in global_stats.items():
            if name not in merged_names:
                merged_stats[name] = data
        
        logger.info(f"前两轮合并完成: {len(global_stats)} 个名字 -> {len(merged_stats)} 个角色")
        
        # ========== 第三阶段：单独判断互引对 ==========
        logger.info("开始第三阶段：单独判断仍未合并的互引角色对...")
        
        # 从合并后的 merged_stats 重新检测互引对
        final_mutual_pairs = _detect_mutual_other_names(merged_stats, conflict_pairs)
        
        if final_mutual_pairs:
            logger.info(f"检测到 {len(final_mutual_pairs)} 对仍未合并的互引角色，将逐对判断")
            
            # 准备详细信息函数
            def prepare_char_info(name, stats):
                """从 merged_stats 准备角色的完整信息"""
                avg_score = stats['total_score'] / stats['count'] if stats['count'] > 0 else 0
                
                # 提取 summaries 文本（处理 tuple 格式）
                summaries = stats.get('summaries', [])
                if summaries and isinstance(summaries[0], (tuple, list)) and len(summaries[0]) == 2:
                    summaries = [s[1] for s in summaries[:10]]
                else:
                    summaries = summaries[:10]
                
                # 提取 interactions 文本
                interactions = stats.get('interactions', [])
                if interactions and isinstance(interactions[0], (tuple, list)) and len(interactions[0]) == 2:
                    interactions = [s[1] for s in interactions[:10]]
                else:
                    interactions = interactions[:10]
                
                return {
                    "name": name,
                    "avg_score": round(avg_score, 1),
                    "count": stats['count'],
                    "other_names": list(stats.get('other_names', set()))[:10],
                    "summaries": summaries,
                    "features": stats.get('features', [])[:10],
                    "relationships": stats.get('relationships', [])[:10],
                    "interactions": interactions,
                }
            
            # 逐对判断（并行处理）
            additional_merges = []
            
            def judge_pair(pair_data):
                name_a, name_b, reason = pair_data
                if name_a not in merged_stats or name_b not in merged_stats:
                    return None  # 已被合并
                
                char_a_info = prepare_char_info(name_a, merged_stats[name_a])
                char_b_info = prepare_char_info(name_b, merged_stats[name_b])
                
                should_merge, main_name, judge_reason = _judge_single_pair_merge(
                    char_a_info, char_b_info, conflict_pairs
                )
                
                if should_merge and main_name:
                    return (main_name, [n for n in [name_a, name_b] if n != main_name], judge_reason)
                else:
                    logger.info(f"拒绝合并: {name_a} 和 {name_b} - {judge_reason}")
                    return None
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(judge_pair, pair): pair 
                    for pair in final_mutual_pairs
                }
                
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                   total=len(futures), desc="单对判断", unit="对"):
                    try:
                        result = future.result()
                        if result:
                            additional_merges.append(result)
                    except Exception as e:
                        pair = futures[future]
                        logger.warning(f"判断失败 ({pair[0]} vs {pair[1]}): {e}")
            
            # 应用额外的合并
            if additional_merges:
                logger.info(f"第三阶段发现 {len(additional_merges)} 对需要合并的角色")
                for main_name, aliases_to_merge, reason in additional_merges:
                    logger.info(f"合并角色: {main_name} <- {aliases_to_merge} ({reason[:50]})")
                    
                    # 确保 main_name 在 merged_stats 中
                    if main_name not in merged_stats:
                        # 使用第一个别名的数据作为主数据
                        if aliases_to_merge and aliases_to_merge[0] in merged_stats:
                            merged_stats[main_name] = merged_stats[aliases_to_merge[0]]
                        else:
                            continue
                    
                    # 合并别名数据
                    for alias in aliases_to_merge:
                        if alias in merged_stats and alias != main_name:
                            alias_data = merged_stats[alias]
                            merged_stats[main_name]["total_score"] += alias_data.get("total_score", 0)
                            merged_stats[main_name]["count"] += alias_data.get("count", 0)
                            merged_stats[main_name].setdefault("chunk_scores", []).extend(alias_data.get("chunk_scores", []))
                            merged_stats[main_name]["summaries"].extend(alias_data.get("summaries", []))
                            merged_stats[main_name]["types"].update(alias_data.get("types", set()))
                            merged_stats[main_name]["other_names"].add(alias)
                            merged_stats[main_name]["other_names"].update(alias_data.get("other_names", set()))
                            merged_stats[main_name]["appearances"].extend(alias_data.get("appearances", []))
                            merged_stats[main_name]["features"].extend(alias_data.get("features", []))
                            merged_stats[main_name]["relationships"].extend(alias_data.get("relationships", []))
                            merged_stats[main_name]["interactions"].extend(alias_data.get("interactions", []))
                            merged_stats[main_name]["emotion_signals"].extend(alias_data.get("emotion_signals", []))
                            # 删除已合并的别名
                            del merged_stats[alias]
                
                logger.info(f"第三阶段合并完成: {len(merged_stats)} 个角色（-{len(additional_merges)}）")
        else:
            logger.info("第三阶段：未发现需要额外处理的互引对")

        # ========== 第四阶段：错别字/近似名的确定性合并（避免同一女主被拆成两人） ==========
        # 仅在“编辑距离很近 + 摘要/互动高度相似”时才合并，避免误合并。
        logger.info("开始第四阶段：错别字/近似名的确定性合并检查...")
        try:
            names = list(merged_stats.keys())
            # 只对较短名字做近似检测（中文/日文常见 2~8 字），外文长名交给证据/AI
            candidates = [n for n in names if 2 <= len(n) <= 8]
            # 按出场频次优先（减少两两比较成本）
            candidates.sort(key=lambda n: merged_stats.get(n, {}).get("count", 0), reverse=True)

            merged_any = 0
            seen = set()
            for i, a in enumerate(candidates):
                if a not in merged_stats or a in seen:
                    continue
                for b in candidates[i + 1 :]:
                    if b not in merged_stats or b in seen:
                        continue
                    # 快速过滤：长度差过大跳过
                    if abs(len(a) - len(b)) > 1:
                        continue
                    dist = _levenshtein_distance(a, b, max_dist=2)
                    if dist > 1:
                        continue
                    # 证据校验：必须过阈值
                    # 选择 count 更高的为主名
                    main = a if merged_stats[a].get("count", 0) >= merged_stats[b].get("count", 0) else b
                    alias = b if main == a else a
                    ok, reason = _should_accept_merge_pair(merged_stats, main, alias, conflict_pairs=conflict_pairs, same_name_prefix_pairs=same_name_prefix_pairs)
                    if not ok:
                        continue

                    logger.info(f"近似名合并: {main} <- {alias}（{reason}）")
                    # 执行合并（字段结构同前面第三阶段）
                    alias_data = merged_stats[alias]
                    merged_stats[main]["total_score"] += alias_data.get("total_score", 0)
                    merged_stats[main]["count"] += alias_data.get("count", 0)
                    merged_stats[main].setdefault("chunk_scores", []).extend(alias_data.get("chunk_scores", []))
                    merged_stats[main]["summaries"].extend(alias_data.get("summaries", []))
                    merged_stats[main]["types"].update(alias_data.get("types", set()))
                    merged_stats[main]["other_names"].add(alias)
                    merged_stats[main]["other_names"].update(alias_data.get("other_names", set()))
                    merged_stats[main]["appearances"].extend(alias_data.get("appearances", []))
                    merged_stats[main]["features"].extend(alias_data.get("features", []))
                    merged_stats[main]["relationships"].extend(alias_data.get("relationships", []))
                    merged_stats[main]["interactions"].extend(alias_data.get("interactions", []))
                    merged_stats[main]["emotion_signals"].extend(alias_data.get("emotion_signals", []))
                    del merged_stats[alias]
                    seen.add(alias)
                    merged_any += 1

            if merged_any:
                logger.info(f"第四阶段完成：近似名额外合并 {merged_any} 次")
            else:
                logger.info("第四阶段完成：未发现满足阈值的近似名合并")
        except Exception as e:
            logger.warning(f"第四阶段近似名合并检查失败: {e}")
        
        logger.info(f"全部合并完成: {len(global_stats)} 个名字 -> {len(merged_stats)} 个角色")
        return merged_stats
        
    except Exception as e:
        logger.error(f"别称合并失败: {e}")
        return global_stats


def identify_heroines(merged_stats):
    """
    最终识别女主角（逐个判断模式）
    对每个候选角色单独调用 AI 判断是否为女主，提供更多上下文信息
    """
    if not merged_stats:
        return {"heroines": [], "analysis": "未发现女性角色"}
    
    # 使用全部证据，不再做均衡抽取
    def _flatten_ordered(raw_list):
        """将带 chunk_index 的列表按时间排序，返回纯文本列表"""
        if not raw_list:
            return []
        first = raw_list[0]
        if isinstance(first, (tuple, list)) and len(first) == 2 and isinstance(first[0], (int, float)):
            return [item[1] for item in sorted(raw_list, key=lambda x: x[0])]
        return raw_list
    
    # HSI：统计学提示（别称合并后计算一次，用于给大模型做先验提示）
    hsi_map, hsi_global = _prepare_hsi_hints(merged_stats)
    try:
        logger.info(
            f"HSI 统计学提示：total={hsi_global.get('total')} | strong_k={hsi_global.get('strong_k')} | threshold≈{hsi_global.get('threshold'):.3f}"
        )
    except Exception:
        pass

    # 计算每个角色的综合得分，准备详细信息
    scored_chars = []
    for name, data in merged_stats.items():
        avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
        frequency_bonus = min(data['count'] * 0.1, 2)
        final_score = avg_score + frequency_bonus
        
        # 使用全量证据（按时间排序，但不抽样）
        full_summaries = _flatten_ordered(data.get('summaries', []))
        full_interactions = _flatten_ordered(data.get('interactions', []))
        full_emotions = _flatten_ordered(data.get('emotion_signals', []))
        
        hsi_hint = hsi_map.get(name, {})
        scored_chars.append({
            "name": name,
            "avg_score": avg_score,
            "count": data['count'],
            "final_score": final_score,
            "chunk_scores": data.get('chunk_scores', []),
            "other_names": list(data.get('other_names', set())),
            "summaries": full_summaries,
            "features": data.get('features', []),
            "appearances": data.get('appearances', []),
            "relationships": data.get('relationships', []),
            "interactions": full_interactions,
            "emotion_signals": full_emotions,
            # ===== 统计学提示字段（供 _judge_single_character 的提示词使用）=====
            "hsi": hsi_hint.get("hsi"),
            "hsi_rank": hsi_hint.get("rank"),
            "hsi_total": hsi_hint.get("total"),
            "hsi_hits": hsi_hint.get("hits"),
            "hsi_density": hsi_hint.get("density"),
            "hsi_text_kb": hsi_hint.get("text_kb"),
            "hsi_threshold": hsi_hint.get("threshold", hsi_global.get("threshold")),
            "hsi_strong_k": hsi_hint.get("strong_k", hsi_global.get("strong_k")),
        })
    
    # 按综合得分排序
    scored_chars.sort(key=lambda x: x['final_score'], reverse=True)

    # 为每个候选角色计算证据档案（数据量上下文、评分分布/趋势、关键证据）
    for char in scored_chars:
        char["_evidence_profile"] = _compute_evidence_profile(
            merged_stats.get(char["name"], {}),
            merged_stats
        )

    # 预筛选：先用传统分数做基础候选，再用 HSI “强候选”兜底（避免漏召回）
    base_candidates = [c for c in scored_chars if c['avg_score'] >= 3.5]
    hsi_threshold = hsi_global.get("threshold", None)
    if hsi_threshold is not None:
        hsi_candidates = [c for c in scored_chars if (c.get("hsi") is not None and c.get("hsi") >= hsi_threshold)]
    else:
        hsi_candidates = []

    # 合并去重（按 name）
    _seen = set()
    potential_heroines = []
    for c in (hsi_candidates + base_candidates):
        n = c.get("name")
        if not n or n in _seen:
            continue
        _seen.add(n)
        potential_heroines.append(c)
    
    # 如果候选太多，提高阈值但保留高频出场/高HSI的角色
    if len(potential_heroines) > 80:
        # 注意：这里依然保留 HSI 强候选，避免“高HSI但avg不够”被剪掉
        trimmed = [c for c in scored_chars if c['avg_score'] >= 5 or c['count'] >= 17]
        if hsi_threshold is not None:
            trimmed_hsi = [c for c in scored_chars if (c.get("hsi") is not None and c.get("hsi") >= hsi_threshold)]
        else:
            trimmed_hsi = []
        _seen = set()
        potential_heroines = []
        for c in (trimmed_hsi + trimmed):
            n = c.get("name")
            if not n or n in _seen:
                continue
            _seen.add(n)
            potential_heroines.append(c)
    
    # 不再分批筛选，直接对每个候选人进行完整的一对一判断
    # 这样可以避免分批筛选时遗漏有明确感情互动的角色
    logger.info(f"预筛选后有 {len(potential_heroines)} 个女主候选人，开始逐个完整判断...")
    
    # 逐个判断每个角色是否为女主（并行处理）
    confirmed_heroines = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_char = {
            executor.submit(_judge_single_character, char, char.get("_evidence_profile")): char
            for char in potential_heroines
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_char), 
                          total=len(potential_heroines), desc="逐个判断女主", unit="人"):
            char = future_to_char[future]
            try:
                result = future.result()
                if result and result.get('is_heroine'):
                    confirmed_heroines.append(result)
            except Exception as e:
                logger.warning(f"判断角色 {char['name']} 时出错: {e}")
    
    logger.info(f"逐个判断完成，确认 {len(confirmed_heroines)} 位女主角")
    
    # 最终排序和汇总分析
    if confirmed_heroines:
        return _final_ranking_and_analysis(confirmed_heroines)
    else:
        return {"heroines": [], "analysis": "未发现符合条件的女主角"}


def _identify_heroine_merge_candidates(heroines_list):
    """
    让大模型识别可能需要合并的女主候选组
    
    参数:
        heroines_list: 女主列表，每个女主包含 name, other_names, features, relationships 等
    
    返回:
        merge_groups: 可能需要合并的女主组列表，如 [["白发少女", "西拉"], ["雪宫主", "伊始神宫宫主"]]
    """
    if len(heroines_list) < 2:
        return []
    
    # 准备简化的女主信息（只包含关键字段）
    simplified_heroines = []
    for h in heroines_list:
        simplified_heroines.append({
            "name": h.get('name', ''),
            "other_names": h.get('other_names', [])[:10],
            "features": h.get('features', [])[:5],
            "relationships": h.get('relationships', [])[:5],
            "count": h.get('count', 0)
        })
    
    system_prompt = """你是专业的小说角色分析专家。你的任务是识别哪些女主角可能是**同一个人的不同称呼**。

## 判断依据（按重要性排序）：
1. **other_names（别名列表）**：如果 A 的别名包含 B 的名字，或 B 的别名包含 A 的名字 → 强烈暗示是同一人
2. **features（外貌特征）**：外貌描写高度一致（如都是"银发红瞳"、"白发少女"等）→ 可能是同一人
3. **relationships（与男主关系）**：关系类型一致（如都是"未婚妻"、"师姐"等）→ 支持是同一人
4. **名称相似性**：一个名字是另一个的子串（如"十六夜"和"黑崎十六夜"）→ 可能是同一人

## 【关键】判断标准：
- 如果 features 外貌描写**几乎相同** + other_names 有重叠 → **应该列为候选**
- 如果名字相似 + features 一致 → **应该列为候选**
- 如果 other_names 互相包含 → **应该列为候选**
- 宁可多列几组候选，也不要遗漏潜在的合并机会

## 输出格式（JSON）：
{
    "merge_groups": [
        {
            "names": ["角色A", "角色B"],
            "reason": "为什么这些角色可能是同一人"
        },
        {
            "names": ["角色C", "角色D", "角色E"],
            "reason": "支持多合一的情况"
        }
    ]
}

## 注意：
- 每个合并组的 names 数组可以包含 2 个或更多名字（支持多合一）
- 只输出**有合理依据**的候选组，不要强行合并明显不同的角色
- 如果没有发现任何需要合并的候选，返回空数组"""

    user_prompt = f"""以下是已识别的女主角列表，请分析哪些可能是同一人的不同称呼：

{json.dumps(simplified_heroines, ensure_ascii=False, indent=2)}

请仔细对比：
1. other_names 是否有重叠或互相包含？
2. features 外貌特征是否高度一致？
3. relationships 与男主的关系是否一致？
4. 名字是否相似（如子串关系）？

请列出所有可能需要合并的女主组。"""

    try:
        response = chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        record_usage(response)
        
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        
        merge_groups_raw = result.get('merge_groups', [])
        
        # 转换为简单的列表格式
        merge_groups = []
        for group in merge_groups_raw:
            names = group.get('names', [])
            reason = group.get('reason', '')
            if len(names) >= 2:
                merge_groups.append((names, reason))
                logger.info(f"识别到候选合并组: {names} - {reason[:60]}...")
        
        return merge_groups
        
    except Exception as e:
        logger.warning(f"识别女主合并候选失败: {e}")
        return []


def _judge_heroine_group_merge(heroines_data_list):
    """
    判断一组女主是否应该合并为一个
    
    参数:
        heroines_data_list: 女主详细数据列表（包含完整的 summaries, features, interactions 等）
    
    返回:
        should_merge (bool): 是否应该合并
        main_name (str): 如果合并，哪个作为主名
        reason (str): 判断理由
    """
    if len(heroines_data_list) < 2:
        return False, "", "只有一个角色，无需合并"
    
    names = [h.get('name', '') for h in heroines_data_list]
    
    system_prompt = """你是专业的小说角色分析专家。你的任务是判断一组女主角是否是**同一个人**。

## 【最高优先级】辈分冲突 = 绝对不同人！
在判断任何其他证据之前，**必须先检查辈分冲突**：
- 同姓但辈分称呼不同（如"XX太太/XX夫人" vs "XX小姐/XX+名字"）→ **绝对不是同一人**（这是母女/婆媳/姑侄关系），**必须拒绝合并**！
- 一方是"母亲/太太/夫人/妈妈/阿姨"，另一方是其"女儿/晚辈"→ **绝对不合并**！
- 即使 summaries 中出现相似的事件（如共同参加同一场景），只要辈分不同就是不同角色！
- 例如：一条太太（母亲）和一条郁子（女儿）虽都与男主互动，但绝对是两个人！

## 判断依据（按重要性排序）：
1. **辈分冲突检查**：一旦检测到辈分差异 → **立即拒绝合并**，不看其他证据
2. **summaries（剧情摘要）**：内容高度相似或完全相同 → 几乎肯定是同一人
3. **features（外貌特征）**：外貌描写完全一致 → 很可能是同一人
4. **other_names（别名列表）**：互相包含对方的名字 → 强烈暗示是同一人
5. **relationships（与男主关系）**：关系类型一致 → 支持是同一人
6. **interactions（互动记录）**：互动内容相似且不矛盾 → 支持是同一人

## 【关键】判断标准：
- 如果 summaries 内容**大量重叠**或**描述同一事件** → **必须合并**！
- 如果 features 外貌描写**完全一致** + other_names 互相包含 → **应该合并**
- 如果名字相似（如"十六夜"和"黑崎十六夜"）+ features/summaries 一致 → **应该合并**
- 在发现**明确矛盾**时必须拒绝合并：
  - **辈分不同**（XX太太 vs XX小姐/XX+名字 = 母女关系）→ 绝对不合并！
  - 外貌完全不同（如一个黑发一个白发，且描述详细）
  - 身份明显矛盾（如一个是公主一个是平民，且故事线不同）
  - summaries 描述了完全不同的故事情节

## 输出格式（JSON）：
{
    "should_merge": true/false,
    "main_name": "如果合并，选择最正式/最常用的名字作为主名",
    "confidence": "high/medium/low",
    "reason": "判断理由（必须说明是基于哪些证据得出结论，引用具体内容）"
}

## 注意：
- **辈分冲突是硬性否决条件**：即使其他证据都支持合并，只要辈分不同就绝对拒绝
- **宁可合并也不要错过**：在不存在辈分冲突的前提下，如果证据倾向于是同一人，应该合并
- **仔细对比 summaries**：这是最重要的证据，如果描述同一事件或高度相似 → 必须合并
- **引用具体内容**：在 reason 中引用具体的外貌描写、剧情片段来支持判断"""

    user_prompt = f"""请判断以下 {len(heroines_data_list)} 个女主角是否是同一人：

"""
    
    for i, heroine in enumerate(heroines_data_list):
        user_prompt += f"""
【女主 {i+1}】：{heroine.get('name', '')}
- 别名：{heroine.get('other_names', [])}
- 外貌特征：{heroine.get('features', [])[:10]}
- 与男主关系：{heroine.get('relationships', [])[:10]}
- 剧情摘要（前20条）：{heroine.get('summaries', [])[:20]}
- 互动记录（前10条）：{heroine.get('interactions', [])[:10]}

"""
    
    user_prompt += """
请仔细对比：
1. summaries 是否描述了同一事件或高度相似？（最重要！）
2. features 外貌描写是否完全一致？
3. other_names 是否互相包含？
4. relationships 与男主的关系是否一致？
5. interactions 互动内容是否相似且不矛盾？

如果以上有 2-3 项明确一致，且没有明显矛盾，应该合并。"""

    try:
        response = chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        record_usage(response)
        
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        
        should_merge = result.get('should_merge', False)
        main_name = result.get('main_name', names[0])
        reason = result.get('reason', '未提供理由')
        
        return should_merge, main_name, reason
        
    except Exception as e:
        logger.warning(f"判断女主组合并失败 ({names}): {e}")
        return False, "", f"API 调用失败: {e}"


def merge_heroines_final(heroines_result, merged_stats):
    """
    女主判定后的最终合并检查
    
    流程：
    1. 将所有女主的关键信息提供给大模型，让其识别可能的合并候选
    2. 对每个候选组，单独调用大模型进行详细判断
    3. 执行合并并更新女主列表
    
    参数:
        heroines_result: identify_heroines 的返回结果
        merged_stats: 合并后的角色统计数据
    
    返回:
        更新后的 heroines_result
    """
    heroines = heroines_result.get('heroines', [])

    def _to_list(value):
        """将别名字段统一为 list，兼容 None/set/tuple/单值。"""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (set, tuple)):
            return list(value)
        return [value]

    def _sync_alias_fields(heroine):
        """
        统一 heroine 的 aliases / other_names 字段，避免字段缺失导致 KeyError。
        两个字段保持同一份去重后的列表，便于后续逻辑兼容。
        """
        if not isinstance(heroine, dict):
            return
        name = str(heroine.get("name", "")).strip()
        merged_aliases = []
        seen_aliases = set()
        for alias in _to_list(heroine.get("aliases")) + _to_list(heroine.get("other_names")):
            alias_str = str(alias).strip()
            if not alias_str or alias_str == name or alias_str in seen_aliases:
                continue
            seen_aliases.add(alias_str)
            merged_aliases.append(alias_str)
        heroine["aliases"] = list(merged_aliases)
        heroine["other_names"] = list(merged_aliases)
    
    if len(heroines) < 2:
        logger.info("女主数量少于2个，无需进行最终合并检查")
        return heroines_result

    # 先统一字段，避免后续在只含 aliases 的数据上访问 other_names 报错
    for h in heroines:
        _sync_alias_fields(h)
    
    logger.info(f"开始女主最终合并检查（当前有 {len(heroines)} 位女主）...")
    
    # 第一步：识别可能的合并候选
    merge_candidates = _identify_heroine_merge_candidates(heroines)

    # 额外候选：基于“名字出现在对方别名里/共享别名”的确定性候选生成
    # 注意：这里只是把候选组补全，是否合并仍交给 AI 二次判定（不做程序强行合并）。
    try:
        name_to_aliases = {}
        for h in heroines:
            n = _normalize_person_name(h.get("name", ""))
            if not n:
                continue
            aliases = set()
            for a in (h.get("aliases", []) or []):
                an = _normalize_person_name(a)
                if an:
                    aliases.add(an)
            for a in (h.get("other_names", []) or []):
                an = _normalize_person_name(a)
                if an:
                    aliases.add(an)
            name_to_aliases[n] = aliases

        existing_pairs = set()
        for names, _reason in merge_candidates:
            if not names or len(names) < 2:
                continue
            key = tuple(sorted([_normalize_person_name(x) for x in names if x]))
            if len(key) >= 2:
                existing_pairs.add(key)

        extra = []
        names_list = [h.get("name", "") for h in heroines]
        for i, a in enumerate(names_list):
            a = _normalize_person_name(a)
            if not a:
                continue
            for b in names_list[i + 1 :]:
                b = _normalize_person_name(b)
                if not b:
                    continue
                a_aliases = name_to_aliases.get(a, set())
                b_aliases = name_to_aliases.get(b, set())
                # 候选规则：别名互含/共享别名
                reasons = []
                if a in b_aliases:
                    reasons.append(f"{b} 的别名包含 {a}")
                if b in a_aliases:
                    reasons.append(f"{a} 的别名包含 {b}")
                shared = a_aliases & b_aliases
                if shared:
                    reasons.append(f"共享别名：{list(shared)[:3]}")
                if reasons:
                    key = tuple(sorted([a, b]))
                    if key not in existing_pairs:
                        extra.append(([a, b], "；".join(reasons)))
                        existing_pairs.add(key)

        if extra:
            logger.info(f"别名规则补充候选 {len(extra)} 组")
            merge_candidates.extend(extra)
    except Exception as e:
        logger.warning(f"补充候选生成失败: {e}")
    
    if not merge_candidates:
        logger.info("未发现需要合并的女主候选")
        return heroines_result
    
    logger.info(f"识别到 {len(merge_candidates)} 组可能需要合并的女主候选")
    
    # 辈分冲突安全检查（防止母女/长辈-晚辈被错误合并）
    _HEROINE_ELDER_KW = {'太太', '夫人', '妈妈', '母亲', '婆婆', '奶奶', '外婆', '阿姨', '姑姑', '姨妈'}
    _HEROINE_YOUNGER_KW = {'小姐', '姐姐', '妹妹', '姐', '妹', '女儿', '孙女', '外孙女', '学妹'}

    def _extract_elder_surname(name):
        """若名字以 ELDER 关键词结尾，返回 (姓氏前缀, True)；否则 (None, False)"""
        for kw in sorted(_HEROINE_ELDER_KW, key=len, reverse=True):
            if name.endswith(kw) and len(name) > len(kw):
                return name[:-len(kw)], True
        return None, False

    def _has_heroine_generation_conflict(names):
        """
        检测候选合并组内是否存在辈分冲突，返回应移除的名字集合。
        规则：
          1. 经典冲突：ELDER 关键词 vs YOUNGER 关键词（同姓）
          2. 扩展冲突：'X太太/X夫人' vs 'X+其他名'（同姓前缀但一方含辈分词）
        """
        to_remove = set()
        elder_names = {}
        for n in names:
            prefix, is_elder = _extract_elder_surname(n)
            if is_elder and prefix:
                elder_names[n] = prefix

        for elder_n, prefix in elder_names.items():
            for other_n in names:
                if other_n == elder_n:
                    continue
                other_prefix, other_is_elder = _extract_elder_surname(other_n)
                if other_is_elder and other_prefix == prefix:
                    continue
                shares_surname = (
                    other_n.startswith(prefix) and len(other_n) > len(prefix)
                )
                has_younger = any(kw in other_n for kw in _HEROINE_YOUNGER_KW)
                if shares_surname or has_younger:
                    to_remove.add(elder_n)
                    to_remove.add(other_n)
                    logger.warning(
                        f"女主合并安全检查：辈分冲突 {elder_n} vs {other_n}，从候选组移除"
                    )
        return to_remove

    # 第二步：逐组判断是否应该合并
    confirmed_merges = []
    
    for candidate_names, candidate_reason in merge_candidates:
        logger.info(f"正在判断候选组: {candidate_names} ({candidate_reason[:40]}...)")
        
        # 辈分冲突过滤：移除存在辈分冲突的名字对
        conflict_names = _has_heroine_generation_conflict(candidate_names)
        if conflict_names:
            filtered = [n for n in candidate_names if n not in conflict_names]
            if len(filtered) < 2:
                logger.info(f"✗ 辈分冲突自动拒绝: {candidate_names}（冲突: {conflict_names}）")
                continue
            logger.info(f"辈分冲突过滤后剩余: {filtered}（移除: {conflict_names}）")
            candidate_names = filtered

        # 获取这些女主的完整数据
        heroines_data = []
        for name in candidate_names:
            for h in heroines:
                if h.get('name') == name:
                    heroines_data.append(h)
                    break
        
        if len(heroines_data) < len(candidate_names):
            logger.warning(f"候选组 {candidate_names} 中有角色未找到，跳过")
            continue
        
        # 调用大模型判断
        should_merge, main_name, reason = _judge_heroine_group_merge(heroines_data)
        
        if should_merge and main_name:
            confirmed_merges.append((candidate_names, main_name, reason))
            logger.info(f"✓ 确认合并: {candidate_names} -> {main_name} ({reason[:50]}...)")
        else:
            logger.info(f"✗ 拒绝合并: {candidate_names} - {reason[:50]}...")
    
    if not confirmed_merges:
        logger.info("所有候选组均不需要合并")
        return heroines_result
    
    # 第三步：执行合并
    logger.info(f"开始执行 {len(confirmed_merges)} 组女主合并...")
    
    # 创建名字到女主数据的映射
    name_to_heroine = {h.get('name'): h for h in heroines}
    merged_names = set()
    new_heroines = []
    
    for merge_names, main_name, reason in confirmed_merges:
        if main_name in merged_names:
            continue  # 已经被合并过了
        
        # 获取主女主数据
        main_heroine = name_to_heroine.get(main_name)
        if not main_heroine:
            # 主名不在列表中，使用第一个找到的
            for name in merge_names:
                if name in name_to_heroine:
                    main_heroine = name_to_heroine[name]
                    main_name = name
                    break
        
        if not main_heroine:
            logger.warning(f"合并组 {merge_names} 无法找到有效数据，跳过")
            continue
        
        # 合并其他女主的数据
        for name in merge_names:
            if name == main_name:
                continue
            
            if name in name_to_heroine:
                other_heroine = name_to_heroine[name]
                
                # 合并 other_names
                main_aliases = _to_list(main_heroine.get('other_names'))
                main_aliases.append(name)
                main_aliases.extend(_to_list(other_heroine.get('other_names')))
                main_aliases.extend(_to_list(other_heroine.get('aliases')))
                main_heroine['other_names'] = main_aliases
                
                # 合并其他字段
                for field in ['summaries', 'features', 'relationships', 'interactions', 'emotion_signals']:
                    if field in other_heroine:
                        if field not in main_heroine:
                            main_heroine[field] = []
                        main_heroine[field].extend(other_heroine.get(field, []))
                
                # 合并 count
                main_heroine['count'] = main_heroine.get('count', 0) + other_heroine.get('count', 0)
                
                # 标记为已合并
                merged_names.add(name)
        
        # 去重 other_names
        _sync_alias_fields(main_heroine)
        
        # 标记主名也已处理
        merged_names.add(main_name)
        new_heroines.append(main_heroine)
        
        logger.info(f"合并完成: {merge_names} -> {main_name} (别名: {main_heroine.get('other_names', [])})")
    
    # 添加未被合并的女主
    for h in heroines:
        if h.get('name') not in merged_names:
            new_heroines.append(h)
    
    # 更新结果
    heroines_result['heroines'] = new_heroines
    
    logger.info(f"女主最终合并完成: {len(heroines)} 位 -> {len(new_heroines)} 位")
    
    # 重新排序和分析
    if new_heroines:
        heroines_result = _final_ranking_and_analysis(new_heroines)
    
    return heroines_result


def _judge_single_character(char_data, evidence_profile=None, max_retries=2):
    """
    判断单个角色是否为女主角
    提供该角色的完整信息，让 AI 专注分析这一个角色
    尽量保留完整数据，只在极端情况下截断
    """

    # 只在数据量极大时进行轻微截断（避免超出 API 上下文限制）
    def _light_truncate(items, max_count=100):
        """轻微截断：只在超过 100 条时才截断，保留开头和结尾"""
        if not items or len(items) <= max_count:
            return items
        # 保留开头 60% 和结尾 40%
        head_count = int(max_count * 0.6)
        tail_count = max_count - head_count
        return items[:head_count] + items[-tail_count:]

    # 使用完整数据，只在极端情况下轻微截断
    interactions = _light_truncate(char_data.get('interactions', []))
    emotions = _light_truncate(char_data.get('emotion_signals', []))
    summaries = _light_truncate(char_data.get('summaries', []))

    system_prompt = """你是一个专业的男性向小说分析师，负责判断**单个女性角色**是否为女主角。

## 核心判断标准：【双向感情互动 或 特殊条件】

### 女主角的定义（满足以下条件之一即可）：

#### 情况一：双向感情互动（最常见）
- 男主和女方都有明确的感情表现（双向暧昧、互相喜欢、恋爱关系）
- 男主对该角色表现出特殊关注、好感、爱意
- 双方有亲密互动且男主是主动或配合的

#### 情况二：单方主动但有实际亲密行为
- 虽然是单相思，但做出了实际行动且成功发生亲密关系：
  - 强上/推倒男主并发生关系 → 是女主！
  - 主动亲吻男主且男主接受 → 是女主！
  - 主动献身/投怀送抱且男主接受 → 是女主！
- 关键：必须有”实际发生”的亲密行为，而非仅有单相思的心理活动

##### 补充（本工具易漏判的类型，请务必纳入 action）：
- **专属女仆/认主仪式/长期亲密侍奉**：如出现”宣誓效忠、认主、亲吻脚尖/跪拜、洗脚按摩、同床、暗示昨日耕耘、被称为专属女仆”等，
  且伴随**明确的亲密身体接触或性关系暗示/确认**，应判为女主（action 或 mutual）。
- **胁迫/契约导致的实质关系**：即便起初是羞辱、权力压制、契约胁迫，只要后续出现**持续的亲密关系/性关系**并进入”亲密女性圈”，也应判为女主（action）。

#### 情况三：高价值单相思（特殊条件）
- 虽然目前是单相思，但同时满足以下条件：
  1. 【高颜值】外貌描写突出（美女/绝色/倾城/天姿国色/仙姿玉貌等）
  2. 【特殊身份】身份地位特殊，包括但不限于：
     - 现代/都市：公主/大小姐/女王/学生会长/校花/偶像/财阀千金/总裁/女CEO
     - 玄幻/修仙：圣女/圣子/仙子/魔女/宗主/掌门/长老之女/帝国公主/女帝/妖女/魔尊/天骄/大能之女/古族公主/神女/剑仙/丹师/炼器师
     - 西幻/奇幻：公主/女骑士/圣女/女祭司/魔法师/精灵族/龙族/吸血鬼/女公爵/女伯爵/圣殿骑士/教廷圣女
     - 历史/古代：公主/郡主/王妃/皇后/贵妃/世家嫡女/将门之女/相府千金
     - 游戏/异世界：女神/NPC女主/公会会长/勇者/魔王/圣者
  3. 【大量互动】与男主有非常多的互动（count >= 20 或互动记录丰富）
- 这类角色虽暂时单相思，但明显是作者重点塑造的潜在女主 → 是女主！

### 【重要】不是女主的情况：
- **普通单相思**：颜值/身份普通，只是暗恋男主 → 不是女主
- **互动很少的单相思**：虽喜欢男主但出场/互动很少 → 不是女主
- **工具人/炮灰追求者**：追求男主但被明确拒绝且无后续发展 → 不是女主
- 纯粹的路人/背景角色

### 特别注意（避免误判）：
- **只有主仆/工作服务**但没有亲密身体接触/性关系暗示/确认、也没有双向暧昧发展 → 不是女主

## 【特别注意】数据量和感情发展趋势的意义

1. **出场次数远超中位数的角色**：如果一个角色的出场次数是中位数的5倍以上，说明作者投入了大量篇幅描写她。这类角色即使早期评分低（前期为”背景角色”），也极可能在后期发展出重要感情线。
2. **评分上升趋势**：如果前半段评分低但后半段评分显著上升（差值≥2分），说明感情在逐步发展。不要因为前期的”无互动”记录否定后期的亲密证据。
3. **关键亲密证据的权重**：即使80条记录中只有5条显示”发生关系/亲吻/告白”，这5条的证据权重远大于其他75条”无互动”记录。一次确认的肉体关系足以判定为女主。
4. **信号稀释陷阱**：长篇小说中，角色从陌生/敌对到亲密是常见模式。不要被大量早期”无感情线”的记录干扰判断。请重点看【关键亲密/恋爱证据摘要】部分。

### 判断关键点：
1. 男主对她的态度是什么？（喜欢/暧昧/无感/排斥）
2. 是否有双向的感情发展？
3. 如果是单方面喜欢：
   - 有没有发生实际的亲密行为？
   - 颜值/身份是否特殊？互动是否足够多？

## 判断依据（按重要性排序）：
1. 【最重要】关键亲密/恋爱证据摘要 - 如果提供了此部分，请优先阅读！
2. 【最重要】interactions - 重点看男主的反应和态度，以及互动数量
3. 【最重要】summaries - 看是否有双向感情发展/实际亲密行为
4. 【重要】relationships - 最终关系类型（恋人/暧昧 vs 单相思/追求者）
5. 【重要】features/appearances - 外貌和身份描写（判断是否高价值角色）
6. 【参考】emotion_signals - 注意区分是双向还是单向
7. 【参考】count - 出场频次（高频+单相思可能是重要角色）

## 输出格式（JSON）：
{
    “is_heroine”: true/false,
    “confidence”: “high/medium/low”,
    “name”: “角色名字”,
    “aliases”: [“别名1”, “别名2”],
    “relationship_type”: “正宫/侧室/暧昧/青梅竹马/红颜知己/初恋/专属女仆/情人/高价值追求者等”,
    “male_lead_attitude”: “男主对她的态度（喜欢/暧昧/好感/无感/排斥）”,
    “is_mutual”: true/false,
    “heroine_reason”: “mutual/action/high_value/none（判定为女主的原因类型）”,
    “key_interactions”: “与男主最关键的感情互动总结（如果有）”,
    “emotion_evidence”: “感情证据总结（重点说明是双向还是单向）”,
    “character_traits”: “性格特点”,
    “summary”: “角色总结（50字以内）”,
    “reason”: “判断理由（必须说明为什么是/不是女主）”
}

判断原则：双向感情 > 单相思+实际行动 > 单相思+高颜值高身份大量互动 > 普通单相思（不算）"""

    # 统计学提示（HSI）
    hsi = char_data.get("hsi", None)
    hsi_rank = char_data.get("hsi_rank", None)
    hsi_total = char_data.get("hsi_total", None)
    hsi_hits = char_data.get("hsi_hits", None)
    hsi_density = char_data.get("hsi_density", None)
    hsi_text_kb = char_data.get("hsi_text_kb", None)
    hsi_threshold = char_data.get("hsi_threshold", None)
    hsi_strong_k = char_data.get("hsi_strong_k", None)

    hsi_block = ""
    try:
        if hsi is not None and hsi_rank is not None and hsi_total is not None:
            hsi_block = f"""
【统计学提示（HSI）】（仅作先验提示，不可替代证据判断）
- HSI={float(hsi):.3f} | 排名 {hsi_rank}/{hsi_total}
- hits={hsi_hits} | density={float(hsi_density):.3f} | text_kb={float(hsi_text_kb):.1f}
- 推荐”强候选阈值” t≈{float(hsi_threshold):.3f}（约取 Top {hsi_strong_k}）
解释：HSI 越高，通常意味着”出场持续 + 与男主感情/亲密相关证据更密集”，更可能是女主；但仍必须结合 interactions/summaries 等其他特征具体内容判断，避免把亲属误判成女主。
"""
    except Exception:
        hsi_block = ""

    # 证据档案提示（数据量上下文 + 评分分布/趋势 + 关键证据）
    evidence_block = ""
    try:
        if evidence_profile:
            parts = []

            # 数据量背景
            vc = evidence_profile.get("volume_context", {})
            if vc.get("median") is not None:
                parts.append(
                    f"【数据量背景】该角色出场 {vc['count']} 次，"
                    f"是全部 {vc.get('total_chars', '?')} 个女性角色中位数（{vc['median']}次）的 {vc['ratio_to_median']} 倍，"
                    f"排名前 {max(100 - vc['percentile'], 0.1):.1f}%。"
                )
                if vc.get("ratio_to_median", 0) >= 5:
                    parts.append("→ 出场次数远超其他角色，作者对该角色投入了大量笔墨，极可能是核心女主之一。")
                elif vc.get("ratio_to_median", 0) >= 3:
                    parts.append("→ 出场次数显著高于其他角色，很可能是重要女性角色。")

            # 评分分布
            sd = evidence_profile.get("score_distribution", {})
            if any(sd.values()):
                parts.append(
                    f"【评分分布】高分(8-10): {sd.get('8-10', 0)}次 | "
                    f"中高(6-7): {sd.get('6-7', 0)}次 | "
                    f"中等(4-5): {sd.get('4-5', 0)}次 | "
                    f"低分(1-3): {sd.get('1-3', 0)}次"
                )

            # 评分趋势
            trend = evidence_profile.get("score_trend", "")
            if trend and trend not in ("无逐块分数数据", "数据不足"):
                parts.append(f"【评分趋势】{trend}")

            # 关键亲密/恋爱证据摘要（最重要，放在最前面让 AI 优先读取）
            peaks = evidence_profile.get("peak_evidence", [])
            if peaks:
                peak_lines = ["【关键亲密/恋爱证据摘要】（以下是从全部数据中提取的最强感情证据，请重点关注！）"]
                for i, p in enumerate(peaks[:10], 1):
                    peak_lines.append(f"  {i}. {p}")
                # 将关键证据放到最前面
                parts = peak_lines + parts

            evidence_block = "\n".join(parts)
    except Exception:
        evidence_block = ""

    user_prompt = f"""请判断以下女性角色是否为女主角：

【角色名】{char_data['name']}
【别称】{', '.join(char_data.get('other_names', [])) or '无'}
【重要性评分】{char_data['avg_score']:.1f}/10
【出场频次】{char_data['count']} 次
{hsi_block}
{evidence_block}

【与男主的互动记录】（重点分析男主的态度和反应！）
{json.dumps(interactions, ensure_ascii=False, indent=2)}

【剧情摘要】（重点看是否有双向感情发展或实际亲密行为）
{json.dumps(summaries, ensure_ascii=False, indent=2)}

【感情信号】（注意区分是双向还是单向！）
{json.dumps(emotions, ensure_ascii=False, indent=2)}

【关系类型】
{json.dumps(char_data.get('relationships', [])[:10], ensure_ascii=False, indent=2)}

【外貌特征】（判断是否高颜值）
{json.dumps(char_data.get('features', [])[:5], ensure_ascii=False, indent=2)}

【身份背景】（判断是否特殊身份）
{json.dumps(char_data.get('appearances', [])[:5], ensure_ascii=False, indent=2)}

请根据以上信息，判断该角色是否为女主角。

关键问题：
1. 男主对她是什么态度？（喜欢/暧昧/无感/排斥）
2. 感情是双向的还是她单方面喜欢男主？
3. 如果是单相思：
   a. 有没有发生实际的亲密行为（如强上男主/发生关系）？→ 是女主
   b. 是否同时满足：高颜值 + 特殊身份 + 大量互动？→ 是女主（高价值追求者）

【判断原则】
- 双向感情 → 女主
- 单相思 + 实际亲密行为 → 女主
- 单相思 + 高颜值 + 特殊身份 + 大量互动 → 女主（高价值追求者）
- 普通单相思（颜值/身份普通或互动少）→ 不是女主"""

    for retry in range(max_retries):
        try:
            response = chat_completion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=3000  # 增加输出限制，适应完整数据
            )
            record_usage(response)
            
            content = response.choices[0].message.content.strip()
            content = re.sub(r'```json\s*|\s*```', '', content)
            
            result = json.loads(content)
            result['avg_score'] = char_data['avg_score']
            result['count'] = char_data['count']
            return result
            
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(1)
            else:
                logger.warning(f"判断角色 {char_data['name']} 失败: {e}")
                # 降级：根据分数和出场次数自动判断（宁多勿漏）
                if char_data['avg_score'] >= 6 or (char_data['avg_score'] >= 5 and char_data['count'] >= 30):
                    return {
                        "is_heroine": True,
                        "confidence": "low",
                        "name": char_data['name'],
                        "aliases": char_data.get('other_names', []),
                        "reason": "API调用失败，根据评分/出场次数自动判定"
                    }
                # 降级兜底：有大量强亲密证据时也自动判定
                ep = evidence_profile or {}
                peak_evidence = ep.get("peak_evidence", [])
                chunk_scores = char_data.get("chunk_scores", [])
                high_scores = [s for _, s in chunk_scores if isinstance(s, (int, float)) and s >= 8] if chunk_scores else []
                if len(peak_evidence) >= 3 and len(high_scores) >= 5:
                    return {
                        "is_heroine": True,
                        "confidence": "medium",
                        "name": char_data['name'],
                        "aliases": char_data.get('other_names', []),
                        "heroine_reason": "action",
                        "reason": f"API调用失败，但检测到 {len(peak_evidence)} 条强亲密证据和 {len(high_scores)} 次高评分，自动判定"
                    }
                return None
    return None


def _final_ranking_and_analysis(confirmed_heroines):
    """
    对确认的女主角进行最终排序和整体分析
    """
    
    system_prompt = """你是专业的小说分析师，请对已确认的女主角进行排序和整体分析。

## 任务：
1. 根据与男主的感情深度对女主角排序（importance_rank）
2. 分析整体女主体系（单女主/后宫/大后宫等）
3. 分析女主之间的关系

## 女主类型（按重要性排序）：
1. mutual - 双向感情互动（最重要）
2. action - 单相思但发生了实际亲密行为
3. high_value - 高颜值+特殊身份+大量互动的追求者

## 输出格式（JSON）：
{
    "heroines": [
        {
            "name": "女主角名字",
            "aliases": ["别名1", "别名2"],
            "importance_rank": 1,
            "relationship_type": "与男主的关系类型",
            "is_mutual": true/false,
            "heroine_type": "mutual/action/high_value",
            "key_interactions": "关键互动",
            "character_traits": "性格特点",
            "summary": "角色总结"
        }
    ],
    "analysis": "整体女主体系分析",
    "novel_type": "单女主/双女主/小后宫(3-5人)/中后宫(6-10人)/大后宫(10人以上)"
}

importance_rank 从1开始，按感情深度排序：双向感情 > 单相思+行动 > 高价值追求者。"""

    user_prompt = f"""以下是已确认的 {len(confirmed_heroines)} 位女主角，请进行排序和整体分析：

{json.dumps(confirmed_heroines, ensure_ascii=False, indent=2)}

请根据每位女主与男主的感情深度进行排序，并分析整体女主体系。

排序优先级：
1. 双向感情（mutual）最高
2. 单相思+实际亲密行为（action）次之
3. 高价值追求者（high_value）再次"""

    try:
        response = chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=3000
        )
        record_usage(response)
        
        content = response.choices[0].message.content.strip()
        content = re.sub(r'```json\s*|\s*```', '', content)
        
        result = json.loads(content)
        return result
        
    except Exception as e:
        logger.error(f"最终排序分析失败: {e}")
        # 降级：按 confidence 和 avg_score 排序
        sorted_heroines = sorted(
            confirmed_heroines, 
            key=lambda x: (x.get('confidence', 'low') == 'high', x.get('avg_score', 0)),
            reverse=True
        )
        return {
            "heroines": [
                {
                    "name": h.get('name', ''),
                    "aliases": h.get('aliases', []),
                    "importance_rank": i + 1,
                    "relationship_type": h.get('relationship_type', '未知'),
                    "key_interactions": h.get('key_interactions', ''),
                    "character_traits": h.get('character_traits', ''),
                    "summary": h.get('summary', '')
                }
                for i, h in enumerate(sorted_heroines)
            ],
            "analysis": "自动排序（API调用失败）",
            "novel_type": f"{'单女主' if len(sorted_heroines) == 1 else '后宫'}"
        }


def _identify_heroines_multi_batch(candidates):
    """
    分批次识别女主（候选人 > 25）
    先分批筛选，再汇总确认
    """
    BATCH_SIZE = 15  # 减小批次大小，因为每个角色数据可能很大
    all_potential_heroines = []
    
    # 对每个角色的数据进行截断，避免请求过大
    def _truncate_for_batch(char_data, max_items=10):
        """截断角色数据用于分批筛选，保留最关键的信息"""
        return {
            "name": char_data['name'],
            "avg_score": char_data['avg_score'],
            "count": char_data['count'],
            "other_names": char_data.get('other_names', [])[:5],
            "relationships": char_data.get('relationships', [])[:5],
            # 只保留前后各几条，确保覆盖开头和结尾的关键发展
            "interactions": (char_data.get('interactions', [])[:max_items] + 
                           char_data.get('interactions', [])[-3:] if len(char_data.get('interactions', [])) > max_items else char_data.get('interactions', [])),
            "emotion_signals": (char_data.get('emotion_signals', [])[:max_items] + 
                              char_data.get('emotion_signals', [])[-3:] if len(char_data.get('emotion_signals', [])) > max_items else char_data.get('emotion_signals', [])),
            "summaries": (char_data.get('summaries', [])[:max_items] + 
                        char_data.get('summaries', [])[-3:] if len(char_data.get('summaries', [])) > max_items else char_data.get('summaries', []))
        }
    
    # 第一轮：分批筛选
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(candidates) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"分批筛选中 ({batch_num}/{total_batches})...")
        
        # 截断批次数据
        truncated_batch = [_truncate_for_batch(c) for c in batch]
        
        system_prompt = """你是小说分析师。请从候选角色中筛选出**可能是女主角**的角色。

## 核心判断标准：双向感情互动 或 特殊条件

### 判断依据（重点看这几个字段）：
1. interactions - 与男主的互动记录（重点看男主的态度和反应！）
2. summaries - 剧情摘要中是否有双向感情发展或实际亲密行为
3. relationships - 关系类型（恋人/暧昧 vs 单相思/追求者）
4. features/appearances - 外貌和身份（判断是否高价值角色）
5. count - 出场频次

### 判断逻辑：
- 有双向感情互动/恋爱关系 → high confidence（必须列入！）
- 有双向暧昧/互相好感 → medium confidence（列入）
- 单相思但发生了实际亲密行为（强上/发生关系） → high confidence（列入！）
- 单相思但高颜值+特殊身份+大量互动（count>=13） → medium confidence（列入！）
- 普通单相思（颜值/身份普通或互动少） → 不列入
- 纯路人/无互动 → 不列入

### 【重要】高价值单相思的判断：
- 颜值描写：美女/绝色/倾城/天姿国色/仙姿玉貌等
- 特殊身份（按题材）：
  * 现代：公主/大小姐/校花/偶像/财阀千金/女CEO
  * 玄幻/修仙：圣女/仙子/魔女/宗主/女帝/妖女/天骄/神女/古族公主
  * 西幻：公主/女骑士/圣女/精灵族/龙族/吸血鬼/女公爵
  * 历史：公主/郡主/王妃/皇后/世家嫡女/将门之女
- 大量互动：count >= 13 或互动记录丰富
- 三者同时满足才算高价值追求者

输出 JSON 格式：
{
    "potential_heroines": [
        {
            "name": "角色名",
            "confidence": "high/medium/low",
            "is_mutual": true/false,
            "heroine_type": "mutual/action/high_value",
            "reason": "判断理由"
        }
    ]
}

普通单相思不列入！必须有双向感情、实际亲密行为、或高颜值+特殊身份+大量互动！"""

        user_prompt = f"""请筛选以下角色中可能是女主角的：

注意：数据已截断，只显示部分记录。请根据现有信息判断。

**关键判断点：**
1. 男主对该角色是什么态度？（喜欢/暧昧/无感/排斥）
2. 是双向感情还是单相思？
3. 如果是单相思：
   - 有没有发生实际亲密行为？→ 列入
   - 是否高颜值+特殊身份+大量互动？→ 列入（高价值追求者）

{json.dumps(truncated_batch, ensure_ascii=False, indent=2)}

【判断原则】
- 双向感情 → 列入
- 单相思+实际亲密行为 → 列入
- 单相思+高颜值+特殊身份+大量互动 → 列入（高价值追求者）
- 普通单相思 → 不列入"""

        try:
            response = chat_completion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4000  # 增加输出限制
            )
            record_usage(response)
            
            content = response.choices[0].message.content.strip()
            content = re.sub(r'```json\s*|\s*```', '', content)
            batch_result = json.loads(content)
            
            for h in batch_result.get('potential_heroines', []):
                # 找到原始数据（完整版）
                for c in batch:
                    if c['name'] == h.get('name'):
                        c['confidence'] = h.get('confidence', 'medium')
                        all_potential_heroines.append(c)
                        break
                        
        except Exception as e:
            logger.warning(f"批次 {batch_num} 筛选失败: {e}，保留高分角色")
            # 失败时保留该批次中高分的（降低阈值，宁多勿漏）
            for c in batch:
                if c['avg_score'] >= 5 or c['count'] >= 30:
                    all_potential_heroines.append(c)
    
    logger.info(f"分批筛选后剩余 {len(all_potential_heroines)} 个候选")
    
    # 第二轮：最终确认和排序
    if len(all_potential_heroines) <= 30:
        # 直接最终确认
        return _final_heroine_confirmation(all_potential_heroines)
    else:
        # 还是太多，再按分数筛选一轮
        all_potential_heroines.sort(key=lambda x: x['final_score'], reverse=True)
        top_candidates = all_potential_heroines[:30]
        return _final_heroine_confirmation(top_candidates)


def _final_heroine_confirmation(candidates):
    """最终确认女主角列表并排序"""
    
    system_prompt = """你是专业的小说分析师，请对筛选后的女主角候选人进行最终确认和排序。

## 核心判断标准：【双向感情互动 或 特殊条件】

### 确认女主的依据（满足任一即可）：
1. 感情是双向的（男主和女方都有感情表现）
2. 虽是单相思但发生了实际亲密行为（如强上男主/发生关系）
3. 虽是单相思但同时满足：高颜值 + 特殊身份 + 大量互动（高价值追求者）

### 【重要】排除标准（不列入）：
- 普通单相思（颜值/身份普通或互动少）→ 不是女主
- 追求男主但被明确拒绝且无后续发展 → 不是女主
- 只有工作/任务关系，无感情互动 → 不是女主

## 输出格式（JSON）：
{
    "heroines": [
        {
            "name": "女主角名字",
            "aliases": ["别名1", "别名2"],
            "importance_rank": 1,
            "relationship_type": "正宫/侧室/暧昧/青梅竹马/红颜知己/初恋/高价值追求者/etc",
            "is_mutual": true/false,
            "heroine_type": "mutual/action/high_value",
            "key_interactions": "与男主最关键的感情互动总结",
            "character_traits": "性格特点",
            "summary": "角色总结"
        }
    ],
    "analysis": "女主体系分析（类型判断、女主关系、感情线发展等）",
    "novel_type": "单女主/双女主/小后宫(3-5人)/中后宫(6-10人)/大后宫(10人以上)"
}

按与男主感情深度排序（双向 > 单相思+行动 > 高价值追求者），importance_rank 从1开始。
普通单相思不算女主！必须有双向感情、实际亲密行为、或高颜值+特殊身份+大量互动！"""

    user_prompt = f"""请对以下 {len(candidates)} 个女主角候选人进行最终确认：

{json.dumps(candidates, ensure_ascii=False, indent=2)}

请：
1. 确认每个角色是否真的是女主角
2. 女主判断标准：
   - 双向感情 → 女主
   - 单相思+实际亲密行为 → 女主
   - 单相思+高颜值+特殊身份+大量互动 → 女主（高价值追求者）
   - 普通单相思 → 不是女主
3. 按与男主感情深度排序（双向 > 单相思+行动 > 高价值追求者）
4. 分析女主体系类型

【重要】普通单相思不算女主！必须满足上述三种条件之一！"""

    try:
        logger.info(f"最终确认女主角（{len(candidates)} 个候选）...")
        response = chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=5000
        )
        record_usage(response)
        
        content = response.choices[0].message.content.strip()
        content = re.sub(r'```json\s*|\s*```', '', content)
        
        result = json.loads(content)
        logger.info(f"最终确认 {len(result.get('heroines', []))} 位女主角")
        return result
        
    except Exception as e:
        logger.error(f"最终确认失败: {e}")
        return {
            "heroines": [{"name": c["name"], "importance_rank": i+1, "aliases": c.get("other_names", [])} 
                        for i, c in enumerate(candidates)],
            "analysis": "自动识别（API调用失败）",
            "novel_type": "未知"
        }


def generate_final_report(heroine_result, merged_stats, male_protagonist=None):
    """生成最终的详细报告"""
    
    heroines = heroine_result.get('heroines', [])
    analysis = heroine_result.get('analysis', '')
    
    report_lines = [
        "=" * 70,
        "小说主角分析报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
    ]
    
    # 男主角部分
    if male_protagonist:
        report_lines.extend([
            "",
            "【男主角】",
            "-" * 70,
            f"  姓名：{male_protagonist.get('name', '未知')}",
            f"  别称：{', '.join(male_protagonist.get('other_names', [])) or '无'}",
            f"  身份：{male_protagonist.get('identity', '未知')}",
            f"  出现频次：{male_protagonist.get('count', 0)} 次",
        ])
        summaries = male_protagonist.get('summaries', [])
        if summaries:
            report_lines.append("  主要事迹：")
            for s in summaries[:2]:
                report_lines.append(f"    - {s[:80]}...")
    
    report_lines.extend([
        "",
        "=" * 70,
        "【女主体系分析】",
        analysis,
        "",
        "-" * 70,
        f"【共识别出 {len(heroines)} 位女主角】",
        "-" * 70,
    ])
    
    for heroine in heroines:
        name = heroine.get('name', '未知')
        aliases = heroine.get('aliases', [])
        rank = heroine.get('importance_rank', 0)
        rel_type = heroine.get('relationship_type', '未知')
        traits = heroine.get('character_traits', '未知')
        summary = heroine.get('summary', '无')
        
        # 从 merged_stats 获取更多信息
        stats = merged_stats.get(name, {})
        count = stats.get('count', 0)
        avg_score = stats['total_score'] / stats['count'] if stats.get('count', 0) > 0 else 0
        
        report_lines.extend([
            "",
            f"【第{rank}女主】{name}",
            f"  别称：{', '.join(aliases) if aliases else '无'}",
            f"  重要性评分：{avg_score:.1f}/10",
            f"  出场频次：{count} 次",
            f"  与男主关系：{rel_type}",
            f"  性格特点：{traits}",
            f"  角色简介：{summary}",
        ])
        
        # 添加一些具体的剧情摘要
        summaries = stats.get('summaries', [])[:3]
        if summaries:
            report_lines.append("  主要剧情：")
            for i, s in enumerate(summaries, 1):
                report_lines.append(f"    {i}. {s[:100]}...")
    
    report_lines.extend([
        "",
        "=" * 70,
        "【完整角色统计】",
        "=" * 70,
    ])
    
    # 按分数排序所有角色
    all_chars = []
    for name, data in merged_stats.items():
        avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
        all_chars.append((name, avg_score, data['count']))
    all_chars.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    for name, score, count in all_chars[:20]:
        report_lines.append(f"  {name}: 评分 {score:.1f}, 出现 {count} 次")
    
    return "\n".join(report_lines)


def export_results(merged_stats, heroine_result, final_report, male_protagonist=None, filename_prefix="heroine_analysis"):
    """导出最终结果"""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 详细数据 JSON
    detailed_data = {
        "analysis_time": timestamp_str,
        "male_protagonist": male_protagonist,
        "heroine_result": heroine_result,
        "all_female_characters": {}
    }
    
    def extract_sorted_content(raw_data):
        """提取带 chunk_index 的数据，按时间排序后返回纯文本列表"""
        if not raw_data:
            return []
        first = raw_data[0]
        # 检查是否为带索引的格式
        if isinstance(first, (tuple, list)) and len(first) == 2 and isinstance(first[0], (int, float)):
            sorted_data = sorted(raw_data, key=lambda x: x[0])
            return [s[1] for s in sorted_data]
        return raw_data
    
    for name, data in merged_stats.items():
        avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
        
        # 处理所有带 chunk_index 的字段，按时间排序后提取纯文本
        all_summaries = extract_sorted_content(data.get('summaries', []))
        all_interactions = extract_sorted_content(data.get('interactions', []))
        all_emotions = extract_sorted_content(data.get('emotion_signals', []))
        
        detailed_data["all_female_characters"][name] = {
            "avg_score": avg_score,
            "count": data['count'],
            "total_score": data['total_score'],
            "other_names": list(data.get('other_names', set())),
            "summaries": all_summaries,  # 保留全部（按时间排序）
            "features": data.get('features', []),
            "relationships": data.get('relationships', []),
            "interactions": all_interactions,  # 保留全部（按时间排序）
            "emotion_signals": all_emotions  # 保留全部（按时间排序）
        }
    
    detailed_file = f"{OUTPUT_DIR}/{filename_prefix}_detailed_{timestamp_str}.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_data, f, ensure_ascii=False, indent=2)
    detailed_snapshot_file = f"{OUTPUT_DIR}/{filename_prefix}_detail_snapshot_{timestamp_str}.json"
    shutil.copyfile(detailed_file, detailed_snapshot_file)
    
    # 男女主特点精简版 JSON（便于其他程序直接读取主角特征）
    protagonists_summary = {
        "analysis_time": timestamp_str,
        "male_protagonist": {
            "name": male_protagonist.get("name") if male_protagonist else None,
            "aliases": male_protagonist.get("other_names") if male_protagonist else [],
            "identity": male_protagonist.get("identity") if male_protagonist else None,
            "count": male_protagonist.get("count") if male_protagonist else 0,
            "summaries": male_protagonist.get("summaries") if male_protagonist else []
        },
        "heroines": []
    }

    for h in heroine_result.get("heroines", []):
        name = h.get("name")
        stats = merged_stats.get(name, {})
        protagonists_summary["heroines"].append({
            "name": name,
            "aliases": h.get("aliases", []),
            "importance_rank": h.get("importance_rank"),
            "relationship_type": h.get("relationship_type"),
            "character_traits": h.get("character_traits"),
            "summary": h.get("summary"),
            "count": stats.get("count", 0),
            "avg_score": stats["total_score"] / stats["count"] if stats.get("count", 0) else 0,
            "features": stats.get("features", []),
            "relationships": stats.get("relationships", []),
            "interactions": extract_sorted_content(stats.get("interactions", [])),
            "emotion_signals": extract_sorted_content(stats.get("emotion_signals", [])),
            "summaries": extract_sorted_content(stats.get("summaries", [])),
        })

    protagonists_file = f"{OUTPUT_DIR}/{filename_prefix}_protagonists_{timestamp_str}.json"
    with open(protagonists_file, 'w', encoding='utf-8') as f:
        json.dump(protagonists_summary, f, ensure_ascii=False, indent=2)
    
    # 文本报告
    report_file = f"{OUTPUT_DIR}/{filename_prefix}_report_{timestamp_str}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(final_report)
    
    logger.info(f"详细数据已保存到: {detailed_file}")
    logger.info(f"详细快照已保存到: {detailed_snapshot_file}")
    logger.info(f"分析报告已保存到: {report_file}")
    
    return detailed_file, detailed_snapshot_file, report_file


def main(novel_path=None, book_name=None, run_id=None):
    global NOVEL_FILE_PATH, clean_filename, OUTPUT_DIR, logger

    # ---- 彻底重新初始化，防止跨小说状态残留 ----
    NOVEL_FILE_PATH = None
    clean_filename = None
    OUTPUT_DIR = None

    base = get_base_dir()
    results_base = os.path.join(base, "results")

    if novel_path:
        os.environ["NOVEL_PATH"] = novel_path

    NOVEL_FILE_PATH = novel_path or os.environ.get("NOVEL_PATH", os.path.join(base, "novels", "default.txt"))
    print(f"★ 正在处理: {NOVEL_FILE_PATH}")
    clean_filename = (book_name or os.path.splitext(os.path.basename(NOVEL_FILE_PATH))[0]).strip()
    init_token_tracker(clean_filename, run_id=run_id)

    resume_dir = find_latest_checkpoint_dir(clean_filename)
    if resume_dir:
        OUTPUT_DIR = resume_dir
        print(f"★ 发现断点，将在此目录续跑: {OUTPUT_DIR}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        OUTPUT_DIR = os.path.join(results_base, f"{clean_filename}_heroine_{timestamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 重新配置 logger（支持多次调用不残留旧 handler）
    logger = logging.getLogger("protagonist")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    _fh = logging.FileHandler(os.path.join(OUTPUT_DIR, "analysis.log"), encoding='utf-8')
    _fh.setFormatter(_fmt)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_sh)

    try:
        print("=" * 70)
        print(f"小说女主角识别工具（专注版 - {MAX_WORKERS}线程并行）")
        print("=" * 70)
        
        # 1. 验证配置
        validate_config()
        
        # 2. 读取文件
        text = read_novel(NOVEL_FILE_PATH)
        print(f"? 全书共 {len(text):,} 字")
        
        # 3. 切分文本
        chunks = split_text_by_length(text, CHUNK_SIZE)
        total_chunks = len(chunks)
        print(f"? 总共 {total_chunks} 个分析块")
        
        # 4. 尝试加载断点（包含女性角色和男主信息，及阶段标记）
        global_stats, male_protagonist_stats, completed_chunks, progress_flags, loaded_merged_stats, loaded_heroine_result, loaded_male_final = load_checkpoint()

        # -------- 断点升级：旧版本男主信息过于精简（identity 仅1条、summaries 仅3条）--------
        # 说明：
        # - 新版本会把 identities/summaries 全量收集并导出；
        # - 但旧断点可能已把 male_protagonist_stats 截断到很少条，且 report_generated=True 会直接退出。
        # 策略：
        # - 如果已生成报告但男主 final 缺少 identities 字段，则强制重新生成报告（不重跑扫描/合并/女主识别）。
        try:
            if progress_flags.get("report_generated") and loaded_male_final:
                if not isinstance(loaded_male_final, dict) or "identities" not in loaded_male_final:
                    logger.warning("检测到旧断点男主信息精简版，将触发一次报告重生成以写入全量 identities/summaries（若扫描阶段曾被截断，需清空断点重跑扫描才能真正补齐）。")
                    progress_flags["report_generated"] = False
        except Exception:
            pass

        # 断点纠错：避免 progress_flags["scanned"]=True 但实际仍有遗漏块，导致跳过扫描
        if progress_flags.get("scanned") and len(completed_chunks) < total_chunks:
            logger.warning(
                f"断点异常：阶段一标记为已完成(scanned=True)，但 completed_chunks={len(completed_chunks)}/{total_chunks}，将自动纠正并补扫遗漏块"
            )
            progress_flags["scanned"] = False
        
        # 如果已完成且报告已生成，直接退出，避免重复分析
        if progress_flags.get("report_generated") and len(completed_chunks) >= total_chunks:
            print("★ 检测到已完成且报告已生成，跳过所有分析。")
            if progress_flags.get("report_files"):
                print(f"★ 报告文件: {progress_flags.get('report_files')}")
            return 0

        # 阶段一：扫描
        if progress_flags.get("scanned"):
            print("★ 阶段一已完成，跳过扫描。")
            chunks_to_process = []
        else:
            # 只处理尚未成功完成的块（不在 completed_chunks 中的块）
            chunks_to_process = [(chunks[i], i) for i in range(len(chunks)) if i not in completed_chunks]
        
        print(f"? 剩余 {len(chunks_to_process)} 个块待分析（已完成 {len(completed_chunks)} 块）...")
        print(f"★ 阶段标记: 扫描={'是' if progress_flags.get('scanned') else '否'} | 男主识别={'是' if progress_flags.get('male_identified') else '否'} | 别称合并={'是' if progress_flags.get('alias_merged') else '否'} | 女主识别={'是' if progress_flags.get('heroines_identified') else '否'} | 报告生成={'是' if progress_flags.get('report_generated') else '否'}")
        if progress_flags.get("report_generated") and progress_flags.get("report_files"):
            print(f"★ 上次报告文件: {progress_flags.get('report_files')}")
        print("\n【阶段一】扫描全书，识别男主和女性角色...")
        print("=" * 70)

        if not progress_flags.get("scanned") and chunks_to_process:
            def _apply_scan_result(result: dict, chunk_idx: int) -> bool:
                """
                将单块分析结果写入 global_stats / male_protagonist_stats，并在成功时标记 completed_chunks。
                返回：本块是否成功。
                """
                # 检查是否成功
                if not result or not result.get("_success", False):
                    error_msg = (result or {}).get("_error", "未知错误")
                    logger.warning(f"块 {chunk_idx} 分析失败: {error_msg}，将在补漏阶段重试")
                    return False

                # 处理男主信息
                male_proto = result.get("male_protagonist")
                if male_proto:
                    raw_name = male_proto.get("name", "")
                    male_names = _split_multi_names(raw_name)
                    if len(male_names) >= 2:
                        logger.warning(f"男主字段疑似包含多个名字，将拆分处理: '{raw_name}' -> {male_names}")
                    for name in male_names[:2]:
                        if not name:
                            continue
                        if name not in male_protagonist_stats:
                            male_protagonist_stats[name] = {
                                "count": 0,
                                "other_names": set(),
                                "identities": [],
                                # summaries 存储 (chunk_index, summary) 便于后续按时间线导出
                                "summaries": [],
                            }
                        male_protagonist_stats[name]["count"] += 1
                        for other_name in male_proto.get("other_names", []):
                            on = _normalize_person_name(other_name)
                            if on and on != name:
                                male_protagonist_stats[name]["other_names"].add(on)
                        identity = (male_proto.get("identity", "") or "").strip()
                        if identity and identity not in male_protagonist_stats[name]["identities"]:
                            # 不再限制数量：收集全量 identity 线索（去重）
                            male_protagonist_stats[name]["identities"].append(identity)
                        summary = (male_proto.get("summary", "") or "").strip()
                        if summary:
                            # 不再限制数量：收集全量 summaries（带 chunk_index、去重）
                            existing = [
                                s[1] if isinstance(s, (tuple, list)) and len(s) == 2 else str(s)
                                for s in (male_protagonist_stats[name].get("summaries", []) or [])
                            ]
                            if summary not in existing:
                                male_protagonist_stats[name]["summaries"].append((chunk_idx, summary))

                # 处理女性角色信息
                chars = result.get("female_characters", [])
                for char in chars:
                    raw_name = char.get("name", "")
                    names = _split_multi_names(raw_name)
                    if not names:
                        continue
                    if len(names) >= 2:
                        logger.warning(f"女性角色字段疑似包含多个名字，将拆分处理: '{raw_name}' -> {names}")

                    score = char.get("score", 0)
                    for name in names[:3]:
                        if not name:
                            continue
                        if name not in global_stats:
                            global_stats[name] = {
                                "total_score": 0,
                                "count": 0,
                                "chunk_scores": [],  # 存储 (chunk_index, score) 用于评分分布和趋势分析
                                "summaries": [],  # 存储 (chunk_index, summary)
                                "types": set(),
                                "other_names": set(),
                                "appearances": [],
                                "features": [],
                                "relationships": [],
                                "interactions": [],  # 与男主互动记录
                                "emotion_signals": [],  # 感情信号记录
                            }

                        global_stats[name]["total_score"] += score
                        global_stats[name]["count"] += 1
                        global_stats[name]["chunk_scores"].append((chunk_idx, score))

                        # 收集其他名字
                        for other_name in char.get("other_names", []):
                            on = _normalize_person_name(other_name)
                            if on and on != name:
                                global_stats[name]["other_names"].add(on)

                        # 收集摘要（带 chunk_index，全部保留）
                        summary = (char.get("summary", "") or "").strip()
                        char_chunk_idx = char.get("chunk_index", chunk_idx)
                        if summary:
                            existing_summaries = [s[1] if isinstance(s, tuple) else s for s in global_stats[name]["summaries"]]
                            if summary not in existing_summaries:
                                global_stats[name]["summaries"].append((char_chunk_idx, summary))

                        # 收集外貌特征
                        appearance = (char.get("appearance", "") or "").strip()
                        if appearance and appearance not in global_stats[name]["features"]:
                            if len(global_stats[name]["features"]) < 10:
                                global_stats[name]["features"].append(appearance)

                        # 收集身份信息
                        identity = (char.get("identity", "") or "").strip()
                        if identity and identity not in global_stats[name]["appearances"]:
                            if len(global_stats[name]["appearances"]) < 10:
                                global_stats[name]["appearances"].append(identity)

                        # 收集关系类型
                        relationship_type = (char.get("relationship_type", "") or "").strip()
                        if relationship_type and relationship_type not in global_stats[name]["relationships"]:
                            if len(global_stats[name]["relationships"]) < 10:
                                global_stats[name]["relationships"].append(relationship_type)

                        # 收集与男主的互动记录（带 chunk_index，全部保留）
                        interaction = (char.get("interaction_with_male_lead", "") or "").strip()
                        if interaction:
                            existing_interactions = [s[1] if isinstance(s, (tuple, list)) else s for s in global_stats[name]["interactions"]]
                            if interaction not in existing_interactions:
                                global_stats[name]["interactions"].append((char_chunk_idx, interaction))

                        # 收集感情信号（带 chunk_index，全部保留）
                        emotion = (char.get("emotion_signals", "") or "").strip()
                        if emotion:
                            existing_emotions = [s[1] if isinstance(s, (tuple, list)) else s for s in global_stats[name]["emotion_signals"]]
                            if emotion not in existing_emotions:
                                global_stats[name]["emotion_signals"].append((char_chunk_idx, emotion))

                completed_chunks.add(chunk_idx)
                if len(completed_chunks) % 5 == 0 or len(completed_chunks) == total_chunks:
                    save_checkpoint(
                        global_stats,
                        male_protagonist_stats,
                        max(completed_chunks) if completed_chunks else -1,
                        progress_flags,
                        completed_chunks=completed_chunks,
                    )
                return True

            failed_chunks = []  # 记录失败的块
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_chunk = {
                    executor.submit(analyze_chunk_for_heroines, chunk_data[0], chunk_data[1], total_chunks): chunk_data[1] 
                    for chunk_data in chunks_to_process
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_chunk), 
                                 total=len(chunks_to_process), desc="扫描中", unit="块"):
                    chunk_idx = future_to_chunk[future]
                    try:
                        result = future.result()
                        ok = _apply_scan_result(result, chunk_idx)
                        if not ok:
                            failed_chunks.append(chunk_idx)
                            
                    except Exception as exc:
                        logger.error(f"块 {chunk_idx} 处理异常: {exc}")
                        failed_chunks.append(chunk_idx)
            
            # ========== 补漏：扫描结束后立即检查遗漏块并补扫 ==========
            # 说明：某些情况下 API 会慢/超时导致单块失败。以前需要下次运行才重试，这里改为当次补齐。
            MAX_PATCH_ROUNDS = 3
            for round_no in range(1, MAX_PATCH_ROUNDS + 1):
                missing = [i for i in range(total_chunks) if i not in completed_chunks]
                if not missing:
                    break

                print(f"\n🔁 补漏扫描：发现 {len(missing)} 个遗漏块，开始第 {round_no}/{MAX_PATCH_ROUNDS} 轮补扫（降低并发避免限速）...")
                logger.warning(f"补漏扫描第{round_no}轮：遗漏块={missing[:20]}{'...' if len(missing) > 20 else ''}")

                retry_workers = max(1, min(3, MAX_WORKERS))  # 降低并发，减少限速导致的慢/超时
                retry_failed = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=retry_workers) as executor:
                    future_to_idx = {
                        executor.submit(analyze_chunk_for_heroines, chunks[i], i, total_chunks): i
                        for i in missing
                    }
                    for future in tqdm(
                        concurrent.futures.as_completed(future_to_idx),
                        total=len(missing),
                        desc=f"补漏中(第{round_no}轮)",
                        unit="块",
                    ):
                        idx = future_to_idx[future]
                        try:
                            res = future.result()
                            ok = _apply_scan_result(res, idx)
                            if not ok:
                                retry_failed.append(idx)
                        except Exception as exc:
                            logger.error(f"补漏块 {idx} 处理异常: {exc}")
                            retry_failed.append(idx)

                # 轮次结束后保存断点
                save_checkpoint(
                    global_stats,
                    male_protagonist_stats,
                    max(completed_chunks) if completed_chunks else -1,
                    progress_flags,
                    completed_chunks=completed_chunks,
                )

                if retry_failed:
                    logger.warning(f"补漏第{round_no}轮仍失败 {len(retry_failed)} 块：{sorted(retry_failed)}")
                    time.sleep(2)  # 给 API 一点缓冲时间再进入下一轮

            # 输出失败块信息
            if failed_chunks:
                logger.warning(f"本次扫描有 {len(failed_chunks)} 个块失败: {sorted(failed_chunks)}")
                # 注意：已经做过补漏扫描，但仍可能有极少数块因网络/限速/内容异常失败
                remaining_missing = [i for i in range(total_chunks) if i not in completed_chunks]
                if remaining_missing:
                    print(f"\n⚠ 仍有 {len(remaining_missing)} 个块分析失败，将在下次运行时继续重试：{remaining_missing[:20]}{'...' if len(remaining_missing) > 20 else ''}")
            
            # 只有所有块都成功完成时才标记阶段一完成
            if len(completed_chunks) >= total_chunks:
                progress_flags["scanned"] = True
                logger.info("所有块扫描完成，标记阶段一完成")
            else:
                logger.info(f"扫描进度: {len(completed_chunks)}/{total_chunks} 块完成")
            
            save_checkpoint(global_stats, male_protagonist_stats, max(completed_chunks) if completed_chunks else -1, progress_flags, completed_chunks=completed_chunks)

        # 识别男主（简单逻辑：出现次数最多的）
        print("\n" + "=" * 70)
        print(f"【阶段二】识别男主角（发现 {len(male_protagonist_stats)} 个候选）...")
        
        if progress_flags.get("male_identified") and loaded_male_final:
            # 旧断点升级：如果缺少 identities 字段，则用现有 male_protagonist_stats 重新汇总一次
            if isinstance(loaded_male_final, dict) and ("identities" in loaded_male_final):
                male_protagonist = loaded_male_final
                print(f"★ 阶段二已完成，跳过男主识别。男主: {male_protagonist.get('name')}")
            else:
                male_protagonist = identify_male_protagonist(male_protagonist_stats)
                print(f"★ 阶段二（升级）已重新汇总男主: {male_protagonist.get('name') if male_protagonist else '未知'}")
                progress_flags["male_identified"] = True
                save_checkpoint(global_stats, male_protagonist_stats, total_chunks - 1, progress_flags, merged_stats=loaded_merged_stats, heroine_result=loaded_heroine_result, male_protagonist_final=male_protagonist, completed_chunks=completed_chunks)
        else:
            male_protagonist = identify_male_protagonist(male_protagonist_stats)
            if male_protagonist:
                print(f"  ? 男主角: {male_protagonist['name']}")
                if male_protagonist.get('other_names'):
                    print(f"    别称: {', '.join(male_protagonist['other_names'])}")
            progress_flags["male_identified"] = True
            save_checkpoint(global_stats, male_protagonist_stats, total_chunks - 1, progress_flags, merged_stats=None, heroine_result=None, male_protagonist_final=male_protagonist, completed_chunks=completed_chunks)
        
        print("\n" + "=" * 70)
        print(f"【阶段三】别称识别与合并（共发现 {len(global_stats)} 个女性角色名）...")
        
        # 进行别称合并
        if progress_flags.get("alias_merged") and loaded_merged_stats is not None:
            merged_stats = loaded_merged_stats
            print("★ 阶段三已完成，跳过别称合并。")
        else:
            merged_stats = merge_aliases(global_stats)
            progress_flags["alias_merged"] = True
            save_checkpoint(global_stats, male_protagonist_stats, total_chunks - 1, progress_flags, merged_stats=merged_stats, heroine_result=None, male_protagonist_final=male_protagonist, completed_chunks=completed_chunks)
        
        print("\n" + "=" * 70)
        print("【阶段四】最终女主角识别...")
        
        # 识别女主角
        if progress_flags.get("heroines_identified") and loaded_heroine_result is not None:
            heroine_result = loaded_heroine_result
            print("★ 阶段四已完成，跳过女主识别。")
        else:
            heroine_result = identify_heroines(merged_stats)
            progress_flags["heroines_identified"] = True
            save_checkpoint(global_stats, male_protagonist_stats, total_chunks - 1, progress_flags, merged_stats=merged_stats, heroine_result=heroine_result, male_protagonist_final=male_protagonist, completed_chunks=completed_chunks)
        
        # 女主最终合并检查
        if progress_flags.get("heroines_final_merged"):
            print("★ 女主最终合并已完成，跳过。")
        else:
            print("\n" + "=" * 70)
            print("【阶段四点五】女主最终合并检查...")
            heroine_result = merge_heroines_final(heroine_result, merged_stats)
            progress_flags["heroines_final_merged"] = True
            save_checkpoint(global_stats, male_protagonist_stats, total_chunks - 1, progress_flags, merged_stats=merged_stats, heroine_result=heroine_result, male_protagonist_final=male_protagonist, completed_chunks=completed_chunks)
        
        print("\n" + "=" * 70)
        print("【阶段五】生成分析报告...")
        
        # 生成报告
        if progress_flags.get("report_generated") and progress_flags.get("report_files"):
            print("★ 阶段五已完成，跳过报告生成。")
            detailed_file = progress_flags["report_files"].get("detailed")
            detailed_snapshot_file = progress_flags["report_files"].get("detailed_snapshot")
            report_file = progress_flags["report_files"].get("report")
        else:
            final_report = generate_final_report(heroine_result, merged_stats, male_protagonist)
            
            # 获取文件名用于保存
            clean_filename = os.path.splitext(os.path.basename(NOVEL_FILE_PATH))[0]
            
            detailed_file, detailed_snapshot_file, report_file = export_results(
                merged_stats, 
                heroine_result,
                final_report,
                male_protagonist=male_protagonist,
                filename_prefix=clean_filename
            )
            progress_flags["report_generated"] = True
            progress_flags["report_files"] = {
                "detailed": detailed_file,
                "detailed_snapshot": detailed_snapshot_file,
                "report": report_file
            }
            save_checkpoint(global_stats, male_protagonist_stats, total_chunks - 1, progress_flags, merged_stats=merged_stats, heroine_result=heroine_result, male_protagonist_final=male_protagonist, completed_chunks=completed_chunks)
        
        print("\n" + "=" * 70)
        print("? 分析完成！")
        print(f"? 详细数据: {detailed_file}")
        print(f"? 分析报告: {report_file}")
        if token_tracker is not None:
            snap = token_tracker.snapshot()
            print(f"? Token 统计: 输入 {snap.get('input', 0)} ，输出 {snap.get('output', 0)} ，总计 {snap.get('total', 0)}")
            token_tracker.flush(status="finished")
        print("=" * 70)
        
        # 打印识别结果摘要
        print("\n【主角识别结果】")
        print("-" * 40)
        
        # 显示男主
        if male_protagonist:
            aliases_str = f"（{', '.join(male_protagonist.get('other_names', []))}）" if male_protagonist.get('other_names') else ""
            print(f"  ★ 男主: {male_protagonist.get('name', '未知')}{aliases_str}")
            print(f"     身份: {male_protagonist.get('identity', '未知')}")
        
        # 显示女主
        heroines = heroine_result.get('heroines', [])
        if heroines:
            print()
            for h in heroines:
                aliases_str = f"（{', '.join(h.get('aliases', []))}）" if h.get('aliases') else ""
                print(f"  {h.get('importance_rank', '?')}. {h.get('name', '未知')}{aliases_str}")
                print(f"     关系: {h.get('relationship_type', '未知')}")
        print("-" * 40)
        print(f"\n【分析】{heroine_result.get('analysis', '')[:200]}...")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        print(f"\n? 错误: {e}")
        try:
            if token_tracker is not None:
                token_tracker.flush(status="error", reason=str(e))
        except Exception:
            pass
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
