import os
import re
import json
import time
import glob
import logging
import concurrent.futures
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import threading
from openai import OpenAI
try:
    from openai import APIStatusError
except Exception:
    APIStatusError = Exception
from tqdm import tqdm
from api_client import make_chat_completion
from token_tracker import create_default_tracker
from shared_utils import get_base_dir, read_file_safely
from text_anchor import build_chunk_manifest, save_chunk_manifest


# ================= 配置区域 =================
# API Key 支持池化：优先读取 API_KEY_POOL（逗号分隔），否则读取 API_KEY
API_KEY_POOL = [
    k.strip()
    for k in os.environ.get("API_KEY_POOL", os.environ.get("API_KEY", "")).split(",")
    if k.strip()
]
API_KEY = API_KEY_POOL[0] if API_KEY_POOL else ""
BASE_URL = os.environ.get("BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("MODEL_NAME", "deepseek-chat")
CHUNK_SIZE = 6000
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "1200"))
_base_workers = int(os.environ.get("MAX_WORKERS", "6"))
MAX_WORKERS = _base_workers + 4
ENABLE_FACT_BOOST = os.environ.get("ENABLE_FACT_BOOST", "1").strip() == "1"
FACT_BOOST_MAX_CALLS_PER_CHUNK = int(os.environ.get("FACT_BOOST_MAX_CALLS_PER_CHUNK", "2"))
DIM_BOOST_MAX_PER_CHUNK = int(os.environ.get("DIM_BOOST_MAX_PER_CHUNK", "3"))
# 扫描完成后自动“补扫遗漏/失败块”的轮数（避免因网络/格式偶发导致的缺块）
RESCAN_ROUNDS = int(os.environ.get("RESCAN_ROUNDS", "3"))
# 补扫阶段线程数（默认更保守，降低被限流/超时概率）
RESCAN_MAX_WORKERS = int(os.environ.get("RESCAN_MAX_WORKERS", "4"))
ENABLE_GLOBAL_RESCAN = os.environ.get("ENABLE_GLOBAL_RESCAN", "1").strip() == "1"
MAX_MIDDLE_SUMMARY_CALLS = int(os.environ.get("MAX_MIDDLE_SUMMARY_CALLS", "10"))
# ---- 全局补扫优化 配置 ----
RESCAN_MAX_HITS = int(os.environ.get("RESCAN_MAX_HITS", "4"))
RESCAN_PRE_FILTER_THRESHOLD = float(os.environ.get("RESCAN_PRE_FILTER_THRESHOLD", "1.0"))
RESCAN_MAX_WINDOW = int(os.environ.get("RESCAN_MAX_WINDOW", "2000"))
RESCAN_MAX_PROMPT_HEROINES = int(os.environ.get("RESCAN_MAX_PROMPT_HEROINES", "4"))

RULES_FILE = os.path.join(get_base_dir(), "config", "rules.json")
LEARNED_KEYWORDS_DIR = os.path.join(get_base_dir(), "results", "learned_keywords")
SEED_FILE = os.path.join(get_base_dir(), "config", "seed_keywords.json")

SNAPSHOT_LOCK = threading.Lock()
DETAIL_FILE_LOCK = threading.Lock()
CHECKPOINT_LOCK = threading.RLock()
PROGRESS_LOCK = threading.Lock()
MIDDLE_SUMMARY_LOCK = threading.Lock()

# 以下变量在 main() 中按每本小说重新初始化
NOVEL_FILE_PATH = None
CHECKPOINT_FILE = None
clean_filename = None
OUTPUT_DIR = None
logger = logging.getLogger(__name__)
CURRENT_CHUNK_PLAN_METADATA = None
CHUNK_SUMMARIES = {}
_ACTIVE_PROGRESS_STATE = None
_middle_summary_calls = 0
_ACTIVE_DETAIL_PATH = None


def find_latest_scan_checkpoint(prefix: str):
    """找到同名小说最近的扫描断点目录"""
    safe_prefix = glob.escape(prefix)
    results_dir = os.path.join(get_base_dir(), "results")
    pattern = os.path.join(results_dir, f"{safe_prefix}_scan_*", "latest_checkpoint.json")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    return os.path.dirname(latest)


def _resolve_detail_path(explicit_detail_path=None, checkpoint_detail_path=None):
    if explicit_detail_path:
        return explicit_detail_path
    if checkpoint_detail_path:
        return checkpoint_detail_path
    if _ACTIVE_DETAIL_PATH:
        return _ACTIVE_DETAIL_PATH
    return _find_latest_detail_file()


def _peek_checkpoint_detail_path():
    if not CHECKPOINT_FILE or not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return (data or {}).get("detail_path")


def _dedupe_names(values):
    seen = set()
    out = []
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_keyword_words(words, enforce_length_bounds=False):
    normalized = []
    seen = set()
    for word in words or []:
        text = str(word).strip()
        if not text:
            continue
        if enforce_length_bounds and (len(text) < 2 or len(text) > 10):
            continue
        if text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _to_name_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return _dedupe_names([value])
    if isinstance(value, dict):
        names = []
        if value.get("name"):
            names.append(value.get("name"))
        names.extend(_to_name_list(value.get("other_names")))
        names.extend(_to_name_list(value.get("aliases")))
        return _dedupe_names(names)
    if isinstance(value, (list, tuple, set)):
        return _dedupe_names([item for item in value if item])
    return _dedupe_names([value])


def _normalize_male_identity(male_protagonist):
    names = []
    raw = male_protagonist
    if isinstance(raw, dict):
        if raw.get("name"):
            names.append(raw.get("name"))
        elif raw:
            first_key = next(iter(raw.keys()), None)
            if isinstance(first_key, str):
                names.append(first_key)
        names.extend(_to_name_list(raw.get("other_names")))
        names.extend(_to_name_list(raw.get("aliases")))
    else:
        names.extend(_to_name_list(raw))

    names = _dedupe_names(names)
    if not names:
        return None

    primary_name = names[0]
    aliases = [name for name in names[1:] if name != primary_name]
    return {
        "name": primary_name,
        "aliases": aliases,
        "other_names": aliases[:],
        "all_names": [primary_name] + aliases,
    }


def _male_identity_prompt_text(male_identity):
    identity = _normalize_male_identity(male_identity)
    if not identity:
        return ""
    text = f"【男主】{identity['name']}"
    if identity["aliases"]:
        text += f"（别名：{', '.join(identity['aliases'][:10])}）"
    return text


def _build_chunk_plan_metadata(text=None, chunks=None, chunk_size=None, overlap=None, chunk_manifest=None):
    if chunk_manifest is not None:
        return {
            "chunk_size": int(chunk_manifest.get("chunk_size", chunk_size if chunk_size is not None else CHUNK_SIZE)),
            "chunk_overlap": int(chunk_manifest.get("chunk_overlap", overlap if overlap is not None else CHUNK_OVERLAP)),
            "chunk_count": int(chunk_manifest.get("chunk_count", len((chunk_manifest or {}).get("chunks", [])))),
            "text_length": int(chunk_manifest.get("text_length", len((chunk_manifest or {}).get("full_text", "")))),
            "chunking_mode": chunk_manifest.get("chunking_mode", "paragraph_window_v1"),
            "version": int(chunk_manifest.get("version", 1)),
            "signature": chunk_manifest.get("signature", ""),
        }
    return {
        "chunk_size": int(chunk_size if chunk_size is not None else CHUNK_SIZE),
        "chunk_overlap": int(overlap if overlap is not None else CHUNK_OVERLAP),
        "chunk_count": len(chunks or []),
        "text_length": len(text or ""),
    }

# ---- API 调用封装：统一收敛到 api_client.py（只需修改 api_client.py 即可全局生效）----
MAX_403_RETRIES = 3
MAX_TIMEOUT_RETRIES = 3  # 连续超时 3 次则标记 key 不可用
REQUEST_TIMEOUT = 120  # 请求超时时间（秒）

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


chat_completion = make_chat_completion(
    openai_client_factory=_openai_client_factory,
    api_key_pool=API_KEY_POOL,
    base_url=BASE_URL,
    request_timeout=REQUEST_TIMEOUT,
    max_retries=5,
    max_403_retries=MAX_403_RETRIES,
    max_timeout_retries=MAX_TIMEOUT_RETRIES,
    base_delay=2,
    logger=logger,
)

token_tracker = None


def init_token_tracker(book_name, run_id=None):
    global token_tracker
    token_tracker = create_default_tracker(
        "novel_scan.py",
        book_name=book_name,
        out_path=os.path.join(get_base_dir(), "results", "token_usage.json"),
        run_id=run_id,
    )
    return token_tracker


def record_usage(resp):
    try:
        if token_tracker is not None:
            token_tracker.record(resp)
    except Exception:
        pass

def load_rules():
    """加载 JSON 规则文件"""
    if not os.path.exists(RULES_FILE):
        logger.error(f"找不到规则文件: {RULES_FILE}")
        return None, None
    
    try:
        data = json.loads(read_file_safely(RULES_FILE))
        return data.get("categories", []), data.get("glossary", [])
    except Exception as e:
        logger.error(f"规则文件解析失败: {e}")
        return None, None


# ---------------- 断点续传 ----------------
def save_checkpoint(all_issues, all_heroine_facts, processed_chunks, extra_relations_all=None, failed_chunks=None,
                    current_chunk_idx=None, chunk_plan_metadata=None, chunk_summaries=None, heroine_profiles=None,
                    detail_path=None, rescan_done_chunks=None, rescan_completed=None):
    """保存扫描进度
    
    Args:
        current_chunk_idx: 当前刚完成的块索引（0-based），用于日志显示
    """
    data = {
        "issues": all_issues,
        "heroine_facts": all_heroine_facts,
        "extra_relations": extra_relations_all or [],
        "processed_chunks": list(sorted(processed_chunks)),
        "failed_chunks": list(sorted(failed_chunks or [])),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    effective_chunk_plan = chunk_plan_metadata if chunk_plan_metadata is not None else CURRENT_CHUNK_PLAN_METADATA
    effective_chunk_summaries = chunk_summaries if chunk_summaries is not None else CHUNK_SUMMARIES
    if effective_chunk_plan:
        data["chunk_plan"] = effective_chunk_plan
    if effective_chunk_summaries:
        data["chunk_summaries"] = {str(k): v for k, v in sorted(effective_chunk_summaries.items()) if v}
    if heroine_profiles is not None:
        data["heroine_profiles"] = heroine_profiles
    effective_detail_path = detail_path if detail_path is not None else _ACTIVE_DETAIL_PATH
    if effective_detail_path:
        data["detail_path"] = effective_detail_path
    if rescan_done_chunks is not None:
        data["rescan_done_chunks"] = list(sorted(rescan_done_chunks))
    if rescan_completed is not None:
        data["rescan_completed"] = bool(rescan_completed)
    try:
        with CHECKPOINT_LOCK:
            with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 优先显示当前完成的块，否则显示进度统计
        if current_chunk_idx is not None:
            logger.info(f"✅ 断点已保存: {CHECKPOINT_FILE}（刚完成第 {current_chunk_idx + 1} 块，累计完成 {len(processed_chunks)} 块，失败 {len(failed_chunks or [])} 块）")
        elif processed_chunks:
            logger.info(f"✅ 断点已保存: {CHECKPOINT_FILE}（累计完成 {len(processed_chunks)} 块，失败 {len(failed_chunks or [])} 块）")
        else:
            logger.info(f"✅ 断点已保存: {CHECKPOINT_FILE}（当前无已完成块）")
    except Exception as e:
        logger.error(f"保存断点失败: {e}")


def load_checkpoint():
    """加载扫描进度，若不存在返回空"""
    global CHUNK_SUMMARIES
    if not CHECKPOINT_FILE or not os.path.exists(CHECKPOINT_FILE):
        CHUNK_SUMMARIES = {}
        return [], [], set(), [], set(), None, None, set(), False
    try:
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        issues = data.get("issues", [])
        heroine_facts = data.get("heroine_facts", [])
        # 兼容旧版 heroine_status
        if not heroine_facts and data.get("heroine_status"):
            heroine_facts = []  # 旧版断点不兼容，重新扫描
        processed_chunks = set(data.get("processed_chunks", []))
        extra_relations = data.get("extra_relations", [])
        failed_chunks = set(data.get("failed_chunks", []))
        heroine_profiles = data.get("heroine_profiles") if "heroine_profiles" in data else None
        detail_path = data.get("detail_path")
        rescan_done_chunks = set(data.get("rescan_done_chunks", []))
        rescan_completed = bool(data.get("rescan_completed", False))
        CHUNK_SUMMARIES = {
            int(k): str(v)
            for k, v in (data.get("chunk_summaries", {}) or {}).items()
            if str(k).strip() and str(v).strip()
        }
        saved_chunk_plan = data.get("chunk_plan")
        if saved_chunk_plan and CURRENT_CHUNK_PLAN_METADATA and saved_chunk_plan != CURRENT_CHUNK_PLAN_METADATA:
            logger.warning("⚠️ 检测到切块配置或文本长度变化，旧断点不再复用，将从头开始扫描。")
            CHUNK_SUMMARIES = {}
            return [], [], set(), [], set(), None, None, set(), False
        logger.info(f"📂 已加载断点，已完成 {len(processed_chunks)} 个片段，失败 {len(failed_chunks)} 个片段")
        return issues, heroine_facts, processed_chunks, extra_relations, failed_chunks, heroine_profiles, detail_path, rescan_done_chunks, rescan_completed
    except Exception as e:
        logger.error(f"加载断点失败: {e}，将从头开始")
        CHUNK_SUMMARIES = {}
        return [], [], set(), [], set(), None, None, set(), False


def _init_chunk_progress(total, initial=0, desc="", processed_chunks=None, failed_chunks=None):
    bar = tqdm(total=total, initial=initial, desc=desc)
    state = {
        "bar": bar,
        "counted": set(processed_chunks or []),
    }
    bar.set_postfix({"success": len(processed_chunks or []), "failed": len(failed_chunks or [])})
    return state


def _advance_chunk_progress(idx, processed_chunks, failed_chunks, progress_state):
    if not progress_state:
        return
    with PROGRESS_LOCK:
        counted = progress_state.setdefault("counted", set())
        bar = progress_state.get("bar")
        if bar is None:
            return
        if idx not in counted:
            counted.add(idx)
            bar.update(1)
        bar.set_postfix({"success": len(processed_chunks or []), "failed": len(failed_chunks or [])})


def _close_chunk_progress(progress_state):
    if not progress_state:
        return
    with PROGRESS_LOCK:
        bar = progress_state.get("bar")
        if bar is not None:
            bar.close()
    
# 兼容 novel4.py 输出的路径与数据结构的女主/男主提取
# 优先使用本函数，覆盖上方旧版 find_heroines 定义

def find_heroines(detail_path=None):
    """自动查找此前分析生成的详细报告，提取女主名单和男主姓名
    兼容 novel4.py 的输出路径与数据格式
    返回: (heroines: list[str], male_protagonist: dict|None)
    """
    latest_file = _resolve_detail_path(explicit_detail_path=detail_path)
    base = get_base_dir()
    _results_dir = os.path.join(base, "results")
    patterns = [
        os.path.join(_results_dir, "**", f"{clean_filename}*_detailed_*.json"),
        os.path.join(_results_dir, "**", "*_detailed_*.json"),
        os.path.join(base, "**", f"{clean_filename}*_detailed_*.json"),
        os.path.join(base, "**", "*_detailed_*.json"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if os.path.isfile(f)]

    # 优先保留文件名中包含 clean_filename 的文件
    if clean_filename:
        filtered = [f for f in files if clean_filename in os.path.basename(f)]
        if filtered:
            files = filtered

    if latest_file and os.path.isfile(latest_file):
        files = [latest_file]

    if not files:
        logger.warning("⚠️ 未找到该小说的角色分析报告，将无法针对特定女主进行‘初/残’判定。")
        return [], None

    latest_file = max(files, key=os.path.getctime)
    logger.info(f"📎 载入角色名单自: {latest_file}")

    def parse_heroines(file_path):
        try:
            data = json.loads(read_file_safely(file_path))
        except Exception as e:
            logger.error(f"角色文件读取失败: {e}")
            return [], None

        male_identity = None
        mp = data.get("male_protagonist") or data.get("male_protagonist_stats")
        male_identity = _normalize_male_identity(mp)

        # novel4: heroine_result
        hr = data.get("heroine_result", {}).get("heroines", [])
        if hr:
            names = [h.get("name") for h in hr if h.get("name")]
            if names:
                return names, male_identity

        # novel4: all_female_characters（按 avg_score/count 排序）
        afc = data.get("all_female_characters", {})
        if afc:
            sorted_chars = sorted(afc.items(), key=lambda x: (x[1].get("avg_score", 0), x[1].get("count", 0)), reverse=True)
            names = [name for name, _ in sorted_chars[:15]]
            if names:
                return names, male_identity

        # 旧格式: character_details
        details = data.get("character_details", {})
        if details:
            names = []
            for name, info in details.items():
                types = info.get("types", [])
                if ("女主" in types) or ("女配" in types and info.get("avg_score", 0) > 5) or ("女" in ''.join(types) and info.get("avg_score", 0) >= 6):
                    names.append(name)
            if names:
                return names, male_identity

        # 兜底: all_characters / characters，按分数排序
        for key in ["all_characters", "characters"]:
            details = data.get(key, {})
            if details:
                sorted_chars = sorted(details.items(), key=lambda x: (x[1].get("avg_score", 0), x[1].get("count", 0)), reverse=True)
                names = [name for name, _ in sorted_chars[:15]]
                if names:
                    return names, male_identity

        return [], male_identity

    heroines, male_identity = parse_heroines(latest_file)
    logger.info(f"💡 锁定关注女主: {heroines}")
    if male_identity and male_identity.get("name"):
        logger.info(f"👑 识别到男主: {male_identity['name']}")
    return heroines, male_identity

_PROMPT_CHECKLIST = [
    "□ 本片段中出现名字的每个女主，是否都在 heroine_facts 中有对应条目？",
    "□ 每个条目的五个维度是否都已逐项检查过（即使为空也必须输出 []）？",
    "□ 文中出现“丈夫/夫君/未婚夫/前夫/男友/恋人/婚约/嫁给/纳妾”等关系线索时，是否已检查并写入 partner_relations？",
    "□ 文中出现亲密肉体接触或更进一步行为时，是否已区分写入 physical_contacts 或 sexual_relations，而不是遗漏？",
    "□ 文中出现“喜欢/爱/心动/倾心/钟情/表白/告白”等情感线索时，是否已检查 romantic_feelings？",
    "□ 你是否执行了【双向锚定】检查：不仅看女主做了什么，也看了别人对女主做了什么、别人如何描述她的关系身份？",
    "□ partner_relations 的每一条是否都检查了 forced 字段；若 evidence 明确含“被迫/强迫/威逼/胁迫/逼婚/包办/卖嫁/不得不嫁”等线索，是否已标记 forced=true，且未遗漏该条？",
    "□ physical_contacts 与 partner_relations 是否都已做“性别与男主校验”：只记录非男主男性；男主、女性对象、性别不明对象是否都已排除？",
    "□ children_info 是否已检查怀孕/生子/流产/收养/领养/继子女/认作女儿等线索，并正确区分 father、origin、is_biological、conception_method，且没有反推 romantic_feelings 或 sexual_relations？",
    "□ 每条 evidence 是否都是当前片段原文的连续子串，且 heroine_facts 中没有混入推测、假设、比喻、传闻、劝说或弱证据内容？",
]


def _render_categories(categories):
    lines = []
    for cat in categories or []:
        if not isinstance(cat, dict):
            continue
        lines.append(f"[{cat.get('name', '')}]: {cat.get('description', '')}")
        for point in cat.get("points", []) or []:
            if not isinstance(point, dict):
                continue
            lines.append(f"- {point.get('name', '')}: {point.get('description', '')}")
    return "\n".join(lines).strip()


def _render_glossary(glossary):
    lines = []
    for term in glossary or []:
        if not isinstance(term, dict):
            continue
        lines.append(f"- {term.get('term', '')}: {term.get('definition', '')}")
    return "\n".join(lines).strip()


def build_prompt(categories, glossary, heroines, male_protagonist=None):
    """构建三层扫描 Prompt：保留原长文规则，只做重排与局部增补。"""
    male_prompt_text = _male_identity_prompt_text(male_protagonist)
    categories_text = _render_categories(categories)
    glossary_text = _render_glossary(glossary)
    checklist_text = "\n".join(_PROMPT_CHECKLIST)

    layer_one = "你是一个深谙'男性向网文'逻辑的扫书专家。你的任务是为读者排毒。请注意：网文的道德逻辑与现实世界不同，请务必执行以下双重标准。\n"

    layer_one += "\n【任务二：女主事实抽取（仅抽取事实，不做判断）】\n"
    if heroines:
        layer_one += f"请重点关注以下女性角色：{', '.join(heroines)}\n"
    else:
        layer_one += "请关注文中的主要女性角色。\n"
    if male_prompt_text:
        layer_one += f"{male_prompt_text}\n"

    layer_one += """
【重要】本任务只抽取客观事实，不做任何"初/处"判断！

【⚠️⚠️⚠️ evidence 字段硬规则（最高优先级！必须严格遵守！）】

evidence 必须是【小说片段原文】的连续子串，禁止任何改写！

1. **严格逐字复制**：evidence 只能从当前小说片段中复制一段"连续子串"
   - 禁止改写、同义替换、补字、删字、调整语序
   - 禁止添加省略号（……/.../…/..），除非原文就包含
   - 禁止把多句话拼接成一句
   - 如果原文有换行，evidence 必须保持换行（不要替换成空格）

2. **evidence 选择规则**：
   - 优先选包含关键词的句子片段：怀孕/孩子/生下/流产/亲生/谁的孩子/几个月/肚子/丈夫/未婚夫/亲吻/抚摸 等
   - 建议长度 12~80 字（但允许更短/更长，只要是原文连续子串即可）
   - 若找不到合适 evidence 或不确定，必须输出空字符串 ""

3. **禁止事项（违反将导致后续定位失败）**：
   - ❌ 把"她怀了他的孩子"改写为"她怀孕了"
   - ❌ 把"小月说：'我怀孕了'"简化为"我怀孕了"
   - ❌ 把原文"生下一个女儿"加省略号变成"生下……女儿"
   - ❌ 把原文中的换行符替换为空格

【⚠️ 事实抽取硬规则（必须严格遵守 - 宁可漏报，不要错挂！）】

1. **主体锚定原则**（防止错挂主体）：
   - heroine_facts 以精确为先：证据不足/主体不明/对话劝说类比/假设转述一律不要写入 heroine_facts！
   - 引号对话里出现"我/你/我们/她"等代词时，必须先判断说话人和指代对象！
   - 若说话人不是该女主，或指代对象不明确，**不得**作为该女主的事实写入 heroine_facts！
   - 示例错误：「"你说不定也会怀上孩子"」→ 这是劝说/假设，不是事实陈述！
   - 示例错误：「"我们都怀孕了"」中的"我们"→ 必须确认说话人身份，不能假设是女主！

【双向锚定抽取原则】
- 不仅看女主做了什么，也要看别人对女主做了什么、别人如何描述她的关系身份。
- 不仅看女主主动行为，也要看女主被动承受、被指认、被定义的关系线索。
- 例如：“她被订婚了”“她被迫嫁给某人”“他亲了她”“她怀了谁的孩子”等，都要反向锚定回女主后再判断是否入表。

2. **语用类型过滤**（防止劝说/类比/假设污染）：
   - 以下语气词/句式表明**不是事实陈述**，禁止写入 heroine_facts：
     * 劝说建议类："说不定/难道/想让/应该/可以/不如/要不"
     * 类比假设类："更像/差不多/迟早/可能/也许/大概/估计/如果/假如/万一"
     * 反问质疑类："难道/莫非/怎么会/不会吧"
     * 传闻转述类："听说/据说/有人说/传言"
   - 除非同段有**强事实陈述句**明确锚定女主，否则禁止写入！

3. **身世倒装过滤**（防止把女主的出身当成女主的孩子）：
   - "某人生下了{女主名}"或"{女主名}是某人的女儿/孩子"→ 这是女主的身世，不是女主的孩子！
   - 这类信息**绝对禁止**写入 children_info！

请从文本中抽取以下5类事实（若无则输出空数组 []，**禁止用"无"作为占位符**）：

1. **性关系事实**（sexual_relations）：
   - 女主与谁发生过性关系？是男主还是其他人？
   - 抽取细节：是否明确描写破处/初夜？是否被强迫？
   - 必须有原文证据支持
   - 【重要排除】“强奸未遂/企图强奸/没得逞/未遂”等描述属于受害或风险事实，不等于已发生性关系；不得仅凭这些描述写入 sexual_relations
   - 【隐含性证据支持】若无同房描写，但有以下**强后果证据**且指向女主，可记录为隐含证据：
     * "怀了X的孩子/孩子是X的/亲子鉴定确认父亲是X"
     * "流产/打胎/堕胎"明确指向女主（前提：未出现试管/魔法/收养等否定）
   - 新增字段：
     * evidence_level: "explicit"（明确描写床戏/性行为） 或 "implicit"（通过后果推断）
     * implicit_type（当evidence_level="implicit"时）: "pregnancy_paternity"/"abortion"/"cohabitation"/"virginity_statement"/"other"
   - 【若无性关系记录，输出空数组 []】

2. **孩子来源事实**（children_info）：
   - 女主是否有孩子？孩子是否为亲生？
   - 【关键】is_biological 定义（生物学意义）：
     * is_biological = true：女主是生母/基因母亲（不论受孕方式，只要是亲生就算）
     * is_biological = false：非亲生孩子（收养/领养/继子女/认作女儿/捡来的/徒弟当女儿等）
   - 新增字段（扩展记录，必须填写）：
     * child_status: "born"（已出生）/ "pregnant"（仅怀孕提及）/ "adopted"（收养）/ "unknown"
     * conception_method: "sex"（性行为）/ "ivf_or_ai"（试管/人工授精）/ "surrogacy"（代孕）/ "magic"（魔法/神赐）/ "unknown"
     * speech_act: "asserted_fact"（事实陈述）/ "hypothesis"（假设）/ "advice_persuasion"（劝说建议）/ "comparison"（类比）/ "rumor"（传闻）
     * evidence_strength: "strong"（明确事实）/ "medium"（较明确）/ "weak"（模糊/推测）
     * speaker（可选）：若 evidence 来源于引号对话，给出说话人是谁
   - 【⚠️ 语用过滤规则】只有 speech_act="asserted_fact" 且 evidence_strength≠"weak" 才允许写入 heroine_facts！其他类型不要写入！
   - 【⚠️ 仅怀孕提及≠有孩子】"我们都怀孕了/肚子里的孩子/怀上了"等表述，若无后续"生下/出生/分娩"证据，必须标注 child_status="pregnant"，且必须标注 evidence_strength="weak"
   - 【重要】若证据出现“没有性行为的情况下怀孕/试管婴儿/人工授精/供精/胚胎移植”等，必须写入 children_info，并优先标注 conception_method="ivf_or_ai"；不得反推 sexual_relations，不得据此推断“自愿伴侣关系”
   - 【若无符合条件的孩子记录，输出空数组 []】

3. **肉体接触事实**（physical_contacts）：
   - 【只关注非男主男性】：女主被哪些**非男主的男性**触碰过身体？
   - 接触类型：猥亵/亲吻/爱抚/强抱/摸/抚摸等实际肉体接触
   - 【强约束：必须确认对方为男性】只要证据中没有明确"男性线索"（如：他/男人/男声/公子/少爷/王爷/殿下/郎君/丈夫/未婚夫/前夫/男友 等），宁可不记录，输出空数组 []
   - 【强约束：禁止女性触碰误入】任何"女性触碰/闺蜜贴贴/姐妹拥抱/女女亲吻/女主触碰自己/女主触碰别人"等，都不属于本字段，禁止输出到 physical_contacts
   - 【排除】：
     * 男主的任何接触 → 不记录
     * 女性的接触 → 不记录
     * 仅被围观/注视 → 不记录
     * 心灵感应/精神连接 → 不记录
   - 必须有原文证据支持
   - 【若无非男主男性的肉体接触，输出空数组 []】

4. **感情事实**（romantic_feelings）：
   - 女主爱过/喜欢过/动心过哪些**非男主的男性**？
   - 被迫/厌恶/无感的不算
   - 新增字段（建议填写）：
     * speech_act: "asserted_fact"/"hypothesis"/"advice_persuasion"/"comparison"/"rumor"
     * evidence_strength: "strong"/"medium"/"weak"
   - 只有 speech_act="asserted_fact" 且 evidence_strength 不是 "weak" 才允许写入
   - 必须有原文证据支持
   - 【若无非男主感情记录，输出空数组 []】

5. **伴侣关系事实**（partner_relations）：
   - 女主有过哪些**非男主的男性伴侣**？（男友/丈夫/未婚夫/恋人）
   - 关系状态：正式交往/已完婚/订婚未完婚/被迫订婚等
   - 【强约束：必须判断是否强迫/非自愿】每条 partner_relations 都必须标注 forced（true/false）：
     * forced = true：证据明确出现“被迫/强迫/威逼/胁迫/包办婚约/被卖嫁/不得不嫁”等非自愿关系
     * forced = false：证据明确自愿，或未出现强迫证据（禁止臆测）
   - 【重要】即使是被迫关系也要照常抽取，不得漏记；只是在该条将 forced 置为 true
   - 【漏检校验】若文本已明确“被迫/强迫/威逼/胁迫/逼婚/包办”等，并且对象是女主与非男主男性伴侣，则必须输出该条 partner_relations，禁止遗漏
   - 【重要】若证据是“强奸未遂/企图强奸/没得逞/被迫受孕且无性行为/试管受孕”等，属于受害或被迫线索；可记录伴侣关系线索但应优先 forced=true，且不得改写为“恋爱/动心/自愿亲密”
   - 【强约束：必须确认对方为男性】若只有"闺蜜/姐姐/妹妹/女伴"等女性对象，或证据缺乏男性线索，则输出空数组 []
   - 【partner 命名规范】：
     * 优先使用明确的人名（如"张三"、"李四"）
     * 若无明确人名但能确定身份/关系，使用【女主名+关系】格式，如：
       - "墨水心的父亲"（而非"某个男人"）
       - "林黛玉的前夫"（而非"他"）
       - "王小姐的未婚夫"（而非"那个人"）
     * 禁止使用"某个男人/某人/他/那个人"等泛指词！
     * 只有完全无法从上下文推断出任何身份时，才可用"未知"
   - 新增字段（建议填写）：
     * forced: true/false（伴侣关系是否被迫；建议每条都填）
     * speech_act: "asserted_fact"/"hypothesis"/"advice_persuasion"/"comparison"/"rumor"
     * evidence_strength: "strong"/"medium"/"weak"
   - 只有 speech_act="asserted_fact" 且 evidence_strength 不是 "weak" 才允许写入
   - 必须有原文证据支持
   - 【若无非男主伴侣记录，输出空数组 []】

⚠️【严禁占位符】：
- 当某类事实不存在时，必须输出空数组 []
- **禁止**输出 {"partner": "无", ...} 或 {"target": "无", ...} 这样的占位符！
- "无"、"没有"、"不存在" 等字符串**不是角色名**，不要作为 partner/target 输出！

【任务三：补充取证（可选输出）】
- 记录其他可能有用的互动细节（如被搭讪、围观、骚扰等）
"""

    highest_law_block = """
【⚠️ 最高判定法则（必须严格遵守）】
【执行顺序】任务2优先于任务1。请先完成任务2的事实抽取，再使用以下法则执行任务1的扫雷与扫郁闷点判定。

1. 施害主体判定（男主豁免权）：
   - 任何【雷点】（如绿帽、亵女、猥亵），**必须由男主以外的人实施**才算数。
   - **男主绝对豁免**：男主对女主做的任何事（调情、强吻、发生关系），统称为“剧情需要”或“后宫收容”，**绝对不属于毒点**。
   - 判定公式：行为人 == 男主 ? 忽略 : 判定。

2. 关于“非处/非初/破鞋/接盘/NTR”等相关毒点的取证规则（只用于任务一 issues）：

身份≠事实：人妻/夫人/未婚妻/前任/订婚不等于发生过性关系或动心。

只认实锤：仅当出现明确性行为描写（同房/圆房/破身/失身等）或强后果证据（怀了某男的孩子/亲子鉴定/流产打胎且无试管/魔法/收养等否定）才可作为“非处”依据。

只有弱暗示时可以在 issues 里“疑似”上报，但必须在 reason 写明“不确定/证据弱”。

不得把推测写进 heroine_facts（任务二只抽事实）。

3. 绿帽（NTR）判定：
   - 仅限：(女主) + (与男主以外的男人) + (发生实质性/暧昧关系)。
   - 排除：男主睡了女主的亲戚/闺蜜（这是后宫扩充，不是绿帽）。

4. 无罪推定原则：
   - 除非有**确凿证据**（实锤描写）证明她是“非处”或“爱过别人”，否则**一律默认为“是”**（True）。
   - 禁止因为“她有未婚夫”就直接推断“她不干净”。

5. 关于【送女】判定：
   - 仅当男主**主动**把自己喜欢的女人送给别人睡才算。女主自己走失、被抓、或正常分手，不算送女。

6. 视角提示：
   - 请从**男性纯爱/后宫党**的视角审视。对于“处女人妻”（有名无实）这种高价值设定，请务必准确识别，不要误判为雷点。
"""

    layer_two = f"\n{highest_law_block}\n【任务一：扫雷与扫郁闷点】\n请检测是否存在以下情节：\n"
    if categories_text:
        layer_two += f"\n{categories_text}\n"
    layer_two += """
【覆盖与完备性要求】
- **宁可多报，不要漏报**：所有疑似雷点/郁闷点都要输出，哪怕证据较弱，也要在 reason 里说明“不确定”。
- **不要合并省略**：同一片段中出现多个点时，每个点都单独输出一条 issues。
- 如果输出空间紧张，缩短 content/evidence 文本，但不要删条目。
"""
    layer_two += "\n判定标准（Glossary）：\n"
    if glossary_text:
        layer_two += f"{glossary_text}\n"

    layer_three = """
\n【输出要求（务必遵守字段名与格式，只能输出 JSON 对象）】
必须输出一个 JSON 对象，包含以下字段（若无请输出空数组 []），**不要因为篇幅限制省略任何条目**：

1. "issues": 发现的雷点/郁闷点（若行为人是男主，直接忽略，不要输出）。
   - 每条必须包含：category, type, content, reason
   - chunk_index 由系统补全，无需返回

2. "heroine_facts": 女主事实抽取（核心输出！仅抽取事实，不做判断）
   - 每条必须包含 name 和 facts 对象
   - facts 包含5个数组：sexual_relations, children_info, physical_contacts, romantic_feelings, partner_relations
   - 示例（有事实时 - 注意新增字段）：
     {
       "name": "角色名",
       "facts": {
         "sexual_relations": [
           {"partner": "男主名", "is_male_lead": true, "forced": false, "detail": "初夜/破处等细节", "evidence": "原文摘录", "evidence_level": "explicit"},
           {"partner": "张三", "is_male_lead": false, "forced": false, "detail": "通过怀孕推断发生过性关系", "evidence": "她怀了张三的孩子", "evidence_level": "implicit", "implicit_type": "pregnancy_paternity"}
         ],
         "children_info": [
           {"child_name": "小明", "father": "张三", "is_biological": true, "origin": "正常生育", "detail": "亲生孩子", "evidence": "她生下了小明", "child_status": "born", "conception_method": "sex", "speech_act": "asserted_fact", "evidence_strength": "strong"},
           {"child_name": "养女名", "father": "未知", "is_biological": false, "origin": "收养", "detail": "非亲生，收养的", "evidence": "原文明确说不是亲生女儿", "child_status": "adopted", "conception_method": "unknown", "speech_act": "asserted_fact", "evidence_strength": "strong"}
         ],
         "physical_contacts": [
           {"partner": "李四", "is_male_lead": false, "contact_type": "猥亵/亲吻/爱抚/强抱等", "detail": "说明", "evidence": "原文摘录"}
         ],
         "romantic_feelings": [
           {"target": "王五", "is_male_lead": false, "mutual": false, "detail": "说明", "evidence": "原文摘录"}
         ],
         "partner_relations": [
           {"partner": "赵六", "is_male_lead": false, "relationship": "男友/丈夫/未婚夫", "status": "已完婚/订婚未完婚/正式交往", "forced": false, "detail": "说明", "evidence": "原文摘录"},
           {"partner": "墨水心的父亲", "is_male_lead": false, "relationship": "丈夫", "status": "已完婚", "forced": true, "detail": "被迫嫁给某人（无明确姓名但能从上下文确定是谁的父亲）", "evidence": "原文摘录"}
         ]
       }
     }
   - 示例（无事实时，**正确做法**）：
     {
       "name": "角色名",
       "facts": {
         "sexual_relations": [],
         "children_info": [],
         "physical_contacts": [],
         "romantic_feelings": [],
         "partner_relations": []
       }
     }
   - ⚠️【错误示例，禁止这样写】：
     {"partner": "无", ...}  ← 错！"无"不是角色名！
     {"target": "没有", ...} ← 错！应该输出空数组 []
     {"child_status": "pregnant", "evidence": "我们都怀孕了", "speech_act": "hypothesis"}  ← 错！speech_act不是asserted_fact时不要写入！

   - 【重要】is_male_lead 字段必须准确标注！male_lead/男主 = true，其他男性 = false
   - 【⚠️ evidence 必须是原文连续子串！】禁止改写、禁止添加省略号、禁止调整语序！若原文有换行必须保留！
   - 【重要】若某类事实无，则该数组**必须为空 []**，禁止用占位符！
   - 【重要】physical_contacts 只记录**非男主男性**的接触，男主的接触和女性的接触都不记录！
   - 【重要】children_info 只写入 speech_act="asserted_fact" 且 evidence_strength≠"weak" 的记录！

3. "extra_relations": 其他补充互动（可选）
   - 每条包含：female, male, is_male_lead, detail, evidence
   - 记录围观/搭讪/骚扰等可能有用但不属于上述5类的互动

顶层 JSON 示例片段（新增字段必须明确写出）：
{
  "extra_relations": [],
  "_reasoning": "这里写你的内部自检，后处理会使用 pop 丢弃。",
  "_context_summary": "这里写下一块可复用的上下文摘要。"
}

4. "_reasoning": 必须输出。用于记录你的内部自检、取舍和漏检排查；后处理会使用 pop 丢弃。
5. "_context_summary": 必须输出。用 1-3 句概括当前片段，供下一块承接上下文；后处理会提取并 pop。
"""

    return "\n".join([layer_one, layer_two, layer_three, "【输出前自检（必须逐项执行）】", "在输出 JSON 前，请确认：", checklist_text])

# ---------------- 结构化事实抽取增强：后处理 + 可选二次补抽 ----------------
_FACT_KEYWORDS = {
    "sexual_relations": re.compile(r"(同房|圆房|破身|失身|春宵|上床|睡了|交合|行房|发生关系|失贞|贞洁|初夜)", re.I),
    "children_info": re.compile(r"(怀孕|有孕|身孕|孕肚|小腹|产下|生下|分娩|生产|流产|胎儿|孩子|儿子|女儿|宝宝|收养|领养|养女|养子|继子|继女|认作女儿|认作儿子|当作女儿|当作儿子|过继|寄养|托孤|遗孤)", re.I),
    "partner_relations": re.compile(r"(丈夫|夫君|夫婿|相公|老公|前夫|未婚夫|男友|男朋友|恋人|订婚|婚约|成亲|完婚|结婚|离婚|再婚|出嫁|嫁给|被迫|强迫|威逼|胁迫|逼婚|包办|强娶|卖嫁|卖给|不得不嫁|强奸|强暴|性侵|未遂|没得逞|企图强奸)", re.I),
    "physical_contacts": re.compile(r"(亲吻|吻|强吻|摸|抚摸|揉|搂|抱|强抱|按在|猥亵|扒|掀裙|撕衣|脱衣|摸胸|摸腿|摸臀|摸腰|爱抚|牵手)", re.I),
    "romantic_feelings": re.compile(r"(喜欢|爱上|爱慕|心动|动心|倾心|钟情|中意|表白|告白|恋慕|芳心暗许)", re.I),
}
_MALE_HINTS = re.compile(r"(他|男人|男声|公子|少爷|王爷|殿下|郎君|夫君|相公|丈夫|未婚夫|前夫|男友|父亲|爹|爸|哥哥|师兄)", re.I)
_FEMALE_HINTS = re.compile(r"(她|女人|女声|小姐|夫人|姑娘|丫鬟|侍女|姐姐|妹妹|闺蜜|嫂|姨|婶|母亲|妈|娘|奶奶)", re.I)
_PLACEHOLDER_NAMES = {"无", "没有", "不存在", "null", "none", "n/a", "na", "未知", "空", "暂无", "无记录"}
_FACT_DIMENSIONS = ["sexual_relations", "children_info", "physical_contacts", "romantic_feelings", "partner_relations"]
_SEED_KEYWORDS = {
    "sexual_relations": ["云雨", "承欢", "洞房", "同寝", "侍寝", "临幸", "春风", "入幕", "男欢女爱", "肌肤之亲", "清白", "处子"],
    "children_info": [],
    "physical_contacts": ["拉手", "握手", "搭肩", "抚脸", "碰触"],
    "romantic_feelings": ["动情", "情愫", "暗恋", "相思", "思慕", "牵挂", "眷恋", "深情", "爱意", "情根", "芳心", "心仪", "春心", "情窦"],
    "partner_relations": ["夫人", "娘子", "夫家", "妾", "通房", "侧妃", "外室", "姘头", "聘礼", "退婚", "和离", "休妻"],
}

# ---- 全局补扫优化 数据结构 ----
@dataclass
class CoOccurrenceHit:
    """共现命中：女主名与维度关键词在chunk中的共现位置。"""
    heroine_name: str
    dimension: str
    chunk_idx: int
    anchor_pos: int       # 女主名在chunk中的字符位置
    keyword_pos: int      # 维度关键词匹配位置
    score: float = 0.0

@dataclass
class ChunkHeroineEntry:
    """同一chunk中，一个女主在一个锚点区域需要查询的维度集合。"""
    heroine_name: str
    dimensions: list       # 待查空维度
    anchor_pos: int        # 代表性锚点（同一女主分散hit按距离拆分后，组内中位数）
    is_free_rider: bool = False
    rescan_context: str = ""

@dataclass
class ProximityCluster:
    """按锚点近邻聚类的一组entry，共享一个文本窗口和一次API调用。"""
    entries: list          # list[ChunkHeroineEntry]
    window_start: int
    window_end: int
    all_dimensions: set = field(default_factory=set)

# “泛指/指代不明”的人名（会导致后续误判），用于伴侣/感情等字段过滤
_VAGUE_PERSON_NAME_RE = re.compile(
    r"^(某(个|位|名)?|一(个|位|名)?|那(个)?|这(个)?|另(外)?一(个|位|名)?|别(的)?|其(他)?)(男人|男性|男子|男士|男的|男生)$"
    r"|^(某人|那人|这个人|那个人|另(外)?一(个|位|名)?人|别人|其他人)$"
    r"|^(他|他人)$"
)

# 假设/劝说/类比语气（用于过滤“非事实关系”）
_NONFACT_SPEECH_ACT = {"hypothesis", "advice_persuasion", "comparison", "rumor", "question"}
_HYPOTHETICAL_CONDITIONAL_HINTS = ("如果", "假如", "万一", "要是", "倘若", "若是", "就算", "即便", "哪怕")
_HYPOTHETICAL_SCENARIO_HINTS = (
    "想想", "场景", "你想", "会不会", "能忍受", "难受吗", "恶心",
    "以后", "将来", "后来", "可能会", "最后会", "投入了别人的怀抱",
)
_RELATION_EVENT_HINTS = ("谈恋爱", "恋爱", "在一起", "结婚", "爱上", "男友", "男朋友", "丈夫", "未婚夫", "恋人")
_FUTURE_HINTS = ("以后", "将来", "未来", "后来")
_SPECULATIVE_HINTS = (
    "可能会", "会被", "会不会", "你想", "你觉得", "想想", "场景",
    "要和", "要跟", "只能", "无奈地", "想看到这样的场景",
)
_FACT_ANCHORS = ("曾经", "已经", "当时", "此前", "已", "确实", "目前")
_FORCED_RELATION_HINTS = re.compile(
    r"(被迫|强迫|威逼|胁迫|逼婚|包办|强娶|被卖|卖嫁|卖给|不得不嫁|逼着嫁|被迫订婚|被迫成亲|被迫结婚|联姻|指婚|许配|强奸|强暴|性侵|未遂|没得逞|企图强奸|企图侵犯|收买.*父母|逼迫受孕)",
    re.I,
)
_NO_SEX_CONCEPTION_HINTS = re.compile(
    r"(没有性行为|无性行为|未发生性行为|不需要性行为|试管婴儿|试管|人工授精|人工受孕|供精|精子库|胚胎移植)",
    re.I,
)


def _is_nonfact_relation_entry(entry: dict) -> bool:
    """识别关系/感情条目是否为“假设、劝说、类比、传闻”而非事实。"""
    if not isinstance(entry, dict):
        return True

    speech_act = _norm_text(entry.get("speech_act", "")).lower()
    if speech_act and speech_act != "asserted_fact":
        if speech_act in _NONFACT_SPEECH_ACT:
            return True
        # 非空且非 asserted_fact，保守视为低置信度
        return True

    evidence_strength = _norm_text(entry.get("evidence_strength", "")).lower()
    if evidence_strength in {"weak", "low"}:
        return True

    blob = "\n".join([
        _norm_text(entry.get("relationship", "")),
        _norm_text(entry.get("relation", "")),
        _norm_text(entry.get("relation_type", "")),
        _norm_text(entry.get("status", "")),
        _norm_text(entry.get("detail", "")),
        _norm_text(entry.get("evidence", "")),
    ])
    if not blob:
        return True

    has_conditional = any(k in blob for k in _HYPOTHETICAL_CONDITIONAL_HINTS)
    has_scenario = any(k in blob for k in _HYPOTHETICAL_SCENARIO_HINTS)
    if has_conditional and has_scenario:
        return True

    # 未来场景 + 推测语气 + 关系事件（即使没有“如果/就算”也按非事实处理）
    has_relation_event = any(k in blob for k in _RELATION_EVENT_HINTS)
    has_future_hint = any(k in blob for k in _FUTURE_HINTS)
    has_speculative = any(k in blob for k in _SPECULATIVE_HINTS)
    if has_relation_event and has_future_hint and has_speculative:
        if not any(k in blob for k in _FACT_ANCHORS):
            return True

    # 明显“想象未来/反问劝说”语境，且缺少已发生锚点时，按非事实处理
    if has_scenario and ("?" in blob or "？" in blob):
        if not any(k in blob for k in _FACT_ANCHORS):
            return True

    return False


def _norm_text(x):
    return str(x or "").strip()


def _coerce_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    s = _norm_text(v).lower()
    if s in {"true", "1", "yes", "y", "是", "真"}:
        return True
    if s in {"false", "0", "no", "n", "否", "假", ""}:
        return False
    return default


def _is_placeholder_name(name: str) -> bool:
    if not name:
        return True
    v = _norm_text(name).lower()
    if v in _PLACEHOLDER_NAMES:
        return True
    if len(v) <= 1 and v in {"无", "空"}:
        return True
    return False


def _is_vague_person_name(name: str) -> bool:
    """如“某个男人/某人/他”这类泛指对象名。"""
    s = _norm_text(name)
    if not s:
        return True
    try:
        return _VAGUE_PERSON_NAME_RE.match(s) is not None
    except Exception:
        return False


def _is_male_lead_name(name: str, male_protagonist: str) -> bool:
    n = _norm_text(name)
    if not n:
        return False
    if n in {"男主", "主角"}:
        return True
    male_identity = _normalize_male_identity(male_protagonist)
    if not male_identity:
        return False
    for candidate in male_identity.get("all_names", []):
        ml = _norm_text(candidate)
        if ml and ((ml in n) or (n in ml)):
            return True
    return False


def _likely_female_partner(partner: str, evidence: str, detail: str, heroine_names):
    p = _norm_text(partner)
    if not p:
        return True
    if p in set(heroine_names or []):
        return True
    text = f"{_norm_text(evidence)}\n{_norm_text(detail)}"
    if _MALE_HINTS.search(text):
        return False
    if _FEMALE_HINTS.search(text):
        return True
    return True


def _dedupe_by_signature(items, sig_fn):
    out = []
    seen = set()
    for it in items or []:
        try:
            sig = sig_fn(it)
        except Exception:
            sig = None
        if sig is not None and sig in seen:
            continue
        if sig is not None:
            seen.add(sig)
        out.append(it)
    return out


def _postprocess_heroine_facts(heroine_facts, heroine_names, male_protagonist, chunk_index):
    cleaned = []
    heroine_name_set = set([h for h in (heroine_names or []) if h])
    for item in heroine_facts or []:
        # 【容错】如果 item 是字符串，尝试转换为 dict
        if isinstance(item, str):
            item = {"name": item, "facts": {}}
        if not isinstance(item, dict):
            continue  # 跳过无法处理的类型
        name = _norm_text(item.get("name", ""))
        if not name:
            continue

        facts = item.setdefault("facts", {})
        # 【容错】facts 本身也可能是字符串/None
        if not isinstance(facts, dict):
            facts = {}
            item["facts"] = facts
        for key in ["sexual_relations", "children_info", "physical_contacts", "romantic_feelings", "partner_relations"]:
            facts.setdefault(key, [])
        item["chunk_index"] = chunk_index

        # sexual_relations：去占位、修正 is_male_lead
        sr_out = []
        for sr in facts.get("sexual_relations", []) or []:
            # 【容错】跳过非 dict 元素
            if not isinstance(sr, dict):
                continue
            partner = _norm_text(sr.get("partner", ""))
            if _is_placeholder_name(partner):
                continue
            sr["chunk_index"] = chunk_index
            sr["is_male_lead"] = bool(sr.get("is_male_lead", False) or _is_male_lead_name(partner, male_protagonist))
            sr_out.append(sr)
        facts["sexual_relations"] = _dedupe_by_signature(
            sr_out,
            lambda x: ("sr", _norm_text(x.get("partner")), _norm_text(x.get("detail")), _norm_text(x.get("evidence"))),
        )

        # children_info：补 origin；father/child_name 的占位归一为"未知"
        ci_out = []
        for ci in facts.get("children_info", []) or []:
            # 【容错】跳过非 dict 元素
            if not isinstance(ci, dict):
                continue
            ci["chunk_index"] = chunk_index
            ci.setdefault("origin", "未知")
            for k in ["father", "child_name"]:
                if _is_placeholder_name(_norm_text(ci.get(k, ""))):
                    ci[k] = "未知"
            ci_out.append(ci)
        facts["children_info"] = _dedupe_by_signature(
            ci_out,
            lambda x: ("ci", _norm_text(x.get("child_name")), _norm_text(x.get("father")), _norm_text(x.get("origin")), _norm_text(x.get("evidence"))),
        )

        # physical_contacts：严格过滤（非男主 + 非女触碰）
        pc_out = []
        for pc in facts.get("physical_contacts", []) or []:
            # 【容错】跳过非 dict 元素
            if not isinstance(pc, dict):
                continue
            partner = _norm_text(pc.get("partner", ""))
            evidence = _norm_text(pc.get("evidence", ""))
            detail = _norm_text(pc.get("detail", ""))
            if _is_placeholder_name(partner) or not evidence:
                continue
            pc["chunk_index"] = chunk_index
            pc["is_male_lead"] = bool(pc.get("is_male_lead", False) or _is_male_lead_name(partner, male_protagonist))
            if pc.get("is_male_lead", False):
                continue
            partner_gender_context = "\n".join([
                detail,
                _norm_text(pr.get("relationship", "")),
                _norm_text(pr.get("status", "")),
            ])
            if _likely_female_partner(partner, evidence, partner_gender_context, heroine_name_set | {name}):
                continue
            pc_out.append(pc)
        facts["physical_contacts"] = _dedupe_by_signature(
            pc_out,
            lambda x: ("pc", _norm_text(x.get("partner")), _norm_text(x.get("contact_type")), _norm_text(x.get("evidence"))),
        )

        # romantic_feelings：去占位、修正 is_male_lead
        rf_out = []
        for rf in facts.get("romantic_feelings", []) or []:
            # 【容错】跳过非 dict 元素
            if not isinstance(rf, dict):
                continue
            target = _norm_text(rf.get("target", ""))
            if _is_placeholder_name(target):
                continue
            if _is_nonfact_relation_entry(rf):
                continue
            rf["chunk_index"] = chunk_index
            rf["is_male_lead"] = bool(rf.get("is_male_lead", False) or _is_male_lead_name(target, male_protagonist))
            rf_out.append(rf)
        facts["romantic_feelings"] = _dedupe_by_signature(
            rf_out,
            lambda x: ("rf", _norm_text(x.get("target")), _norm_text(x.get("detail")), _norm_text(x.get("evidence"))),
        )

        # partner_relations：严格过滤（非男主 + 非女伴侣）
        pr_out = []
        for pr in facts.get("partner_relations", []) or []:
            # 【容错】跳过非 dict 元素
            if not isinstance(pr, dict):
                continue
            partner = _norm_text(pr.get("partner", ""))
            evidence = _norm_text(pr.get("evidence", ""))
            detail = _norm_text(pr.get("detail", ""))
            # 禁止“某个男人/他”等泛指对象进入 partner_relations（会导致后续误判）
            if _is_placeholder_name(partner) or _is_vague_person_name(partner) or not evidence:
                continue
            if _is_nonfact_relation_entry(pr):
                continue
            pr["chunk_index"] = chunk_index
            pr["is_male_lead"] = bool(pr.get("is_male_lead", False) or _is_male_lead_name(partner, male_protagonist))
            if pr.get("is_male_lead", False):
                continue
            if _likely_female_partner(partner, evidence, detail, heroine_name_set | {name}):
                continue
            # 兜底纠偏：若文本出现强迫线索，则 forced 必须为 true
            forced_blob = "\n".join([
                _norm_text(pr.get("relationship", "")),
                _norm_text(pr.get("status", "")),
                detail,
                evidence,
            ])
            forced_by_hint = _FORCED_RELATION_HINTS.search(forced_blob) is not None
            no_sex_conception_by_hint = _NO_SEX_CONCEPTION_HINTS.search(forced_blob) is not None
            pr["forced"] = True if forced_by_hint else _coerce_bool(pr.get("forced"), default=False)
            if no_sex_conception_by_hint:
                # “无性行为受孕/试管”不应被后续当作自愿亲密关系线索
                pr["forced"] = True
            pr_out.append(pr)
        facts["partner_relations"] = _dedupe_by_signature(
            pr_out,
            lambda x: ("pr", _norm_text(x.get("partner")), _norm_text(x.get("relationship")), _norm_text(x.get("status")), _norm_text(x.get("evidence"))),
        )

        cleaned.append(item)
    return cleaned


def _normalize_keyword_payload(payload, enforce_length_bounds=False):
    normalized = {dim: [] for dim in _FACT_DIMENSIONS}
    if not isinstance(payload, dict):
        return normalized
    for dim in _FACT_DIMENSIONS:
        normalized[dim] = _normalize_keyword_words(
            payload.get(dim, []),
            enforce_length_bounds=enforce_length_bounds,
        )
    return normalized


def _compile_keyword_regex(keywords):
    words = [re.escape(word) for word in _normalize_keyword_words(keywords) if str(word).strip()]
    if not words:
        return re.compile(r"$^")
    return re.compile("|".join(words), re.I)


def _merge_static_keyword_regex(static_regex, extra_keywords):
    extra_words = _normalize_keyword_words(extra_keywords)
    if not extra_words:
        return static_regex
    extra_pattern = "|".join(re.escape(word) for word in extra_words)
    combined_pattern = f"({static_regex.pattern}|{extra_pattern})"
    return re.compile(combined_pattern, static_regex.flags)


def _ensure_seed_file():
    with SNAPSHOT_LOCK:
        os.makedirs(LEARNED_KEYWORDS_DIR, exist_ok=True)
        if os.path.exists(SEED_FILE):
            return SEED_FILE
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "keywords": _normalize_keyword_payload(_SEED_KEYWORDS),
        }
        with open(SEED_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return SEED_FILE


def _find_latest_learned_file():
    pattern = os.path.join(LEARNED_KEYWORDS_DIR, "learned_*.json")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_dynamic_keywords():
    latest = _find_latest_learned_file()
    if not latest or not os.path.exists(latest):
        return {dim: [] for dim in _FACT_DIMENSIONS}
    try:
        data = read_file_safely(latest)
        payload = json.loads(data)
    except Exception:
        return {dim: [] for dim in _FACT_DIMENSIONS}
    return _normalize_keyword_payload(payload.get("keywords", {}))


def get_effective_keywords():
    _ensure_seed_file()
    try:
        seed_payload = json.loads(read_file_safely(SEED_FILE))
        seed_keywords = _normalize_keyword_payload(seed_payload.get("keywords", {}))
    except Exception:
        seed_keywords = _normalize_keyword_payload(_SEED_KEYWORDS)
    dynamic_keywords = _load_dynamic_keywords()
    effective = {}
    for dim, static_regex in _FACT_KEYWORDS.items():
        merged = _normalize_keyword_words((seed_keywords.get(dim) or []) + (dynamic_keywords.get(dim) or []))
        effective[dim] = _merge_static_keyword_regex(static_regex, merged)
    return effective


def save_learned_keywords(learned_keywords, source_phase="unknown", novel_name=None):
    payload_keywords = _normalize_keyword_payload(learned_keywords, enforce_length_bounds=True)
    if not any(payload_keywords.values()):
        return None
    with SNAPSHOT_LOCK:
        os.makedirs(LEARNED_KEYWORDS_DIR, exist_ok=True)
        if not os.path.exists(SEED_FILE):
            payload = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "keywords": _normalize_keyword_payload(_SEED_KEYWORDS),
            }
            with open(SEED_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        merged = _normalize_keyword_payload(_load_dynamic_keywords(), enforce_length_bounds=True)
        for dim in _FACT_DIMENSIONS:
            merged[dim] = _normalize_keyword_words(
                (merged.get(dim) or []) + (payload_keywords.get(dim) or []),
                enforce_length_bounds=True,
            )
        os.makedirs(LEARNED_KEYWORDS_DIR, exist_ok=True)
        filename = f"learned_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{source_phase}.json"
        path = os.path.join(LEARNED_KEYWORDS_DIR, filename)
        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_phase": source_phase,
            "novel_name": novel_name or clean_filename or "",
            "keywords": merged,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path


def _extract_fact_texts(extracted_facts):
    blobs = []
    for fact in extracted_facts or []:
        if isinstance(fact, dict):
            for key in ("evidence", "detail", "partner", "target", "father", "origin", "relationship", "status"):
                value = _norm_text(fact.get(key, ""))
                if value:
                    blobs.append(value)
        elif fact:
            blobs.append(_norm_text(fact))
    return "\n".join(blobs)


def check_uncovered_keywords(text_chunk, extracted_facts, dimension, effective_keywords):
    regex = (effective_keywords or {}).get(dimension)
    if regex is None:
        return False, []
    chunk_text = text_chunk or ""
    all_positions = [(m.start(), m.end(), m.group(0)) for m in regex.finditer(chunk_text)]
    if not all_positions:
        return False, []

    covered_indices = set()
    for fact in extracted_facts or []:
        if not isinstance(fact, dict):
            continue
        evidence = _norm_text(fact.get("evidence", ""))
        if not evidence:
            continue
        ev_start = 0
        while True:
            pos = chunk_text.find(evidence, ev_start)
            if pos == -1:
                break
            ev_end = pos + len(evidence)
            for idx, (kw_start, kw_end, _keyword) in enumerate(all_positions):
                if pos <= kw_start and kw_end <= ev_end:
                    covered_indices.add(idx)
            ev_start = pos + 1

    uncovered_indices = [idx for idx in range(len(all_positions)) if idx not in covered_indices]
    uncovered_ratio = len(uncovered_indices) / len(all_positions) if all_positions else 0.0

    contexts = []
    for idx in uncovered_indices[:5]:
        pos_start, pos_end, _keyword = all_positions[idx]
        ctx_start = max(0, pos_start - 50)
        ctx_end = min(len(chunk_text), pos_end + 50)
        snippet = chunk_text[ctx_start:ctx_end].strip()
        if snippet and snippet not in contexts:
            contexts.append(snippet)

    should_boost = uncovered_ratio > 0.5 and len(uncovered_indices) >= 3
    return should_boost, contexts


def _get_dimension_rules(dimension):
    rules = {
        "sexual_relations": "只补抽 sexual_relations；若只是未遂/企图/威胁，不得写成已发生性关系。",
        "children_info": "只补抽 children_info；必须区分 father、origin、is_biological、child_status、conception_method。",
        "physical_contacts": "只补抽 physical_contacts；只记录非男主男性对女主的实际肉体接触。",
        "romantic_feelings": "只补抽 romantic_feelings；只记录女主对非男主男性的明确情感事实。",
        "partner_relations": "只补抽 partner_relations；每条都必须输出 forced 字段，并排除女性对象与男主。",
    }
    return rules.get(dimension, "只补抽当前指定维度，其他维度保持空数组。")


def _build_fact_dimension_semantics_block():
    return """
【五维语义定义（统一口径，防止误判）】
1. sexual_relations：指女主与某男性之间已经发生的性行为事实。
   - 包含：明确同房、圆房、失身、破身、床戏描写；或由怀孕亲子归属、流产打胎等强后果反推的既成性关系。
   - 不包含：强奸未遂、企图侵犯、差点发生、暧昧调情、单纯亲吻、单纯搂抱、试探性语言。
   - 重点字段：partner、is_male_lead、forced、detail、evidence、evidence_level、implicit_type。

2. children_info：指女主与孩子、生育、怀孕、收养、亲缘来源相关的事实。
   - 包含：已生子女、仅怀孕、流产/堕胎、收养/领养、继子女、认作女儿、父亲归属、受孕方式。
   - 不包含：女主自己的身世倒装、假设“以后会怀孕”、劝说式“你也会怀上”。
   - 重点字段：child_name、father、is_biological、origin、child_status、conception_method、speech_act、evidence_strength。

3. physical_contacts：指非男主男性对女主已经发生的实际身体接触事实。
   - 包含：强抱、亲吻、猥亵、抚摸、摸腿、摸腰、按住、撕衣等现实肉体接触。
   - 不包含：男主接触、女性接触、女女贴贴、围观注视、语言调戏、精神连接、女主摸自己或她人。
   - 重点字段：partner、is_male_lead、contact_type、detail、evidence。

4. romantic_feelings：指女主对非男主男性产生过明确情感倾向的事实。
   - 包含：喜欢、爱上、动心、倾心、钟情、表白、承认爱慕等明确情感表达。
   - 不包含：被迫依附、第三方猜测、传闻、假设、劝说、单纯好感不明、别人喜欢女主但女主无回应。
   - 重点字段：target、is_male_lead、mutual、detail、evidence、speech_act、evidence_strength。

5. partner_relations：指女主与非男主男性之间已经成立或被明确指认的伴侣/婚约/婚姻关系事实。
   - 包含：男友、恋人、丈夫、前夫、未婚夫、婚约对象、被迫订婚、被迫成亲、联姻、妾室关系。
   - 不包含：仅有追求意图、单方示爱、纯暧昧、没有男性锚点的关系称呼、女性伴侣。
   - 重点字段：partner、is_male_lead、relationship、status、forced、detail、evidence、speech_act、evidence_strength。
""".strip()


# ---- 按维度拆分的语义定义（供多维度补扫 prompt 按需选取） ----
_DIMENSION_SEMANTIC_DEFS = {
    "sexual_relations": (
        "sexual_relations：指女主与某男性之间已经发生的性行为事实。\n"
        "   - 包含：明确同房、圆房、失身、破身、床戏描写；或由怀孕亲子归属、流产打胎等强后果反推的既成性关系。\n"
        "   - 不包含：强奸未遂、企图侵犯、差点发生、暧昧调情、单纯亲吻、单纯搂抱、试探性语言。\n"
        "   - 重点字段：partner、is_male_lead、forced、detail、evidence、evidence_level、implicit_type。"
    ),
    "children_info": (
        "children_info：指女主与孩子、生育、怀孕、收养、亲缘来源相关的事实。\n"
        "   - 包含：已生子女、仅怀孕、流产/堕胎、收养/领养、继子女、认作女儿、父亲归属、受孕方式。\n"
        '   - 不包含：女主自己的身世倒装、假设"以后会怀孕"、劝说式"你也会怀上"。\n'
        "   - 重点字段：child_name、father、is_biological、origin、child_status、conception_method、speech_act、evidence_strength。"
    ),
    "physical_contacts": (
        "physical_contacts：指非男主男性对女主已经发生的实际身体接触事实。\n"
        "   - 包含：强抱、亲吻、猥亵、抚摸、摸腿、摸腰、按住、撕衣等现实肉体接触。\n"
        "   - 不包含：男主接触、女性接触、女女贴贴、围观注视、语言调戏、精神连接、女主摸自己或她人。\n"
        "   - 重点字段：partner、is_male_lead、contact_type、detail、evidence。"
    ),
    "romantic_feelings": (
        "romantic_feelings：指女主对非男主男性产生过明确情感倾向的事实。\n"
        "   - 包含：喜欢、爱上、动心、倾心、钟情、表白、承认爱慕等明确情感表达。\n"
        "   - 不包含：被迫依附、第三方猜测、传闻、假设、劝说、单纯好感不明、别人喜欢女主但女主无回应。\n"
        "   - 重点字段：target、is_male_lead、mutual、detail、evidence、speech_act、evidence_strength。"
    ),
    "partner_relations": (
        "partner_relations：指女主与非男主男性之间已经成立或被明确指认的伴侣/婚约/婚姻关系事实。\n"
        "   - 包含：男友、恋人、丈夫、前夫、未婚夫、婚约对象、被迫订婚、被迫成亲、联姻、妾室关系。\n"
        "   - 不包含：仅有追求意图、单方示爱、纯暧昧、没有男性锚点的关系称呼、女性伴侣。\n"
        "   - 重点字段：partner、is_male_lead、relationship、status、forced、detail、evidence、speech_act、evidence_strength。"
    ),
}


def _build_fact_dimension_semantics_block_for_dims(dimensions):
    """只返回指定维度的语义定义（用于多维度补扫 prompt 精简）。"""
    if not dimensions:
        return _build_fact_dimension_semantics_block()
    lines = ["【维度语义定义（仅限当前回扫维度）】"]
    for i, dim in enumerate(dimensions, 1):
        text = _DIMENSION_SEMANTIC_DEFS.get(dim, "")
        if text:
            lines.append(f"{i}. {text}")
    return "\n".join(lines)


def _build_dynamic_keyword_rules_block(dimension=None):
    dim_text = dimension or "当前目标维度"
    return f"""
【动态词库提取规则】
- 你可以在输出 facts 的同时，为 {dim_text} 提取少量“新关键词/新表达方式”到 `_learned_keywords`。
- 新关键词的定义：能稳定指向某个维度事实的短语或表达方式，而不是某一次性的完整句子。
- 关键词必须按维度归类；当前 prompt 只优先收集与 {dim_text} 直接相关的表达。
- 关键词长度建议 2-10 个字；去掉前后空白；同义重复只保留一个更稳定的写法。

可收关键词示例：
- sexual_relations：云雨、承欢、春宵、失贞、珠胎暗结（若语境稳定指向既成性关系）
- children_info：有喜、珠胎、身怀六甲、养作义女、过继
- physical_contacts：揽入怀中、捏住下巴、隔衣揉捏、按倒在榻
- romantic_feelings：芳心暗许、情根深种、倾慕已久
- partner_relations：纳为妾室、指腹为婚、结成道侣、招作赘婿

不可收内容示例：
- 人名、称呼、代词：张三、她、他、夫人、小姐
- 一次性整句 evidence：她昨夜被迫订婚给张三
- 纯剧情摘要、纯情绪词、泛化动词：发生了、这样那样、难过、生气、纠缠
- 跨维度模糊词：关系、接触、喜欢他一下子

输出要求：
- 若没有新的稳定表达方式，`_learned_keywords` 对应维度输出空数组。
- 不要把 evidence 原文整句原样塞进 `_learned_keywords`。
- 不要编造当前文本里没有出现的新表达。
""".strip()


def _build_dimension_workflow_block(mode, dimension):
    if mode == "targeted_rescan":
        return f"""
【执行步骤】
步骤1：先根据回扫线索定位可能相关的句段，但要记住“以下信息仅供参考线索，请以原文为准”。
步骤2：再回到当前文本逐句核实，确认目标女主、目标男性、目标维度三者是否真的在原文中锚定成立。
步骤3：做主体锚定、语用过滤、性别过滤、男主过滤，排除假设、劝说、传闻、女性对象、男主对象。
步骤4：若确认当前维度存在事实，再输出该维度 facts；若没有实锤，就保持该维度为空数组。
步骤5：最后检查当前文本里是否出现了能够稳定指向本维度的新表达方式，再决定是否写入 `_learned_keywords`。
""".strip()
    return f"""
【执行步骤】
步骤1：先识别当前片段里出现的目标女主，并确认哪些句子与当前维度 {dimension} 真正相关。
步骤2：再查看“重点检查区域”中的未覆盖关键词上下文，优先核查这些位置是否存在被主扫描漏掉的事实。
步骤3：对候选句做主体锚定、语用过滤、性别过滤、男主过滤，排除假设、劝说、传闻、未遂、女性对象和性别不明对象。
步骤4：只输出当前维度的 facts，其他四个维度保持空数组，避免跨维度扩写。
步骤5：最后检查当前文本里是否出现了能够稳定指向本维度的新表达方式，再决定是否写入 `_learned_keywords`。
""".strip()


def _build_profile_dimension_semantics_block():
    return """
【五维事实解释口径】
- sexual_relations：这表示女主已经与某男性发生过性行为事实，或有足够强的后果证据可反推既成性关系。
- children_info：这表示女主与怀孕、生子、流产、收养、父亲归属、亲生/非亲生等有关的既有事实。
- physical_contacts：这表示非男主男性对女主已经发生过现实身体接触，不包含男主接触和女性接触。
- romantic_feelings：这表示女主对非男主男性已经出现明确情感倾向，而不是第三方猜测或传闻。
- partner_relations：这表示女主与非男主男性已经存在或被明确指认的婚约/恋爱/婚姻/被迫伴侣关系。
""".strip()


def build_dimension_boost_prompt(dimension, heroines, male_protagonist=None, keyword_contexts=None):
    heroines_text = "、".join([h for h in (heroines or []) if h]) or "当前片段提到的女主"
    context_text = "\n".join([f"- {item}" for item in (keyword_contexts or []) if item]) or "- 无额外线索"
    male_prompt_text = _male_identity_prompt_text(male_protagonist) or "【男主】未显式提供"
    semantics_block = _build_fact_dimension_semantics_block()
    keyword_rules_block = _build_dynamic_keyword_rules_block(dimension)
    workflow_block = _build_dimension_workflow_block("dimension_boost", dimension)
    return f"""
你是“结构化事实补抽器”。你必须输出严格 JSON（json_object），不要输出任何解释文字。

【补抽目标】
- 当前维度：{dimension}
- 关注女主：{heroines_text}
- {male_prompt_text}
- {_get_dimension_rules(dimension)}

{semantics_block}

【当前维度重点】
- 当前只允许补抽 {dimension}。
- 你可以参考上面的五维统一口径来避免把相邻维度的事实误挂到 {dimension}。
- 若文本只支持其他维度而不支持 {dimension}，当前维度必须保持空数组。

【重点检查区域】
{context_text}

{keyword_rules_block}

{workflow_block}

【硬约束】
- 只补抽 {dimension}，其他四个维度必须输出空数组 []。
- evidence 必须是当前文本的连续子串，不得改写。
- partner_relations 必须输出 forced 字段。
- 只记录非男主男性；女性对象、男主、性别不明对象必须排除。
- children_info 需严格区分 is_biological、origin、conception_method，不得反推 romantic_feelings 或 sexual_relations。

输出 JSON：
{{
  "heroine_facts": [
    {{
      "name": "角色名",
      "facts": {{
        "sexual_relations": [],
        "children_info": [],
        "physical_contacts": [],
        "romantic_feelings": [],
        "partner_relations": []
      }}
    }}
  ],
  "_learned_keywords": {{
    "{dimension}": ["新关键词"]
  }}
}}
""".strip()


def _merge_dimension_facts(base_facts, boost_facts, dimension):
    merged = {
        item.get("name"): item
        for item in (base_facts or [])
        if isinstance(item, dict) and item.get("name")
    }
    for extra in boost_facts or []:
        if not isinstance(extra, dict):
            continue
        name = extra.get("name")
        if not name:
            continue
        if name not in merged:
            facts = {key: [] for key in _FACT_DIMENSIONS}
            facts[dimension] = list(((extra.get("facts") or {}).get(dimension, [])) or [])
            merged[name] = {"name": name, "facts": facts, "chunk_index": extra.get("chunk_index")}
            continue
        base = merged[name].setdefault("facts", {})
        for key in _FACT_DIMENSIONS:
            base.setdefault(key, [])
        base[dimension].extend(((extra.get("facts") or {}).get(dimension, [])) or [])
        base[dimension] = _dedupe_by_signature(
            base[dimension],
            lambda x, _k=dimension: (
                _k,
                _norm_text(x.get("partner") or x.get("target") or x.get("father") or x.get("child_name")),
                _norm_text(x.get("relationship") or x.get("contact_type") or x.get("detail")),
                _norm_text(x.get("evidence")),
            ),
        )
    return list(merged.values())


def build_fact_boost_prompt(heroines, male_protagonist=None):
    heroines_text = "、".join([h for h in (heroines or []) if h])
    ml = male_protagonist or ""
    return f"""
你是"结构化事实抽取器"。只做抽取，不做判断，不做推理。
你必须输出严格 JSON（json_object），不要输出任何解释文字。

抽取对象：仅限这些女主：{heroines_text}
男主名（如有）：{ml}

【⚠️ evidence 字段硬规则（最高优先级！）】
evidence 必须是【文本】的连续子串，禁止任何改写！
- 禁止改写、同义替换、补字、删字、调整语序
- 禁止添加省略号（……/.../…），除非原文就包含
- 若原文有换行，evidence 必须保持换行
- 建议长度 12~80 字（但只要是原文连续子串即可）

抽取规则（核心）：
1) 只抽取文本中明确出现的事实；evidence 必须是原文的连续子串（严格逐字复制！）。
2) 占位符禁止：partner/target 不能是"无/没有/不存在/未知/空/null"等；若不知道则不要输出该条。
3) physical_contacts / partner_relations：只记录"非男主男性"。如证据缺乏男性线索（他/男人/公子/少爷/王爷/丈夫/未婚夫/前夫/男友等）则不要输出；女性触碰禁止输出到这两类。
4) partner_relations 额外要求：每条都要判断是否强迫并写 forced（true/false）。若 evidence 明确是被迫/强迫/威逼/胁迫/包办/逼婚等非自愿关系，forced=true；否则 forced=false（无证据不臆测）。
5) partner_relations 不漏检要求：先检查“被迫/强迫/威逼/胁迫/逼婚/包办/卖嫁/不得不嫁”等线索；若已明确指向女主与非男主男性伴侣，必须输出对应条目，不能漏。
6) children_info：出现"怀孕/生下/女儿/儿子/孩子/收养/领养/养女/继子/认作女儿/过继"等均可记录；origin 不明确时填"未知"；is_biological 不确定可为 null 或省略。
7) “强奸未遂/企图强奸/没得逞”属于受害或风险事实，不等于已发生性关系；不要仅凭此写入 sexual_relations。
8) “无性行为怀孕/试管婴儿/人工授精/供精/胚胎移植”必须优先写入 children_info（conception_method 优先 ivf_or_ai），不得反推 sexual_relations 或“自愿恋爱/动心”。

输出格式：
{{"heroine_facts":[{{"name":"女主名","facts":{{"sexual_relations":[],"children_info":[],"physical_contacts":[],"romantic_feelings":[],"partner_relations":[]}}}}]}}
""".strip()

def read_novel(file_path):
    """读取小说文件（UTF-8 -> GB18030 兜底）"""
    return read_file_safely(file_path)

def split_text(text, chunk_size, overlap=None):
    if overlap is None:
        overlap = CHUNK_OVERLAP
    try:
        chunk_size = int(chunk_size)
    except Exception:
        chunk_size = CHUNK_SIZE
    try:
        overlap = int(overlap)
    except Exception:
        overlap = CHUNK_OVERLAP
    manifest = build_chunk_manifest(text or "", chunk_size=chunk_size, overlap=overlap)
    return [entry.get("text", "") for entry in manifest.get("chunks", [])]


# ---------------- JSON 容错解析（避免 200 OK 但返回非 JSON/空串导致高失败率） ----------------
def _safe_json_loads(text):
    """
    尽最大可能把模型输出解析成 JSON。
    主要应对：空串/非JSON提示语/```json```包裹/夹杂控制字符/截断。
    """
    def _normalize_fullwidth_json_punct(src: str) -> str:
        """
        将常见“全角 JSON 结构符号”归一为半角，以降低解析失败率。
        重要：只在【字符串字面量之外】替换，避免污染 evidence/detail 等正文内容。
        """
        if not src:
            return src
        mapping = {
            "［": "[",
            "］": "]",
            "｛": "{",
            "｝": "}",
            "，": ",",
            "：": ":",
            "“": '"',
            "”": '"',
        }
        out = []
        in_str = False
        escape = False
        for ch in src:
            if in_str:
                out.append(ch)
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            # 非字符串态：允许替换结构符号
            if ch == '"':
                in_str = True
                out.append(ch)
                continue
            out.append(mapping.get(ch, ch))
        return "".join(out)

    if text is None:
        raise json.JSONDecodeError("empty", "", 0)
    s = str(text).strip()
    if not s:
        raise json.JSONDecodeError("empty", "", 0)
    s = re.sub(r"```json\s*|\s*```", "", s).strip()
    s = _normalize_fullwidth_json_punct(s)

    # ---- 预处理修复：弱模型常见格式错误 ----
    # 修复1：布尔/null 值后的多余引号（"true",  →  true,）
    # 原因：弱模型有时在 true/false/null 后多输出一个 " 字符，如：is_biological": true",
    s = re.sub(r'\b(true|false|null)"(\s*[,\}\]])', r'\1\2', s)
    # 修复2：数字值后的多余引号（123",  →  123,）
    s = re.sub(r'(\d)"(\s*[,\}\]])', r'\1\2', s)

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
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    # 5) 更激进修复：移除所有 true/false/null/数字 后的悬挂引号（不限制后续字符），再重试
    # 针对弱模型系统性 bug：true" 后面可能跟换行/空格再接逗号
    aggressive = re.sub(r'\b(true|false|null)"', r'\1', cleaned)
    aggressive = re.sub(r'(\d)"', r'\1', aggressive)
    if aggressive != cleaned:
        try:
            return json.loads(aggressive)
        except json.JSONDecodeError:
            pass
        la = aggressive.find("{")
        ra = aggressive.rfind("}")
        if la != -1 and ra != -1 and ra > la:
            try:
                return json.loads(aggressive[la:ra + 1])
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("unable to parse json", cleaned[:200], 0)


def _call_json_chat_completion(messages, max_tokens, temperature=0.1, log_prefix="chat"):
    try:
        response = chat_completion(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.warning(f"{log_prefix} response_format 不可用，降级调用: {e}")
        response = chat_completion(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    record_usage(response)
    return response

def _normalize_issue(issue, chunk_index):
    """确保 issues 输出包含 category/type/content/reason/chunk_index"""
    # 兼容：模型偶尔会返回字符串列表或其他非 dict 结构
    if isinstance(issue, str):
        return {
            "category": "",
            "type": "",
            "content": issue,
            "reason": "",
            "chunk_index": chunk_index,
        }
    if not isinstance(issue, dict):
        return {
            "category": "",
            "type": "",
            "content": str(issue),
            "reason": "",
            "chunk_index": chunk_index,
        }
    return {
        "category": issue.get("category", ""),
        "type": issue.get("type", ""),
        "content": issue.get("content", ""),
        "reason": issue.get("reason", ""),
        "chunk_index": chunk_index,
    }


def scan_chunk(text_chunk, index, total, system_prompt, heroines, male_protagonist=None, fact_boost_prompt=None, context_summary=""):
    """分析单个块，返回 issues、heroine_facts、extra_relations 与下一块可复用摘要。"""
    summary_block = ""
    if context_summary:
        summary_block = f"\n前文摘要（仅用于承接上下文，不可替代当前片段证据）：\n{context_summary}\n"
    user_prompt = f"""
这是小说的第 {index + 1}/{total} 部分。
请根据 System Prompt 中的规则进行分析。先完成 heroine_facts 的五维抽取，再输出 issues。
{summary_block}
小说片段：
{text_chunk[:]}
"""

    retries = 3
    last_err = ""
    for _ in range(retries):
        try:
            response = _call_json_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=6000,
                log_prefix=f"chunk {index}",
            )
            content = response.choices[0].message.content

            try:
                data = _safe_json_loads(content)
            except Exception as je:
                snippet = (str(content)[:200] if content is not None else "None")
                logger.warning(f"chunk {index} 返回非JSON/空内容，前200字符: {snippet!r}")
                raise je

            if not isinstance(data, dict):
                snippet = (str(content)[:200] if content is not None else "None")
                logger.warning(f"chunk {index} 顶层JSON不是对象(dict)，类型={type(data).__name__}，前200字符: {snippet!r}")
                raise ValueError(f"top_level_json_not_object:{type(data).__name__}")

            data.pop("_reasoning", None)
            next_summary = _norm_text(data.pop("_context_summary", ""))
            data.pop("_learned_keywords", None)

            raw_issues = data.get("issues", [])
            if raw_issues is None:
                raw_issues = []
            elif isinstance(raw_issues, dict):
                raw_issues = [raw_issues]
            elif isinstance(raw_issues, str):
                raw_issues = [raw_issues]
            issues = [_normalize_issue(item, index + 1) for item in (raw_issues or [])]

            heroine_facts = data.get("heroine_facts", [])
            if not isinstance(heroine_facts, list):
                heroine_facts = [heroine_facts] if heroine_facts else []

            old_heroine_status = data.get("heroine_status", [])
            if not isinstance(old_heroine_status, list):
                old_heroine_status = [old_heroine_status] if old_heroine_status else []
            for item in old_heroine_status:
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "")
                if name and not any((isinstance(h, dict) and h.get("name") == name) for h in heroine_facts):
                    heroine_facts.append({
                        "name": name,
                        "chunk_index": index + 1,
                        "facts": {key: [] for key in _FACT_DIMENSIONS},
                        "_legacy_evidence": item.get("evidence", ""),
                    })

            heroine_facts = _postprocess_heroine_facts(heroine_facts, heroines, male_protagonist, index + 1)

            mentioned = [h for h in (heroines or []) if h and h in text_chunk]
            if mentioned and DIM_BOOST_MAX_PER_CHUNK > 0:
                effective_keywords = get_effective_keywords()
                boost_dims = []
                for dimension in _FACT_DIMENSIONS:
                    if len(boost_dims) >= DIM_BOOST_MAX_PER_CHUNK:
                        break
                    extracted_facts = []
                    for item in heroine_facts:
                        if not isinstance(item, dict):
                            continue
                        if item.get("name") not in mentioned:
                            continue
                        extracted_facts.extend(((item.get("facts") or {}).get(dimension, [])) or [])
                    should_boost, keyword_contexts = check_uncovered_keywords(
                        text_chunk,
                        extracted_facts=extracted_facts,
                        dimension=dimension,
                        effective_keywords=effective_keywords,
                    )
                    if should_boost:
                        boost_dims.append((dimension, keyword_contexts))

                for dimension, keyword_contexts in boost_dims:
                    boost_prompt = build_dimension_boost_prompt(
                        dimension,
                        mentioned,
                        male_protagonist=male_protagonist,
                        keyword_contexts=keyword_contexts,
                    )
                    try:
                        boost_resp = _call_json_chat_completion(
                            messages=[
                                {"role": "system", "content": boost_prompt},
                                {"role": "user", "content": f"文本：\n{text_chunk}"},
                            ],
                            temperature=0.1,
                            max_tokens=4000,
                            log_prefix=f"chunk {index} dim_boost {dimension}",
                        )
                        boost_data = _safe_json_loads(boost_resp.choices[0].message.content)
                        if not isinstance(boost_data, dict):
                            continue
                        boost_data.pop("_reasoning", None)
                        boost_learned = boost_data.pop("_learned_keywords", {})
                        if isinstance(boost_learned, dict) and any(boost_learned.values()):
                            save_learned_keywords(boost_learned, source_phase="dim_boost", novel_name=clean_filename)
                        boost_facts = boost_data.get("heroine_facts", [])
                        if not isinstance(boost_facts, list):
                            boost_facts = [boost_facts] if boost_facts else []
                        boost_facts = _postprocess_heroine_facts(boost_facts, heroines, male_protagonist, index + 1)
                        heroine_facts = _merge_dimension_facts(heroine_facts, boost_facts, dimension)
                    except Exception as boost_exc:
                        logger.warning(f"chunk {index} 维度补抽失败 dim={dimension}: {boost_exc}")

            extra_relations = data.get("extra_relations", [])
            if not isinstance(extra_relations, list):
                extra_relations = []
            extra_relations = [item for item in extra_relations if isinstance(item, dict)]
            for item in extra_relations:
                item.setdefault("chunk_index", index + 1)
                item.setdefault("evidence", "")

            return issues, heroine_facts, extra_relations, next_summary, True, False, ""

        except Exception as e:
            last_err = str(e)
            fatal_markers = ["所有 API_KEY 均不可用", "请设置 API_KEY", "API key 无效"]
            if any(m in last_err for m in fatal_markers):
                return [], [], [], "", False, True, last_err
            time.sleep(1)
            continue
    return [], [], [], "", False, False, last_err


def _empty_fact_bucket():
    return {key: [] for key in _FACT_DIMENSIONS}


def _fact_signature(dimension, item):
    if not isinstance(item, dict):
        return (dimension, str(item))
    return (
        dimension,
        _norm_text(item.get("partner") or item.get("target") or item.get("father") or item.get("child_name")),
        _norm_text(item.get("relationship") or item.get("contact_type") or item.get("detail")),
        _norm_text(item.get("status")),
        _norm_text(item.get("evidence")),
    )


def _partition_indices_for_thread_blocks(total_chunks, max_workers):
    total_chunks = max(0, int(total_chunks or 0))
    if total_chunks <= 0:
        return []
    worker_count = max(1, min(int(max_workers or 1), total_chunks))
    base, extra = divmod(total_chunks, worker_count)
    blocks = []
    start = 0
    for worker_idx in range(worker_count):
        block_size = base + (1 if worker_idx < extra else 0)
        if block_size <= 0:
            continue
        stop = start + block_size
        blocks.append(list(range(start, stop)))
        start = stop
    return blocks


def generate_context_summary(text_chunk, heroines=None, male_protagonist=None, previous_summary=""):
    heroines_text = "、".join([name for name in (heroines or []) if name]) or "当前片段中的女主"
    male_prompt_text = _male_identity_prompt_text(male_protagonist) or "【男主】未显式提供"
    prev_block = f"\n前文摘要：{previous_summary}\n" if previous_summary else "\n"
    prompt = f"""
你是“上下文摘要器”。

【摘要目标】
- 该摘要只服务于后续 chunk 的上下文承接，帮助模型理解人物、事件和关系延续。
- 不能把摘要中的内容当作新证据，后续 facts 仍必须以当前 chunk 原文为准。

【保留重点】
- 当前片段和前文摘要中涉及的主要人物，尤其是目标女主。
- 关键事件：婚约、被迫关系、怀孕、孩子、亲密接触、情感变化、关系转折、冲突升级与缓和。
- 尤其注意：婚约/被迫关系/怀孕/孩子/亲密接触 这类会影响后续理解的连续信息。
- 人物关系变化、时间推进、身份变化、已发生的重要后果。
- 对后续理解剧情连续性有帮助的信息，而不是逐句复述正文。

【禁止事项】
- 不要抽取结构化事实，不要输出 issues，不要输出人物评价。
- 不要编造原文中没有出现的关系或事件。
- 不要把纯环境描写、抒情、议论、无关配角支线当作摘要重点。

【输出要求】
- 请为后续扫描生成 1-3 句承接摘要，只保留人物关系、婚约、怀孕、亲密接触、孩子归属等关键信息。
- 摘要目标长度为 150-200字，只在这个范围内尽量压缩关键信息，不要额外解释长度要求。
- 摘要应尽量紧凑，优先写“谁和谁、发生了什么、关系怎么变了”。
- 关注女主：{heroines_text}
- {male_prompt_text}
- 只输出 JSON，不要输出额外解释。
{prev_block}
输出格式：
{{
  "_context_summary": "简洁摘要"
}}
""".strip()
    try:
        response = _call_json_chat_completion(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"当前片段：\n{text_chunk}"},
            ],
            temperature=0.1,
            max_tokens=800,
            log_prefix="context_summary",
        )
        data = _safe_json_loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"生成前情提要失败，退化为无上下文扫描: {e}")
        return ""
    if not isinstance(data, dict):
        return ""
    return _norm_text(data.get("_context_summary") or data.get("context_summary") or data.get("summary") or "")


def _process_thread_block(block_id, block_indices, chunks, system_prompt, heroines, male_protagonist=None,
                          fact_boost_prompt=None, all_issues=None, all_heroine_facts=None,
                          extra_relations_all=None, processed_chunks=None, failed_chunks=None):
    global _middle_summary_calls
    if not block_indices:
        return {
            "block_id": block_id,
            "boundary_idx": None,
            "boundary_seed_summary": "",
            "last_summary": "",
            "boundary_needs_finalize": False,
            "fatal_error": "",
        }

    boundary_idx = None
    carry_summary = ""
    seed_summary = ""
    boundary_needs_finalize = False
    indices_to_scan = list(block_indices)
    processed_chunks = processed_chunks if processed_chunks is not None else set()
    failed_chunks = failed_chunks if failed_chunks is not None else set()
    all_issues = all_issues if all_issues is not None else []
    all_heroine_facts = all_heroine_facts if all_heroine_facts is not None else []
    extra_relations_all = extra_relations_all if extra_relations_all is not None else []

    if block_id > 0:
        boundary_idx = block_indices[0]
        predecessor_summary = ""
        if boundary_idx > 0:
            with CHECKPOINT_LOCK:
                predecessor_summary = CHUNK_SUMMARIES.get(boundary_idx - 1, "")
        if boundary_idx in processed_chunks:
            with CHECKPOINT_LOCK:
                carry_summary = CHUNK_SUMMARIES.get(boundary_idx, predecessor_summary)
            indices_to_scan = block_indices[1:]
        elif predecessor_summary:
            carry_summary = predecessor_summary
        else:
            seed_summary = generate_context_summary(
                chunks[boundary_idx],
                heroines=heroines,
                male_protagonist=male_protagonist,
            )
            carry_summary = seed_summary
            indices_to_scan = block_indices[1:]
            boundary_needs_finalize = True

    total = len(chunks)
    fatal_error = ""
    for idx in indices_to_scan:
        if idx in processed_chunks:
            with CHECKPOINT_LOCK:
                cached_summary = CHUNK_SUMMARIES.get(idx, "")
            carry_summary = cached_summary if cached_summary else ""
            continue

        if not carry_summary and idx > 0:
            with CHECKPOINT_LOCK:
                cached_prev = CHUNK_SUMMARIES.get(idx - 1, "")

            if cached_prev:
                carry_summary = cached_prev
            else:
                should_generate_middle_summary = False
                with MIDDLE_SUMMARY_LOCK:
                    if _middle_summary_calls < MAX_MIDDLE_SUMMARY_CALLS:
                        _middle_summary_calls += 1
                        should_generate_middle_summary = True

                if should_generate_middle_summary:
                    generated_summary = generate_context_summary(
                        chunks[idx - 1],
                        heroines=heroines,
                        male_protagonist=male_protagonist,
                    )
                    if generated_summary:
                        carry_summary = generated_summary
                        with CHECKPOINT_LOCK:
                            existing_summary = CHUNK_SUMMARIES.get(idx - 1, "")
                            if existing_summary:
                                carry_summary = existing_summary
                            else:
                                CHUNK_SUMMARIES[idx - 1] = generated_summary

        issues, heroine_facts, extra_rel, next_summary, ok, fatal, err_msg = scan_chunk(
            chunks[idx],
            idx,
            total,
            system_prompt,
            heroines,
            male_protagonist,
            fact_boost_prompt,
            context_summary=carry_summary,
        )
        if fatal:
            with CHECKPOINT_LOCK:
                failed_chunks.add(idx)
                save_checkpoint(
                    all_issues,
                    all_heroine_facts,
                    processed_chunks,
                    extra_relations_all,
                    failed_chunks=failed_chunks,
                    current_chunk_idx=idx,
                )
            fatal_error = err_msg or "所有 API_KEY 均不可用"
            break
        _commit_chunk_result(
            idx,
            issues,
            heroine_facts,
            extra_rel,
            next_summary,
            ok,
            err_msg,
            all_issues=all_issues,
            all_heroine_facts=all_heroine_facts,
            extra_relations_all=extra_relations_all,
            processed_chunks=processed_chunks,
            failed_chunks=failed_chunks,
        )
        if ok and next_summary:
            carry_summary = next_summary or carry_summary

    return {
        "block_id": block_id,
        "boundary_idx": boundary_idx,
        "boundary_seed_summary": seed_summary,
        "last_summary": carry_summary,
        "boundary_needs_finalize": boundary_needs_finalize,
        "fatal_error": fatal_error,
    }


def _merge_scan_success(all_issues, all_heroine_facts, extra_relations_all, processed_chunks, failed_chunks,
                        idx, issues, heroine_facts, extra_rel, ok, err_msg=""):
    if ok:
        if issues:
            all_issues.extend(issues)
        if heroine_facts:
            all_heroine_facts.extend(heroine_facts)
        if extra_rel:
            extra_relations_all.extend(extra_rel)
        processed_chunks.add(idx)
        failed_chunks.discard(idx)
        return
    failed_chunks.add(idx)
    if err_msg:
        logger.warning(f"chunk {idx} 扫描失败（未计入完成）：{err_msg}")


def _commit_chunk_result(idx, issues, heroine_facts, extra_rel, next_summary, ok, err_msg="",
                         all_issues=None, all_heroine_facts=None, extra_relations_all=None,
                         processed_chunks=None, failed_chunks=None, progress_state=None):
    all_issues = all_issues if all_issues is not None else []
    all_heroine_facts = all_heroine_facts if all_heroine_facts is not None else []
    extra_relations_all = extra_relations_all if extra_relations_all is not None else []
    processed_chunks = processed_chunks if processed_chunks is not None else set()
    failed_chunks = failed_chunks if failed_chunks is not None else set()
    with CHECKPOINT_LOCK:
        _merge_scan_success(
            all_issues,
            all_heroine_facts,
            extra_relations_all,
            processed_chunks,
            failed_chunks,
            idx,
            issues,
            heroine_facts,
            extra_rel,
            ok,
            err_msg,
        )
        if ok and next_summary:
            CHUNK_SUMMARIES[idx] = next_summary
        save_checkpoint(
            all_issues,
            all_heroine_facts,
            processed_chunks,
            extra_relations_all,
            failed_chunks=failed_chunks,
            current_chunk_idx=idx,
        )
        _advance_chunk_progress(idx, processed_chunks, failed_chunks, progress_state or _ACTIVE_PROGRESS_STATE)


def _run_initial_thread_block_scan(chunks, system_prompt, heroines, male_protagonist, fact_boost_prompt,
                                   all_issues, all_heroine_facts, extra_relations_all, processed_chunks, failed_chunks):
    global _ACTIVE_PROGRESS_STATE
    if not chunks:
        return None

    blocks = _partition_indices_for_thread_blocks(len(chunks), MAX_WORKERS)
    if not blocks:
        return None

    workers = min(len(blocks), max(1, int(MAX_WORKERS or 1)))
    logger.info(f"首轮线程块扫描：blocks={len(blocks)} workers={workers}")
    block_results = {}
    fatal_error_msg = ""
    progress_state = _init_chunk_progress(
        total=len(chunks),
        initial=len(processed_chunks),
        desc="首扫中",
        processed_chunks=processed_chunks,
        failed_chunks=failed_chunks,
    )
    previous_progress_state = _ACTIVE_PROGRESS_STATE
    _ACTIVE_PROGRESS_STATE = progress_state
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_thread_block,
                    block_id,
                    block_indices,
                    chunks,
                    system_prompt,
                    heroines,
                    male_protagonist,
                    fact_boost_prompt,
                    all_issues,
                    all_heroine_facts,
                    extra_relations_all,
                    processed_chunks,
                    failed_chunks,
                ): block_id
                for block_id, block_indices in enumerate(blocks)
                if block_indices
            }
            for future in concurrent.futures.as_completed(futures):
                block_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error(f"线程块 {block_id} 崩溃: {exc}", exc_info=True)
                    if not fatal_error_msg:
                        fatal_error_msg = str(exc)
                    continue
                block_results[block_id] = result
                if result.get("fatal_error"):
                    return result.get("fatal_error")

        for block_id in sorted(block_results.keys()):
            if block_id <= 0:
                continue
            result = block_results[block_id]
            boundary_idx = result.get("boundary_idx")
            if (
                boundary_idx is None
                or boundary_idx in processed_chunks
                or not result.get("boundary_needs_finalize")
            ):
                continue
            predecessor_summary = ""
            if boundary_idx > 0:
                with CHECKPOINT_LOCK:
                    predecessor_summary = CHUNK_SUMMARIES.get(boundary_idx - 1, "")
            if not predecessor_summary:
                predecessor_summary = (
                    block_results.get(block_id - 1, {}).get("last_summary")
                    or result.get("boundary_seed_summary", "")
                )
            issues, heroine_facts, extra_rel, _next_summary, ok, fatal, err_msg = scan_chunk(
                chunks[boundary_idx],
                boundary_idx,
                len(chunks),
                system_prompt,
                heroines,
                male_protagonist,
                fact_boost_prompt,
                context_summary=predecessor_summary,
            )
            if fatal:
                with CHECKPOINT_LOCK:
                    failed_chunks.add(boundary_idx)
                    save_checkpoint(
                        all_issues,
                        all_heroine_facts,
                        processed_chunks,
                        extra_relations_all,
                        failed_chunks=failed_chunks,
                        current_chunk_idx=boundary_idx,
                    )
                    _advance_chunk_progress(boundary_idx, processed_chunks, failed_chunks, progress_state)
                return err_msg or "所有 API_KEY 均不可用"
            _commit_chunk_result(
                boundary_idx,
                issues,
                heroine_facts,
                extra_rel,
                _next_summary,
                ok,
                err_msg,
                all_issues=all_issues,
                all_heroine_facts=all_heroine_facts,
                extra_relations_all=extra_relations_all,
                processed_chunks=processed_chunks,
                failed_chunks=failed_chunks,
                progress_state=progress_state,
            )

        if fatal_error_msg:
            return fatal_error_msg
        return None
    finally:
        _close_chunk_progress(progress_state)
        _ACTIVE_PROGRESS_STATE = previous_progress_state


def merge_heroine_facts_by_name(heroine_facts, heroine_names=None):
    merged = {}
    heroine_names = heroine_names or []

    def _canonical_name(name):
        for target in heroine_names:
            if not target:
                continue
            if target == name or target in name or name in target:
                return target
        return name

    for item in heroine_facts or []:
        if not isinstance(item, dict):
            continue
        raw_name = _norm_text(item.get("name"))
        if not raw_name:
            continue
        name = _canonical_name(raw_name)
        merged.setdefault(name, _empty_fact_bucket())
        facts = item.get("facts") or {}
        for dimension in _FACT_DIMENSIONS:
            merged[name][dimension].extend((facts.get(dimension) or []))

    for name in heroine_names:
        merged.setdefault(name, _empty_fact_bucket())

    for name, facts in merged.items():
        for dimension in _FACT_DIMENSIONS:
            facts[dimension] = _dedupe_by_signature(
                facts.get(dimension, []),
                lambda x, _dimension=dimension: _fact_signature(_dimension, x),
            )
    return merged


def _pick_first_nonempty_text(items, default=""):
    for item in items or []:
        text = _norm_text(item)
        if text:
            return text
    return default


def _merge_profile_texts(*values):
    merged = []
    seen = set()
    for value in values:
        text = _norm_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return "；".join(merged)


def _normalize_profile_report_summary(report_summary, detail_json_data=None):
    detail_json_data = detail_json_data or {}
    summaries = detail_json_data.get("summaries") or []
    features = detail_json_data.get("features") or []
    relationships = detail_json_data.get("relationships") or []
    interactions = detail_json_data.get("interactions") or []
    emotion_signals = detail_json_data.get("emotion_signals") or []

    fallback_identity = _pick_first_nonempty_text(summaries)
    fallback_features = _merge_profile_texts(
        _pick_first_nonempty_text(features),
        _pick_first_nonempty_text(emotion_signals),
    )
    fallback_relationship = _pick_first_nonempty_text(relationships) or _pick_first_nonempty_text(interactions)
    fallback_key_events = _pick_first_nonempty_text(interactions) or _pick_first_nonempty_text(summaries)

    if isinstance(report_summary, dict):
        legacy_features = _merge_profile_texts(
            report_summary.get("appearance"),
            report_summary.get("personality"),
            report_summary.get("traits"),
        )
        normalized = {
            "identity": _norm_text(report_summary.get("identity")) or fallback_identity,
            "features": _norm_text(report_summary.get("features")) or legacy_features or fallback_features,
            "relationship_with_protagonist": (
                _norm_text(report_summary.get("relationship_with_protagonist") or report_summary.get("relationship"))
                or fallback_relationship
            ),
            "key_events": _norm_text(report_summary.get("key_events")) or fallback_key_events,
        }
    else:
        normalized = {
            "identity": _norm_text(report_summary) or fallback_identity,
            "features": fallback_features,
            "relationship_with_protagonist": fallback_relationship,
            "key_events": fallback_key_events,
        }

    return {key: _norm_text(value) for key, value in normalized.items()}


def _match_detail_profile_data(heroine_name, all_female_characters):
    if not heroine_name or not isinstance(all_female_characters, dict):
        return {}
    if heroine_name in all_female_characters:
        return all_female_characters.get(heroine_name) or {}

    for key, char_data in all_female_characters.items():
        aliases = char_data.get("other_names", []) or []
        if heroine_name in aliases or heroine_name in key or key in heroine_name:
            return char_data or {}
    return {}


def generate_single_heroine_profile(heroine_name, facts, male_protagonist=None, detail_json_data=None):
    facts = facts or _empty_fact_bucket()
    detail_json_data = detail_json_data or {}
    male_prompt_text = _male_identity_prompt_text(male_protagonist) or "【男主】未显式提供"
    semantics_block = _build_profile_dimension_semantics_block()
    detail_payload = {
        "summaries": detail_json_data.get("summaries") or [],
        "features": detail_json_data.get("features") or [],
        "relationships": detail_json_data.get("relationships") or [],
        "interactions": detail_json_data.get("interactions") or [],
        "emotion_signals": detail_json_data.get("emotion_signals") or [],
    }
    prompt = f"""
你是一个小说角色分析助手。请根据以下已知信息，为【{heroine_name}】生成总结，并输出严格 JSON。
- 当前女主：{heroine_name}
- {male_prompt_text}

{semantics_block}

【信息来源一：角色分析报告】
- 这部分是人物展示信息，可用于补充身份背景、特点、与男主关系、关键事件。
- 这些线索只能用于总结展示层信息，不能反向伪造五维事实。

【信息来源二：结构化五维事实】
- 这部分负责告诉你：女主是否和非男主男性发生过性行为、是否有孩子或怀孕线索、是否被非男主男性身体接触、是否对非男主男性产生感情、是否存在非男主男性伴侣关系。

【任务拆解】
- 先读取五维结构化事实，再区分哪些是已经发生的事实、哪些只是缺失或为空。
- 再结合角色分析报告，把这些事实整理成角色层面的关键信息：身份背景、人物特点、与男主关系、关键事件，以及她和谁发生过关系、是否有孩子或怀孕线索、是否被非男主男性接触、是否爱过别人、是否存在婚约或被迫关系。
- 不得编造或推测结构化事实中不存在的信息；没有事实就老实写“未见明确记录/暂无明确证据”。

请输出JSON，包含两部分：

1. "report_summary"：供 report.py 展示的角色总结，字段必须是 identity、features、relationship_with_protagonist、key_events。
   - identity：身份背景（家族、出身、阵营、社会位置等）
   - features：人物特点，合并外貌、性格、标签等展示信息
   - relationship_with_protagonist：与男主的关系发展
   - key_events：涉及该角色的关键事件

2. "rescan_context"：供后续全局定向回扫使用的精简线索，严格100-150字，格式接近“身份：...；已知关系：...；已知事件：...”。

【写作要求】
- report_summary：必须输出为对象，至少包含 identity、features、relationship_with_protagonist、key_events 四个子字段。优先让字段简洁、可展示，不要输出长段分析。
- rescan_context：只保留身份、关系、对象、事件锚点，不要写成长段评论。
- 若某维度为空，不要编造成“清白”；只表述“当前扫描未见明确记录”。

输出格式：
{{
  "report_summary": {{
    "identity": "身份背景描述",
    "features": "人物特点描述",
    "relationship_with_protagonist": "与男主关系描述",
    "key_events": "关键事件概述"
  }},
  "rescan_context": "严格100-150字的回扫线索"
}}
""".strip()
    response = _call_json_chat_completion(
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"name": heroine_name, "facts": facts, "detail_json_data": detail_payload},
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0.1,
        max_tokens=1200,
        log_prefix=f"profile {heroine_name}",
    )
    data = _safe_json_loads(response.choices[0].message.content)
    if not isinstance(data, dict):
        return {
            "report_summary": _normalize_profile_report_summary("", detail_json_data),
            "rescan_context": "",
        }
    return {
        "report_summary": _normalize_profile_report_summary(data.get("report_summary", {}), detail_json_data),
        "rescan_context": _norm_text(data.get("rescan_context", "")),
    }


def generate_heroine_profiles(all_heroine_facts, heroines, male_protagonist=None, checkpoint_callback=None):
    merged = merge_heroine_facts_by_name(all_heroine_facts, heroine_names=heroines)
    if not merged:
        return {}

    all_female_characters = {}
    detail_path = _resolve_detail_path()
    if detail_path and os.path.exists(detail_path):
        try:
            with open(detail_path, "r", encoding="utf-8") as f:
                detail_data = json.load(f)
            all_female_characters = (detail_data or {}).get("all_female_characters") or {}
        except Exception as exc:
            logger.warning(f"读取 detail 文件失败，女主展示画像将缺少补充信息: {exc}")

    profiles = {}
    workers = max(1, min(int(RESCAN_MAX_WORKERS or 1), len(merged)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                generate_single_heroine_profile,
                name,
                facts,
                male_protagonist,
                _match_detail_profile_data(name, all_female_characters),
            ): name
            for name, facts in merged.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                profiles[name] = future.result()
            except Exception as exc:
                logger.warning(f"女主总结失败 {name}: {exc}")
                profiles[name] = {
                    "report_summary": _normalize_profile_report_summary({}, {}),
                    "rescan_context": "",
                }
            if callable(checkpoint_callback):
                checkpoint_callback(heroine_name=name, heroine_profiles=profiles)
    return profiles


def _save_heroine_profiles_to_detail(heroine_profiles):
    if not heroine_profiles:
        return
    detail_path = _resolve_detail_path()
    if not detail_path or not os.path.exists(detail_path):
        logger.warning("未找到对应的 *_detailed_*.json，无法写入 profile_for_report。")
        return

    with DETAIL_FILE_LOCK:
        try:
            with open(detail_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"读取 detail 文件失败: {exc}")
            return

        all_chars = data.setdefault("all_female_characters", {})
        for heroine_name, profile in (heroine_profiles or {}).items():
            if not heroine_name:
                continue
            target_key = heroine_name
            if target_key not in all_chars:
                for key, char_data in all_chars.items():
                    aliases = char_data.get("other_names", []) or []
                    if heroine_name == key or heroine_name in aliases or heroine_name in key or key in heroine_name:
                        target_key = key
                        break
            entry = all_chars.setdefault(target_key, {
                "avg_score": 0,
                "count": 0,
                "total_score": 0,
                "other_names": [],
                "summaries": [],
                "features": [],
                "relationships": [],
                "interactions": [],
                "emotion_signals": [],
            })
            entry["profile_for_report"] = _normalize_profile_report_summary(
                (profile or {}).get("report_summary", {}),
                entry,
            )
            if heroine_name != target_key and heroine_name not in (entry.get("other_names") or []):
                entry.setdefault("other_names", []).append(heroine_name)

        try:
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"写入 detail 文件失败: {exc}")


def find_cooccurrence_chunks(chunks, heroine_name, dimension, effective_keywords=None, candidate_indices=None, max_hits=8, window=200):
    regex = (effective_keywords or get_effective_keywords()).get(dimension)
    if regex is None:
        return []
    indices = sorted(candidate_indices) if candidate_indices is not None else list(range(len(chunks or [])))
    hits = []
    for idx in indices:
        if idx < 0 or idx >= len(chunks or []):
            continue
        chunk = chunks[idx] or ""
        name_positions = [m.start() for m in re.finditer(re.escape(heroine_name), chunk)]
        if not name_positions:
            continue
        for pos in name_positions:
            search_start = max(0, pos - window)
            search_end = min(len(chunk), pos + len(heroine_name) + window)
            if regex.search(chunk[search_start:search_end]):
                hits.append(idx)
                break
        if len(hits) >= max_hits:
            break
    return hits


def find_cooccurrence_chunks_with_positions(chunks, heroine_name, dimension, effective_keywords=None, candidate_indices=None, max_hits=None, window=200):
    """同 find_cooccurrence_chunks，但返回 list[CoOccurrenceHit]（含锚点/关键词位置）。"""
    if max_hits is None:
        max_hits = RESCAN_MAX_HITS
    regex = (effective_keywords or get_effective_keywords()).get(dimension)
    if regex is None:
        return []
    indices = sorted(candidate_indices) if candidate_indices is not None else list(range(len(chunks or [])))
    hits = []
    for idx in indices:
        if idx < 0 or idx >= len(chunks or []):
            continue
        chunk = chunks[idx] or ""
        name_positions = [m.start() for m in re.finditer(re.escape(heroine_name), chunk)]
        if not name_positions:
            continue
        for pos in name_positions:
            search_start = max(0, pos - window)
            search_end = min(len(chunk), pos + len(heroine_name) + window)
            m = regex.search(chunk[search_start:search_end])
            if m:
                hits.append(CoOccurrenceHit(
                    heroine_name=heroine_name,
                    dimension=dimension,
                    chunk_idx=idx,
                    anchor_pos=pos,
                    keyword_pos=search_start + m.start(),
                ))
                break  # 同原逻辑：每个chunk只记一个hit
        if len(hits) >= max_hits:
            break
    return hits


def _score_hit(hit, chunk, heroine_name):
    """仅用正向信号对共现hit评分。分值越高越有可能包含目标维度事实。"""
    score = 0.0
    window_start = max(0, hit.anchor_pos - 300)
    window_end = min(len(chunk), hit.anchor_pos + len(heroine_name) + 300)
    window = chunk[window_start:window_end]

    # 信号1：锚点-关键词距离（0-3分）
    distance = abs(hit.anchor_pos - hit.keyword_pos)
    if distance < 50:
        score += 3.0
    elif distance < 100:
        score += 2.0
    elif distance < 200:
        score += 1.0

    # 信号2：女主名在窗口内出现频次（0-2分）
    name_count = window.count(heroine_name)
    score += min(name_count * 0.5, 2.0)

    # 信号3：维度关键词密度（0-2分）
    dim_regex = _FACT_KEYWORDS.get(hit.dimension)
    if dim_regex:
        kw_matches = len(dim_regex.findall(window))
        score += min(kw_matches * 0.5, 2.0)

    # 信号4：男性线索词出现（0-1.5分）
    if _MALE_HINTS.search(window):
        score += 1.5

    return score


def _build_chunk_hit_map(chunks, merged, heroines, effective_keywords, candidate_indices, max_hits=None):
    """遍历所有女主的空维度，构建 chunk_idx → list[CoOccurrenceHit] 映射。"""
    if max_hits is None:
        max_hits = RESCAN_MAX_HITS
    chunk_hit_map = defaultdict(list)
    for heroine_name, facts in merged.items():
        for dimension in _FACT_DIMENSIONS:
            if facts.get(dimension):
                continue
            hits = find_cooccurrence_chunks_with_positions(
                chunks, heroine_name, dimension,
                effective_keywords=effective_keywords,
                candidate_indices=candidate_indices,
                max_hits=max_hits,
            )
            for hit in hits:
                chunk_hit_map[hit.chunk_idx].append(hit)
    return dict(chunk_hit_map)


def _hits_to_entries(hits, heroine_profiles, merge_threshold=None):
    """将同一chunk的CoOccurrenceHit列表聚合为ChunkHeroineEntry列表。
    同一女主的hit按锚点距离拆分：距离<merge_threshold合为一个entry，否则拆分。"""
    if merge_threshold is None:
        merge_threshold = RESCAN_MAX_WINDOW
    by_heroine = defaultdict(list)
    for hit in hits:
        by_heroine[hit.heroine_name].append(hit)

    entries = []
    for heroine_name, heroine_hits in by_heroine.items():
        # 按锚点排序
        heroine_hits.sort(key=lambda h: h.anchor_pos)
        # 按距离分组
        groups = [[heroine_hits[0]]]
        for h in heroine_hits[1:]:
            if h.anchor_pos - groups[-1][-1].anchor_pos < merge_threshold:
                groups[-1].append(h)
            else:
                groups.append([h])
        # 每组生成一个entry
        profile = (heroine_profiles or {}).get(heroine_name, {})
        rescan_context = _norm_text((profile or {}).get("rescan_context", ""))
        for group in groups:
            dims = list(dict.fromkeys(h.dimension for h in group))  # 去重保序
            positions = sorted(h.anchor_pos for h in group)
            anchor = positions[len(positions) // 2]  # 组内中位数
            entries.append(ChunkHeroineEntry(
                heroine_name=heroine_name,
                dimensions=dims,
                anchor_pos=anchor,
                rescan_context=rescan_context,
            ))
    return entries


def _cluster_by_proximity(entries, chunk_len, max_window=None, merge_threshold=None):
    """按锚点近邻聚类：距离<merge_threshold的entry合为一簇。"""
    if max_window is None:
        max_window = RESCAN_MAX_WINDOW
    if merge_threshold is None:
        merge_threshold = max_window
    if not entries:
        return []
    sorted_entries = sorted(entries, key=lambda e: e.anchor_pos)
    clusters = []
    current_group = [sorted_entries[0]]
    for entry in sorted_entries[1:]:
        group_min = current_group[0].anchor_pos
        if entry.anchor_pos - group_min <= merge_threshold:
            current_group.append(entry)
        else:
            clusters.append(current_group)
            current_group = [entry]
    clusters.append(current_group)

    result = []
    for group in clusters:
        min_anchor = min(e.anchor_pos for e in group)
        max_anchor = max(e.anchor_pos for e in group)
        max_name_len = max(len(e.heroine_name) for e in group)
        all_dims = set()
        for e in group:
            all_dims.update(e.dimensions)
        result.append(ProximityCluster(
            entries=list(group),
            window_start=min_anchor,
            window_end=max_anchor + max_name_len,
            all_dimensions=all_dims,
        ))
    return result


def _add_free_riders(cluster, chunk, merged, heroines, max_free_riders=2, max_prompt_heroines=None):
    """为簇添加"顺风车女主"：名字出现在窗口文本中但不在当前簇内的女主。"""
    if max_prompt_heroines is None:
        max_prompt_heroines = RESCAN_MAX_PROMPT_HEROINES
    current_names = {e.heroine_name for e in cluster.entries}
    if len(current_names) >= max_prompt_heroines:
        return cluster
    # 提取窗口文本区域
    padding = 500
    ws = max(0, cluster.window_start - padding)
    we = min(len(chunk), cluster.window_end + padding)
    window_text = chunk[ws:we]
    added = 0
    for h_name in heroines:
        if h_name in current_names:
            continue
        if added >= max_free_riders:
            break
        if len(current_names) + added >= max_prompt_heroines:
            break
        if h_name in window_text:
            # 只查顺风车女主的空维度（已有事实的维度不浪费token）
            rider_empty_dims = [d for d in cluster.all_dimensions if not (merged.get(h_name, {}).get(d))]
            if not rider_empty_dims:
                continue
            # 找到名字在窗口中的位置作为锚点
            pos_in_window = window_text.find(h_name)
            anchor = ws + pos_in_window if pos_in_window >= 0 else cluster.window_start
            cluster.entries.append(ChunkHeroineEntry(
                heroine_name=h_name,
                dimensions=rider_empty_dims,
                anchor_pos=anchor,
                is_free_rider=True,
            ))
            added += 1
    return cluster


def _extract_focused_window(chunk, cluster, padding=500, max_window=None):
    """提取簇覆盖区域 ± padding 的文本窗口，对齐句子边界。"""
    if max_window is None:
        max_window = RESCAN_MAX_WINDOW
    if not chunk:
        return ""
    min_pos = cluster.window_start
    max_pos = cluster.window_end

    raw_start = max(0, min_pos - padding)
    raw_end = min(len(chunk), max_pos + padding)

    # 如果窗口已覆盖全chunk或超出max_window后仍然比chunk短不了多少，直接返回全chunk
    if raw_end - raw_start >= len(chunk) * 0.85:
        return chunk

    # 超出max_window时从两侧向中心收缩
    if raw_end - raw_start > max_window:
        center = (min_pos + max_pos) // 2
        raw_start = max(0, center - max_window // 2)
        raw_end = min(len(chunk), raw_start + max_window)
        raw_start = max(0, raw_end - max_window)

    # 对齐句子边界
    sentence_ends = "。！？\n"
    # start 向前找句子结束符
    for i in range(raw_start, min(raw_start + 100, min_pos)):
        if chunk[i] in sentence_ends:
            raw_start = i + 1
            break
    # end 向后找句子结束符
    for i in range(raw_end - 1, max(raw_end - 100, max_pos), -1):
        if chunk[i] in sentence_ends:
            raw_end = i + 1
            break

    return chunk[raw_start:raw_end]


def build_targeted_rescan_prompt(heroine_name, dimension, male_protagonist=None, rescan_context=""):
    male_prompt_text = _male_identity_prompt_text(male_protagonist) or "【男主】未显式提供"
    extra_context = rescan_context or "无额外线索"
    semantics_block = _build_fact_dimension_semantics_block()
    keyword_rules_block = _build_dynamic_keyword_rules_block(dimension)
    workflow_block = _build_dimension_workflow_block("targeted_rescan", dimension)
    return f"""
你是“定向回扫器”。请仅为指定女主的指定维度补抽事实，并输出严格 JSON。
- 当前女主：{heroine_name}
- 当前维度：{dimension}
- {male_prompt_text}
- 回扫线索：{extra_context}
- 以下信息仅供参考线索，请以原文为准。
- {_get_dimension_rules(dimension)}

{semantics_block}

【当前回扫重点】
- 你只需要检查【{heroine_name}】在【{dimension}】上的遗漏事实。
- 其他维度即使在文本里顺手看到了，也不要输出，避免覆盖主扫描结果。
- 回扫线索只帮助你定位，不构成证据。

{keyword_rules_block}

{workflow_block}

输出格式：
{{
  "heroine_facts": [
    {{
      "name": "{heroine_name}",
      "facts": {{
        "sexual_relations": [],
        "children_info": [],
        "physical_contacts": [],
        "romantic_feelings": [],
        "partner_relations": []
      }}
    }}
  ],
  "_learned_keywords": {{
    "{dimension}": ["新关键词"]
  }}
}}
""".strip()


# ---- 全局补扫优化：多女主多维度 prompt 常量前缀 ----
_RESCAN_OPT_CONSTANT_PREFIX = """你是"多目标定向回扫器"。请仅为指定的女主和指定维度补抽事实，并输出严格 JSON。
- 以下信息仅供参考线索，请以原文为准。

【⚠️ evidence 字段硬规则（最高优先级！）】
evidence 必须是【文本】的连续子串，禁止任何改写！
- 禁止改写、同义替换、补字、删字、调整语序
- 禁止添加省略号（……/.../…），除非原文就包含
- 若原文有换行，evidence 必须保持换行
- 建议长度 12~80 字（但只要是原文连续子串即可）

【硬约束】
- evidence 必须是当前文本的连续子串，不得改写。
- partner_relations 必须输出 forced 字段（true/false）。
- physical_contacts / partner_relations：只记录"非男主男性"对女主的事实。女性对象、男主、性别不明对象必须排除。
- children_info 需严格区分 is_biological、origin、conception_method。
- 占位符禁止：partner/target 不能是"无/没有/不存在/未知/空/null"等。
- "强奸未遂/企图强奸/没得逞"不等于已发生性关系，不得仅凭此写入 sexual_relations。
- "无性行为怀孕/试管婴儿/人工授精"写入 children_info（conception_method=ivf_or_ai），不得反推 sexual_relations。""".strip()


def build_multi_heroine_rescan_prompt(cluster, male_protagonist=None, heroine_contexts=None):
    """构建多女主多维度回扫prompt。按三层结构排列以最大化prefix caching。"""
    # 第1层：常量前缀
    layer1 = _RESCAN_OPT_CONSTANT_PREFIX

    # 第2层：半常量（同维度组合相同）
    male_prompt_text = _male_identity_prompt_text(male_protagonist) or "【男主】未显式提供"
    sorted_dims = sorted(cluster.all_dimensions, key=lambda d: _FACT_DIMENSIONS.index(d) if d in _FACT_DIMENSIONS else 99)
    semantics_block = _build_fact_dimension_semantics_block_for_dims(sorted_dims)
    # 多维度的关键词规则和工作流
    dim_rules_parts = []
    for dim in sorted_dims:
        dim_rules_parts.append(f"- {_get_dimension_rules(dim)}")
    dim_rules_text = "\n".join(dim_rules_parts)
    keyword_rules_block = _build_dynamic_keyword_rules_block("、".join(sorted_dims))
    workflow_block = _build_dimension_workflow_block("targeted_rescan", "、".join(sorted_dims))

    layer2 = f"""- {male_prompt_text}
{dim_rules_text}

{semantics_block}

{keyword_rules_block}

{workflow_block}"""

    # 第3层：变量（每次调用不同）
    targets = []
    for entry in cluster.entries:
        dims_str = "、".join(entry.dimensions)
        ctx = (heroine_contexts or {}).get(entry.heroine_name, "") or "无额外线索"
        rider_tag = "（顺带检查）" if entry.is_free_rider else ""
        targets.append(f"- 女主：{entry.heroine_name}{rider_tag}，待补维度：{dims_str}\n  回扫线索：{ctx}")
    targets_text = "\n".join(targets)

    # 输出格式模板
    heroine_format_parts = []
    for entry in cluster.entries:
        facts_dict_parts = []
        for dim in sorted_dims:
            facts_dict_parts.append(f'        "{dim}": []')
        facts_str = ",\n".join(facts_dict_parts)
        heroine_format_parts.append(
            f'    {{\n      "name": "{entry.heroine_name}",\n'
            f'      "facts": {{\n{facts_str}\n      }}\n    }}'
        )
    heroines_format = ",\n".join(heroine_format_parts)
    kw_parts = []
    for dim in sorted_dims:
        kw_parts.append(f'    "{dim}": ["新关键词"]')
    kw_str = ",\n".join(kw_parts)

    layer3 = f"""【回扫目标】
{targets_text}

【当前回扫重点】
- 你只需要检查上述女主在上述维度上的遗漏事实。
- 其他维度即使在文本里顺手看到了，也不要输出，避免覆盖主扫描结果。
- 回扫线索只帮助你定位，不构成证据。

输出格式：
{{
  "heroine_facts": [
{heroines_format}
  ],
  "_learned_keywords": {{
{kw_str}
  }}
}}"""

    return f"{layer1}\n\n{layer2}\n\n{layer3}"


def _estimate_max_tokens(cluster):
    """根据簇中女主×维度对数估算max_tokens。"""
    n_pairs = sum(len(e.dimensions) for e in cluster.entries)
    return min(200 + n_pairs * 800, 6000)


def _parse_multi_heroine_response(data, cluster, heroines, male_protagonist, chunk_idx):
    """解析多女主多维度回扫的JSON响应。返回 (boost_facts, learned_keywords)。"""
    if not isinstance(data, dict):
        return [], {}
    data.pop("_reasoning", None)
    data.pop("_context_summary", None)
    learned_keywords = data.pop("_learned_keywords", {})
    if not isinstance(learned_keywords, dict):
        learned_keywords = {}
    boost_facts = data.get("heroine_facts", [])
    if not isinstance(boost_facts, list):
        boost_facts = [boost_facts] if boost_facts else []

    # 过滤：只保留簇内目标女主 + 查询维度的输出
    allowed_dims = cluster.all_dimensions
    cluster_names = {e.heroine_name for e in cluster.entries}
    filtered = []
    for item in boost_facts:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        # 模糊匹配簇内女主名
        matched_name = None
        for cn in cluster_names:
            if cn == name or cn in name or name in cn:
                matched_name = cn
                break
        if not matched_name:
            continue
        facts = item.get("facts", {})
        if not isinstance(facts, dict):
            continue
        clean_facts = {}
        for dim in _FACT_DIMENSIONS:
            if dim in allowed_dims and isinstance(facts.get(dim), list) and facts[dim]:
                clean_facts[dim] = facts[dim]
            else:
                clean_facts[dim] = []
        filtered.append({"name": matched_name, "facts": clean_facts, "chunk_index": chunk_idx + 1})

    # 后处理
    filtered = _postprocess_heroine_facts(filtered, heroines, male_protagonist, chunk_idx + 1)
    return filtered, learned_keywords


def _merge_multi_dimension_results(merged, boost_facts, dimensions, heroines):
    """将多维度回扫结果合并回merged字典（对每个维度分别去重）。"""
    merged_list = [{"name": name, "facts": values} for name, values in merged.items()]
    for dim in dimensions:
        merged_list = _merge_dimension_facts(merged_list, boost_facts, dim)
    new_merged = merge_heroine_facts_by_name(merged_list, heroine_names=heroines)
    return new_merged


def global_dimension_rescan(chunks, processed_chunks, all_heroine_facts, heroine_profiles, heroines, male_protagonist=None, checkpoint_callback=None, rescan_done_chunks=None):
    if not ENABLE_GLOBAL_RESCAN:
        return all_heroine_facts

    rescan_done_chunks = set(rescan_done_chunks or [])

    merged = merge_heroine_facts_by_name(all_heroine_facts, heroine_names=heroines)
    effective_keywords = get_effective_keywords()
    candidate_indices = set(processed_chunks or [])

    # Phase 1: 构建 chunk-centric hit map（排除已处理的补扫 chunk）
    actual_candidates = candidate_indices - rescan_done_chunks
    chunk_hit_map = _build_chunk_hit_map(chunks, merged, heroines, effective_keywords, actual_candidates)
    if rescan_done_chunks:
        logger.info(f"全局补扫：续传模式，跳过 {len(rescan_done_chunks)} 个已处理chunk")
    if not chunk_hit_map:
        logger.info("全局补扫：无共现命中，跳过。")
        return [{"name": name, "facts": facts} for name, facts in merged.items()]

    # Phase 2: 评分并过滤低分 hit
    for chunk_idx in list(chunk_hit_map.keys()):
        scored = []
        for hit in chunk_hit_map[chunk_idx]:
            hit.score = _score_hit(hit, chunks[chunk_idx], hit.heroine_name)
            if hit.score >= RESCAN_PRE_FILTER_THRESHOLD:
                scored.append(hit)
        if scored:
            chunk_hit_map[chunk_idx] = scored
        else:
            del chunk_hit_map[chunk_idx]

    if not chunk_hit_map:
        logger.info("全局补扫：评分过滤后无命中，跳过。")
        return [{"name": name, "facts": facts} for name, facts in merged.items()]

    # Phase 3: 按总分排序，二次排序按维度组合（利于prefix caching）
    def _chunk_sort_key(idx):
        hits = chunk_hit_map[idx]
        dim_set = frozenset(h.dimension for h in hits)
        total_score = sum(h.score for h in hits)
        return (dim_set, -total_score)
    sorted_chunks = sorted(chunk_hit_map.keys(), key=_chunk_sort_key)

    total_calls = 0
    # Phase 4: 遍历每个 chunk
    for chunk_idx in sorted_chunks:
        hits = chunk_hit_map.get(chunk_idx, [])
        # 早停：剔除已填充维度的 hit
        hits = [h for h in hits if not merged.get(h.heroine_name, {}).get(h.dimension)]
        if not hits:
            continue

        # hit → entry（按距离拆分同一女主）
        entries = _hits_to_entries(hits, heroine_profiles)
        # 近邻聚类
        clusters = _cluster_by_proximity(entries, len(chunks[chunk_idx]))

        chunk_had_error = False
        for cluster in clusters:
            # 添加顺风车女主
            cluster = _add_free_riders(cluster, chunks[chunk_idx], merged, heroines)
            # 更新 all_dimensions（顺风车可能带入新维度）
            cluster.all_dimensions = set()
            for e in cluster.entries:
                cluster.all_dimensions.update(e.dimensions)

            # 提取聚焦窗口
            window_text = _extract_focused_window(chunks[chunk_idx], cluster)
            # 构建 prompt
            heroine_contexts = {e.heroine_name: e.rescan_context for e in cluster.entries}
            prompt = build_multi_heroine_rescan_prompt(cluster, male_protagonist, heroine_contexts)
            n_heroines = len(cluster.entries)
            n_dims = len(cluster.all_dimensions)

            try:
                response = _call_json_chat_completion(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"文本：\n{window_text}"},
                    ],
                    temperature=0.1,
                    max_tokens=_estimate_max_tokens(cluster),
                    log_prefix=f"global_rescan_opt chunk {chunk_idx} ({n_heroines}h x {n_dims}d)",
                )
                total_calls += 1
                data = _safe_json_loads(response.choices[0].message.content)
                if not isinstance(data, dict):
                    continue

                boost_facts, learned_kw = _parse_multi_heroine_response(
                    data, cluster, heroines, male_protagonist, chunk_idx
                )

                # 保存学习到的关键词
                if isinstance(learned_kw, dict) and any(learned_kw.values()):
                    save_learned_keywords(learned_kw, source_phase="global_rescan_opt", novel_name=clean_filename)

                # 合并结果
                merged = _merge_multi_dimension_results(merged, boost_facts, list(cluster.all_dimensions), heroines)
            except Exception as e:
                chunk_had_error = True
                logger.warning(f"全局定向回扫(优化)失败 chunk {chunk_idx} ({n_heroines}h x {n_dims}d): {e}")
                continue

        # 只有全部 cluster 成功才标记为已完成
        if not chunk_had_error:
            rescan_done_chunks.add(chunk_idx)

        # chunk 级别的 checkpoint（包含 rescan_done_chunks）
        if callable(checkpoint_callback):
            checkpoint_callback(
                all_heroine_facts=[{"name": n, "facts": f} for n, f in merged.items()],
                rescan_done_chunks=rescan_done_chunks,
            )

    logger.info(f"全局补扫优化完成，共 {total_calls} 次API调用")
    return [{"name": name, "facts": facts} for name, facts in merged.items()]


def generate_report(all_issues, all_heroine_facts, heroines):
    """生成最终报告（事实抽取版）"""
    def classify_issue(item):
        cat = item.get('category', '')
        t = item.get('type', '')
        text = f"{cat}{t}"
        if any(k in text for k in ['雷', '毒', '严重']):
            return 'lei'
        if any(k in text for k in ['郁闷', '不爽']):
            return 'yumen'
        return 'other'

    lei_points = [x for x in all_issues if classify_issue(x) == 'lei']
    yumen_points = [x for x in all_issues if classify_issue(x) == 'yumen']
    
    report = f"🔍 小说深度扫描报告：{clean_filename}\n"
    report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    report += "="*60 + "\n\n"
    
    # 1. 女主事实抽取报告
    report += "❤️ 【女主事实抽取报告（仅展示抽取到的事实，最终判定由 reviewer 完成）】\n"
    report += "-"*30 + "\n"
    
    # 汇总每个女主的事实
    heroine_facts_merged = {}
    for item in all_heroine_facts:
        name = item.get('name', '')
        if not name:
            continue
        # 模糊匹配
        target_name = name
        for h_name in heroines:
            if h_name in name or name in h_name:
                target_name = h_name
                break
        
        if target_name not in heroine_facts_merged:
            heroine_facts_merged[target_name] = {
                "sexual_relations": [],
                "children_info": [],
                "physical_contacts": [],
                "romantic_feelings": [],
                "partner_relations": [],
            }
        
        facts = item.get("facts", {})
        for key in heroine_facts_merged[target_name]:
            heroine_facts_merged[target_name][key].extend(facts.get(key, []))

    # 输出女主报告
    for name in heroines:
        if name not in heroine_facts_merged:
            heroine_facts_merged[name] = {
                "sexual_relations": [],
                "children_info": [],
                "physical_contacts": [],
                "romantic_feelings": [],
                "partner_relations": [],
            }
    
    for name, facts in heroine_facts_merged.items():
        report += f"角色：{name}\n"
        
        if facts["sexual_relations"]:
            report += f"  - 性关系: {len(facts['sexual_relations'])} 条记录\n"
            for r in facts["sexual_relations"][:2]:  # 最多显示2条
                partner = r.get("partner", "未知")
                is_ml = "男主" if r.get("is_male_lead") else "非男主"
                report += f"    · 与{partner}({is_ml}): {r.get('detail', '')[:30]}\n"
        
        if facts["children_info"]:
            report += f"  - 孩子: {len(facts['children_info'])} 条记录\n"
            for c in facts["children_info"][:2]:
                origin = c.get("origin", "未知")
                report += f"    · 来源:{origin}, {c.get('detail', '')[:30]}\n"
        
        if facts["physical_contacts"]:
            non_ml_contacts = [c for c in facts["physical_contacts"] if not c.get("is_male_lead")]
            if non_ml_contacts:
                report += f"  - 非男主肉体接触: {len(non_ml_contacts)} 条记录\n"
        
        if facts["romantic_feelings"]:
            non_ml_feelings = [f for f in facts["romantic_feelings"] if not f.get("is_male_lead")]
            if non_ml_feelings:
                report += f"  - 非男主感情: {len(non_ml_feelings)} 条记录\n"
        
        if facts["partner_relations"]:
            non_ml_partners = [p for p in facts["partner_relations"] if not p.get("is_male_lead")]
            if non_ml_partners:
                report += f"  - 非男主伴侣: {len(non_ml_partners)} 条记录\n"
        
        report += "\n"

    report += "\n" + "="*60 + "\n\n"
    
    # 2. 毒点统计
    report += f"💀 【毒点扫描结果】\n"
    report += f"🔴 严重雷点: {len(lei_points)} 处\n"
    report += f"🟠 郁闷点: {len(yumen_points)} 处\n\n"
    
    if lei_points:
        report += "--- 严重雷点详情 ---\n"
        for i, item in enumerate(lei_points, 1):
            report += f"{i}. [{item.get('type')}] (第 {item['chunk_index']} 块)\n"
            report += f"   原文: \"{item.get('content', 'N/A')[:80]}...\"\n"
            report += f"   分析: {item.get('reason', 'N/A')}\n\n"
            
    if yumen_points:
        report += "--- 郁闷点详情 ---\n"
        for i, item in enumerate(yumen_points, 1):
            report += f"{i}. [{item.get('type')}] (第 {item['chunk_index']} 块)\n"
            report += f"   分析: {item.get('reason', 'N/A')}\n\n"

    return report


# ---------------- 追加 detail.json 交叉取证 ----------------
def _find_latest_detail_file():
    """寻找当前小说对应的 *_detailed_*.json（protagonist.py 生成）"""
    base = get_base_dir()
    _results_dir = os.path.join(base, "results")
    patterns = [
        os.path.join(_results_dir, "**", f"{clean_filename}*_detailed_*.json"),
        os.path.join(_results_dir, "**", "*_detailed_*.json"),
        os.path.join(base, "**", f"{clean_filename}*_detailed_*.json"),
        os.path.join(base, "**", "*_detailed_*.json"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    # 优先包含 clean_filename 的文件，再取最近修改
    prioritized = [f for f in files if clean_filename in os.path.basename(f)]
    if prioritized:
        files = prioritized
    latest = max(files, key=os.path.getmtime)
    return latest


def _append_to_detail_file(heroine_facts, extra_relations, male_protagonist=None):
    """
    将扫描阶段抽取的结构化事实写入 protagonist 生成的 detail.json
    - heroine_facts: 女主结构化事实列表
    - extra_relations: 其他补充互动
    
    【重要】写入时做别名归一化：
    - 如果 name 是某个已有 key 的别名（通过 other_names 或包含关系匹配），合并到那个 key
    - 避免同一角色的证据被分散到多个 key
    """
    if not heroine_facts and not extra_relations:
        return
    detail_path = _resolve_detail_path()
    if not detail_path or not os.path.exists(detail_path):
        logger.warning("未找到对应的 *_detailed_*.json，无法追加取证数据。")
        return
    with DETAIL_FILE_LOCK:
        try:
            with open(detail_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"读取 detail 文件失败: {e}")
            return

        all_chars = data.setdefault("all_female_characters", {})

        def _find_canonical_key(name: str) -> str:
            """
            查找 name 对应的 canonical key（别名归一化）
            优先级：
            1. 精确匹配已有 key
            2. 在已有 key 的 other_names 中找到
            3. 包含关系匹配（优先匹配更长的 key）
            4. 都没有则返回 name 本身
            """
            name = name.strip()
            if not name:
                return name
            if name in all_chars:
                return name
            for key, char_data in all_chars.items():
                other_names = char_data.get("other_names", []) or []
                if name in other_names:
                    return key
            candidates = []
            for key in all_chars.keys():
                if (name in key) or (key in name):
                    candidates.append(key)
            if candidates:
                return max(candidates, key=len)
            return name

        def _ensure_entry(name):
            if name not in all_chars:
                all_chars[name] = {
                    "avg_score": 0,
                    "count": 0,
                    "total_score": 0,
                    "other_names": [],
                    "summaries": [],
                    "features": [],
                    "relationships": [],
                    "interactions": [],
                    "emotion_signals": [],
                }
            entry = all_chars[name]
            entry.setdefault("purity_facts", {
                "sexual_relations": [],
                "children_info": [],
                "physical_contacts": [],
                "romantic_feelings": [],
                "partner_relations": [],
            })
            entry.setdefault("non_male_male_interactions", [])
            entry.setdefault("male_lead_intimacy", [])
            entry.setdefault("virginity_mentions", [])
            entry.setdefault("children_info", [])
            return entry

        for item in heroine_facts:
            raw_name = item.get("name", "")
            if not raw_name:
                continue
            name = _find_canonical_key(raw_name)
            entry = _ensure_entry(name)
            if raw_name != name and raw_name not in entry.get("other_names", []):
                entry.setdefault("other_names", []).append(raw_name)
            facts = item.get("facts", {})
            purity_facts = entry["purity_facts"]

            for key in ["sexual_relations", "children_info", "physical_contacts", "romantic_feelings", "partner_relations"]:
                for fact in facts.get(key, []):
                    if fact not in purity_facts[key]:
                        purity_facts[key].append(fact)

            for sr in facts.get("sexual_relations", []):
                if sr.get("is_male_lead"):
                    text = f"(第{sr.get('chunk_index', '?')}块) 与男主亲密: {sr.get('detail', '')} | 证据: {sr.get('evidence', '')}"
                    if text not in entry["male_lead_intimacy"]:
                        entry["male_lead_intimacy"].append(text)

            for pc in facts.get("physical_contacts", []):
                if not pc.get("is_male_lead"):
                    partner = pc.get("partner", "未知")
                    text = f"(第{pc.get('chunk_index', '?')}块) 与{partner}接触: {pc.get('contact_type', '')} {pc.get('detail', '')} | 证据: {pc.get('evidence', '')}"
                    if text not in entry["non_male_male_interactions"]:
                        entry["non_male_male_interactions"].append(text)

        for item in extra_relations:
            female = item.get("female") or "未知女性"
            male = item.get("male") or "未知男性"
            is_male_lead = item.get("is_male_lead", False)
            detail = item.get("detail", "")
            idx = item.get("chunk_index")
            evidence = item.get("evidence") or ""
            text = f"(第{idx}块) 与{male}互动: {detail}" if idx else f"与{male}互动: {detail}"
            if evidence:
                text += f" | 证据: {evidence}"
            entry = _ensure_entry(female)
            if is_male_lead:
                if text and text not in entry["male_lead_intimacy"]:
                    entry["male_lead_intimacy"].append(text)
            else:
                if text and text not in entry["non_male_male_interactions"]:
                    entry["non_male_male_interactions"].append(text)

        try:
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"已向 detail 文件追加结构化事实: {detail_path}")
        except Exception as e:
            logger.warning(f"写入 detail 文件失败: {e}")

def _run_scan_for_indices(chunks, indices, system_prompt, heroines, male_protagonist, fact_boost_prompt,
                          all_issues, all_heroine_facts, extra_relations_all, processed_chunks, failed_chunks,
                          phase_name="补扫"):
    global _ACTIVE_PROGRESS_STATE
    """
    对指定 indices（0-based）执行扫描。
    - 成功：写入 all_issues/all_heroine_facts/extra_relations_all，并将 idx 加入 processed_chunks，且从 failed_chunks 移除
    - 失败：idx 保持/加入 failed_chunks，但不加入 processed_chunks（避免“假完成”）
    - 致命错误：立即保存断点并返回 (fatal_error_msg)
    """
    if not indices:
        return None
    pending_indices = [i for i in indices if i not in processed_chunks]
    if not pending_indices:
        return None

    workers = max(1, min(int(RESCAN_MAX_WORKERS or 1), int(MAX_WORKERS or 1), len(pending_indices)))
    print(f"🔁 {phase_name}阶段：准备扫描 {len(pending_indices)} 个片段，线程数={workers}")
    logger.info(f"{phase_name}阶段：indices={sorted(pending_indices)[:50]}{'...' if len(pending_indices) > 50 else ''}")

    progress_state = _init_chunk_progress(
        total=len(pending_indices),
        initial=0,
        desc=f"{phase_name}中",
        processed_chunks=processed_chunks,
        failed_chunks=failed_chunks,
    )
    previous_progress_state = _ACTIVE_PROGRESS_STATE
    _ACTIVE_PROGRESS_STATE = progress_state
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _rescan_worker_task,
                    i,
                    chunks,
                    system_prompt,
                    heroines,
                    male_protagonist,
                    fact_boost_prompt,
                ): i
                for i in pending_indices
            }
            fatal_error = None
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    idx, issues, heroine_facts, extra_rel, next_summary, ok, fatal, err_msg = future.result()
                except Exception as e:
                    ok, fatal, err_msg = False, False, str(e)
                    issues, heroine_facts, extra_rel = [], [], []
                    next_summary = ""

                if fatal:
                    fatal_error = err_msg or "所有 API_KEY 均不可用"
                    with CHECKPOINT_LOCK:
                        failed_chunks.add(idx)
                        save_checkpoint(
                            all_issues,
                            all_heroine_facts,
                            processed_chunks,
                            extra_relations_all,
                            failed_chunks=failed_chunks,
                            current_chunk_idx=idx,
                        )
                        _advance_chunk_progress(idx, processed_chunks, failed_chunks, progress_state)
                    logger.error(f"❌ 致命错误，终止{phase_name}：chunk={idx} err={fatal_error}")
                    break

                _commit_chunk_result(
                    idx,
                    issues,
                    heroine_facts,
                    extra_rel,
                    next_summary,
                    ok,
                    err_msg,
                    all_issues=all_issues,
                    all_heroine_facts=all_heroine_facts,
                    extra_relations_all=extra_relations_all,
                    processed_chunks=processed_chunks,
                    failed_chunks=failed_chunks,
                    progress_state=progress_state,
                )

            return fatal_error
    finally:
        _close_chunk_progress(progress_state)
        _ACTIVE_PROGRESS_STATE = previous_progress_state


def _rescan_worker_task(idx, chunks, system_prompt, heroines, male_protagonist=None, fact_boost_prompt=None):
    context_summary = ""
    if idx > 0:
        with CHECKPOINT_LOCK:
            context_summary = CHUNK_SUMMARIES.get(idx - 1, "")
        if not context_summary:
            generated_summary = generate_context_summary(
                chunks[idx - 1],
                heroines=heroines,
                male_protagonist=male_protagonist,
            )
            if generated_summary:
                with CHECKPOINT_LOCK:
                    existing_summary = CHUNK_SUMMARIES.get(idx - 1, "")
                    if existing_summary:
                        context_summary = existing_summary
                    else:
                        CHUNK_SUMMARIES[idx - 1] = generated_summary
                        context_summary = generated_summary

    issues, heroine_facts, extra_rel, next_summary, ok, fatal, err_msg = scan_chunk(
        chunks[idx],
        idx,
        len(chunks),
        system_prompt,
        heroines,
        male_protagonist,
        fact_boost_prompt,
        context_summary=context_summary,
    )
    return idx, issues, heroine_facts, extra_rel, next_summary, ok, fatal, err_msg

def main(novel_path=None, book_name=None, run_id=None, detail_path=None):
    global NOVEL_FILE_PATH, clean_filename, OUTPUT_DIR, CHECKPOINT_FILE, logger, CURRENT_CHUNK_PLAN_METADATA, CHUNK_SUMMARIES, _middle_summary_calls, _ACTIVE_DETAIL_PATH

    # ---- 彻底重新初始化，防止跨小说状态残留 ----
    NOVEL_FILE_PATH = None
    clean_filename = None
    OUTPUT_DIR = None
    CHECKPOINT_FILE = None
    CHUNK_SUMMARIES = {}
    _middle_summary_calls = 0
    _ACTIVE_DETAIL_PATH = None

    base = get_base_dir()
    results_base = os.path.join(base, "results")

    if novel_path:
        os.environ["NOVEL_PATH"] = novel_path

    NOVEL_FILE_PATH = novel_path or os.environ.get("NOVEL_PATH", os.path.join(base, "novels", "default.txt"))
    clean_filename = (book_name or os.path.splitext(os.path.basename(NOVEL_FILE_PATH))[0]).strip()
    init_token_tracker(clean_filename, run_id=run_id)

    resume_dir = find_latest_scan_checkpoint(clean_filename)
    if resume_dir:
        OUTPUT_DIR = resume_dir
        CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "latest_checkpoint.json")
        print(f"★ 发现断点，将复用目录: {OUTPUT_DIR}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        OUTPUT_DIR = os.path.join(results_base, f"{clean_filename}_scan_{timestamp}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "latest_checkpoint.json")
        print(f"★ 未找到断点，创建新目录: {OUTPUT_DIR}")

    # 重新配置 logger（支持多次调用不残留旧 handler）
    logger = logging.getLogger("novel_scan")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    _fh = logging.FileHandler(os.path.join(OUTPUT_DIR, "scan.log"), encoding='utf-8')
    _fh.setFormatter(_fmt)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_sh)
    _ACTIVE_DETAIL_PATH = _resolve_detail_path(
        explicit_detail_path=detail_path,
        checkpoint_detail_path=_peek_checkpoint_detail_path(),
    )

    print(f"🚀 开始深度扫描《{clean_filename}》...")
    
    # 1. 加载规则
    categories, glossary = load_rules()
    if not categories:
        print("❌ 无法加载规则，终止运行。")
        return

    # 2. 寻找女主名单和男主姓名
    heroines, male_protagonist = find_heroines(detail_path=_ACTIVE_DETAIL_PATH)
    
    # 3. 构建 Prompt
    system_prompt = build_prompt(categories, glossary, heroines, male_protagonist)
    fact_boost_prompt = None
    
    # 4. 读取与切分
    if not os.path.exists(NOVEL_FILE_PATH):
        print(f"❌ 文件不存在: {NOVEL_FILE_PATH}")
        return
    text = read_novel(NOVEL_FILE_PATH)
    chunk_manifest = build_chunk_manifest(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunk_manifest_path = os.path.join(OUTPUT_DIR, "chunk_manifest.json")
    save_chunk_manifest(chunk_manifest, chunk_manifest_path)
    chunks = [entry.get("text", "") for entry in chunk_manifest.get("chunks", [])]
    CURRENT_CHUNK_PLAN_METADATA = _build_chunk_plan_metadata(chunk_manifest=chunk_manifest)
    print(f"📚 文本已切分为 {len(chunks)} 个片段")
    
    # 断点续传：加载历史进度
    (
        all_issues,
        all_heroine_facts,
        processed_chunks,
        extra_relations_all,
        failed_chunks,
        heroine_profiles,
        checkpoint_detail_path,
        rescan_done_chunks,
        rescan_completed,
    ) = load_checkpoint()
    _ACTIVE_DETAIL_PATH = _resolve_detail_path(
        explicit_detail_path=detail_path,
        checkpoint_detail_path=checkpoint_detail_path,
    )
    if processed_chunks:
        print(f"⏸️ 检测到断点，已完成 {len(processed_chunks)} 个片段，将继续首扫剩余片段")
    else:
        processed_chunks = set()
        extra_relations_all = []
        failed_chunks = set()
    
    # 若已全部扫描完，直接跳过扫描阶段
    if len(processed_chunks) >= len(chunks):
        print("✅ 检测到全部片段均已扫描，跳过扫描阶段。")
    else:
        print(f"⚡ 启动线程块首扫，线程数={MAX_WORKERS}...")
        fatal_error = _run_initial_thread_block_scan(
            chunks=chunks,
            system_prompt=system_prompt,
            heroines=heroines,
            male_protagonist=male_protagonist,
            fact_boost_prompt=fact_boost_prompt,
            all_issues=all_issues,
            all_heroine_facts=all_heroine_facts,
            extra_relations_all=extra_relations_all,
            processed_chunks=processed_chunks,
            failed_chunks=failed_chunks,
        )
        if fatal_error:
            print(f"❌ 扫描终止：{fatal_error}")
            return

    # 5.5 扫描完后复查：对遗漏/失败片段进行补扫（多轮）
    # 说明：processed_chunks 存 0-based index；issues 等内部 chunk_index 仍用 1-based 展示
    all_indices = set(range(len(chunks)))
    missing = all_indices - set(processed_chunks)
    if missing:
        logger.warning(f"📌 发现遗漏片段：{len(missing)} 个（将进入补扫流程）")
    if failed_chunks:
        logger.warning(f"📌 发现失败片段：{len(failed_chunks)} 个（将进入补扫流程）")

    if RESCAN_ROUNDS > 0 and (missing or failed_chunks):
        for round_no in range(1, RESCAN_ROUNDS + 1):
            pending = sorted(list(all_indices - set(processed_chunks)))
            if not pending:
                break
            print(f"🧩 补扫轮次 {round_no}/{RESCAN_ROUNDS}：待补扫 {len(pending)} 个片段（含失败/遗漏）")
            fatal = _run_scan_for_indices(
                chunks=chunks,
                indices=pending,
                system_prompt=system_prompt,
                heroines=heroines,
                male_protagonist=male_protagonist,
                fact_boost_prompt=fact_boost_prompt,
                all_issues=all_issues,
                all_heroine_facts=all_heroine_facts,
                extra_relations_all=extra_relations_all,
                processed_chunks=processed_chunks,
                failed_chunks=failed_chunks,
                phase_name=f"补扫(第{round_no}轮)",
            )
            if fatal:
                print(f"❌ 补扫终止：{fatal}")
                return

        # 补扫后最终复查
        final_missing = all_indices - set(processed_chunks)
        if final_missing:
            logger.warning(f"⚠️ 补扫后仍有遗漏/失败片段：{len(final_missing)} 个（已写入断点 failed_chunks）")
            print(f"⚠️ 补扫完成但仍有 {len(final_missing)} 个片段未成功（详见 {CHECKPOINT_FILE} 的 failed_chunks）")
        else:
            print("✅ 补扫完成：所有片段均已成功扫描。")

    if heroine_profiles is None:
        heroine_profiles = generate_heroine_profiles(
            all_heroine_facts,
            heroines,
            male_protagonist,
            checkpoint_callback=lambda **_kwargs: save_checkpoint(
                all_issues,
                all_heroine_facts,
                processed_chunks,
                extra_relations_all,
                failed_chunks=failed_chunks,
                heroine_profiles=_kwargs.get("heroine_profiles"),
            ),
        )
    if heroine_profiles:
        _save_heroine_profiles_to_detail(heroine_profiles)

    if ENABLE_GLOBAL_RESCAN and not rescan_completed:
        all_heroine_facts = global_dimension_rescan(
            chunks=chunks,
            processed_chunks=processed_chunks,
            all_heroine_facts=all_heroine_facts,
            heroine_profiles=heroine_profiles,
            heroines=heroines,
            male_protagonist=male_protagonist,
            checkpoint_callback=lambda all_heroine_facts=None, rescan_done_chunks=None, **_kwargs: save_checkpoint(
                all_issues,
                all_heroine_facts or [],
                processed_chunks,
                extra_relations_all,
                failed_chunks=failed_chunks,
                heroine_profiles=heroine_profiles,
                rescan_done_chunks=rescan_done_chunks,
            ),
            rescan_done_chunks=rescan_done_chunks,
        )
        rescan_completed = True
        save_checkpoint(
            all_issues,
            all_heroine_facts,
            processed_chunks,
            extra_relations_all,
            failed_chunks=failed_chunks,
            heroine_profiles=heroine_profiles,
            rescan_done_chunks=rescan_done_chunks,
            rescan_completed=True,
        )
    elif rescan_completed:
        logger.info("全局补扫：检测到断点标记 rescan_completed=true，跳过补扫阶段。")

    # 6. 追加 detail.json 结构化事实
    _append_to_detail_file(all_heroine_facts, extra_relations_all, male_protagonist)

    # 7. 保存与报告
    raw_data = {
        "issues": all_issues,
        "heroine_facts": all_heroine_facts,
        "extra_relations": extra_relations_all,
        "heroine_profiles": heroine_profiles,
        "detail_path": _ACTIVE_DETAIL_PATH,
        "chunk_manifest_file": chunk_manifest_path,
        "chunk_plan": CURRENT_CHUNK_PLAN_METADATA,
    }
    with open(f"{OUTPUT_DIR}/raw_data.json", 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
        
    report = generate_report(all_issues, all_heroine_facts, heroines)
    report_file = f"{OUTPUT_DIR}/FULL_REPORT.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print("\n" + "="*60)
    print("✅ 分析完成！")
    print(f"📄 报告: {report_file}")
    if token_tracker is not None:
        snap = token_tracker.snapshot()
        print(f"🔢 Token 统计: 输入 {snap.get('input', 0)} ，输出 {snap.get('output', 0)} ，总计 {snap.get('total', 0)}")
        token_tracker.flush(status="finished")
    print("="*60)
    print(report[:2000]) # 打印前2000字预览

if __name__ == "__main__":
    main()
