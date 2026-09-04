import os
import glob
import json
import argparse
from datetime import datetime
import concurrent.futures
import time
from tqdm import tqdm
import threading
import re
import logging
from api_client import make_chat_completion
from shared_utils import get_base_dir, read_file_safely
from token_tracker import create_default_tracker

try:
    from openai import OpenAI
    from openai import APIStatusError
except Exception:
    OpenAI = None  # 若未安装 openai，润色功能将不可用
    APIStatusError = Exception

# token 统计
token_tracker = None


def init_token_tracker(book_name, run_id=None):
    global token_tracker
    token_tracker = create_default_tracker(
        "report.py",
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


def get_report_logger():
    global _REPORT_LOGGER
    if _REPORT_LOGGER is not None:
        return _REPORT_LOGGER
    logger = logging.getLogger("report_generation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        try:
            os.makedirs(os.path.dirname(REPORT_RUN_LOG_PATH), exist_ok=True)
            fh = logging.FileHandler(REPORT_RUN_LOG_PATH, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(fh)
        except Exception:
            pass
    _REPORT_LOGGER = logger
    return logger


def log_report(msg: str):
    print(msg)
    try:
        get_report_logger().info(msg)
    except Exception:
        pass


# === 配置 ===
BASE_URL = os.environ.get("BASE_URL", "https://api.deepseek.com")
# 模型选择：可修改为其他已部署模型或通过环境变量 MODEL_NAME 覆盖
MODEL = os.environ.get("MODEL_NAME", "deepseek-chat")
# 并发线程数（用于润色并发），可通过环境变量 MAX_WORKERS 覆盖
_base_workers = int(os.environ.get("MAX_WORKERS", "4"))
MAX_WORKERS = _base_workers + 4

# API Key 支持池化：优先 API_KEY_POOL（逗号分隔），否则回退 API_KEY
API_KEY_POOL = [
    k.strip()
    for k in os.environ.get("API_KEY_POOL", os.environ.get("API_KEY", "")).split(",")
    if k.strip()
]
API_KEY = API_KEY_POOL[0] if API_KEY_POOL else ""
BASE_DIR = get_base_dir()
RESULTS_DIR = os.environ.get("RESULTS_DIR") or os.path.join(BASE_DIR, "results")
REPORT_CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "report_checkpoint.json")
REPORT_RUN_LOG_PATH = os.path.join(RESULTS_DIR, "report_generation.log")
_REPORT_LOGGER = None


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
    logger=None,  # report.py 里原本用 print 输出
)


def find_latest(pattern: str, base_dir: str = RESULTS_DIR):
    paths = glob.glob(os.path.join(base_dir, "**", pattern), recursive=True)
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def find_detailed_json(book_key: str, base_dir: str = RESULTS_DIR, detail_path: str = None):
    """
    根据书名 key 查找对应的 *_detailed_*.json 文件
    优先匹配同名书籍的文件
    """
    if detail_path:
        return detail_path
    if not book_key:
        return find_latest("*_detailed_*.json", base_dir)
    
    # 尝试精确匹配书名
    pattern = os.path.join(base_dir, "**", f"{book_key}*_detailed_*.json")
    paths = glob.glob(pattern, recursive=True)
    if paths:
        return max(paths, key=os.path.getmtime)
    
    # 降级：查找任意 detailed 文件
    return find_latest("*_detailed_*.json", base_dir)


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    return json.loads(read_file_safely(path))


def _safe_mtime(path: str):
    try:
        if path and os.path.exists(path):
            return os.path.getmtime(path)
    except Exception:
        pass
    return None


def load_report_checkpoint(path: str = REPORT_CHECKPOINT_FILE) -> dict:
    data = load_json(path)
    if not isinstance(data, dict):
        data = {}
    jobs = data.get("jobs")
    if not isinstance(jobs, dict):
        data["jobs"] = {}
    return data


def save_report_checkpoint(data: dict, path: str = REPORT_CHECKPOINT_FILE):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_report(f"\u5199\u5165\u62a5\u544a\u68c0\u67e5\u70b9\u5931\u8d25: {e}")


def extract_book_key_from_path(path: str) -> str:
    """
    从文件名推断书名 key:
    例: 《书名》_protagonists_20251213_120000.json -> 《书名》
    例: 《书名》_detailed_20251213_120000.json -> 《书名》
    """
    if not path:
        return ""
    base = os.path.basename(path)
    for marker in ["_protagonists_", "_detailed_", "_scan_", "_heroine_"]:
        if marker in base:
            return base.split(marker, 1)[0]
    return os.path.splitext(base)[0]


def infer_book_key_from_results_dir(path: str) -> str:
    """
    当 reviewer 文件名是 VERIFIED_SUMMARY_*.json 这种“无书名”的情况时，
    尝试从其所在 results 子目录名推断书名：
      例：.../《书名》_scan_2025xxxx/VERIFIED_SUMMARY_xxx.json -> 《书名》
    """
    if not path:
        return ""
    try:
        parent = os.path.basename(os.path.dirname(path))
        for marker in ["_scan_", "_heroine_", "_protagonist_", "_review_", "_verify_"]:
            if marker in parent:
                return parent.split(marker, 1)[0]
        # 兜底：如果目录名本身就以《...》开头，取到第一个》为止
        if "《" in parent and "》" in parent:
            l = parent.find("《")
            r = parent.find("》", l + 1)
            if l != -1 and r != -1 and r > l:
                return parent[l : r + 1]
        return ""
    except Exception:
        return ""


def sanitize_book_key(key: str) -> str:
    """将书名 key 转为文件友好格式"""
    if not key:
        return ""
    return re.sub(r"[^\w\-\.\u4e00-\u9fa5]+", "_", key)


def format_book_title_for_filename(book_key: str) -> str:
    """
    生成用于文件名展示的“书名标题”：
    - 如果 book_key 已含《...》前缀（常见：`《书名》完结+番外`），避免再套一层《》
    - 否则使用《书名》包裹
    """
    if not book_key:
        return ""
    s = str(book_key).strip()
    if s.startswith("《") and "》" in s:
        return s
    return f"《{s}》"


def summarize_appearance(name: str, features: list) -> str:
    """调用大模型总结女主外貌特点"""
    if not features:
        return "未描述"
    if not OpenAI or not API_KEY:
        # 无法调用大模型，直接拼接返回
        return '；'.join(features[:3])
    
    features_text = '\n'.join([f"- {f}" for f in features])
    system_prompt = "你是一个专业的小说角色外貌总结助手。请根据提供的外貌描写片段，总结出简洁的外貌特点（30字以内），只保留关键的外貌特征如：容貌、身材、发色、穿着等。不要添加性格描写。"
    user_prompt = f"角色：{name}\n外貌描写片段：\n{features_text}\n\n请用30字以内总结外貌特点："
    
    try:
        resp = chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=100,
        )
        record_usage(resp)
        result = resp.choices[0].message.content.strip()
        return result if result else '；'.join(features[:2])
    except Exception as e:
        print(f"外貌总结失败({name}): {e}")
        return '；'.join(features[:2])


def build_features_map(detailed_data) -> dict:
    """从 all_female_characters 构建 name -> features 映射"""
    features_map = {}
    if not detailed_data:
        return features_map
    
    all_chars = detailed_data.get("all_female_characters", {})
    for name, info in all_chars.items():
        features = info.get("features", [])
        if features:
            features_map[name] = features
        # 也保存 other_names 用于模糊匹配
        other_names = info.get("other_names", [])
        for alias in other_names:
            if alias and alias not in features_map:
                features_map[alias] = features
    
    return features_map


def build_report_sections(detailed_data, reviewer, features_map=None):
    """
    整合 detailed_data (*_detailed_*.json) 与 reviewer (VERIFIED_SUMMARY_*.json)
    新格式：女主名字 | 肉体洁 | 精神洁 | 是否被推倒 | 女主特点（身份、外貌、与男主关系）
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [f"精简报告 生成时间：{ts}", "=" * 60]

    # 从 reviewer 构建：肉体/精神状态 + 推倒判定（支持四维度洁度判定 + 验证信息）
    purity_map = {}  # name -> {is_virgin, virgin_status, has_other_contact, contact_status, no_partner, partner_status, verification, ...}
    if reviewer:
        for item in reviewer.get("heroines_purity", []):
            name = item.get("name")
            if name:
                verification = item.get("verification", {})
                purity_map[name] = {
                    # 四维度洁度判定
                    "is_virgin": item.get("is_virgin", True),
                    "virgin_status": item.get("virgin_status", item.get("body_status", "❓ 未知")),
                    "has_other_contact": item.get("has_other_contact", False),
                    "contact_status": item.get("contact_status", "❓ 未知"),
                    "no_partner": item.get("no_partner", True),
                    "partner_status": item.get("partner_status", "❓ 未知"),
                    # 精神洁度
                    "is_spirit_clean": item.get("is_spirit_clean", True),
                    "spirit_status": item.get("spirit_status", "❓ 未知"),
                    # 综合
                    "is_clean": item.get("is_clean", True),
                    "pushed": item.get("pushed_by_male_lead"),
                    "pushed_reason": item.get("pushed_reason", "未判定"),
                    "summary": item.get("summary", ""),
                    # 兼容旧字段
                    "body_status": item.get("body_status", "❓ 未知"),
                    # 验证信息
                    "verification": {
                        "method": verification.get("method", "unknown"),
                        "llm_agreed": verification.get("llm_agreed", True),
                        "rounds": verification.get("rounds", 0),
                        "first_disagreement": verification.get("first_disagreement", ""),
                        "second_reason": verification.get("second_reason", ""),
                        "program_result": verification.get("program_result", {}),
                        "llm_first_result": verification.get("llm_first_result", {}),
                    },
                }

    # 男主信息
    male_lines = []
    male_lines.append("\n【男主信息】")
    mp = detailed_data.get("male_protagonist") if detailed_data else None
    if mp:
        male_lines.append(f"- 男主：{mp.get('name') or '未识别'}")
        other_names = mp.get("other_names", [])
        if other_names:
            male_lines.append(f"  别名：{', '.join(other_names[:10])}")  # 限制别名数量
        if mp.get("identity"):
            male_lines.append(f"  身份：{mp.get('identity')}")
        summaries = mp.get("summaries", [])
        if summaries:
            male_lines.append("  主要剧情：")
            for s in summaries[:3]:
                male_lines.append(f"    · {s}")

    # 女主信息：整合 detailed + reviewer
    heroine_blocks = []
    heroines = []
    if detailed_data:
        heroines = detailed_data.get("heroine_result", {}).get("heroines", [])
    
    # 构建外貌特征映射（如果未提供）
    if features_map is None:
        features_map = build_features_map(detailed_data)
    
    if heroines:
        for h in heroines:
            lines = []
            name = h.get('name') or '未知'
            rank = h.get('importance_rank', '?')
            aliases_list = h.get('aliases', [])
            aliases = ', '.join(aliases_list[:5]) if aliases_list else '无'
            rel = h.get('relationship_type') or '未知'
            traits = h.get('character_traits') or ''
            summary = h.get('summary') or ''
            
            # 从 features_map 获取外貌描写（支持别名匹配）
            features = features_map.get(name, [])
            if not features:
                for alias in aliases_list:
                    if alias in features_map:
                        features = features_map[alias]
                        break
            
            # 从 summary 中提取身份（通常是第一句）
            identity = ''
            if summary:
                first_sentence = summary.split('。')[0].split('，')[0].strip()
                if first_sentence and len(first_sentence) < 50:
                    identity = first_sentence
            
            # 性格直接使用 traits
            personality = traits or '未描述'
            
            # 从 purity_map 获取初/处和推倒信息（支持模糊匹配）
            purity_info = purity_map.get(name)
            if not purity_info:
                # 尝试模糊匹配
                for pname, pinfo in purity_map.items():
                    if name in pname or pname in name:
                        purity_info = pinfo
                        break
            
            if purity_info:
                # 四维度洁度判定
                virgin_status = purity_info.get("virgin_status", purity_info.get("body_status", "❓ 未知"))
                contact_status = purity_info.get("contact_status", "❓ 未知")
                partner_status = purity_info.get("partner_status", "❓ 未知")
                spirit_status = purity_info.get("spirit_status", "❓ 未知")
                pushed = purity_info.get("pushed")
                pushed_reason = purity_info.get("pushed_reason", "未判定")
                verification = purity_info.get("verification", {})
            else:
                virgin_status = "❓ 未知"
                contact_status = "❓ 未知"
                partner_status = "❓ 未知"
                spirit_status = "❓ 未知"
                pushed = None
                pushed_reason = "未判定"
                verification = {}
            
            # 推倒状态显示
            if pushed is True:
                pushed_flag = "✅ 是"
            elif pushed is False:
                pushed_flag = "❌ 否"
            else:
                pushed_flag = "❓ 未知"

            # 新格式输出（四维度洁度判定）
            lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"【{rank}】{name}")
            lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append(f"  别名：{aliases}")
            lines.append(f"  是否处女：{virgin_status}")
            lines.append(f"  非男主接触：{contact_status}")
            lines.append(f"  有无男伴：{partner_status}")
            lines.append(f"  精神洁：{spirit_status}")
            lines.append(f"  被推倒：{pushed_flag}")
            
            # 显示验证信息
            method = verification.get("method", "")
            if method == "program_with_llm_agree":
                lines.append(f"  验证：✅ LLM确认程序判定")
            elif method == "program_confirmed_after_second_round":
                lines.append(f"  验证：⚠️ 二次校验后确认")
                if verification.get("first_disagreement"):
                    lines.append(f"    初次分歧：{verification.get('first_disagreement', '')[:60]}")
            elif method == "llm_override_after_disagreement":
                lines.append(f"  验证：⚠️ 程序与LLM存在分歧")
                # 显示程序原判定
                prog = verification.get("program_result", {})
                if prog:
                    lines.append(f"  ┌─ 程序判定（被LLM修正）：")
                    lines.append(f"  │  处女：{prog.get('virgin_status', '?')}")
                    lines.append(f"  │  接触：{prog.get('contact_status', '?')}")
                    lines.append(f"  │  男伴：{prog.get('partner_status', '?')}")
                    lines.append(f"  │  精神：{prog.get('spirit_status', '?')}")
                    lines.append(f"  └─────────────────")
                if verification.get("first_disagreement"):
                    lines.append(f"    分歧原因：{verification.get('first_disagreement', '')[:80]}")
                if verification.get("second_reason"):
                    lines.append(f"    最终理由：{verification.get('second_reason', '')[:80]}")
            elif method == "no_facts_default_clean":
                lines.append(f"  验证：ℹ️ 无事实记录，默认全初")
            
            lines.append(f"  ────────────────")
            if identity:
                lines.append(f"  身份：{identity}")
            # 外貌：使用 features 并标记待总结
            lines.append(f"  外貌：__APPEARANCE__{name}__")
            lines.append(f"  性格：{personality}")
            lines.append(f"  与男主关系：{rel}")
            if summary:
                lines.append(f"  概要：{summary}")
            if pushed_reason and pushed_reason != "未判定":
                lines.append(f"  推倒说明：{pushed_reason}")
            
            heroine_blocks.append("\n".join(lines))
    
    # 补充：reviewer 中有但 detailed 中没有的女主
    detailed_names = {h.get('name') for h in heroines if h.get('name')}
    for pname, pinfo in purity_map.items():
        if pname not in detailed_names:
            # 检查是否已通过模糊匹配处理
            already_matched = any(pname in dn or dn in pname for dn in detailed_names)
            if not already_matched:
                lines = []
                lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                lines.append(f"【?】{pname}")
                lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                # 四维度洁度判定
                lines.append(f"  是否处女：{pinfo.get('virgin_status', pinfo.get('body_status', '❓ 未知'))}")
                lines.append(f"  非男主接触：{pinfo.get('contact_status', '❓ 未知')}")
                lines.append(f"  有无男伴：{pinfo.get('partner_status', '❓ 未知')}")
                lines.append(f"  精神洁：{pinfo.get('spirit_status', '❓ 未知')}")
                pushed = pinfo.get("pushed")
                if pushed is True:
                    pushed_flag = "✅ 是"
                elif pushed is False:
                    pushed_flag = "❌ 否"
                else:
                    pushed_flag = "❓ 未知"
                lines.append(f"  被推倒：{pushed_flag}")
                
                # 显示验证信息
                verification = pinfo.get("verification", {})
                method = verification.get("method", "")
                if method == "program_with_llm_agree":
                    lines.append(f"  验证：✅ LLM确认程序判定")
                elif method == "program_confirmed_after_second_round":
                    lines.append(f"  验证：⚠️ 二次校验后确认")
                elif method == "llm_override_after_disagreement":
                    lines.append(f"  验证：⚠️ 程序与LLM存在分歧")
                    prog = verification.get("program_result", {})
                    if prog:
                        lines.append(f"  ┌─ 程序判定（被LLM修正）：")
                        lines.append(f"  │  处女：{prog.get('virgin_status', '?')}")
                        lines.append(f"  │  接触：{prog.get('contact_status', '?')}")
                        lines.append(f"  │  男伴：{prog.get('partner_status', '?')}")
                        lines.append(f"  │  精神：{prog.get('spirit_status', '?')}")
                        lines.append(f"  └─────────────────")
                    if verification.get("first_disagreement"):
                        lines.append(f"    分歧原因：{verification.get('first_disagreement', '')[:80]}")
                elif method == "no_facts_default_clean":
                    lines.append(f"  验证：ℹ️ 无事实记录，默认全初")
                
                lines.append(f"  ────────────────")
                lines.append(f"  （详细信息未找到）")
                heroine_blocks.append("\n".join(lines))

    # 雷/郁闷原文，保持未经润色
    risk = []
    if reviewer:
        lei_points = reviewer.get("lei_points", [])
        yumen_points = reviewer.get("yumen_points", [])
        if lei_points:
            risk.append("【严重雷点】")
            for i, p in enumerate(lei_points, 1):
                risk.append(f"{i}. [{p.get('type','')}] @chunk {p.get('chunk_index')}")
                risk.append(f"   原文：{p.get('content','')}")
                if p.get("review_comment"):
                    risk.append(f"   裁决：{p.get('review_comment')}")
            risk.append("")
        if yumen_points:
            risk.append("【郁闷点】")
            for i, p in enumerate(yumen_points, 1):
                risk.append(f"{i}. [{p.get('type','')}] @chunk {p.get('chunk_index')}")
                risk.append(f"   原文：{p.get('content','')}")
                if p.get("review_comment"):
                    risk.append(f"   裁决：{p.get('review_comment')}")

    return header, male_lines, heroine_blocks, ("\n".join(risk) if risk else "")


# =========================== 新版报告（男主→女主→毒/雷点） ===========================
def _bool_mark(v):
    """把 True/False/None 映射成 ✅/❌/❓"""
    if v is True:
        return "✅"
    if v is False:
        return "❌"
    return "❓"


def build_purity_map(reviewer: dict) -> dict:
    """
    从 novel_reviewer.py 的 VERIFIED_SUMMARY_*.json 构建洁度映射：
      name -> {is_virgin, is_spirit_clean, no_partner, has_other_contact, ...}
    """
    m = {}
    if not reviewer:
        return m
    for item in reviewer.get("heroines_purity", []) or []:
        name = item.get("name")
        if not name:
            continue
        m[name] = {
            "is_virgin": item.get("is_virgin"),
            "is_spirit_clean": item.get("is_spirit_clean"),
            "no_partner": item.get("no_partner"),
            "has_other_contact": item.get("has_other_contact"),
            # status fields
            "virgin_status": item.get("virgin_status", ""),
            "spirit_status": item.get("spirit_status", ""),
            "partner_status": item.get("partner_status", ""),
            "contact_status": item.get("contact_status", ""),
            # conflict detail (LLM vs rule)
            "virgin_judgement_conflict": bool(item.get("virgin_judgement_conflict", False)),
            "llm_virgin_status": item.get("llm_virgin_status", ""),
            "llm_virgin_reason": item.get("llm_virgin_reason", ""),
            "rule_virgin_status": item.get("rule_virgin_status", ""),
            "rule_virgin_reason": item.get("rule_virgin_reason", ""),
            "contact_judgement_conflict": bool(item.get("contact_judgement_conflict", False)),
            "llm_contact_status": item.get("llm_contact_status", ""),
            "llm_contact_reason": item.get("llm_contact_reason", ""),
            "rule_contact_status": item.get("rule_contact_status", ""),
            "rule_contact_reason": item.get("rule_contact_reason", ""),
            "partner_judgement_conflict": bool(item.get("partner_judgement_conflict", False)),
            "llm_partner_status": item.get("llm_partner_status", ""),
            "llm_partner_reason": item.get("llm_partner_reason", ""),
            "rule_partner_status": item.get("rule_partner_status", ""),
            "rule_partner_reason": item.get("rule_partner_reason", ""),
            "spirit_judgement_conflict": bool(item.get("spirit_judgement_conflict", False)),
            "llm_spirit_status": item.get("llm_spirit_status", ""),
            "llm_spirit_reason": item.get("llm_spirit_reason", ""),
            "rule_spirit_status": item.get("rule_spirit_status", ""),
            "rule_spirit_reason": item.get("rule_spirit_reason", ""),
            "pushed_by_male_lead": item.get("pushed_by_male_lead"),
            "pushed_reason": item.get("pushed_reason", ""),
            # original 4-dim summary
            "is_leak_heroine": item.get("is_leak_heroine"),
            "leak_reason": item.get("leak_reason", ""),
            "summary": item.get("summary", ""),
        }
    return m


def summarize_male_profile_llm(male_obj: dict, model: str = None) -> dict:
    """
    输入：*_detailed_*.json 输出中的 male_protagonist 字段
    输出：{identity, personality, experience}
    """
    male_obj = male_obj or {}
    name = male_obj.get("name") or "男主"
    identity = (male_obj.get("identity") or "").strip()
    # 按你的要求：男主不读取 other_names（别称不输出）
    aliases = male_obj.get("aliases") or []
    summaries = male_obj.get("summaries") or []

    # 无法调用大模型时：尽量用原始信息拼
    if not OpenAI or not API_KEY:
        exp = "；".join([s.strip() for s in summaries[:3] if s and str(s).strip()])
        return {
            "identity": identity or "未描述",
            "personality": "未描述",
            "experience": exp or "未描述",
            "aliases": aliases,
        }

    system_prompt = (
        "你是小说人物信息总结助手。请基于提供的原始线索，提炼男主的：身份、性格特点、经历概括。"
        "要求：不杜撰，不加入原文没有的事实；输出简洁；经历用1-3句概括。"
        "请只输出 JSON。"
    )
    user_prompt = json.dumps(
        {
            "name": name,
            "identity": identity,
            # 你要求尽量读全：这里保留到 100
            "aliases": aliases[:100],
            "summaries": summaries[:100],
        },
        ensure_ascii=False,
        indent=2,
    )
    try:
        resp = chat_completion(
            model=model or MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        record_usage(resp)
        data = json.loads(resp.choices[0].message.content.strip())
        return {
            "identity": (data.get("identity") or identity or "未描述").strip(),
            "personality": (data.get("personality") or "未描述").strip(),
            "experience": (data.get("experience") or "未描述").strip(),
            "aliases": aliases,
        }
    except Exception:
        exp = "；".join([str(s).strip() for s in summaries[:3] if s and str(s).strip()])
        return {
            "identity": identity or "未描述",
            "personality": "未描述",
            "experience": exp or "未描述",
            "aliases": aliases,
        }


def _pick_first_str(items, default=""):
    for x in items or []:
        s = (str(x) if x is not None else "").strip()
        if s:
            return s
    return default


def _match_female_evidence(name: str, aliases: list, all_female_characters: dict):
    """
    在 detailed_data["all_female_characters"] 里为某个女主找证据条目。
    """
    if not name or not all_female_characters:
        return None
    if name in all_female_characters:
        return all_female_characters.get(name)
    for a in aliases or []:
        if a in all_female_characters:
            return all_female_characters.get(a)
    # other_names 反向匹配
    for _k, v in (all_female_characters or {}).items():
        others = (v or {}).get("other_names", []) or []
        if name in others:
            return v
        for a in aliases or []:
            if a in others:
                return v
    # 子串兜底（至少 3 字，避免误配）
    if len(name) >= 3:
        for k, v in (all_female_characters or {}).items():
            if k and (name in k or k in name):
                return v
    return None


def _normalize_profile_for_report(profile_for_report: dict) -> dict:
    profile_for_report = profile_for_report or {}
    if not isinstance(profile_for_report, dict):
        return {}
    feature_candidates = []
    primary_features = (str(profile_for_report.get("features") or "")).strip()
    if primary_features:
        feature_candidates.append(primary_features)
    else:
        for key in ("appearance", "personality", "traits"):
            value = (str(profile_for_report.get(key) or "")).strip()
            if value and value not in feature_candidates:
                feature_candidates.append(value)
    return {
        "identity": (str(profile_for_report.get("identity") or "")).strip(),
        "features": "；".join(feature_candidates),
        "relationship_with_protagonist": (
            str(
                profile_for_report.get("relationship_with_protagonist")
                or profile_for_report.get("relationship")
                or ""
            )
        ).strip(),
        "key_events": (str(profile_for_report.get("key_events") or "")).strip(),
    }


def summarize_heroine_profile_llm(
    name: str,
    heroine_meta: dict,
    heroine_evidence: dict,
    model: str = None,
) -> dict:
    """
    输入：
    - heroine_meta：*_detailed_*.json 的 heroine_result.heroines[] 单条（关系/性格/概要/别名）
    - heroine_evidence：*_detailed_*.json 的 all_female_characters[name]（features/summaries/interactions 等证据）
    输出：{relationship, appearance, traits}
    """
    heroine_meta = heroine_meta or {}
    heroine_evidence = heroine_evidence or {}

    rel_type = heroine_meta.get("relationship_type") or ""
    traits = heroine_meta.get("character_traits") or ""
    summary = heroine_meta.get("summary") or ""
    aliases = heroine_meta.get("aliases") or heroine_meta.get("other_names") or []

    relationships = heroine_evidence.get("relationships") or []
    features = heroine_evidence.get("features") or []
    interactions = heroine_evidence.get("interactions") or []
    emotions = heroine_evidence.get("emotion_signals") or []
    summaries = heroine_evidence.get("summaries") or []

    # 无法调用大模型时：尽量回退
    if not OpenAI or not API_KEY:
        return {
            "relationship": rel_type or _pick_first_str(relationships, "未描述") or "未描述",
            "appearance": "；".join([f for f in features[:3] if f]) or "未描述",
            "traits": traits or (summary.split("。")[0] if summary else "未描述"),
        }

    system_prompt = (
        "你是小说女主信息总结助手。请基于提供的线索，总结：与男主关系、外貌描写、特点（身份/性格/标签）。"
        "要求：不杜撅，不加入不存在的事实；外貌只写外貌；特点不超过40字；关系尽量短。只输出 JSON。"
    )
    user_payload = {
        "name": name,
        "aliases": aliases[:30],
        "relationship_type": rel_type,
        "relationships": relationships[:30],
        "character_traits_raw": traits,
        "summary_raw": summary,
        "features_raw": features[:80],
        "key_interactions_raw": interactions[:120],
        "emotion_signals_raw": emotions[:60],
        "summaries_raw": summaries[:120],
    }
    try:
        resp = chat_completion(
            model=model or MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        record_usage(resp)
        data = json.loads(resp.choices[0].message.content.strip())
        return {
            "relationship": (data.get("relationship") or rel_type or "未描述").strip(),
            "appearance": (data.get("appearance") or "未描述").strip(),
            "traits": (data.get("traits") or traits or "未描述").strip(),
        }
    except Exception:
        return {
            "relationship": rel_type or _pick_first_str(relationships, "未描述") or "未描述",
            "appearance": "；".join([f for f in features[:3] if f]) or "未描述",
            "traits": traits or (summary.split("。")[0] if summary else "未描述"),
        }


def build_report_v2(book_key: str, detailed_data: dict, reviewer: dict) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [f"书名：{book_key or '未识别'}", f"报告生成时间：{ts}", "=" * 60]

    # ---------------- 男主 ----------------
    male_obj = (detailed_data or {}).get("male_protagonist") or {}
    male_name = male_obj.get("name") or (reviewer or {}).get("male_lead") or "未识别"
    male_summary = summarize_male_profile_llm(male_obj)
    male_lines = [
        "【男主】",
        f"姓名：{male_name}",
        f"身份：{male_summary.get('identity', '未描述')}",
        f"性格：{male_summary.get('personality', '未描述')}",
        f"经历：{male_summary.get('experience', '未描述')}",
    ]
    # 按你的要求：男主不输出别称（不读取 other_names）

    # ---------------- 女主 ----------------
    purity_map = build_purity_map(reviewer)
    heroine_result = (detailed_data or {}).get("heroine_result") or {}
    heroines = heroine_result.get("heroines") or []
    all_female_characters = (detailed_data or {}).get("all_female_characters") or {}
    # 按 importance_rank 排序（没有就放最后）
    def _rank_key(h):
        try:
            return int(h.get("importance_rank", 9999))
        except Exception:
            return 9999
    heroines = sorted(heroines, key=_rank_key)

    heroine_lines = ["", "【女主】"]

    # 并发总结（避免女主多时太慢）
    profile_cache = {}
    if heroines:
        def _work(h):
            n = h.get("name") or ""
            if not n:
                return (
                    "",
                    {
                        "identity": "未描述",
                        "features": "未描述",
                        "relationship_with_protagonist": "未描述",
                        "key_events": "未描述",
                    },
                )
            aliases = h.get("aliases") or h.get("other_names") or []
            evid = _match_female_evidence(n, aliases, all_female_characters) or {}
            profile_for_report = _normalize_profile_for_report(evid.get("profile_for_report"))
            if any(profile_for_report.values()):
                return (n, profile_for_report)
            return (n, summarize_heroine_profile_llm(n, h, evid))

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for n, prof in tqdm(ex.map(_work, heroines), total=len(heroines), desc="女主总结"):
                if n:
                    profile_cache[n] = prof

    for h in heroines:
        name = h.get("name") or "未知"
        prof = _normalize_profile_for_report(profile_cache.get(name, {}))
        p = purity_map.get(name, None)
        # 初摸：无非男主接触 => ✅
        virgin = _bool_mark(p.get("is_virgin")) if p else "❓"
        spirit = _bool_mark(p.get("is_spirit_clean")) if p else "❓"
        first_marriage = _bool_mark(p.get("no_partner")) if p else "❓"
        first_touch = _bool_mark((not p.get("has_other_contact")) if p and p.get("has_other_contact") is not None else None) if p else "\u2753"
        purity_summary = (p.get("summary") or "").strip() if p else ""
        pushed = _bool_mark(p.get("pushed_by_male_lead")) if p else "❓"
        pushed_reason = (p.get("pushed_reason") or "").strip() if p else ""
        leak = _bool_mark(p.get("is_leak_heroine")) if p else "❓"
        leak_reason = (p.get("leak_reason") or "").strip() if p else ""
        virgin_conflict = bool(p.get("virgin_judgement_conflict", False)) if p else False
        llm_virgin_status = (p.get("llm_virgin_status") or p.get("virgin_status") or "\u672a\u77e5") if p else "\u672a\u77e5"
        llm_virgin_reason = (p.get("llm_virgin_reason") or "").strip() if p else ""
        rule_virgin_status = (p.get("rule_virgin_status") or "\u672a\u77e5") if p else "\u672a\u77e5"
        rule_virgin_reason = (p.get("rule_virgin_reason") or "").strip() if p else ""
        contact_conflict = bool(p.get("contact_judgement_conflict", False)) if p else False
        llm_contact_status = (p.get("llm_contact_status") or p.get("contact_status") or "\u672a\u77e5") if p else "\u672a\u77e5"
        llm_contact_reason = (p.get("llm_contact_reason") or "").strip() if p else ""
        rule_contact_status = (p.get("rule_contact_status") or "\u672a\u77e5") if p else "\u672a\u77e5"
        rule_contact_reason = (p.get("rule_contact_reason") or "").strip() if p else ""
        partner_conflict = bool(p.get("partner_judgement_conflict", False)) if p else False
        llm_partner_status = (p.get("llm_partner_status") or p.get("partner_status") or "\u672a\u77e5") if p else "\u672a\u77e5"
        llm_partner_reason = (p.get("llm_partner_reason") or "").strip() if p else ""
        rule_partner_status = (p.get("rule_partner_status") or "\u672a\u77e5") if p else "\u672a\u77e5"
        rule_partner_reason = (p.get("rule_partner_reason") or "").strip() if p else ""
        spirit_conflict = bool(p.get("spirit_judgement_conflict", False)) if p else False
        llm_spirit_status = (p.get("llm_spirit_status") or p.get("spirit_status") or "\u672a\u77e5") if p else "\u672a\u77e5"
        llm_spirit_reason = (p.get("llm_spirit_reason") or "").strip() if p else ""
        rule_spirit_status = (p.get("rule_spirit_status") or "\u672a\u77e5") if p else "\u672a\u77e5"
        rule_spirit_reason = (p.get("rule_spirit_reason") or "").strip() if p else ""
        missing_desc = "\u672a\u63cf\u8ff0"
        empty_text = "\uff08\u65e0\uff09"

        block = [
            "",
            f"{name}:",
            f"\u5904\u5973\uff1a{virgin}",
            f"\u7cbe\u795e\u521d\uff1a{spirit}",
            f"\u521d\u5a5a\uff1a{first_marriage}",
            f"\u521d\u6478\uff1a{first_touch}",
            f"\u8eab\u4efd\uff1a{prof.get('identity') or missing_desc}",
            f"\u4e0e\u7537\u4e3b\u5173\u7cfb\uff1a{prof.get('relationship_with_protagonist') or missing_desc}",
            f"\u7279\u70b9\uff1a{prof.get('features') or missing_desc}",
            f"\u5173\u952e\u4e8b\u4ef6\uff1a{prof.get('key_events') or missing_desc}",
            f"\u56db\u7ef4\u7eaf\u6d01\u5ea6summary\uff1a{purity_summary or empty_text}",
            f"是否被推倒：{pushed}",
            f"推倒说明：{pushed_reason or empty_text}",
            f"是否漏女：{leak}",
            f"漏女说明：{leak_reason or empty_text}",
        ]

        if virgin_conflict:
            block.extend([
                "\u5904\u5973\u5224\u5b9a\u51b2\u7a81\uff1a\u26a0\ufe0f \u89c4\u5219\u4e0e\u5927\u6a21\u578b\u4e0d\u4e00\u81f4",
                f"  \u5927\u6a21\u578b\u5224\u65ad\uff1a{llm_virgin_status}",
                f"  \u5927\u6a21\u578b\u7406\u7531\uff1a{llm_virgin_reason or empty_text}",
                f"  \u89c4\u5219\u5224\u65ad\uff1a{rule_virgin_status}",
                f"  \u89c4\u5219\u7406\u7531\uff1a{rule_virgin_reason or empty_text}",
            ])
        if contact_conflict:
            block.extend([
                "\u63a5\u89e6\u5224\u5b9a\u51b2\u7a81\uff1a\u26a0\ufe0f \u89c4\u5219\u4e0e\u5927\u6a21\u578b\u4e0d\u4e00\u81f4",
                f"  \u5927\u6a21\u578b\u5224\u65ad\uff1a{llm_contact_status}",
                f"  \u5927\u6a21\u578b\u7406\u7531\uff1a{llm_contact_reason or empty_text}",
                f"  \u89c4\u5219\u5224\u65ad\uff1a{rule_contact_status}",
                f"  \u89c4\u5219\u7406\u7531\uff1a{rule_contact_reason or empty_text}",
            ])
        if partner_conflict:
            block.extend([
                "\u7537\u4f34\u5224\u5b9a\u51b2\u7a81\uff1a\u26a0\ufe0f \u89c4\u5219\u4e0e\u5927\u6a21\u578b\u4e0d\u4e00\u81f4",
                f"  \u5927\u6a21\u578b\u5224\u65ad\uff1a{llm_partner_status}",
                f"  \u5927\u6a21\u578b\u7406\u7531\uff1a{llm_partner_reason or empty_text}",
                f"  \u89c4\u5219\u5224\u65ad\uff1a{rule_partner_status}",
                f"  \u89c4\u5219\u7406\u7531\uff1a{rule_partner_reason or empty_text}",
            ])
        if spirit_conflict:
            block.extend([
                "\u7cbe\u795e\u521d\u5224\u5b9a\u51b2\u7a81\uff1a\u26a0\ufe0f \u89c4\u5219\u4e0e\u5927\u6a21\u578b\u4e0d\u4e00\u81f4",
                f"  \u5927\u6a21\u578b\u5224\u65ad\uff1a{llm_spirit_status}",
                f"  \u5927\u6a21\u578b\u7406\u7531\uff1a{llm_spirit_reason or empty_text}",
                f"  \u89c4\u5219\u5224\u65ad\uff1a{rule_spirit_status}",
                f"  \u89c4\u5219\u7406\u7531\uff1a{rule_spirit_reason or empty_text}",
            ])

        heroine_lines.extend(block)

    # ---------------- 毒点/雷点（原样输出，不润色） ----------------
    risk_lines = ["", "【毒点】"]
    yumen = (reviewer or {}).get("yumen_points") or []
    if yumen:
        for i, p in enumerate(yumen, 1):
            risk_lines.append(f"{i}. [{p.get('type','')}] @chunk {p.get('chunk_index')}")
            risk_lines.append(f"   原文：{p.get('content','')}")
            if p.get("review_comment"):
                risk_lines.append(f"   裁决：{p.get('review_comment')}")
    else:
        risk_lines.append("（无）")

    risk_lines.extend(["", "【雷点】"])
    lei = (reviewer or {}).get("lei_points") or []
    if lei:
        for i, p in enumerate(lei, 1):
            risk_lines.append(f"{i}. [{p.get('type','')}] @chunk {p.get('chunk_index')}")
            risk_lines.append(f"   原文：{p.get('content','')}")
            if p.get("review_comment"):
                risk_lines.append(f"   裁决：{p.get('review_comment')}")
    else:
        risk_lines.append("（无）")

    return "\n".join([*header, "", *male_lines, *heroine_lines, *risk_lines])


def polish_text(text: str, model: str = None):
    if not OpenAI:
        print("未安装 openai，跳过润色。")
        return text
    if not API_KEY:
        print("未设置 API_KEY，跳过润色。")
        return text
    model = model or MODEL
    system_prompt = "你是专业的中文编辑，请在不改变事实的前提下，提升流畅度与可读性，保持格式和段落结构。"
    user_prompt = f"请润色以下报告：\n{text}"
    try:
        resp = chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        record_usage(resp)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"润色失败，返回原文：{e}")
        return text


def main(novel_path=None, book_name=None, run_id=None, detail_path=None):
    global _REPORT_LOGGER
    _REPORT_LOGGER = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--polish", action="store_true", help="调用大模型润色输出（已默认开启）")
    parser.add_argument("--no-polish", action="store_true", help="禁用润色，直接输出原文")
    parser.add_argument("--skip-existing", action="store_true", help="若已存在同书名报告则跳过生成（旧行为）")
    parser.add_argument("--force-regenerate", action="store_true", help="\u5ffd\u7565\u68c0\u67e5\u70b9\uff0c\u5f3a\u5236\u91cd\u65b0\u751f\u6210\u62a5\u544a")
    parse_cli_args = novel_path is None and book_name is None and run_id is None
    args = parser.parse_args() if parse_cli_args else parser.parse_args([])

    if novel_path:
        os.environ["NOVEL_PATH"] = novel_path
    log_report("=" * 80)
    log_report(f"[START] report generation @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_report(
        f"args: force_regenerate={args.force_regenerate}, "
        f"skip_existing={args.skip_existing}, polish={args.polish}, no_polish={args.no_polish}"
    )

    # 每次调用重新读取 NOVEL_PATH（支持多本小说循环）
    novel_path_env = os.environ.get("NOVEL_PATH")
    env_book_key = ""
    if novel_path_env:
        env_book_key = os.path.splitext(os.path.basename(novel_path_env))[0].strip()

    # 查找数据文件
    reviewer_path = find_latest("VERIFIED_SUMMARY_*.json")
    reviewer = load_json(reviewer_path)
    
    # 先从环境变量/文件名尝试推断书名；若 reviewer 是 VERIFIED_SUMMARY_*.json，则需要进一步纠正
    book_key = env_book_key or extract_book_key_from_path(reviewer_path)
    detailed_path = None
    if book_key:
        detailed_path = find_detailed_json(book_key, detail_path=detail_path)
    elif detail_path:
        detailed_path = detail_path
    if not detailed_path:
        detailed_path = find_latest("*_detailed_*.json")
    
    detailed_data = load_json(detailed_path)

    # 纠正“无书名 reviewer 文件名”的场景：优先用 detailed/protagonists 文件名或 reviewer 目录名反推
    def _looks_like_non_book_key(k: str) -> bool:
        if not k:
            return True
        return k.startswith("VERIFIED_") or k.startswith("AGGREGATED_") or k.startswith("VERIFIED_SUMMARY_")

    if _looks_like_non_book_key(book_key):
        fixed = (
            extract_book_key_from_path(detailed_path)
            or infer_book_key_from_results_dir(reviewer_path)
        )
        if fixed:
            book_key = fixed

    log_report(f"using detailed: {detailed_path or '<not found>'}")
    log_report(f"using reviewer: {reviewer_path or '<not found>'}")
    log_report(f"book key: {book_key or '<unknown>'}")

    resolved_book_name = (book_name or book_key or "").strip()
    if not resolved_book_name:
        novel_path_for_book = novel_path or os.environ.get("NOVEL_PATH", "")
        if novel_path_for_book:
            resolved_book_name = os.path.splitext(os.path.basename(novel_path_for_book))[0].strip()
    if resolved_book_name:
        init_token_tracker(resolved_book_name, run_id=run_id)

    checkpoint_data = load_report_checkpoint()
    jobs = checkpoint_data.get("jobs", {})
    removed_legacy_keys = 0
    if isinstance(jobs, dict):
        for key in list(jobs.keys()):
            if "||" in str(key):
                jobs.pop(key, None)
                removed_legacy_keys += 1
    else:
        jobs = {}
        checkpoint_data["jobs"] = jobs
    if removed_legacy_keys:
        log_report(f"cleaned {removed_legacy_keys} legacy checkpoint keys")
        save_report_checkpoint(checkpoint_data)

    checkpoint_hit = jobs.get(book_key, {}) if isinstance(jobs, dict) else {}
    if not isinstance(checkpoint_hit, dict):
        checkpoint_hit = {}
    checkpoint_status = checkpoint_hit.get("status")
    checkpoint_out = checkpoint_hit.get("out_file")
    saved_reviewer_mtime = checkpoint_hit.get("reviewer_mtime")
    current_reviewer_mtime = _safe_mtime(reviewer_path)

    if not args.force_regenerate:
        if checkpoint_status == "pending":
            log_report("checkpoint status is pending, forcing regenerate")
        elif (
            checkpoint_status == "completed"
            and checkpoint_out
            and os.path.exists(checkpoint_out)
            and current_reviewer_mtime == saved_reviewer_mtime
        ):
            log_report(f"\u2605 \u68c0\u67e5\u70b9\u547d\u4e2d\uff0c\u62a5\u544a\u5df2\u751f\u6210\uff0c\u8df3\u8fc7\uff1a{checkpoint_out}")
            return
        elif checkpoint_status == "completed" and current_reviewer_mtime != saved_reviewer_mtime:
            log_report("checkpoint reviewer_mtime changed, regenerate report")

    book_key_safe = sanitize_book_key(book_key)

    # 若已存在同书名的聚合报告，则可选跳过（默认不跳过，方便你改格式后重生成）
    if book_key_safe:
        # 兼容旧命名和新命名
        title_for_file = format_book_title_for_filename(book_key)
        title_wrapped = f"《{book_key}》" if book_key else ""
        patterns = [
            os.path.join(RESULTS_DIR, f"AGGREGATED_REPORT_{book_key_safe}_*.txt"),
            # 旧：总是外层包《》
            os.path.join(RESULTS_DIR, f"{title_wrapped}扫书报告*.txt") if title_wrapped else "",
            # 新：如果 book_key 本身已包含《》，则不再外包
            os.path.join(RESULTS_DIR, f"{title_for_file}扫书报告*.txt") if title_for_file else "",
        ]
        existing_reports = []
        for p in patterns:
            if not p:
                continue
            existing_reports.extend(glob.glob(p))
        if existing_reports and args.skip_existing:
            existing_reports = sorted(existing_reports)
            log_report(f"★ 已找到该书的聚合报告，跳过生成：{existing_reports[-1]}")
            return

    # 新版报告：男主→女主→毒/雷点（毒/雷点不润色）
    log_report("building report content...")
    report = build_report_v2(book_key, detailed_data, reviewer)
    log_report(f"report content built, chars={len(report)}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if book_key:
        # 新命名： <书名>扫书报告_时间戳.txt（避免《《书名》...》双层括号）
        title_for_file = format_book_title_for_filename(book_key)
        out_file = os.path.join(RESULTS_DIR, f"{title_for_file}扫书报告_{ts}.txt")
    else:
        out_file = os.path.join(RESULTS_DIR, f"AGGREGATED_REPORT_{ts}.txt")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report)
    log_report(f"report written: {out_file}")

    checkpoint_data.setdefault("jobs", {})[book_key] = {
        "book_key": book_key,
        "status": "completed",
        "reviewer_mtime": _safe_mtime(reviewer_path),
        "out_file": out_file,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_report_checkpoint(checkpoint_data)
    log_report(f"checkpoint updated: {REPORT_CHECKPOINT_FILE}")
    log_report(f"\n已生成：{out_file}")
    if token_tracker is not None:
        snap = token_tracker.snapshot()
        log_report(
            f"Token 统计：输入 {snap.get('input', 0)} ，输出 {snap.get('output', 0)} ，总计 {snap.get('total', 0)}"
        )
        token_tracker.flush(status="finished")
    log_report("[END] report generation finished")


if __name__ == "__main__":
    main()
