"""
main.py - PyInstaller 打包入口，替代 run_all_novels.bat。
功能：读取配置 -> 扫描 novels 目录 -> 依次调用四个业务脚本的 main()。
"""

import json
import multiprocessing
import os
import sys
import time
import uuid


_DEFAULT_ENV_SETTINGS = {
    "BASE_URL": "https://api.deepseek.com",
    "MODEL_NAME": "deepseek-chat",
    "MAX_WORKERS": "6",
    "DIM_BOOST_MAX_PER_CHUNK": "3",
    "RESCAN_ROUNDS": "3",
    "MAX_MIDDLE_SUMMARY_CALLS": "10",
    "RESCAN_MAX_HITS": "4",
    "RESCAN_PRE_FILTER_THRESHOLD": "1.0",
    "RESCAN_MAX_WINDOW": "2000",
    "RESCAN_MAX_PROMPT_HEROINES": "4",
}
_PASSTHROUGH_SETTING_KEYS = {"BASE_URL", "MODEL_NAME", "MAX_WORKERS", "RPM_LIMIT", "TPM_LIMIT"}
_VALIDATED_NON_NEGATIVE_INT_KEYS = {
    "DIM_BOOST_MAX_PER_CHUNK": _DEFAULT_ENV_SETTINGS["DIM_BOOST_MAX_PER_CHUNK"],
    "RESCAN_ROUNDS": _DEFAULT_ENV_SETTINGS["RESCAN_ROUNDS"],
    "MAX_MIDDLE_SUMMARY_CALLS": _DEFAULT_ENV_SETTINGS["MAX_MIDDLE_SUMMARY_CALLS"],
    "RESCAN_MAX_HITS": _DEFAULT_ENV_SETTINGS["RESCAN_MAX_HITS"],
    "RESCAN_MAX_WINDOW": _DEFAULT_ENV_SETTINGS["RESCAN_MAX_WINDOW"],
    "RESCAN_MAX_PROMPT_HEROINES": _DEFAULT_ENV_SETTINGS["RESCAN_MAX_PROMPT_HEROINES"],
}
_VALIDATED_NON_NEGATIVE_FLOAT_KEYS = {
    "RESCAN_PRE_FILTER_THRESHOLD": _DEFAULT_ENV_SETTINGS["RESCAN_PRE_FILTER_THRESHOLD"],
}


def get_base_dir():
    """返回程序根目录：打包后为 exe 所在目录，开发时为脚本所在目录。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _read_file_safely(file_path):
    """安全读取文件：先尝试 UTF-8，失败则回退 GB18030。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gb18030") as f:
            return f.read()


def _set_non_negative_int_env(key, raw_value, default_value):
    try:
        parsed = int(raw_value)
        if parsed < 0:
            raise ValueError("negative value")
    except (TypeError, ValueError):
        print(f"[WARN] setting.txt 中 {key}={raw_value!r} 非法，已回退默认值 {default_value}")
        os.environ[key] = default_value
        return

    os.environ[key] = str(parsed)


def _set_non_negative_float_env(key, raw_value, default_value):
    try:
        parsed = float(raw_value)
        if parsed < 0:
            raise ValueError("negative value")
    except (TypeError, ValueError):
        print(f"[WARN] setting.txt 中 {key}={raw_value!r} 非法，已回退默认值 {default_value}")
        os.environ[key] = default_value
        return

    os.environ[key] = str(parsed)


def load_configs(base_dir):
    """
    读取 setting.txt 和 api.txt，注入到 os.environ。
    对应 bat 中的 setting.txt 解析和 API_KEY_POOL 构建逻辑。
    """
    for key, default_value in _DEFAULT_ENV_SETTINGS.items():
        os.environ.setdefault(key, default_value)

    setting_file = os.path.join(base_dir, "config", "setting.txt")
    if os.path.exists(setting_file):
        text = _read_file_safely(setting_file)
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().upper()
            value = value.strip()
            if key in _PASSTHROUGH_SETTING_KEYS:
                os.environ[key] = value
            elif key in _VALIDATED_NON_NEGATIVE_INT_KEYS:
                _set_non_negative_int_env(key, value, _VALIDATED_NON_NEGATIVE_INT_KEYS[key])
            elif key in _VALIDATED_NON_NEGATIVE_FLOAT_KEYS:
                _set_non_negative_float_env(key, value, _VALIDATED_NON_NEGATIVE_FLOAT_KEYS[key])

    api_file = os.path.join(base_dir, "config", "api.txt")
    if not os.path.exists(api_file):
        print(f"[ERROR] 未找到 {api_file}，请创建并写入可用的 API Key（每行一条）。")
        input("按回车键退出...")
        sys.exit(1)

    text = _read_file_safely(api_file)
    keys = [k.strip() for k in text.splitlines() if k.strip()]
    if not keys:
        print(f"[ERROR] {api_file} 中未读取到任何 key")
        input("按回车键退出...")
        sys.exit(1)

    os.environ["API_KEY_POOL"] = ",".join(keys)
    os.environ["API_KEY"] = keys[0]

    print(
        f"配置已加载：BASE_URL={os.environ['BASE_URL']}  "
        f"MODEL_NAME={os.environ['MODEL_NAME']}  "
        f"MAX_WORKERS={os.environ['MAX_WORKERS']}"
    )
    print(
        f"扫描调优配置：DIM_BOOST_MAX_PER_CHUNK={os.environ['DIM_BOOST_MAX_PER_CHUNK']}  "
        f"RESCAN_ROUNDS={os.environ['RESCAN_ROUNDS']}  "
        f"MAX_MIDDLE_SUMMARY_CALLS={os.environ['MAX_MIDDLE_SUMMARY_CALLS']}"
    )
    print(
        f"全局补扫优化：RESCAN_MAX_HITS={os.environ['RESCAN_MAX_HITS']}  "
        f"RESCAN_PRE_FILTER_THRESHOLD={os.environ['RESCAN_PRE_FILTER_THRESHOLD']}  "
        f"RESCAN_MAX_WINDOW={os.environ['RESCAN_MAX_WINDOW']}  "
        f"RESCAN_MAX_PROMPT_HEROINES={os.environ['RESCAN_MAX_PROMPT_HEROINES']}"
    )
    print(f"API Key 数量: {len(keys)}")
    print()


def scan_novels(base_dir):
    """递归扫描 novels 目录下所有 .txt 文件（匹配 bat 的 for /r 行为）。"""
    novels_dir = os.path.join(base_dir, "novels")
    if not os.path.isdir(novels_dir):
        print(f"[ERROR] 未找到 novels 目录: {novels_dir}")
        input("按回车键退出...")
        sys.exit(1)

    novel_files = []
    for root, _dirs, files in os.walk(novels_dir):
        for name in files:
            if name.lower().endswith(".txt"):
                novel_files.append(os.path.join(root, name))

    if not novel_files:
        print(f"[ERROR] novels 目录下没有 txt 文件: {novels_dir}")
        input("按回车键退出...")
        sys.exit(1)

    return novel_files


def _read_json_safely(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        return json.loads(_read_file_safely(file_path))
    except Exception as exc:
        print(f"[WARN] 读取 JSON 失败: {file_path} ({exc})")
        return None


def _report_is_fresh(base_dir, book_name):
    checkpoint_path = os.path.join(base_dir, "results", "report_checkpoint.json")
    checkpoint_data = _read_json_safely(checkpoint_path)
    if not isinstance(checkpoint_data, dict):
        return False, None

    jobs = checkpoint_data.get("jobs", {})
    if not isinstance(jobs, dict):
        return False, None

    job = jobs.get(book_name)
    if not isinstance(job, dict):
        return False, None
    if job.get("status") != "completed":
        return False, None

    out_file = job.get("out_file")
    if not out_file or not os.path.exists(out_file):
        return False, None

    return True, out_file


def print_pending_novels(novel_files):
    print("待扫描书籍：")
    for index, novel_path in enumerate(novel_files, start=1):
        print(f"  {index}. {os.path.basename(novel_path)}")
    print()


def _generate_run_id():
    return f"Run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"


def _get_working_detail_path(protagonist_module, book_name):
    helper = getattr(protagonist_module, "get_latest_report_files", None)
    if not callable(helper):
        return None
    try:
        report_files = helper(book_name) or {}
    except Exception as exc:
        print(f"[WARN] 读取 protagonist report_files 失败: {exc}")
        return None
    return (report_files or {}).get("detailed")


def print_token_summary(base_dir, token_usage_path=None):
    """读取 results/token_usage.json，打印当前运行批次的 Token 用量汇总。"""
    token_log = token_usage_path or os.path.join(base_dir, "results", "token_usage.json")
    if not os.path.exists(token_log):
        return

    print()
    print("Token 用量汇总")
    total = {"input": 0, "output": 0, "total": 0}
    active_run_id = os.environ.get("TOKEN_RUN_ID", "").strip()

    try:
        with open(token_log, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  读取 token 日志失败: {e}")
        return

    books = data.get("books", {}) if isinstance(data, dict) else {}
    if active_run_id:
        for book_entry in books.values():
            if not isinstance(book_entry, dict):
                continue
            runs = book_entry.get("runs", {})
            run_entry = runs.get(active_run_id) if isinstance(runs, dict) else None
            if not isinstance(run_entry, dict):
                continue
            total["input"] += int(run_entry.get("run_total_input", 0))
            total["output"] += int(run_entry.get("run_total_output", 0))
            total["total"] += int(run_entry.get("run_total_tokens", 0))
    else:
        for book_entry in books.values():
            if not isinstance(book_entry, dict):
                continue
            total["input"] += int(book_entry.get("book_total_input", 0))
            total["output"] += int(book_entry.get("book_total_output", 0))
            total["total"] += int(book_entry.get("book_total_tokens", 0))

    print(f"  input:  {total['input']}")
    print(f"  output: {total['output']}")
    print(f"  total:  {total['total']}")


def run():
    base_dir = get_base_dir()
    run_id = _generate_run_id()
    os.environ["TOKEN_RUN_ID"] = run_id

    print("=" * 60)
    print("小说分析全流程工具")
    print("=" * 60)
    print()

    load_configs(base_dir)
    novel_files = scan_novels(base_dir)

    import protagonist
    import novel_scan
    import novel_reviewer
    import report

    total = len(novel_files)
    done = 0
    skipped = 0
    failed = 0

    print_pending_novels(novel_files)

    print(f"扫描 novels 目录下所有 txt，共 {total} 本，依次运行：")
    print("  1) protagonist.py   - 主角识别")
    print("  2) novel_scan.py    - 深度扫描")
    print("  3) novel_reviewer.py - 毒点二审")
    print("  4) report.py        - 生成报告")
    print()

    for novel_path in novel_files:
        os.environ["NOVEL_PATH"] = novel_path
        book_name = os.path.splitext(os.path.basename(novel_path))[0]

        print("=" * 40)
        print(f"正在处理: {os.path.basename(novel_path)}")
        print(f"NOVEL_PATH={novel_path}")

        should_skip, out_file = _report_is_fresh(base_dir, book_name)
        if should_skip:
            skipped += 1
            print(f"★ 检测到该书报告已正常生成，跳过后续流程：{out_file}")
            print()
            continue

        time.sleep(5)

        status = "ok"

        if status == "ok":
            try:
                ret = protagonist.main(novel_path=novel_path, book_name=book_name, run_id=run_id)
                if ret is not None and ret != 0:
                    status = "fail"
            except Exception as e:
                print(f"[protagonist] 异常: {e}")
                status = "fail"

        detail_path = None
        if status == "ok":
            detail_path = _get_working_detail_path(protagonist, book_name)

        if status == "ok":
            try:
                novel_scan.main(novel_path=novel_path, book_name=book_name, run_id=run_id, detail_path=detail_path)
            except Exception as e:
                print(f"[novel_scan] 异常: {e}")
                status = "fail"

        if status == "ok":
            try:
                novel_reviewer.main(novel_path=novel_path, book_name=book_name, run_id=run_id, detail_path=detail_path)
            except Exception as e:
                print(f"[novel_reviewer] 异常: {e}")
                status = "fail"

        if status == "ok":
            try:
                report.main(novel_path=novel_path, book_name=book_name, run_id=run_id, detail_path=detail_path)
            except Exception as e:
                print(f"[report] 异常: {e}")
                status = "fail"

        if status == "ok":
            print(f"成功: {os.path.basename(novel_path)}")
            done += 1
        else:
            print(f"失败: {os.path.basename(novel_path)}")
            failed += 1
        print()

    print("=" * 40)
    print("处理完成")
    print(f"总计: {total} 本  成功: {done}  跳过: {skipped}  失败: {failed}")
    print("=" * 40)

    print_token_summary(base_dir)

    print()
    try:
        input("所有任务完成，按回车键退出...")
    except EOFError:
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run()
