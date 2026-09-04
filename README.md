# NovelDigest

面向男性向长篇 `.txt` 小说的多阶段扫书分析工具。项目按顺序完成主角色识别、正文深度扫描、结果复核和最终报告生成，并把中间产物与可读报告统一输出到 `results/` 目录。

这不是通用文本分析框架，而是围绕「扫书、排雷、角色事实提取、最终汇总」打磨出来的实用型流水线。

## 核心能力

- 批量扫描 `novels/` 目录下的所有 `.txt` 小说。
- 自动识别男主、女主候选及其别名，输出角色中间结构。
- 按分块扫描全书正文，提取雷点、郁闷点和角色相关事实。
- 在复核阶段对扫描结果做二次审核，输出更稳定的汇总结论。
- 自动生成最终可读的扫书报告。
- 记录阶段性中间文件、断点信息、日志和 token 使用情况。
- Windows 下可通过 `run.bat` 启动，首次运行由 `uv` 自动创建 `.venv` 并安装依赖。

## 项目结构

```text
.
├── main.py                # 主流程入口：读取配置 -> 扫描 novels -> 串联四个阶段
├── run.sh                 # macOS / Linux 启动脚本（uv）
├── run.bat                # Windows 启动脚本（uv）
├── protagonist.py         # 主角色识别与女主候选提取
├── novel_scan.py          # 分块扫描正文，提取问题点和结构化事实
├── novel_reviewer.py      # 二次复核与汇总结论生成
├── toxic_reviewer.py      # 毒点二次复核规则
├── report.py              # 生成最终面向阅读的报告
├── api_client.py          # 统一 API 调用、错误重试、限流与 token 统计
├── shared_utils.py        # 共享配置、API 调用封装、通用工具
├── text_anchor.py         # chunk manifest 与证据定位逻辑
├── token_tracker.py       # token 统计
├── config/
│   ├── rules.json         # 扫书规则库：雷点/郁闷点分类及说明
│   ├── seed_keywords.json # 初始关键词种子
│   ├── setting.example.txt# 运行配置模板（复制为 setting.txt）
│   └── api.example.txt    # API Key 模板（复制为 api.txt）
├── novels/                # 输入小说文本目录（不提交）
├── results/               # 输出目录（不提交）
│   └── learned_keywords/  # 扫描阶段生成的增量关键词快照
├── pyproject.toml         # uv 项目与依赖声明
├── uv.lock                # uv 依赖锁定文件
├── .gitignore
└── LICENSE
```

## 快速开始

### Windows

1. 安装 `uv`（推荐，自带 Python 版本管理）或 Python 3.10+。
2. 把待分析的小说 `.txt` 放到 `novels/` 目录。
3. 复制 `config/api.example.txt` 为 `config/api.txt`，每行填写一个可用的 API Key。
4. 按需修改 `config/setting.txt`。
5. 双击 `run.bat`。

`run.bat` 会检查 `uv` 是否存在，然后执行 `uv run main.py`；首次运行时 `uv` 会自动创建 `.venv` 并安装 `pyproject.toml` 中声明的依赖。

### macOS / Linux

```bash
chmod +x run.sh
./run.sh
```

脚本会执行 `uv run main.py`，首次运行自动创建 `.venv` 并安装依赖。

### 手动运行

```powershell
uv sync
uv run main.py
```

## 配置说明

### `config/api.txt`

每行一个 API Key，程序会自动组成 `API_KEY_POOL`。请只在本地保存真实 Key，不要把它提交到公开仓库（`.gitignore` 已忽略该文件）。

### `config/setting.txt`

`main.py` 会从 `config/setting.txt` 读取运行配置并注入环境变量。常用配置：

- `BASE_URL`：OpenAI 兼容接口地址。
- `MODEL_NAME`：调用的模型名称。
- `MAX_WORKERS`：并发线程基线。
- `RPM_LIMIT` / `TPM_LIMIT`：限流相关配置。

其余 `DIM_BOOST_*`、`RESCAN_*`、`MAX_MIDDLE_SUMMARY_CALLS` 主要用于扫描阶段的补扫和增强策略，属于进阶调优项。

### `config/rules.json`

扫书和复核阶段共同使用的规则库，定义要扫描的雷点/郁闷点分类与具体条目。`novel_scan.py` 与 `novel_reviewer.py` 共用同一套标准，避免初扫和二审口径不一致。

### `config/seed_keywords.json`

初始关键词种子，供扫描阶段的关键词增强与补扫逻辑使用。代码会与 `results/learned_keywords/` 下最新生成的 `learned_*.json` 合并，形成当前生效的关键词集合。

## 流程说明

`main.py` 对 `novels/` 下的每本 `.txt` 依次执行四个阶段：

1. `protagonist.py`：识别男主、女主候选及别名，生成角色相关中间文件。
2. `novel_scan.py`：分块扫描正文，按 `config/rules.json` 提取问题点与结构化角色事实，并把新学习到的关键词写入 `results/learned_keywords/`。
3. `novel_reviewer.py`：结合 `config/rules.json` 对扫描结果做二次复核，输出更稳定的汇总结论。
4. `report.py`：读取最新复核结果与角色明细，生成最终可读的扫书报告。

中间产物默认输出到 `results/<书名>_heroine_<timestamp>/` 和 `results/<书名>_scan_<timestamp>/`，最终报告输出到 `results/<书名>扫书报告_<timestamp>.txt`。

## 输出结果概览

- `results/<书名>扫书报告_<timestamp>.txt`：最终可读报告。
- `results/<书名>_scan_<timestamp>/VERIFIED_SUMMARY_<timestamp>.json`：复核阶段汇总。
- `results/<书名>_scan_<timestamp>/raw_data.json`：扫描阶段原始结构化结果。
- `results/<书名>_heroine_<timestamp>/*_detailed_*.json`：角色与事实的详细中间产物。
- `results/learned_keywords/`：扫描阶段学习到的增量关键词快照。
- `results/token_usage.json`：当前运行批次的 token 使用汇总。

## 使用声明

- 禁止将本程序生成、汇总或润色后的报告，在未明确标注「AI 生成」或「AI 辅助生成」的情况下对外售卖。
- 如果基于本程序输出的内容进行商业发布、分发或售卖，必须进行清晰、显著、不可误解的 AI 生成标注。
- 不建议将本程序产出的报告包装成人工原创评测、人工精读结论或纯人工整理成果进行传播。

## 许可证

本项目采用 [GNU Affero General Public License v3.0](LICENSE)（AGPL-3.0-or-later）授权，详见根目录 `LICENSE` 文件。
