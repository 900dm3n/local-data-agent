# Local Data Agent 本地智能数据分析工作台

一个可在本地运行的数据分析 Agent 个人项目：将自然语言问题转为可检查的执行计划，检索业务定义与原文证据，调用只读 SQL 工具，并发布可从计算产物复核的报告。

项目在 AI 编程工具辅助下实现和迭代，重点是显式的 Agent 执行流程、可追溯的数据计算以及可重复的验证。它是研究与工程原型，业务示例使用合成数据。

## 核心能力

- **Agent Loop 与工具调用**：使用 LangChain `create_agent`、原生工具节点和中间件，执行模型请求、工具反馈、计划修订与终止控制。
- **检索增强**：业务文档分块、SQLite FTS5 全文索引、元数据筛选及原文引用；检索证据供模型计划与回答使用。当前不是向量数据库或向量重排系统。
- **可检查计算**：SQLite / CSV / Parquet / JSONL 数据源，SQLite 查询下推、DuckDB 聚合、SQL AST 检查及具名 Parquet 产物。
- **结果核对**：针对示例现金净额口径，用源事件独立复算金额与变化方向，保留报告引用及血缘信息。
- **计划与上下文管理**：声明式 DAG、增量补丁、受影响下游失效、未变结果复用、有界上下文和模型用量记录。
- **持久化与交互**：SQLite 业务状态、LangGraph 检查点、需批准的记忆草稿、本地网页工作台、后台任务、取消与显式接续。

## 运行环境

Python 3.12 或更高版本。Windows PowerShell 示例：

Windows 建议将仓库和测试临时目录放在较短路径，避免深层运行产物触及系统路径长度限制。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m langchain_data_agent doctor
.\.venv\Scripts\python.exe -m langchain_data_agent serve
```

打开 `http://127.0.0.1:8765`，创建离线演示数据并运行离线完整分析。也可运行 `start-workbench.cmd`。

离线命令行示例：

```powershell
.\.venv\Scripts\python.exe -m langchain_data_agent bootstrap --fixture cash-net-v1
.\.venv\Scripts\python.exe -m langchain_data_agent run --mode replay --case cash-net-v1
```

重复 bootstrap 会拒绝覆盖已有源。所有运行产物留在被 Git 忽略的 `var/`，不会随代码上传。

## 真实模型配置

原生框架入口支持 DeepSeek 与 MiMo 的官方 Chat Completions 接口。复制相应无密钥配置样例，填写账户可用的模型 ID：

```powershell
Copy-Item config.deepseek.example.json config.local.json
# 编辑 config.local.json，将 YOUR_MODEL_ID 改为账户可用模型
$agentSecret = Read-Host 'API key' -AsSecureString
$env:DATA_AGENT_API_KEY = [System.Net.NetworkCredential]::new('', $agentSecret).Password
Remove-Variable agentSecret
.\.venv\Scripts\python.exe -m langchain_data_agent doctor --probe
.\.venv\Scripts\python.exe -m langchain_data_agent run --mode live --question '按现金净额比较2026年8月与9月，列出支付、退款及净额变化'
Remove-Item Env:\DATA_AGENT_API_KEY
```

真实模式会向你选择的模型提供商发送问题和被选择的数据预览、字段及文本证据，并产生 API 用量。不要把私人或公司数据用于公开演示。缺失配置或模型请求失败时会明确报错，不会静默换成离线回放。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pip install '.[test]'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/verify_langchain.py
.\.venv\Scripts\python.exe scripts/verify_offline_business.py
```

上述命令默认无需真实 API。`verify_langchain.py --live` 会调用配置的模型，请在理解预算和数据范围后手动启用。

测试覆盖协议与消息历史、计划补丁、查询约束、源事件核对、证据引用、检查点恢复、工作台任务和下载。离线回放使用声明好的动作，不代表模型自主规划；自动化测试通过也不等于模型分析准确率。

## 目录

```text
src/langchain_data_agent/   原生框架适配、中间件、工作台、任务管理
src/local_data_agent/       计划、检索、执行、业务核对、报告与状态
tests/                     核心与框架回归
scripts/                   数据审计和离线/可选真实模型评价
docs/                      架构、演示与边界说明
config.*.example.json       不含密钥和个人路径的配置样例
```

## 当前边界

业务数值核验限定为示例 `CASH_NET_v1`：按悉尼本地事件月计算成功支付减成功退款，以整数分存储金额。尚不是任意业务指标平台。FTS5 检索不等于向量检索；证据相关性不证明因果；SQL 应用约束不等于操作系统级沙箱。后台队列使用单工作线程，外部 API 调用不能保证 exactly-once。模型仍可能生成错误计划或耗尽请求预算。

详见 [架构说明](docs/ARCHITECTURE.md)、[演示说明](docs/DEMO.md) 和 [安全与公开范围](SECURITY.md)。
