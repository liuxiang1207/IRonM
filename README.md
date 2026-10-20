# IronM - 你的健身AI管家

多 Agent 健身管家：训练、营养、恢复、行为等专业 Agent 协作，数字由代码计算、每条建议可溯源、安全规则先行。

## 开发

```bash
make setup     # 安装依赖与 pre-commit
make check     # 提交前与 CI 都跑：锁文件 → ruff → 依赖方向 → mypy → pytest
make schemas   # 契约改动后重新导出 JSON Schema（make check 会比对）
make migrate   # 执行尚未执行的数据库迁移（migrations/NNNN_说明.sql）
make run       # 本地启动，/healthz 存活检查
```

需要 [uv](https://docs.astral.sh/uv/)；Python 3.13 由 uv 自动管理。

## 约定

- **先写需求再动手**：需求记在 `spec/spec.md`，对应验证清单记在 `spec/testing.md`，完成后逐条勾选。
- **依赖只能向下**：分层见 `pyproject.toml` 的 `[tool.importlinter]`；`safety` 只依赖 `contracts` 与 `core`；新增子包必须登记分层。
- **阈值进规则表**：业务阈值写在 `rules/*.yaml`，代码按编号取用；改规则必须同时改测试。
- **出错就报错**：配置、契约、日志字段不合规时直接失败，不静默降级。
- **日志不含健康数据原文**：事件名是固定标识，字段只允许 `*_id` 与运行类信息，异常只记类型。

## 配置

| 环境变量 | 取值 | 默认 |
|---|---|---|
| `IRONM_ENV` | `dev` / `test` / `prod` | `dev` |
| `IRONM_LOG_LEVEL` | `debug` / `info` / `warning` / `error` | `info` |
| `IRONM_DB_PATH` | 数据库文件路径 | `data/ironm.sqlite` |
| `IRONM_RULES_DIR` | 规则表目录 | `rules` |

出现未知的 `IRONM_` 变量、非法取值或规则表格式错误时，启动失败。

## 目录

```
ironm/
  core/        时间来源、标识生成、配置、结构化日志（最底层）
  contracts/   数据契约：记录、未知值、指标结果、Agent 输出、字段归属、快照、计划版本、模型调用
  safety/ rules/ engine/ knowledge/ llm/ skills/ agents/ memory/ orchestrator/ scheduler/ api/
  store/       SQLite 连接、迁移执行器、记录仓储（只追加、去重、全文检索）
migrations/    按编号的数据库迁移脚本
rules/         规则表（YAML，每条带编号、审定状态、来源、参数）
schemas/       契约导出的 JSON Schema
spec/          需求与验证清单
tests/
```
