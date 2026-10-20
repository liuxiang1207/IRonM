# IronM - 你的健身AI管家

多 Agent 健身管家：训练、营养、恢复、行为等专业 Agent 协作，数字由代码计算、每条建议可溯源、安全规则先行。

## 开发

```bash
make setup     # 安装依赖与 pre-commit
make check     # 提交前与 CI 都跑：锁文件 → ruff → 依赖方向 → mypy → pytest
make schemas   # 契约改动后重新导出 JSON Schema（make check 会比对）
make offline   # 急症词表或回答模板改动后重新生成 App 离线版数据（make check 会比对）
make migrate   # 执行尚未执行的数据库迁移（migrations/NNNN_说明.sql）
make sim       # 快进模拟：5 类模拟用户各 8 周，输出汇总与第 8 周末的指标（精度、趋势与可信范围、真实趋势、状态、还差什么）
make eval-models  # 模型选型评测（调用真实接口、产生费用；密钥放 .env）
make run       # 本地启动（启动时自动执行迁移）；/healthz 存活检查
```

需要 [uv](https://docs.astral.sh/uv/)；Python 3.13 由 uv 自动管理。检查 App 离线版需要 Node.js 22.18 以上。

## 约定

- **先写需求再动手**：需求记在 `spec/spec.md`，对应验证清单记在 `spec/testing.md`，完成后逐条勾选。
- **依赖只能向下**：分层见 `pyproject.toml` 的 `[tool.importlinter]`；`safety` 只依赖 `contracts` 与 `core`；新增子包必须登记分层。
- **阈值进规则表**：业务阈值写在 `rules/*.yaml`，代码按编号取用；改规则必须同时改测试。
- **出错就报错**：配置、契约、日志字段不合规时直接失败，不静默降级。
- **全局规则**：每个计算都处理更正（统一走有效记录层）、未知、时区、截止时间，阈值来自规则表且启动时校验（spec §16）。
- **日志不含健康数据原文**：事件名是固定标识，字段只允许 `*_id` 与运行类信息，异常只记类型。

## 配置

| 环境变量 | 取值 | 默认 |
|---|---|---|
| `IRONM_ENV` | `dev` / `test` / `prod` | `dev` |
| `IRONM_LOG_LEVEL` | `debug` / `info` / `warning` / `error` | `info` |
| `IRONM_DB_PATH` | 数据库文件路径 | `data/ironm.sqlite` |
| `IRONM_RULES_DIR` | 规则表目录 | `rules` |

出现未知的 `IRONM_` 变量、非法取值，或规则表、急症词表、拒答词表、回答模板有误时，启动失败。

## 接口

所有接口都需要当前用户：登录功能完成前由请求头 `X-IronM-User` 给出（缺少返回 401）；提交的记录必须属于当前用户。请求体上限 2 MB（超过返回 413）。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 存活检查 |
| POST | `/events` | 提交一条记录：201 已创建、200 重复（只存一次）、409 冲突、422 不合法（只给字段位置与原因，不回显内容） |
| POST | `/events/batch` | 批量提交（离线队列上传），`{"events": [...]}`，每批 1–500 条；逐条处理、逐条返回结果，同批内更正排在原记录之后；提交顺序不影响最终结果 |
| GET | `/energy/today?tz=` | 今天的精力选择与训练量调整（时区用标准名称，如 Asia/Shanghai）；未选择时 `selected: false`，不给调整。精力通过 `POST /events` 提交 `energy_check` 记录 |
| GET | `/metrics?tz=` | 当前用户的指标：四个领域的精度、体重趋势（带可信范围与判断窗口）、近 7 天日均热量；每项含状态与还差什么 |

## 目录

```
ironm/
  core/        时间来源、标识生成、配置、结构化日志（最底层）
  contracts/   数据契约：记录、未知值、指标结果、Agent 输出、字段归属、快照、计划版本、模型调用
  safety/      急症词表、拒答词表、回答模板（*.yaml，只能由人修改）；急症判定、人群安全分层、三档输出与四段式：不依赖其他业务模块，离线可用
  rules/ engine/ knowledge/ llm/ skills/ agents/ memory/ orchestrator/ scheduler/ api/
  store/       SQLite 连接、迁移执行器、记录仓储（只追加、去重、全文检索）
migrations/    按编号的数据库迁移脚本
rules/         规则表（YAML，每条带编号、审定状态、来源、参数）
schemas/       契约导出的 JSON Schema
app/           手机 App（React Native + Expo，TypeScript）；目前只有离线急症判定 src/safety/
sim/           快进模拟：模拟用户（隐藏真值 + 陷阱）、按天生成记录、快进运行器
evals/         模型选型评测：题目、模型配置、判分
spec/          需求与验证清单
tests/
```
