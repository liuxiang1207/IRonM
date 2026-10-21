# uv 不一定在 PATH 里（例如图形化 git 客户端触发 pre-commit 时），按常见安装位置兜底查找
UV ?= $(or $(shell command -v uv 2>/dev/null),$(wildcard $(HOME)/.local/bin/uv),$(wildcard $(HOME)/.cargo/bin/uv))
ifeq ($(strip $(UV)),)
$(error 找不到 uv，请先安装：https://docs.astral.sh/uv/getting-started/installation/)
endif

.PHONY: setup check lock-check lint deps fmt type test run schemas offline migrate eval-models sim demo

setup:
	$(UV) sync --locked
	$(UV) run pre-commit install

# 提交前与 CI 都跑这一项；顺序从快到慢，先失败先停
check: lock-check lint deps type test

# 依赖声明改了却没更新 uv.lock 时直接失败，避免“我这里能跑”
lock-check:
	$(UV) lock --check

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

# 模块依赖方向，规则见 pyproject.toml 的 [tool.importlinter]
deps:
	$(UV) run lint-imports

fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

type:
	$(UV) run mypy

test:
	$(UV) run pytest

# 契约改动后重新导出 JSON Schema；make check 会比对，未导出即失败
schemas:
	$(UV) run python -m ironm.contracts.schema schemas

# 急症词表改动后重新生成 App 离线版数据；make check 会比对，未生成即失败
offline:
	$(UV) run python -m ironm.safety.offline

# 执行尚未执行的数据库迁移（路径取自 IRONM_DB_PATH）
migrate:
	$(UV) run python -m ironm.store.migrate

# 模型选型评测（会调用真实接口并产生费用）；密钥从 .env 读取，可在 ARGS 里指定配置名
eval-models:
	set -a; [ -f .env ] && . ./.env; set +a; $(UV) run python -m evals.model_selection.run $(ARGS)

# 快进模拟：5 类模拟用户各 8 周，写入 data/sim/sim.sqlite 并输出汇总
sim:
	$(UV) run python -m sim

# 冲刺 2 演示：几类模拟用户从建档到第一版计划（含依据卡与基线测试）
demo:
	$(UV) run python -m sim.first_plan_demo

run:
	$(UV) run uvicorn ironm.api.app:app --reload
