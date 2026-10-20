# uv 不一定在 PATH 里（例如图形化 git 客户端触发 pre-commit 时），按常见安装位置兜底查找
UV ?= $(or $(shell command -v uv 2>/dev/null),$(wildcard $(HOME)/.local/bin/uv),$(wildcard $(HOME)/.cargo/bin/uv))
ifeq ($(strip $(UV)),)
$(error 找不到 uv，请先安装：https://docs.astral.sh/uv/getting-started/installation/)
endif

.PHONY: setup check lock-check lint deps fmt type test run schemas migrate

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

# 执行尚未执行的数据库迁移（路径取自 IRONM_DB_PATH）
migrate:
	$(UV) run python -m ironm.store.migrate

run:
	$(UV) run uvicorn ironm.api.app:app --reload
