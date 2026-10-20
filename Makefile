UV ?= uv

.PHONY: setup check lint fmt type test run

setup:
	$(UV) sync
	$(UV) run pre-commit install

check: lint type test

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

type:
	$(UV) run mypy

test:
	$(UV) run pytest

run:
	$(UV) run uvicorn ironm.api.app:app --reload
