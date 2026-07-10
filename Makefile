.PHONY: format lint typecheck test build

format:
	uv run ruff check --fix-only .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy .

test:
	uv run pytest

build:
	uv build
