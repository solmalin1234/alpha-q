.PHONY: install sync train play record mlflow lint format test clean

install:
	uv sync --extra dev

sync:
	uv sync

train:
	uv run python scripts/train.py $(ARGS)

play:
	uv run python scripts/play.py $(ARGS)

record:
	uv run python scripts/record.py $(ARGS)

mlflow:
	uv run mlflow ui --port 5000

lint:
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/

format:
	uv run ruff check --fix src/ tests/ scripts/
	uv run ruff format src/ tests/ scripts/

test:
	uv run pytest tests/ -v

clean:
	rm -rf mlruns/ mlartifacts/ videos/ experiments/outputs/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
