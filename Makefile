.PHONY: help setup lint format type test test-all coverage eval demo serve worker bench-load bench-recovery docker-build clean all changelog cut-release

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies (including the git-sourced libraries)
	uv sync --dev

format: ## Format automatically
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint (no fixes)
	uv run ruff format --check .
	uv run ruff check .

type: ## Type-check
	uv run mypy src

test: ## Run the CI test selection
	uv run pytest -m "not slow"

test-all: ## Run every test, including slow soak tests
	uv run pytest

coverage: ## Tests with the coverage gate used in CI
	uv run pytest -m "not slow" --cov=src --cov-report=term-missing --cov-fail-under=80

eval: ## Run the eval suite against an in-process server
	uv run agent-platform eval

demo: ## Run the end-to-end demo (server, worker crash, recovery)
	uv run agent-platform demo

serve: ## Run the API server
	uv run agent-platform serve

worker: ## Run a job worker
	uv run agent-platform worker

bench-load: ## Load benchmark against a server process
	uv run python benchmarks/load.py

bench-recovery: ## Worker-kill recovery benchmark
	uv run python benchmarks/recovery.py

docker-build: ## Build the image
	docker build -t agent-platform .

clean: ## Remove caches and local data
	rm -rf .pytest_cache .mypy_cache .ruff_cache .pytest-work .platform dist build coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

all: lint type coverage ## Full local gate (mirror of continuous integration)

changelog: ## Regenerate CHANGELOG.md from Conventional Commits (git-cliff)
	git cliff --output CHANGELOG.md && perl -0pi -e 's/\n+\z/\n/' CHANGELOG.md

cut-release: ## Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional)
	scripts/release.sh $(RELEASE_VERSION)
