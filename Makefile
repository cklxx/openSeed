.PHONY: install test lint format clean typecheck

install: ## Install in editable mode with dev deps
	pip install -e ".[dev]"

test: ## Run tests
	pytest -v

lint: ## Lint with ruff
	ruff check src/ tests/

format: ## Auto-format with ruff
	ruff format src/ tests/

typecheck: ## Run mypy type checking
	mypy src/openseed/

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
