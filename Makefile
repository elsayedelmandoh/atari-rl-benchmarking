.PHONY: install test lint format clean

# --- Environment ---
install:
	uv sync
	uv run nbstripout --install

# --- Quality ---
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	uv run pytest tests/ -v

# --- Cleanup ---
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type f -name '*.pyc' -delete; \
	rm -rf .pytest_cache
