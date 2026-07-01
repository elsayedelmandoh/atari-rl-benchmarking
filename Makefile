.PHONY: install install-dev test lint format clean smoke dqn-final ppo-diagnostic sac-diagnostic playback-sac-all figures report

PYTHON ?= python
PROFILE ?= 64_1seed
ENV ?= all

# --- Environment ---
install:
	uv sync --extra torch --extra rl

install-dev:
	uv sync --extra torch --extra rl --extra dev --extra report
	uv run nbstripout --install

# --- Quality ---
lint:
	uv run ruff check app.py src/ tests/

format:
	uv run ruff format app.py src/ tests/

test:
	uv run pytest tests/ -v

# --- Benchmark Runs ---
smoke:
	uv run $(PYTHON) app.py 64_1seed

dqn-final:
	uv run $(PYTHON) app.py 1m_5seeds --algo dqn

ppo-diagnostic:
	uv run $(PYTHON) app.py 1m_1seed_ppo_diagnostic --algo ppo

sac-diagnostic:
	uv run $(PYTHON) app.py 1m_1seed_StaDiscSac_diagnostic --algo discretesac

playback-sac-all:
	uv run $(PYTHON) -m src.inference.record_playback 1m_1seed_StaDiscSac_diagnostic $(ENV)

figures:
	uv run $(PYTHON) evals/analyze_pilot.py

report: figures

# --- Cleanup ---
clean:
	uv run $(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyc')]; shutil.rmtree('.pytest_cache', ignore_errors=True)"
