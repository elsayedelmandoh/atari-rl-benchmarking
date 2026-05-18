# 11 - development workflow

## setup

```bash
# clone & enter
git clone https://github.com/elsayedelmandoh/recon-agent.git
cd recon-agent

# install uv (if not installed)
curl -Ls https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# copy env template and fill in api keys
cp .env.example .env
# edit .env with your keys: OPENAI_API_KEY, SHODAN_API_KEY, etc.
```

---

## running the system

```bash
# basic passive recon
uv run recon-agent run --target example.com

# with explicit scope config
uv run recon-agent run --target example.com --scope scope.json

# enable active modules (requires explicit confirmation)
uv run recon-agent run --target example.com --active

# specify output dir
uv run recon-agent run --target example.com --output ./results/example/

# use a specific llm provider
uv run recon-agent run --target example.com --llm claude
```

---

## development loop

```bash
# run linter + formatter
uv run ruff check . --fix
uv run ruff format .

# run unit tests
uv run pytest tests/unit/ -v

# run integration tests (mocked apis)
uv run pytest tests/integration/ -v

# run full evaluation against ground truth
uv run python tests/eval/run_eval.py --target example_com

# run all tests
uv run pytest -v
```

---

## adding a new sub-agent

1. create `src/recon_agent/tools/new_source_tool.py` - implement api call, return typed dict
2. create `src/recon_agent/agents/sub_agents/new_agent.py` - wrap tool as crewai agent
3. define output schema in `src/recon_agent/schemas/findings.py`
4. register the agent as a tool in `src/recon_agent/agents/planner.py`
5. add dedup/correlation logic in `src/recon_agent/synthesis/` if needed
6. write unit test in `tests/unit/test_new_source_tool.py`

---

## testing strategy

| test type | scope | mocking |
|-----------|-------|---------|
| unit | individual tools, validators, dedup logic | full api mock (responses library) |
| integration | planner → sub-agents → synthesis flow | mocked api responses, real logic |
| evaluation | end-to-end against ground truth targets | no mocking - real apis, authorized targets |

---

## git workflow

```
main              # stable, never broken
├── dev           # integration branch
│   ├── feat/dns-agent
│   ├── feat/network-agent
│   ├── feat/synthesis-layer
│   └── fix/rate-limiter-bug
```

- feature branches off `dev`, pr back to `dev`
- `dev` merges to `main` after eval run passes
- commit messages: `feat:`, `fix:`, `chore:`, `docs:`, `test:`

---

## common commands cheatsheet

```bash
uv run recon-agent run --target <domain>     # full passive recon
uv run pytest tests/unit/ -v                 # unit tests
uv run ruff check . --fix                    # lint + autofix
uv run python tests/eval/run_eval.py         # evaluation
docker compose up                            # start containerized env
```
