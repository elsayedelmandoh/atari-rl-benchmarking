# 09 - technology stack

## core stack

| layer | technology | version | purpose |
|-------|-----------|---------|---------|
| language | python | 3.11+ | primary implementation language |
| agent framework | crewai | latest | multi-agent orchestration, role-based agents |
| llm provider | ollama (llama-3.1-8b) | - | planner + synthesis agent reasoning |
| llm provider (alt) | ollama (qwen2.5-7b) | - | configurable alternative for planner |
| data validation | pydantic | v2 | typed schemas for all inter-agent data |
| async execution | asyncio + aiohttp | stdlib | concurrent sub-agent execution |
| cli | prompt_toolkit | latest | clean cli interface with flags |
| env management | python-dotenv | latest | api key management |

---

## osint & data source libraries

| library | source | purpose |
|---------|--------|---------|
| dnspython | pypi | dns record resolution (a, mx, ns, txt, aaaa) |
| python-whois | pypi | whois queries |
| ipwhois | pypi | asn/ip ownership lookups |
| shodan | pypi (official) | shodan api client |
| censys | pypi (official) | censys api client |
| requests / httpx | pypi | http api calls (crt.sh, hunter.io, virustotal) |
| pygithub | pypi | github search api (dorking) |
| beautifulsoup4 | pypi | html parsing if needed for web scraping fallback |

---

## output & reporting

| tool | purpose |
|------|---------|
| jinja2 | markdown report templating |
| rich | terminal output formatting (progress bars, tables) |
| jsonlines | recon trace logging |

---

## development tooling

| tool | purpose |
|------|---------|
| uv | fast python package manager (recommended over pip) |
| pytest | unit + integration tests |
| pytest-asyncio | async test support |
| ruff | linter + formatter |
| pre-commit | enforce code quality on commit |
| docker | reproducible environment for testing |
| github actions | ci pipeline (lint + test on push) |

---

## external apis required (api keys needed)

| api | free tier sufficient? | notes |
|-----|----------------------|-------|
| ollama | yes (llama-3.1-8b) | primary llm |
| shodan | yes (limited) | 1 req/sec, basic search |
| censys | yes (researcher) | 0.2 req/sec, 250 queries/month |
| virustotal | yes | 4 req/min |
| github | yes (authenticated) | 30 req/min search |
| hunter.io | limited (25/month free) | sufficient for evaluation |
| crt.sh | no key needed | fully public api |

all api keys stored in `.env`, never committed.

---

## infrastructure

- **compute**: developer laptop (m1 mac / linux x86) - no gpu required
- **storage**: local filesystem only (no cloud storage for output files)
- **networking**: outbound https to external apis only
- **containers**: docker compose for reproducible test environment

---

## not in the stack (and why)

| rejected | reason |
|---------|--------|
| langchain | heavier than crewai for this use case; crewai's role model fits better |
| langgraph | better for complex stateful workflows; overkill for v1, reserved for v2 |
| elasticsearch | no need for indexed search at this scale; json files are sufficient |
| nmap (python-nmap) | active scanning - excluded from passive mode, optional in active mode |
| proprietary llm apis (openai, anthropic) | cost, privacy, data sent to external servers - open source preferred |
