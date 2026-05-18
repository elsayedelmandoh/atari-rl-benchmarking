# 03 - related work

## existing tools & frameworks

### osint frameworks

| tool | what it does | limitation |
|------|-------------|-----------|
| **maltego** | graph-based osint visualization and correlation | expensive, manual, no llm-based reasoning |
| **spiderfoot** | automated osint with 200+ modules | no agent reasoning, static workflow, not adaptive |
| **recon-ng** | modular recon framework (like metasploit for osint) | cli-only, no synthesis layer, purely additive |
| **theHarvester** | email/subdomain/ip harvesting | single-purpose, no orchestration |
| **amass** | deep subdomain enumeration via passive and active methods | excellent at one task, not generalizable |
| **shodan cli** | search engine for internet-connected devices | query-only, requires manual interpretation |

**gap:** none of these combine tool orchestration + intelligent planning + result synthesis into a unified agent system.

### ai-powered security research

- **penTestGPT (2023)**: uses llm to guide pentesters through phases, but relies on human execution of tools. no autonomous tool use.
- **autoattack / auto-pentest papers**: academic work on automating phases of pentesting using rl or llm-guided agents. mostly proof-of-concept, limited osint integration.
- **vulnhuntr**: static code analysis with llm, entirely different domain (code, not recon).
- **nuclei + ai**: template-based scanning, active only, no osint layer.

### agentic ai frameworks relevant to this project

| framework | relevance |
|-----------|-----------|
| **langchain agents** | tool-use, planning, memory - directly applicable |
| **langgraph** | stateful multi-agent graphs - ideal for multi-phase recon orchestration |
| **autogen (microsoft)** | multi-agent conversation framework - useful for sub-agent delegation |
| **crewai** | role-based agent teams - maps well to specialized recon agents (dns agent, socmint agent, etc.) |

### academic literature

- **"automated reconnaissance in adversarial environments" (various)**: mostly network-layer focus, minimal nlp/llm integration.
- **"osint in cyber threat intelligence" (hutchins et al., lockheed martin cyber kill chain)**: foundational framing for where recon sits in attack lifecycle.
- **"llm agents for cybersecurity tasks" (2024 survey papers)**: emerging body of work confirming feasibility, noting lack of production-ready systems.
- **"graph neural networks for asset correlation in osint"**: relevant to the synthesis layer if graph-based correlation is added.

## key differentiators of this project

1. **agent-native**: not a wrapper around existing tools. the planner reasons about what to collect next based on prior findings.
2. **adaptive recon**: if subdomain discovery surfaces a cloud provider, the agent pivots to query cloud-specific osint sources.
3. **synthesis layer**: raw data from 8+ sources is unified into a single structured profile, not just concatenated.
4. **academic framing**: designed to be reproducible, benchmarkable, and publishable - not just a script.
