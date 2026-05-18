# 08 - architecture

## system overview

```
┌─────────────────────────────────────────────────────────────┐
│                        user / cli                           │
│         target: "example.com"  scope: {...}  mode: passive  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   scope validator                           │
│   validates target against scope config before any queries  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   planner agent (llm)                       │
│   react loop: observe → reason → act                        │
│   decomposes task, delegates to sub-agents, tracks context  │
└───┬──────┬──────┬──────┬──────┬──────┬──────┬──────────────┘
    │      │      │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼      ▼      ▼
┌──────┐ ┌────┐ ┌─────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌──────┐
│ dns  │ │sub │ │ net │ │whois │ │socmint │ │repute│ │email │
│agent │ │dom │ │agent│ │agent │ │ agent  │ │agent │ │agent │
└──┬───┘ └──┬─┘ └──┬──┘ └──┬───┘ └───┬────┘ └──┬───┘ └──┬───┘
   │        │      │       │          │          │         │
   ▼        ▼      ▼       ▼          ▼          ▼         ▼
┌──────┐ ┌────┐ ┌─────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌──────┐
│dns   │ │crt │ │shod │ │whois │ │github  │ │vt    │ │hunter│
│python│ │.sh │ │censys│ │ipwho │ │api     │ │api   │ │.io   │
└──────┘ └────┘ └─────┘ └──────┘ └────────┘ └──────┘ └──────┘
                         │
              structured json outputs
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  synthesis agent (llm)                      │
│   deduplicates, correlates, risk-scores findings            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     output layer                            │
│   target_profile.json  |  target_report.md  |  recon_log   │
└─────────────────────────────────────────────────────────────┘
```

---

## component breakdown

### planner agent
- **framework**: crewai (v1) / langgraph (v2)
- **model**: llama-3.1-8b (local) or qwen2.5-7b (higher reasoning)
- **memory**: shared context store (dict-based in v1, vector store in v2)
- **tool registry**: each sub-agent exposed as a crewai tool
- **logic**: react loop - after each sub-agent returns, the planner updates context and decides whether to delegate more tasks or trigger synthesis

### sub-agents
each sub-agent is a stateless function wrapped as a crewai tool:
- receives: target (domain/ip/org) + scope object
- executes: one or more api calls / dns queries
- returns: typed pydantic json object

### scope validator
- called before every sub-agent tool invocation
- checks query target against: allowed domains, allowed ip ranges (cidr), allowed org names
- raises `OutOfScopeError` on violation - logged, not crashed

### synthesis agent
- receives: list of sub-agent json outputs
- runs: deduplication (e.g., same ip from dns + shodan), correlation (e.g., cert san matches subdomain list), risk scoring (severity tags per finding)
- outputs: final `TargetProfile` pydantic model → serialized to json + rendered to markdown

---

## data flow

```
user input
  → scope validation
  → planner context init
  → sub-agent task queue
  → parallel execution (max 4 concurrent)
  → structured result collection
  → planner context update + adaptive re-planning
  → synthesis trigger (when planner decides coverage is sufficient)
  → output generation
```

---

## key architecture decisions

| decision | choice | reasoning |
|----------|--------|-----------|
| inter-agent communication | shared pydantic context object | type-safe, no free-text ambiguity |
| concurrency model | asyncio + semaphore (max 4) | respects api rate limits, prevents cascade failures |
| llm for planner | ollama (local, cpu inference) | no gpu required, university laptop compatible |
| sub-agent isolation | each agent is a pure function | testable independently, easy to swap out sources |
| output format | json (machine) + markdown (human) | covers both downstream automation and analyst use |
