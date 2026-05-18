# 12 - timeline & milestones

## total duration: 16 weeks (one academic semester)

---

## phase 1 - foundation (weeks 1–3)

**goal:** working skeleton, all external apis connected, scope validator in place

| week | deliverable | tasks |
|------|-------------|-------|
| 1 | project setup | repo, pyproject.toml, .env structure, ci pipeline, ruff config |
| 2 | core schemas + scope validator | pydantic models for all agent outputs, scope validation logic, unit tests |
| 3 | first 3 sub-agents live | dns agent, whois agent, crt.sh subdomain agent - each returning typed json |

**milestone 1 checkpoint:** can query a domain, get dns records + subdomains + whois, all in structured json. scope validator blocks out-of-scope queries.

---

## phase 2 - agent integration (weeks 4–7)

**goal:** all 7 sub-agents implemented, planner can delegate to them

| week | deliverable | tasks |
|------|-------------|-------|
| 4 | network agent + reputation agent | shodan + censys integration, virustotal reputation queries |
| 5 | socmint agent + email agent | github dorking, hunter.io integration |
| 6 | planner agent v1 | crewai orchestrator, react loop, sequential sub-agent delegation |
| 7 | rate limiting + error handling | per-source token buckets, graceful degradation, retry logic |

**milestone 2 checkpoint:** planner can execute a full passive recon run, delegating to all 7 sub-agents, collecting structured outputs. no crashes on api failures.

---

## phase 3 - synthesis layer (weeks 8–10)

**goal:** raw sub-agent outputs transformed into a unified, correlated target profile

| week | deliverable | tasks |
|------|-------------|-------|
| 8 | deduplicator + correlator | cross-source dedup, cert san vs subdomain correlation, ip asset linking |
| 9 | risk scorer + synthesis agent | severity tagging, synthesis agent llm call, targetprofile generation |
| 10 | output renderer | json serialization, markdown report template (jinja2), recon_log jsonl |

**milestone 3 checkpoint:** full end-to-end run produces `target_profile.json` + `target_report.md`. report is readable and actionable.

---

## phase 4 - adaptive planning (weeks 11–12)

**goal:** planner adapts based on discovered findings (not just fixed checklist)

| week | deliverable | tasks |
|------|-------------|-------|
| 11 | adaptive replanning | planner detects s3/cloud patterns, adds follow-up tasks dynamically |
| 12 | planner v2 refinement | prompt engineering, context compression, multi-run context carryover |

**milestone 4 checkpoint:** system pivots correctly on at least 3 types of discovered intelligence (cloud provider, leaked creds, third-party service detected).

---

## phase 5 - evaluation (weeks 13–14)

**goal:** measure system against manually-built ground truth profiles

| week | deliverable | tasks |
|------|-------------|-------|
| 13 | ground truth construction | manually build profiles for 5 authorized targets (ctf domains + personal lab) |
| 14 | eval run + analysis | run eval script on all 5 targets, compute precision/recall/f1, document gaps |

**milestone 5 checkpoint:** evaluation report with quantified performance metrics vs. baseline (manual recon).

---

## phase 6 - final delivery (weeks 15–16)

| week | deliverable | tasks |
|------|-------------|-------|
| 15 | documentation + readme | complete readme, api key setup guide, legal disclaimer, usage examples |
| 16 | final presentation + submission | demo video, slide deck, code cleanup, tag v1.0 release |

---

## risk buffer

- weeks 4–7 have the highest risk (api integration surprises, rate limit issues)
- 2 weeks of buffer built into the timeline (phase 5–6 can absorb delays)
- if github dorking or hunter.io proves unreliable: replace with alternative source without losing more than 3 days

---

## summary gantt (simplified)

```
week:  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
       [---phase 1---][-------phase 2-------][--phase 3--][p4][eval][fin]
```
