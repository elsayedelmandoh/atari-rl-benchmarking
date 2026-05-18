# 06 - proposed solution

## approach

build a **multi-agent recon system** where a central planner agent decomposes a recon task into sub-tasks, delegates them to specialized agents, and synthesizes all findings into a structured target profile. the system operates primarily in passive mode (no active probing), with optional active modules gated behind explicit user authorization.

this is not a tool wrapper. the planner reasons about what to do next based on what's been discovered, simulating how a competent threat actor thinks through initial recon.

---

## agent roles

### 1. planner agent (orchestrator)
- receives: target (domain / ip / org name) + scope config
- responsible for: task decomposition, sub-agent delegation, context management, synthesis trigger
- model: llama-3.1-8b or qwen2.5-7b (configurable)
- reasoning loop: react (reason + act) with tool-call-based sub-agent invocation

### 2. specialized sub-agents

| agent | tools | output |
|-------|-------|--------|
| **dns agent** | dnspython, dig | a/mx/ns/txt/aaaa records |
| **subdomain agent** | crt.sh api, amass (passive), subfinder | subdomain list |
| **network agent** | shodan api, censys api | exposed ports, services, banners |
| **whois/asn agent** | python-whois, ipwhois | ip ownership, asn, org, registrar |
| **socmint agent** | github search api | leaked secrets, config refs, internal hostnames |
| **reputation agent** | virustotal api | domain/ip history, threat intel tags |
| **email agent** | hunter.io api | email addresses, patterns |
| **synthesis agent** | llm + structured json schema | final target profile |

### 3. synthesis agent (reporter)
- receives: all sub-agent structured outputs
- responsible for: deduplication, correlation, risk scoring, markdown report generation
- outputs: `target_profile.json` + `target_report.md`

---

## key design decisions

### decision 1: structured intermediate outputs (not raw text)
each sub-agent returns a typed json object, not a string. this prevents context window overflow when the synthesizer processes all results.

**alternative considered:** feeding raw tool output directly to the planner. rejected - too noisy, burns tokens, degrades reasoning quality.

### decision 2: scope validation layer
before any sub-agent executes a query, a scope validator checks the query target against the user-defined scope. any out-of-scope query is blocked and logged.

**why this matters:** without it, the system can wander into querying unrelated infrastructure - legally and ethically unacceptable.

### decision 3: passive-first, active-optional
all modules default to passive. active modules (nmap-style port scanning against live targets) require `--active` flag + explicit scope confirmation. this matches how real authorized red teams operate.

### decision 4: crewai for agent orchestration (v1)
crewai's role-based agent model maps cleanly to the specialized sub-agent pattern. lower implementation overhead than building langgraph state machines from scratch for v1. langgraph considered for v2 if adaptive workflow control becomes necessary.

---

## adaptive behavior

the planner doesn't just execute a fixed checklist. it adapts:

- if subdomain discovery surfaces an aws s3 bucket pattern, the planner triggers an s3 enumeration sub-task
- if whois reveals a shared hosting provider, the planner notes it and adjusts risk scoring
- if github dorking finds a repo with internal hostnames, those are fed back into the subdomain agent's queue

this feedback loop is what separates an "ai-driven agent" from "a script that calls multiple apis."

---

## output artifacts

```
output/
  target_profile.json     # machine-readable structured profile
  target_report.md        # human-readable analyst report
  recon_log.jsonl         # agent reasoning trace (for eval + debugging)
```
