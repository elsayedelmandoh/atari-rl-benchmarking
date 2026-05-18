# 07 - constraints

## must do

- **scope enforcement**: every query must be validated against the user-defined target scope before execution. no exceptions.
- **passive by default**: all data collection defaults to passive osint. active scanning requires explicit user authorization via cli flag + scope confirmation prompt.
- **rate limit compliance**: respect api rate limits for every external source. implement per-source token buckets. never risk getting an api key banned.
- **structured outputs**: all sub-agent outputs must conform to a defined json schema. no free-text blobs passed between agents.
- **logging**: full reasoning trace (agent decisions, tool calls, results) logged to jsonl for reproducibility and evaluation.
- **authorization documentation**: all test targets must have documented authorization (owned, ctf, public bug bounty scope) before any queries run.
- **graceful degradation**: if a source is unavailable (api down, rate limited, key missing), the agent skips that source and continues - it does not stall or crash.

## must not do

- **no active scanning without authorization**: nmap, nuclei, or any active probe against live targets requires explicit `--active` flag and user confirmation. default mode is passive.
- **no pii storage beyond scope**: do not store personal emails, credentials, or personal data discovered during recon beyond what's needed for the target profile. no databases, no persistent logs of pii.
- **no exploitation**: this tool stops at recon. no vulnerability exploitation, no payload delivery, no post-exploitation. any such requests from the planner agent should be rejected by the system.
- **no querying ips/domains outside defined scope**: even if discovered during recon, out-of-scope assets get noted but not queried.
- **no use against unauthorized targets**: the system must include a disclaimer and authorization confirmation at startup.

## technical constraints

- all api keys stored in `.env` file, never hardcoded, never committed to git
- python 3.10+ required (match queen's university lab environment if applicable)
- must run on a standard developer laptop (no gpu required - llm inference runs locally via ollama, cpu-only)
- latency target: < 10 minutes for a full passive recon run on a standard domain
- context window management: no single llm call should exceed 60% of the model's context window

## legal constraints

- tool is designed for authorized use only
- passive osint queries against public data are generally legal, but vary by jurisdiction
- active scanning is governed by the computer fraud and abuse act (cfaa, us), computer misuse act (uk), and equivalent laws in canada (criminal code s.342.1)
- project documentation must include a clear legal disclaimer

## risk constraints

| constraint | why |
|-----------|-----|
| no auto-execution on discovered subdomains | prevents runaway scope expansion |
| max sub-agents running concurrently: 4 | prevents api rate limit cascade failures |
| max recon duration: 30 minutes (configurable) | prevents infinite loops on edge cases |
| output files written locally only | no auto-upload to any external service |

## ethical framing

this tool is built for:
- authorized penetration testing
- defensive security (know your own attack surface)
- academic research and education

it is not a hacking tool for unauthorized access. the same osint techniques used here are used by threat intelligence teams at every major security company. the difference is authorization.
