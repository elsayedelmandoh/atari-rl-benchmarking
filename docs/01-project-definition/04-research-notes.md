# 04 - research notes

## key findings from literature review

### on agentic planning for security tasks

- llms with tool-use (function calling / react-style agents) can reliably decompose multi-step recon tasks when given a clear system prompt defining the target, scope, and available tools.
- chain-of-thought prompting significantly improves agent decisions on when to pivot (e.g., discovering an unexpected ip range and deciding to enumerate it vs. staying on original scope).
- **finding**: the planner agent needs explicit scope boundaries in its system prompt - without them, models tend to over-explore or hallucinate tool calls.

### on osint source reliability

- shodan and censys are the gold standard for passive port/service discovery but differ in data freshness and coverage. shodan has broader iot coverage; censys has better tls certificate data.
- certificate transparency (crt.sh) is underutilized in automated tooling but is one of the most reliable passive subdomain sources - no rate limiting, fully public.
- github dorking (searching for leaked api keys, internal hostnames, config files) is high-yield but noisy. needs a scoring/filtering layer before being fed to the planner.
- email harvesting from hunter.io / phonebook.cz has diminishing returns for large orgs due to privacy scrubbing, but remains useful for smaller targets.

### on multi-agent architectures

- **crewai** vs **langgraph**: crewai is faster to prototype with role-based agents; langgraph gives finer control over state transitions and is better for complex conditional workflows. for a graduation project, crewai for v1, langgraph for v2 is a reasonable roadmap.
- shared memory between sub-agents is critical. without it, the dns agent and the subdomain agent duplicate work and miss cross-source correlations.
- tool call failure handling is non-trivial: api rate limits, network timeouts, and empty results all need graceful degradation paths. the planner must not stall on a failed sub-agent.

### on output synthesis

- naive concatenation of tool outputs into a single prompt overflows context windows and degrades llm reasoning quality.
- **better approach**: structured intermediate representations (json schemas per source type) that the synthesizer agent processes in a final pass.
- markdown target profiles with severity-tagged findings (info / low / medium / high) are more actionable for analysts than raw json dumps.

### on legal / ethical framing

- passive osint (querying public data sources, not touching target systems) is legally defensible in most jurisdictions when scoped to authorized targets.
- the tool must implement a strict scope validation layer: no queries against ips/domains not in the user-defined target scope.
- academic use: all testing should be done against either owned infrastructure or ctf/intentionally vulnerable targets (hackthebox, tryhackme, testphp.vulnweb.com, etc.).

## open questions (to resolve during development)

1. how to handle dynamic ip ranges in target scope definition? (cidr notation vs domain-only)
2. best strategy for deduplicating assets discovered by multiple sub-agents?
3. how to benchmark recall against a ground-truth manually-built profile? (need evaluation methodology)
4. llm model choice: llama-3.1-8b vs qwen2.5-7b vs mistral-7b for the planner? (latency vs cost vs capability tradeoff)
5. rate limiting strategy: shared token bucket across all sub-agents, or per-agent limits?

## promising implementation patterns

```
react agent loop (for planner):
  observe → reason → act → observe → ...

sub-agent pattern:
  planner issues task → sub-agent executes tool → returns structured result → planner updates context

synthesis pass:
  all sub-agent outputs → synthesizer agent → json target profile → markdown report
```
