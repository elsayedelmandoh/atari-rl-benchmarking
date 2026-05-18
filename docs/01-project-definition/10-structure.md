# 10 - project structure

```
recon-agent/
│
├── .env.example                    # api key template (never commit .env)
├── .gitignore
├── README.md
├── pyproject.toml                  # dependencies + project metadata (uv/pip)
├── docker-compose.yml              # reproducible dev environment
│
├── docs/
│   └── project-definition/
│       ├── 01-problem.md
│       ├── 02-goal.md
│       ├── 03-related-work.md
│       ├── 04-research-notes.md
│       ├── 05-dataset.md
│       ├── 06-solution.md
│       ├── 07-constraints.md
│       ├── 08-architecture.md
│       ├── 09-stack.md
│       ├── 10-structure.md         # this file
│       ├── 11-workflow.md
│       ├── 12-timeline.md
│       ├── 13-references.md
│       └── 14-proposal.md
│
├── src/
│   └── recon_agent/
│       ├── __init__.py
│       │
│       ├── cli.py                  # prompt_toolkit cli entrypoint
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py         # env vars, api keys, global config
│       │   └── scope.py            # scope validation logic
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── planner.py          # orchestrator agent (react loop)
│       │   ├── synthesis.py        # synthesis + report generation agent
│       │   └── sub_agents/
│       │       ├── __init__.py
│       │       ├── dns_agent.py
│       │       ├── subdomain_agent.py
│       │       ├── network_agent.py
│       │       ├── whois_agent.py
│       │       ├── socmint_agent.py
│       │       ├── reputation_agent.py
│       │       └── email_agent.py
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── shodan_tool.py
│       │   ├── censys_tool.py
│       │   ├── crtsh_tool.py
│       │   ├── virustotal_tool.py
│       │   ├── github_tool.py
│       │   ├── hunter_tool.py
│       │   ├── dns_tool.py
│       │   └── whois_tool.py
│       │
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── target.py           # TargetScope, TargetProfile pydantic models
│       │   ├── findings.py         # DnsFindings, NetworkFindings, etc.
│       │   └── report.py           # ReconReport model
│       │
│       ├── synthesis/
│       │   ├── __init__.py
│       │   ├── deduplicator.py     # cross-source dedup logic
│       │   ├── correlator.py       # cross-source correlation (e.g., cert san vs subdomain)
│       │   ├── risk_scorer.py      # severity tagging per finding
│       │   └── renderer.py         # json + markdown output generation
│       │
│       └── utils/
│           ├── __init__.py
│           ├── rate_limiter.py     # per-source token bucket
│           ├── logger.py           # jsonl trace logging
│           └── validators.py       # input sanitization
│
├── tests/
│   ├── unit/
│   │   ├── test_scope.py
│   │   ├── test_dns_tool.py
│   │   ├── test_deduplicator.py
│   │   └── test_risk_scorer.py
│   ├── integration/
│   │   ├── test_planner_flow.py    # mock api responses
│   │   └── test_synthesis.py
│   └── eval/
│       ├── ground_truth/           # manually-built target profiles for eval
│       │   └── example_com.json
│       └── run_eval.py             # precision/recall/f1 evaluation script
│
└── output/                         # gitignored - runtime outputs land here
    ├── target_profile.json
    ├── target_report.md
    └── recon_log.jsonl
```

## key file roles

| file | purpose |
|------|---------|
| `cli.py` | `recon-agent run --target example.com --mode passive` |
| `config/scope.py` | enforces target scope before every tool call |
| `agents/planner.py` | react loop: observe → reason → delegate → synthesize |
| `schemas/target.py` | `TargetProfile` is the single source of truth for output structure |
| `synthesis/correlator.py` | detects cross-source relationships (most interesting logic) |
| `tests/eval/run_eval.py` | computes precision/recall vs ground truth profiles |
