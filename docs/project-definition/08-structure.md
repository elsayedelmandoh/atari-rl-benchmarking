# repository structure and file responsibilities

## project structure

```
atari-games/
├── data/
│   ├── models/
│   ├── predictions/
│   ├── processed/
│   ├── raw/
│   ├── samples/
│   └── vectorizers/
|
├── docs/
│   |
│   ├── 00-requirements/
|   │   ├── 00-quickstart.md           # section overview
|   │   ├── 01-related-work.md         # competitor landscape, prior art
|   │   ├── 02-research-notes.md       # raw observations, exploration logs
|   │   ├── 03-courses.md              # skill gap analysis, learning objectives
|   │   ├── 04-timeline.md             # working milestones & schedule
|   │   └── 05-meeting/                # dated meeting notes (e.g. jun-8-2026.md)
│   |
│   ├── 01-project-definition/
|   │   ├── 00-quickstart.md           # 10-min project overview
|   │   ├── 01-problem.md              # problem statement & context
|   │   ├── 02-goal.md                 # project goals & objectives
|   │   ├── 03-dataset.md              # data sources & specifications
|   │   ├── 04-solution.md             # proposed solution approach
|   │   ├── 05-constraints.md          # should do / should not do
|   │   ├── 06-architecture.md         # system design & data flow
|   │   ├── 07-stack.md                # technology stack & dependencies
|   │   ├── 08-structure.md            # project directory structure
|   │   ├── 09-workflow.md             # development workflow & process
|   │   └── 10-references.md           # academic references & papers
|   |
|   ├── 02-results/
|   │   ├── 00-quickstart.md           # 10-min overview of results
|   │   ├── 01-evaluation.md           # model/solution evaluation metrics
|   │   ├── 02-testing.md              # testing methodology & results
|   │   ├── 03-performance-comparison.md  # benchmarks vs. baselines
|   │   ├── 04-results-analysis.md     # detailed findings & insights
|   │   ├── 05-future-work.md          # next steps & open problems
|   │   └── figures/          
|   |
|   ├── 03-deliverables/
|   │   ├── 00-quickstart.md           # 10-min overview
|   │   ├── 01-proposal.md             # business/project proposal
|   │   ├── 02-presentation.md
|   │   └── 03-paper.md
|   |
|   └── api/
|       ├── 00-quickstart.md           # 10-min api overview
|       └── 01-api-design.md           # api specifications & contracts
|
├── notebooks/
│   ├── 00-quickstart.ipynb
│   ├── 01-data-acquisition/
│   │   └── 00-quickstart.ipynb
│   ├── 02-eda/
│   │   └── 00-quickstart.ipynb
│   ├── 03-data-preprocessing/
│   │   └── 00-quickstart.ipynb
│   ├── 04-feature-engineering/
│   │   └── 00-quickstart.ipynb
│   ├── 05-model-training/
│   │   └── 00-quickstart.ipynb
│   ├── 06-model-evaluation/
│   │   └── 00-quickstart.ipynb
│   └── 07-model-testing/
│       └── 00-quickstart.ipynb
├── src/
│   ├── config/
│   │   └── config.py
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── contract/
│   ├── integration/
│   └── unit/
|
├── .env
├── .env.example
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── app.py
├── pyproject.toml
└── requirements.txt
````

## directory explanation

- app.py: entrypoint for local startup and quick validation.
- src/config: runtime configuration and environment loading.
- src/database: connection helpers, migrations, and repository code.
- src/utils: shared utilities that do not belong in a feature module.
- notebooks: exploratory and iterative work that should later be moved into src/.
- tests: unit and integration coverage for the critical paths.
- data/raw: source data kept as close to the original form as possible.
- data/processed: cleaned and transformed datasets.
- data/samples: small fixture-like datasets for fast iteration.
- data/models: serialized model artifacts.
- data/predictions: output predictions and inference results.
- data/vectorizers: fitted text or feature preprocessing artifacts.

