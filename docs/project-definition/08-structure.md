# repository structure

```
atari-rl-benchmarking/
  app.py                         main benchmark runner (profiles, locks, training, eval, playback)
  pyproject.toml                 project metadata, dependencies, dev scripts
  requirements.txt               pinned dependencies
  CLAUDE.md                      agent behavior rules
  README.md                      project overview, setup, usage, results

  configs/
    algorithms.json               dqn/ppo/discretesac hyperparameters
    benchmark.json                named run profiles (timesteps, seeds, eval episodes)
    envs.json                     game definitions and environment ids
    preprocessing.json            shared preprocessing pipeline config
    defaults.json                 default paths and settings
    model_contract.json           input shape, dtype, range validation specs

  src/
    config/
      __init__.py
      settings.py                 path resolution, logging setup
    utils/
      __init__.py
      env.py                      atari environment construction, wrappers
      seeding.py                  seed and device helpers
    models/
      __init__.py
      dqn.py                      sb3 dqn bridge
      ppo.py                      sb3 ppo bridge
      discrete_sac.py             custom categorical-policy sac
      sb3_bridge.py               shared sb3 training/evaluation harness
      train_eval.py               training and evaluation loop
    evaluation/
      __init__.py
      contract.py                 input validation (shape, dtype, range, etc.)
      metadata.py                 experiment metadata collection
      metrics.py                  reward statistics and diagnostics
      reporting.py                helper functions for report-ready output
    inference/
      __init__.py
      record_playback.py          playback regeneration from checkpoints
    config/
      __init__.py
      loader.py                   config file loading and hashing
      logging_setup.py            structured log configuration

  evals/
    checkpoints/                  profile-scoped checkpoints, csvs, manifests, diagnostics
      <profile>/
        <profile>_results_<timestamp>.csv
        <profile>_manifest_<timestamp>.json
        <game>/<algo>/seed_<seed>/
          final_model.zip           dqn/ppo
          final_model.pt            discretesac
          ppo_diagnostics.csv
          discrete_sac_diagnostics.csv
    figures/
      mean_reward.png
      max_reward.png
      reward_per_hour.png
      training_time.png

  artifacts/
    evaluation/
      playback/                   mp4 videos per profile/game/algo
        <profile>/
          <game>/<algo>/*.mp4
    preparing/
      input_samples/              raw and preprocessed observation samples

  logs/                           tensorboard events and process logs

  tests/
    unit/
    integration/
    contract/

  docs/
    deliverables/
      00-quickstart.md             report writing guide
      01-proposal.md               project proposal
      01-proposal-feedback.md      feedback from proposal submission
      02-report.md                 final project report (markdown)
      02-report.tex                final project report (latex)
    project-definition/
      00-quickstart.md             this folder overview
      01-problem.md                problem statement
      02-goal.md                   goal and success criteria
      03-dataset.md                environment details
      04-solution.md               solution overview
      05-constraints.md            constraints and assumptions
      06-architecture.md           system architecture
      07-stack.md                  technology stack
      08-structure.md              this file
      09-workflow.md               development workflow
      10-references.md             references
    project-reference-guide.md     technical walkthrough of the codebase
```
