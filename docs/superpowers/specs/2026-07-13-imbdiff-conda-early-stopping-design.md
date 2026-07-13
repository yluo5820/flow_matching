# ImbDiff Conda Environment and Early-Stopping Design

## Objective

Use the repository's existing Conda environment as the sole local development
environment and enable a common early-stopping policy across all nine ImbDiff
experiment configs.

## Scope

The configuration change applies only to `configs/imbdiff/*.yaml`. It does not
change the Python-level early-stopping default or alter toy, MNIST, or geometry
experiments.

Each ImbDiff config will contain:

```yaml
early_stopping:
  enabled: true
  patience_steps: 10000
  warmup_steps: 20000
  min_delta: 0.0001
  ema_alpha: 0.01
```

Using one policy preserves comparability across CIFAR-10 and CIFAR-100 DDPM,
x-vloss, CBDM, OC, and CM runs. The warmup avoids stopping during the initial
optimization phase, the exponential moving average smooths minibatch noise,
and patience is measured in training steps.

## Environment consolidation

The documented environment is `.conda/fm_lab`, created June 4 with Python
3.11. The redundant `.venv` was created July 13 at 12:44 by `uv` with Homebrew
Python 3.13. Both directories are git-ignored, and no interactive shell-history
entry accounts for `.venv`, consistent with automated creation during this
task.

Before removing `.venv`, `.conda/fm_lab` must pass:

1. dependency consistency checks;
2. the repository test suite;
3. a one-step CIFAR-10 CM CPU smoke experiment that trains and samples.

Only after all three checks succeed will `.venv` be deleted. Future commands
will use `.conda/fm_lab/bin/...` or executables from an activated Conda
environment.

## Verification

A config regression test will enumerate all nine ImbDiff YAML files and require
the exact early-stopping block above. After the edits, the focused config tests,
lint, and full test suite will run under `.conda/fm_lab`. The final repository
state must be clean after committing and pushing the verified changes.
