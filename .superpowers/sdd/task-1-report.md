# Task 1 Report: Path-Aware Prediction Value Objects

## Implementation summary

- Added canonical `PredictionKind` values and alias normalization.
- Added immutable `PathPrediction` delegation and the runtime-checkable
  `PathPredictionState` protocol.
- Added the runtime-checkable `ConvertibleFlowPath` protocol without changing
  the minimal `FlowPath` contract.
- Added `LinearPredictionState` and `LinearPath.prediction_state` with direct
  source, target, and velocity conversion identities, endpoint clamping,
  broadcasting, and input validation.
- Exported the new public interfaces from `fm_lab.paths`.
- Added tests for aliases, direct identities, image broadcasting, endpoints,
  gradients, validation, and unsupported path detection.

## Files changed

- `fm_lab/paths/prediction.py` (created)
- `tests/test_path_prediction.py` (created)
- `fm_lab/paths/base.py` (modified)
- `fm_lab/paths/linear.py` (modified)
- `fm_lab/paths/__init__.py` (modified)

## RED

Command:

```text
.conda/fm_lab/bin/pytest tests/test_path_prediction.py -q
```

Output:

```text
==================================== ERRORS ====================================
________________ ERROR collecting tests/test_path_prediction.py ________________
ImportError while importing test module '/Users/yluo/Downloads/Projects/Diffusion/flow_matching/.worktrees/continuous-long-tail-objectives/tests/test_path_prediction.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../.conda/fm_lab/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_path_prediction.py:4: in <module>
    from fm_lab.paths import LinearPath, PredictionKind, normalize_prediction_kind
E   ImportError: cannot import name 'PredictionKind' from 'fm_lab.paths' (/Users/yluo/Downloads/Projects/Diffusion/flow_matching/fm_lab/paths/__init__.py)
=========================== short test summary info ============================
ERROR tests/test_path_prediction.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.77s
```

Why it failed: the public `PredictionKind` and normalization API did not yet
exist, so collection failed at the expected missing import before any
implementation was added.

## GREEN

Initial required behavior:

```text
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest tests/test_path_prediction.py -q
..                                                                       [100%]
2 passed in 0.70s
```

Focused regression set after adding the remaining required coverage:

```text
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest tests/test_path_prediction.py tests/test_objectives.py -q
..................................                                       [100%]
34 passed in 1.54s
```

Lint verification:

```text
PYTHONPATH=$PWD .conda/fm_lab/bin/ruff check fm_lab/paths tests/test_path_prediction.py
All checks passed!
```

## Full-suite result

Command:

```text
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest -q
```

Output:

```text
........................................................................ [ 18%]
........................................................................ [ 36%]
........................................................................ [ 54%]
........................................................................ [ 73%]
........................................................................ [ 91%]
.................................                                        [100%]
393 passed in 13.23s
```

## Self-review

- Confirmed `FlowPath` retains only `sample_xt` and `target_velocity`; optional
  conversion is isolated in `ConvertibleFlowPath`.
- Confirmed all three linear source kinds use direct formulas and
  velocity-to-endpoint conversion remains exact at `t=0` and `t=1`.
- Confirmed target/source endpoint inversion remains finite through positive
  denominator clamping.
- Confirmed `PathPrediction` and `LinearPredictionState` are frozen dataclasses.
- Confirmed alias normalization, shape validation, positive `min_denom`, time
  broadcasting, autograd, public exports, and runtime protocol behavior are
  covered by tests.
- Confirmed no unrelated tracked files were changed and `git diff --check`
  reports no whitespace errors.
- Retained the brief's required `str, Enum` inheritance with a targeted Ruff
  exemption rather than changing the public enum compatibility to `StrEnum`.

## Concerns

- The shared `.conda/fm_lab` environment is an editable install pointing at the
  main checkout. Commands after RED therefore set `PYTHONPATH=$PWD` so imports
  definitively exercise this worktree. The `.conda` worktree symlink remains
  untracked and is not part of the commit.

## Review fix: non-finite denominator validation and source conversion coverage

### Changes

- Tightened `LinearPredictionState.min_denom` validation to reject every
  non-finite value as well as zero and negative values.
- Added explicit interior-time assertions for source-to-velocity and
  source-to-target conversion.
- Parameterized denominator validation coverage over zero, a negative value,
  NaN, positive infinity, and negative infinity.

### RED

Command:

```text
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest tests/test_path_prediction.py -q
```

Output before the production fix:

```text
...................FF...                                                 [100%]
FAILED tests/test_path_prediction.py::test_linear_prediction_state_rejects_invalid_min_denom[nan]
FAILED tests/test_path_prediction.py::test_linear_prediction_state_rejects_invalid_min_denom[inf]
2 failed, 22 passed in 0.99s
```

The new source-origin interior conversion assertions passed, while NaN and
positive infinity failed to raise the expected `ValueError`, directly
reproducing the validation defect.

### GREEN

Command:

```text
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest tests/test_path_prediction.py -q
```

Output:

```text
........................                                                 [100%]
24 passed in 0.63s
```

Static checks:

```text
PYTHONPATH=$PWD .conda/fm_lab/bin/ruff check fm_lab/paths/linear.py tests/test_path_prediction.py
All checks passed!

git diff --check
```

Required focused regression set:

```text
PYTHONPATH=$PWD .conda/fm_lab/bin/pytest tests/test_path_prediction.py tests/test_objectives.py -q
........................................                                 [100%]
40 passed in 1.07s
```
