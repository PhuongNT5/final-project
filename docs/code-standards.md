# Code Standards (observed, not aspirational)

This describes conventions actually in use in this repo, verified by grep/read against the code.
It is not a style guide imposed from outside — where the repo is inconsistent, that is stated
below rather than papered over.

## File naming

Two conventions coexist, split by location:

- **Root scripts and Python packages** (`run_kfold_relation.py`, `train_behaviour_detector.py`,
  `unsafenet/`, `eval_utils/`): `snake_case.py`, matching Python module-import requirements (these
  files are imported by other scripts, e.g. `run_kfold_relation.py` does
  `import run_relation_transformer as R`).
- **`deployment/scripts/`**: `kebab-case.py` (e.g. `export-backbone.py`, `jetson-benchmark.py`,
  `verify-onnx-parity.py`). These are invoked as standalone CLI entry points via
  `.venv/bin/python3 deployment/scripts/name.py`, never imported as modules, so kebab-case does
  not conflict with Python import syntax.

When adding a new file: match the convention of the directory you're adding it to.

## File size

Root `run_*.py` scripts vary widely: most are 100-300 lines (`run_kfold_relation.py` = 109,
`run_full_eval.py` = 161, `run_lco_eval.py` = 206), but some experiment-sweep scripts run to
700-800+ lines (`run_finetune_bbox.py` = 781). There is no enforced file-size cap in this codebase
as it stands; new evaluation/metric code should follow the smaller `eval_utils/*.py` modules
(most under 150 lines, one responsibility per file) rather than the larger experiment scripts.

## Module docstrings

Every script observed opens with a triple-quoted module docstring stating: what the script does,
what protocol it implements (e.g. "blind held-out protocol," "pools 516 clips"), and a `Usage:`
line with the exact CLI invocation including default flags. Example pattern
(`run_lco_eval.py`, `run_kfold_relation.py`, `run_relation_transformer.py`):

```python
#!/usr/bin/env python3
"""One-line summary.

Longer explanation of the protocol / methodology, including any known caveats
(e.g. "Trains on CPU — the MPS training bug corrupts Adam updates").

Usage: .venv/bin/python script_name.py [--flag default]
"""
```

Follow this pattern for new scripts — the docstring is load-bearing documentation in this repo,
not boilerplate (e.g. `run_lco_eval.py`'s docstring documents the three-baseline design (A3b) and
is the only place that design is explained).

## Imports

Standard library first, third-party next, project-local imports last — observed consistently
(`run_kfold_relation.py`: `argparse`/`collections` -> `numpy`/`torch`/`sklearn` ->
`run_relation_transformer`/`eval_utils.*`). No `import *`. Project-local cross-script imports use
`import module_name as ShortAlias` (e.g. `import run_relation_transformer as R`).

## CLI scripts use argparse

Every executable entry point (14+ root scripts observed) uses `argparse.ArgumentParser()` with
explicit `--flag` defaults documented inline, not hardcoded config objects. New entry-point scripts
should follow this pattern for consistency with the rest of the repo's repro story.

## Metrics and evaluation code

- `eval_utils/` is the single binding implementation of the thesis evaluation contract — see
  `codebase-summary.md`. New metric code belongs there, not as a one-off calculation inside an
  experiment script (`eval_utils.suite.evaluate_suite()` explicitly reuses `unsafenet.metrics`
  rather than reimplementing, matching the DRY intent even though full consolidation is incomplete
  — see `codebase-summary.md` Unresolved).
- Metric naming is precise and disambiguated in docstrings (e.g. `eval_utils/metrics.py`:
  "`macro_accuracy` = mean per-class recall (the thesis 'macro' metric), NOT macro-F1") — follow
  this precision; "accuracy" and "macro" are overloaded terms in this project's own history (see
  the withdrawn-claims list in `plans/2026-09-06-thesis-2week-final/brief.md` S2-S4) and
  ambiguous naming has caused real number-attribution bugs.

## Error handling

Try/except usage is present but sparse (observed in resource-loading and optional-dependency
paths, not pervasively wrapped around every operation) — this matches the project's YAGNI/KISS
posture (per `.claude/rules/development-rules.md`) rather than defensive-everywhere coding. New
code should add error handling around I/O (file loads, external tool calls) and leave pure
numeric/metric functions unwrapped, consistent with current practice.

## Testing

`eval_utils/tests/` is the only directory with a formal test suite in this repo (pytest-style,
`run_all.py` driver, one `test_*.py` file per module: calibration, classification metrics,
conformal, cost/operating-point, edge cases, power analysis, real-artifact regression, selective
risk, statistics, suite integration, timing). Root `run_*.py` experiment scripts are not unit
tested — they are themselves the "test" in the sense of producing a reproducible numeric result
against real data, verified by independent recomputation (see the QA reports under
`plans/2026-09-06-thesis-2week-final/reports/`) rather than by CI-run unit tests. New evaluation
logic added to `eval_utils/` should include a corresponding test file in `eval_utils/tests/`.

## Output filenames must encode the configuration that produced them (MANDATORY)

Any script that writes an artifact **must** put every varied configuration knob into the output
filename, and must never write a fixed path that a differently-configured run would silently
overwrite. This is not a style preference — silent overwrite has destroyed results in this repo four
separate times:

| script | fixed path it wrote | what it clobbered |
|---|---|---|
| `run_kfold_eval.py` | `kfold-eval.json` | every cluster-seed / seed-count variant (fixed: `--out_tag`) |
| `run_kfold_relation.py` | `kfold_oof_softmax_<tag>.pt` | the canonical `--dedupe none` OOF set (fixed: policy in filename) |
| `tools/build_scene_grouping.py` | `clip_scene_map.json` | the canonical LCO camera grouping (fixed: `--cluster_seed` + suffixed default) |
| historical (S6/S9) | — | two earlier result sets, unrecoverable |

Rules:
1. Any knob exposed as a CLI flag is part of the artifact's identity → part of its filename.
2. The **canonical** configuration may keep the short, unsuffixed name; every other configuration gets
   a suffix. Never the reverse.
3. Adding a flag to a script that writes to a fixed path is incomplete work until the path is
   parameterised in the same change.
4. When a knob was previously hardcoded, exposing it must be verified behaviour-preserving at the old
   default (diff the new output against the committed artifact) before any sweep is run.

## Provenance discipline (a known gap, not a strength — do not claim otherwise)

Several artifacts in `sweep_results/relation_tokens/` currently lack a persisted log of the exact
command that generated them (see `codebase-summary.md`, `system-architecture.md` Unresolved). This
is a known, disclosed gap, not a convention to imitate: new experiment scripts should persist a
`train_<name>.log` (or equivalent) alongside every `--save_logits`-style output, recording the
full CLI invocation, so a future reader does not have to reconstruct provenance from file
modification times.

## Unresolved questions

1. No linter/formatter config (`.flake8`, `pyproject.toml` `[tool.black]`, `.pre-commit-config.yaml`)
   was found at the repo root. Whether one exists in a gitignored location or is simply absent is
   not resolved here.
