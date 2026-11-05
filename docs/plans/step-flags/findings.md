# Findings: per-step preprocess flags (#150)

Spec: `docs/specs/2026-11-05-step-flags.md`. Plan: `docs/plans/step-flags/plan.json`.

Summary: replace `preprocess --legacy-rnt` (and the `LEGACY_RNT` constant) with one opt-in flag per step, named after
the output prefix it writes: `--initial-resample` (field `resample`), `--rescale`, `--ahe`, `--resample` (field
`harmonize_spacing`), `--reshape`. No step code, naming or order changes. task-01: source + tests (5 files, atomic);
task-02: README and examples.

## task-01
- Files: `__main__.py` (five step flags, `resample=args.initial_resample`, `harmonize_spacing=args.resample`),
  `preprocess.py` (LEGACY_RNT and "Legacy RNT" docstring gone), tests. `FOUR_STEPS` lives in `tests/test_preprocess.py`,
  imported by `tests/test_cli.py`. `PRE_RUN` wraps `preprocess.preprocess` to shrink `new_shape`.
- Decision: AC6 "no `--legacy-rnt` in tests/" is read as "no use of the flag". The help test (asserts absence) and the
  AC4 test (passes it, expects exit 2) keep the string by necessity; a plan wording gap, not a code issue.
- Gotcha: argparse prefix abbreviation means `--re` is ambiguous (usage error), never a silent wrong step.
- Suite on the task branch: 842 passed, 4 skipped; ruff clean.

## review-group-1
PASS: no BLOCKING. NOTE (deferred): no test asserts the PRE_RUN seam shrinks output shape; a broken seam would only slow the end-to-end tests.

## task-02
- README/examples now show the five step flags, a flag/step/prefix table and the four-flag full pipeline; the "--rescale alone refuses MR/PT" and reshape-at-native-spacing notes added.
- Suite on the session branch: 842 passed, 4 skipped; ruff clean.
