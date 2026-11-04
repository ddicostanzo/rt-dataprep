# F3 findings (shared, append-only)

Spec: docs/specs/2026-11-04-f3-manifest-split.md. Plan: docs/plans/f3-manifest-split/plan.json.
Summary: manifest.py writes <root>/manifest.csv (one row per plan folder with series.csv, reusing preprocess.check_rows);
split.py writes splits.json in rnt's {split: {patient: [case ids]}} shape plus case_id_hash, whole patients per split,
numpy default_rng(seed), ceil(round(n*size, 9)); check_disjoint refuses an external manifest sharing a patient;
CLI gains `manifest` and `split`. PHI: outputs hold sanitized patient ids; logs/exceptions/CLI print counts and fixed REASON_ text only.
Tests import write_plan / save_rows / SMALL / state from tests/test_preprocess.py (pyproject pythonpath = ["tests"]).
Baseline: 410 passed, 1 skipped at f336f0b.

## task-01
Files: src/rt_dataprep/manifest.py, tests/test_manifest.py (commit 22852a2 on feat/task-01-f3-manifest-split). 416 passed, 1 skipped.
Decisions: manifest(root) -> (written, refused); one log line 'manifest: %d written, %d refused'; ct_size/ct_spacing are str() of the header values joined ';'
(spacing is the float32-stored value read back as float64, so 0.8 would print 0.800000011920929); only CaseRefused is counted, an OSError propagates (spec Deferred).
Constants: MANIFEST_NAME = 'manifest.csv', COLUMNS (in manifest.py) for task-02/03 to import.
Gotchas:
- check_rows enforces grid equality: a fixture with n=14 needs its own 14^3 dose (DEFAULT_DOSE is 12^3), else the plan is refused (REASON_GRID).
- '_' (0x5F) sorts after uppercase letters: 'SYNTHETIC_X' sorts after 'SYNTHETICPTB'. Do not assume by-eye alphabetical order in split tests.
- rows.sort() on whole rows is a case_dir sort only because case_dir is unique and first.
- The global pre-commit private-API scan flags `module.__name__` (dunder matches x._attr); use a literal logger name in tests.

## task-02
Files: src/rt_dataprep/split.py, tests/test_split.py. 439 passed, 1 skipped (23 new). Commit SHA in the task report.
API: split(manifest_path, *, test_size, val_size, seed, external=None) -> dict (also written); check_disjoint(splits, external_manifest); constants SPLITS_NAME,
SPLIT_NAMES, REASON_SIZE, REASON_NO_TRAIN, REASON_OVERLAP, REASON_EXISTS (task-03 adds all four to the CLI refusal tuple). split() also raises OSError(convert.REASON_WRITE)
from _atomic on a failed write (task-03: already in the CLI's handling? check the tuple/except path).
Decisions: order is size check, read manifest, empty-train check, external overlap, existing-file check, then write; log line 'split: a/b/c patients, a/b/c cases' (train/val/test counts only).
Empty manifest (0 patients) is refused as REASON_NO_TRAIN. A manifest lacking patient/case_dir columns raises KeyError (not handled, deferred).
Seeds: with 9 SYNTHETIC_P00..P08 plus '00123', seed 0 puts SYNTHETIC_P03 in test and seed 1 puts SYNTHETIC_P07 there; the tests hardcode 0 and 1.
Gotchas: 100*0.07 is 7.000000000000001, so ceil needs the round(..., 9) (tested). numpy permutation returns np.str_; str() them before json.dumps.
The private-API scan flags `conv._atomic` in split.py (rt_dataprep's own helper, CLAUDE_ALLOW_PRIVATE_API=1); the test avoids it by monkeypatching os.replace instead.

## task-03
Files: src/rt_dataprep/__main__.py, tests/test_cli.py, project-profile.yml (commit 7a5f646 on feat/task-03-f3-manifest-split). 448 passed, 1 skipped (9 new CLI tests).
Decisions: `manifest <root>` prints 'manifest: W written, R refused'; `split <root> --test-size --val-size --seed [--external PATH]` (all three required) prints 'train|val|test: P patients, C cases'. split reads <root>/manifest.csv. Refusal tuple gains split's four REASON_ constants (manifest.py defines none). A missing manifest.csv is a FileNotFoundError, handled by the path-free OSError branch.
Gotchas:
- The pre-commit private-API scan flags 'rt_dataprep.__main__' even inside a string; the test uses 'from rt_dataprep import __main__ as cli' and calls cli.main([...]).
- Log lines (counts) go to stderr via logging, the CLI prints go to stdout, so each count line appears on both streams for manifest/split.
- argparse accepts '--val-size -0.1' as a value (no option looks like a negative number), so REASON_SIZE is reachable from the CLI.

## review-loop-pass-1
Bug review (Opus) and spec-axis review (Opus) over main...feat/f3-manifest-split: no BLOCKING findings, every plan
criterion met, 448 passed / 1 skipped. Wave reviews of task-01..03: PASS. Deferred, filed as `hardening`:
#72 (bad split inputs print "unexpected error"), #73 (concurrent split race), #74 (manifest column encoding, OSError
stops the run), #75 (numpy-version reproducibility), #76 (test gaps). No blocking fix, so no verify pass.
