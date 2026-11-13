# Findings: hardening sprint

Spec: docs/specs/2026-11-12-hardening-sprint.md. Plan: docs/plans/hardening-sprint/plan.json (six tasks; groups [01,03,05], [02,04], [06]).
Execution: sequential in the session worktree on feat/hardening-sprint (max_parallel_worktrees = 1). Each task appends `## <task-id>` with files changed, decisions, gotchas and the red-mutation runs.
Repo gate: at most 5 files per commit; `~/.claude/hooks/pre-commit.sh` must pass before each commit.

## task-01

Files changed: src/rt_dataprep/preprocess.py (index_plan_table), tests/test_preprocess.py (two tests, plus `import pathlib` and `import pyarrow as pa`).
Decisions: `data = source.read_bytes()` sits outside the try; parse is `pq.read_table(pa.BufferReader(data))` catching (OSError, ArrowInvalid, ArrowNotImplementedError, ArrowTypeError) -> False. The write stays outside the try. Docstring updated. The only other caller is the existing direct test, which passes unchanged.
Gotchas: corruption recipe `data[-24:-8] = 0xFF*16` (footer thrift just before the trailing length + PAR1) raised plain OSError in-test on the first try; no fallback to a data page or snappy block was needed. Test (b) expects the patient row ('', 'refused', REASON_IO_ERROR) (the worker's OSError handler yields a patient-level refused row, not 'failed'), and no output patient folder.
Red runs: before the change (a) failed (patient got REASON_IO_ERROR, the OSError escaped pq.read_table) and (b) failed (read_bytes never called, patient preprocessed). Both pass after.
Suite: 2062 passed, 4 skipped, 3 deselected; ruff clean; compileall clean.
