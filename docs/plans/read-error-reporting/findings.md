# Findings: #47 read-error reporting

Spec: `docs/specs/2026-11-06-read-error-reporting.md`. Plan: `docs/plans/read-error-reporting/plan.json`.

Summary: when SimpleITK fails in `preprocess.read_image`/`read_info`, a new helper `_read_failure(path)` re-reads the file
with `open(path, 'rb')` in chunks. Any OSError other than FileNotFoundError propagates (the patient's REASON_IO_ERROR via
`_preprocess_worker`); ENOMEM becomes a bare MemoryError; a clean read or FileNotFoundError keeps
CaseRefused(REASON_FILE_UNREADABLE).

Known facts:
- SimpleITK reports EACCES as "The file ... does not exist" (RuntimeError, no errno): measured in this venv.
- `read_series_csv` (preprocess.py ~605) uses bare `open` too: forced-error raisers must be path-selective.
- Every except between the readers and `_preprocess_worker` catches only CaseRefused.
- PHI: raise after except blocks so nothing is chained; never log an exception message.

## task-01

Decisions:
- `_read_failure(path)` and `READ_CHUNK` (1 MiB, a module constant so a test can shrink it) sit just above `read_image`. The helper holds the
  OSError in a local and raises it after the except block (MemoryError bare for ENOMEM), so `__context__`/`__cause__` are None.
- `read_image`/`read_info` use `read_series`'s shape: catch CaseRefused into `refusal`, call `_read_failure(path)` after the block, `raise refusal`.
  A MemoryError from `quiet_call` is not a CaseRefused, so it never reaches the helper.

Gotchas:
- Tests force errors with a path-selective `pre.open` raiser (`raising=False`, delegating to `builtins.open`); `read_series_csv` also uses bare `open`.
- Chunk test: reads end on the empty read, so a 10-byte file with READ_CHUNK=4 gives four `read(4)` calls.
- 4 skips in tests/test_preprocess.py are pre-existing (torchio not installed in this venv).

## task-02

Decisions:
- Tests only (plus the spec table row and a README troubleshooting row); no src change. Helpers `make_unreadable`/`run_two_patients`/`assert_patient_io_error` sit before the compute-failure test.
- One parametrization `UNREADABLE = ['forced', 'chmod' (skipif root)]` covers A1 and A4; A5 is a forced-only test with workers=1.
- A3 needed no new test: `test_a_series_row_naming_a_missing_file_refuses_with_file_missing` and the missing-ptv test already assert REASON_FILE_MISSING.
- Mutation check: with `_read_failure` returning immediately, all five new tests fail.

Gotchas:
- The forced raiser's PermissionError carries the path in its message, so A5 proves the worker handler never logs or reports exception text.
