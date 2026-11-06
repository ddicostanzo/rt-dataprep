# Preprocess: a file-system error during an image read is the patient's I/O error (#47)

## Problem

`preprocess.read_image` and `preprocess.read_info` wrap SimpleITK in `quiet_call(..., REASON_FILE_UNREADABLE)`.
SimpleITK reports every read failure as a `RuntimeError` without an errno. It reports a file it may not open (EACCES)
as "does not exist" (checked on SimpleITK in this repo's venv). A failing disk or a permission problem therefore
refuses each plan or series as a bad input file, and the run carries on. The report hides a machine problem behind
many per-plan refusals.

The out-of-memory half of #47 already shipped (PR #157). `quiet_call` re-raises a recognized allocation failure as
`MemoryError`, which stops the run with REASON_INTERNAL. This spec doesn't change that.

## Behaviour

When SimpleITK fails inside `read_image` or `read_info`, and only then, preprocess reads the same file with plain
Python in fixed-size chunks to the end, outside `quiet_call`. The result decides how the failure is reported:

- **Python raises an OSError other than FileNotFoundError** (EACCES, EIO, EISDIR, ...): that OSError propagates. The
  existing worker handler turns it into the patient's REASON_IO_ERROR row: no plan rows for that patient, the partial
  folder is removed, and the run carries on with the next patient. Same flow as an OSError from a write today.
- **FileNotFoundError**: the read stays CaseRefused(REASON_FILE_UNREADABLE). A file listed in series.csv but missing
  is broken input, not a failing disk (owner decision). `check_rows`/`check_study_rows` already refuse a listed file
  that is missing (REASON_FILE_MISSING) before any read, so this branch is reached only when the file vanishes between
  that check and the read.
- **ENOMEM** (an OSError that `dicom_io.allocation_failure` recognizes): raised as `MemoryError`, so the run stops, as
  for an out-of-memory read today (owner decision).
- **Python reads the whole file without error**: CaseRefused(REASON_FILE_UNREADABLE), as today. The file is
  malformed.
- **The check itself raises MemoryError**: it propagates, which stops the run.

### The helper

One preprocess-local helper, `_read_failure(path)`, is called by both `read_image` and `read_info` and only after they catch
CaseRefused. It is called after the except block, then `raise refusal`, the same shape as `read_series`. It reads the
file in chunks to the end, then:
- returns on a clean read, so the caller raises its CaseRefused;
- swallows FileNotFoundError and returns;
- raises MemoryError when `allocation_failure(exc)` is true;
- re-raises any other OSError.

Tests target the helper directly (A3, A3b) and spy on it for A6. To force an error they make SimpleITK fail with a
garbage file and patch only preprocess's own `open` (`monkeypatch.setattr(pre, 'open', ..., raising=False)`), never
`builtins.open`. The MemoryError for ENOMEM is raised bare after the except block, so no path-naming OSError is attached.

The happy path does no extra I/O. The check runs only after a SimpleITK failure, and reads in chunks so a large file
doesn't need extra memory. It doesn't run when quiet_call has already raised MemoryError (an allocation failure),
because that is the machine's problem and is handled already.

### Callers

- **preprocess plans, studies and series.** An OSError already propagates to the patient, as the `preprocess_one` and
  `preprocess_study` docstrings say (every except in between catches only CaseRefused). A disk error in one series
  therefore fails the whole patient, including its good plans. `read_series` (registered series) only catches CaseRefused, so an OSError now
  reaches the patient instead of leaving that series out. That is intended: a disk error is not a bad series.
- **manifest.** `manifest.py` imports `read_info` and already catches `(CaseRefused, OSError)` at every call, so its
  behaviour is unchanged.

### PHI

The re-raised OSError's message names the path. It is never logged or written: the preprocess worker logs only the
fixed REASON_IO_ERROR, and manifest catches it silently. The SimpleITK message stays discarded inside `quiet_call`.
The Python OSError is raised outside the except block that caught the SimpleITK failure, so it chains nothing from
SimpleITK.

## Acceptance

- A1. A plan file with its read permission removed (POSIX; skipped when run as root, as in tests/test_dicom_io.py) gives one patient-level REASON_IO_ERROR row for
  that patient and no plan rows. The run continues, and a second patient in the same run is preprocessed.
- A2. A truncated or garbage NIfTI still refuses its plan with REASON_FILE_UNREADABLE (unchanged).
- A3. A file named in series.csv but missing still refuses its plan with REASON_FILE_MISSING (unchanged). A unit test of
  the classification shows that a FileNotFoundError from the Python read keeps CaseRefused(REASON_FILE_UNREADABLE).
- A3b. A forced PermissionError from the Python read (monkeypatched, so it also runs as root) gives the patient's
  REASON_IO_ERROR. A forced ENOMEM OSError raises MemoryError.
- A4. A registered series (with a valid reg matrix, so it is read) whose file cannot be read for permission reasons gives the patient's REASON_IO_ERROR,
  where before it was left out.
- A5. With workers=1, no log line (caplog) and no text in preprocess_report.csv contains the file's path, and the patient
  folder name appears only in the report's existing patient column.
- A6. The happy path makes no extra reads: when SimpleITK succeeds, the Python check doesn't run (a spy test).
- A7. manifest's behaviour is unchanged for an unreadable file (its existing tests pass).

## Deferred

- Peak-memory measurement at 701x701x433 on real data: #209.
- The same classification for convert's DICOM, CT, dose and series reads: #210.
- An EIO the OS reports only to SimpleITK and not to a second Python read (transient errors) still reads as
  unreadable.
- Windows read-only and permission semantics (no Windows runner).
- A plan or study folder that can be listed but not entered (r without x): `is_file()` may raise or return False,
  depending on the Python version (owner decision: defer; tracked in #210).
- A FIFO swapped in for a listed file between the check and the read could make the Python `open()` block where
  SimpleITK failed fast. Race-only, unverified.
- For `read_info`, a header-only failure can't come from an EIO past the header, so reading to the end is wasted I/O.
  It is a known cost on the failure path only; there is no parameter for it.
