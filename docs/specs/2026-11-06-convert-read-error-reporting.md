# Convert: a file-system error while reading an input is the patient's I/O error (#210)

Date: 2026-11-06 · Branch: `feat/convert-io-errors` · Status: REVIEWED

## Background

Split from #47 (PR #211), which taught preprocess to tell a disk or permission error from a malformed file
(spec `docs/specs/2026-11-06-read-error-reporting.md`). Convert still reads its inputs through `quiet_call` and
`dicom_io.read_header_core`, which turn every failure into a fixed refusal, so EACCES or EIO on an input reads as
"unreadable DICOM" for one plan or one file.

Why now (owner): pre-run sanity. A mis-mounted or permission-broken share should show up in `convert_report.csv` as an
I/O error per patient, not as many "unreadable" plans the owner then chases one by one. No real incident so far.

## Status quo

- `_read_one` catches the `OSError(REASON_UNREADABLE)` that `read_header_core` raises for any failure, EACCES and EIO
  included, and counts the file as unreadable. In a plan folder that refuses the plan; in `misc_files/` the file's study
  is salvaged with a force read (#36) and the patient carries on.
- `_read_ct`, `_read_dose`, `series.py`'s series read, `_salvage_study` and masks' `_place_struct` (via `quiet_call`)
  turn an EACCES or EIO into their per-plan or per-series refusal reason.
- Folder checks use `Path.is_dir()` / `Path.is_file()`. For a folder that can be listed but not entered (r without x),
  Python 3.13 (the project venv) raises PermissionError, which already reaches the patient's REASON_IO_ERROR (or, at
  the root, the CLI's fixed "file system error on the source or output root"). Python 3.14 returns False, so the
  entries are skipped silently: probed on this machine with 3.13.16 and 3.14.4.
- `_convert_worker` already turns any OSError from a patient into one REASON_IO_ERROR row for the patient, logging
  only the fixed reason; `__main__` already turns an OSError at the root into its fixed message.

## Constraints

- Safety: no clinical value changes. The change only turns some refusals into a different refusal (whole patient instead
  of one plan); converted output for readable inputs is byte-identical. Failure stays loud: a refusal row, never a
  silent skip.
- PHI: an OSError names the path, which holds the patient id. It is never logged or written; only the fixed
  REASON_IO_ERROR text reaches the log and the report, as today. No new place PHI lands.
- FHIR: not applicable. Audit: the existing `convert_report.csv` row is the record.
- Performance: the happy path does no extra I/O; the Python re-read runs only after a reader has failed.

## Premises (confirmed)

- A file-system error is the patient's, wherever the file sits (plan folder or `misc_files/`): nothing is kept for a
  patient whose tree the machine could not fully read. Same rule as #47.
- A broken share gives one REASON_IO_ERROR row per patient; the run is not stopped early (no circuit breaker).
- organize is out of scope and unchanged.

## Approaches

- **Chosen: re-read after failure.** Move `preprocess._read_failure` to `dicom_io.read_failure(path)` and call it at each
  convert read site after the reader fails. Reuses the #47 helper; zero cost on the happy path.
- Rejected: pre-flight access check before every read (costs a read of every file, a check/use race, misses an EIO deep
  in a file).
- Rejected: a `paths=` argument on `quiet_call` (widens a contract also used for writes and mask runs).

## Behaviour

### `dicom_io.read_failure(path)` (moved, unchanged)

`preprocess._read_failure` and `READ_CHUNK` move to `dicom_io` as `read_failure` and `READ_CHUNK`, same behaviour: read
`path` with `open(path, 'rb')` in fixed chunks to the end; return on a clean read or FileNotFoundError; raise a bare
`MemoryError` when `allocation_failure` recognizes the OSError; re-raise any other OSError, outside the except block.
preprocess calls the moved helper and its behaviour is unchanged. Its tests move their patches from `pre.open`,
`pre.READ_CHUNK` and `pre._read_failure` to `dicom_io.open`, `dicom_io.READ_CHUNK` and `dicom_io.read_failure`.
Every caller goes through the module (`dicom_io.read_failure(...)`, as `_read_one` calls `dicom_io.read_header_core`),
so a test can patch or spy on the one seam.

### Read sites

After the reader fails, and before the existing refusal is raised or recorded, `read_failure` runs on each input file
of that read:

| site | files re-read | refusal kept for a malformed file |
|---|---|---|
| `convert._read_one` (header read) | the file | file counted unreadable |
| `convert._read_ct` | every CT slice | REASON_CT_UNREADABLE |
| `convert._read_dose` (the `extract` read) | the RTDOSE | REASON_DOSE_UNREADABLE |
| `series.py` series read | every slice of the series | series skipped (returns None) |

At each site `dicom_io.read_failure(...)` runs after the except block that caught the reader's failure, not inside
it (as `preprocess.read_image` does), so no reader exception is chained. An OSError raised by `read_failure` propagates to `_convert_worker`: the patient gets one REASON_IO_ERROR row and no
plan rows, and nothing of the patient is published (existing worker path). ENOMEM becomes `MemoryError`, which stops
the run as today: `_convert_worker` turns it into the patient's REASON_INTERNAL row and `convert` raises
`ConvertAborted` (the same path as #47). Only the dose `extract` read is classified, not the `build` step (no file read there).

### Folder checks

A helper in `dicom_io` stats an entry with `path.stat()` (so a test can patch `Path.stat` path-selectively) and returns Python 3.13's `is_file()` / `is_dir()` result on any version: False for
ENOENT, ENOTDIR, EBADF and ELOOP, and for a ValueError (an embedded NUL), the type check otherwise; any other OSError (EACCES, EIO, ...) is raised. Convert's
folder checks use it: the patient-folder listing of the organized root, the plan-folder listing of a patient, the
`misc_files/` check, the file listing in `read_folder_headers`, and the `misc_files/` listing for salvage.

- At the root, the raise reaches `__main__`'s existing `except OSError` (fixed message, exit 1).
- Inside a patient, it reaches `_convert_worker` (the patient's REASON_IO_ERROR row).

## Acceptance criteria

POSIX permission tests are skipped when run as root (as in `tests/test_dicom_io.py`); each has a monkeypatched twin
that also runs as root. Forced-error tests patch the module's `open` path-selectively (other paths go to
`builtins.open`), in binary mode.

- A1. A plan's CT slice with read permission removed (caught by the header re-read in `_read_one`, since `dcmread` fails
  at open; only A3's forced errors reach the CT, dose and series sites) gives exactly one row for the patient, REASON_IO_ERROR, and
  no plan rows; the patient has no output folder.
- A2. A `misc_files/` DICOM file with read permission removed gives the patient's REASON_IO_ERROR (today: salvaged).
- A3. A forced PermissionError (or EIO) from the Python re-read after each of the four readers fails gives the
  patient's REASON_IO_ERROR: one test per site in the table.
- A4. A file at each site that the reader rejects but whose header reads (garbage for the header read; truncated
  pixels for CT and series; a NaN DoseGridScaling for dose) keeps today's refusal (existing tests pass unchanged; one new garbage-file
  test per site where none exists).
- A5. A forced ENOMEM OSError from the re-read gives the patient's REASON_INTERNAL row and `convert` raises
  `ConvertAborted` (workers=1).
- A6. With `Path.is_file` / `Path.is_dir` patched to return False (Python 3.14's behaviour), a plan folder with mode
  0o444 (listable, not enterable) gives the patient's REASON_IO_ERROR. A real chmod alone passes on 3.13 with no code
  change, so the patch is part of the criterion. Root-safe twin: `Path.stat` forced to raise PermissionError for that
  folder's entries only.
- A7. The folder helper: False for a missing entry, a dangling symlink, an entry under a regular file (ENOTDIR) and a
  name with an embedded NUL; raises PermissionError for EACCES.
- A8. With `Path.is_file` / `Path.is_dir` patched to return False, the organized root itself with mode 0o444,
  holding one patient folder, makes the CLI exit 1 with "file system error on
  the source or output root", and the output root does not exist afterwards.
- A9. With workers=1, no log line (caplog) and no text in `convert_report.csv` contains the failing file's path or the
  patient folder name except the report's own patient column.
- A10. The happy path makes no re-read: with every reader succeeding, `read_failure` is not called (a spy).
- A11. preprocess's existing read-error tests pass against the moved helper.

series.py's module docstring and `read_series` docstring, which say no exception escapes, are updated: an OSError
from `read_failure` now does.

## Failure handling

- Malformed file: today's refusal, unchanged.
- Vanished file (FileNotFoundError on the re-read): today's refusal, unchanged.
- Permission or disk error: the patient's REASON_IO_ERROR row; the run continues.
- Out of memory: MemoryError, the run stops (as today).

## Deferred

- An EIO that only the first reader sees and the Python re-read does not (transient): stays the per-plan refusal.
- A FIFO swapped in for a listed file between listing and re-read could block `open()` (as #47).
- Windows permission semantics (no Windows runner).
- organize's reads (out of scope by decision).
- Every unreadable `misc_files/` entry (routine, #36) now costs one Python re-read of that file.
- `_salvage_study` and masks' `_place_struct` get no re-read (owner): the header re-read has already classified both
  files, so only a file that changed between the two reads reaches them, as above.
- The CT/series re-read reads every slice after a failure; a malformed-series refusal on a large series costs one extra
  read of its files.

## Delivery

One PR on `feat/convert-io-errors`, /dev auto-merge after a clean review round (owner's choice). No CI; the pre-push
full pytest run is the gate. Closes #210.
