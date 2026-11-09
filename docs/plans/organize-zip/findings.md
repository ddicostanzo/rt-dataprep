# organize-zip findings

Spec: docs/specs/2026-11-09-organize-zip.md. Plan: docs/plans/organize-zip/plan.json (3 tasks, sequential).
Real data (OSC probe): zips are Deflate only, ~2.5k DICOM members each, unencrypted. Happy path is method 8.
Append-only: each task adds `## <task-id>` with files changed, decisions, gotchas.

## task-01

Files: src/rt_dataprep/organize.py (ZipRefused, REASON_ZIP_TOO_LARGE/UNREADABLE/LOCKED, MAX_UNZIP_BYTES/MAX_ZIP_MEMBERS, `_has_preamble` -> 'dicom'/'zip'/None, `_read_one` needs 'dicom', `_zip_scan`, `_unzip`, `_stage_zips`), tests/test_organize_zip.py (new, 48 tests).

Decisions:
- `_stage_zips(paths, staging)` returns `(staged, zips, zip_members, zips_skipped)`; `zips` and `zip_members` count what was staged, so skipped zips are only in `zips_skipped` (total found = zips + zips_skipped).
- Member number is the `infolist()` position (directories keep their number), so a zip whose first member is a directory stages files from `2`.
- Over-cap total outranks an unreadable zip in a zip-only folder (TOO_LARGE); the cap sums cover zips that scan cleanly. Beside loose DICOM an over-cap total skips every zip.
- Pre-scan (central directory + allow-list) runs on all zips before any write; a bad zip then is refused/skipped before extraction. A zip failing mid-stream removes `staging/<zip #>` (and all earlier ones on refusal). `staging` itself is the caller's and is left in place; it is created lazily, only by the first member written.
- ZipRefused is raised outside any `except` block (and `from None`), so `__context__` is None as well as `__cause__`; the `except _ZIP_ERRORS` clauses bind no exception.
- `_has_preamble` is read twice per file once task-02 wires this in (here and in `_read_one`); `_stage_zips` takes the walked paths and classifies them itself. task-02 may want to pass the kinds through.

Gotchas:
- `zipfile.writestr` resets `flag_bits`, so encrypted/method-9 test zips are made by patching both the central and local headers.
- Writing duplicate member names warns (UserWarning); the traversal test filters it.

## review-1
No BLOCKING. Deferred (file as hardening in 2.9):
- W1: corrupt central dir with negative header_offset passes the allow-list; zf.open raises OSError (Errno 22) → REASON_IO_ERROR instead of unreadable/skip. Fix: `or i.header_offset < 0` in the allow-list.
- N2: _unzip re-reads infolist() instead of the scanned list (TOCTOU on a changing source).
- N3: bzip2/lzma decompress output caps are a recent CPython fix; early 3.12.x may balloon memory on a bzip2 bomb. Document min patch or allow only 0/8.
- N4: no test pins "stream larger than declared file_size → unreadable, nothing staged" (verified by hand).
Task-02 must: wrap _stage_zips in task A's try/except BaseException (an OSError leaves staged files); fold the double _has_preamble read.

## plan-fix-1 (task-02)
AC18 amended: test_organize.py helpers listing/hidden/snapshot ignore the persistent .organize.lock (AC16 keeps it on disk; unlinking an flock file races). Plan bug found by the task-02 implementer before editing.
