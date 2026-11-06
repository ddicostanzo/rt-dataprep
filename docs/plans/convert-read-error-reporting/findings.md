# Findings: convert read-error reporting (#210)

Spec: docs/specs/2026-11-06-convert-read-error-reporting.md. Plan: docs/plans/convert-read-error-reporting/plan.json.

Summary: after a convert input reader fails (header, CT, dose, other series), re-read the file(s) with
dicom_io.read_failure (moved from preprocess). An OSError other than FileNotFoundError propagates to _convert_worker,
which writes the patient's REASON_IO_ERROR row. Folder checks use a dicom_io stat helper with Python 3.13's
is_file/is_dir semantics on every version. Tasks: task-01 (move the helper) -> task-02 (four read sites) -> task-03 (folder checks).

Append under `## <task-id>`: files changed, decisions, gotchas.

## task-01

Files changed: src/rt_dataprep/dicom_io.py (read_failure, READ_CHUNK added after allocation_failure), src/rt_dataprep/preprocess.py
(helper and constant removed; read_image and read_info call dicom_io.read_failure through the module), tests/test_dicom_io.py (helper
tests added), tests/test_preprocess.py (patches retargeted to dicom_io.open / dicom_io.read_failure).

Decisions:
- The two helper-only preprocess tests (chunked read, missing file) moved to test_dicom_io.py; the reader-level preprocess tests
  (garbage file, vanished file, EACCES/EIO, PermissionError, ENOMEM, no-re-read on success) stay, with patches on dicom_io.
- New dicom_io tests also cover a file vanishing between the failure and the re-read, and that a propagated OSError is the same
  object, unchained, opened in 'rb'.

Gotchas:
- dicom_io has no module-level `open`; monkeypatch.setattr(dicom_io, 'open', ..., raising=False) shadows the builtin for that
  module only, so the path-selective raiser must hand every other path to builtins.open.
- Full suite takes about 3m40s (1116 passed, 4 skipped).

## review-1
- No BLOCKING. WARNING: vanished-file tests (test_dicom_io ~:664, test_preprocess ~:288) don't assert the patch was reached (`modes == ['rb']`). NOTE: fail_python_open duplicated in test_dicom_io/test_preprocess (shared helper). NOTE: forced-PermissionError preprocess test checks type only (covered). All deferred.
