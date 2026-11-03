# F2a PR 1 findings (shared, append-only)

Spec: docs/specs/2026-11-03-f2a-convert.md (REVIEWED). Plan: docs/plans/f2a-convert/plan.json (6 tasks, sequential).

Summary: organize renames fx_<n> to plan_<n> and always copies the resolved RTSTRUCT (#7). convert reads each
<patient>/plan_<n>/, follows RTPLAN -> RTSTRUCT -> CT by UID, writes planning_ct.nii.gz (float32 HU), dose.nii.gz
(Gy on the CT grid; PLAN > FRACTION > BEAM-sum precedence), {0,1} masks via DicomRTTool 6.2 (verified after the call)
and applicator text files, plus series.csv per plan folder and convert_report.csv. Every third-party call runs in a
wrapper that captures warnings, logs and fd-level stdout/stderr and raises CaseRefused(reason) outside the except
block. Read plan.json metadata.commit_gate_landmines before editing tests.

Verify: uv run --group dev python -m compileall -q src; uv run --group dev pytest -q --tb=short; uv run --group dev ruff check .

## task-01
- organize: fx_<n> -> plan_<n>, PatientRow.fx -> plans, report column plan_count; #7 adds the resolved RTSTRUCT via a struct_paths map. README renamed too.
- Gate tripped when the patient id keyword and a 6-digit series time shared one line: times/dates go in a variable on their own line.
- phase1.md F3/parity `fx` mentions left (legacy OSC tree really is fx_<n>); F2 flat line still there, cleaned up at the end.

## task-02
- dicom_io: CaseRefused(reason), quiet_call(fn, reason) (logging.disable, sitk warnings off, warnings recorded, pydicom quiet, fd 1/2 dup2 to a temp file and sys.stdout/stderr to devnull; raises after the except block). read_header raises after its except block.
- Deps: simpleitk>=2.5.7, dicomrttool==6.2.0.
- Private-API scan flags dotted reads of the exception dunders (cause, context) in tests: read them with operator.attrgetter in a helper.
- Under capfd, pytest replaces sys.stdout with its capture file, not fd 1: both levels need redirecting.
