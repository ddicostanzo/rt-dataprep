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

## task-03
- tests/synthetic_dicom.py: write_rtstruct_rois(path, ct, rois, contour_images=True, sop_uid=None, **meta); roi dicts have number, name, kind (box|point|empty|channel), interpreted; box=((z0,z1),(y0,y1),(x0,x1)) inclusive indices. write_rtplan gains label=.
- DicomRTTool 6.2 rounds vertices half-up and fills vertex voxels (boundary-inclusive): box vertices sit at the centres of the extreme voxels; half-step corners overcount (40 vs 24). Real contours drawn on voxel edges will rasterize one voxel larger per side: inherited legacy behaviour, recorded for review.
- arg_max=False mask is (z, y, x, nROI+1) with channel 0 background.

## review-group-1 (tasks 01-03): PASS, no BLOCKING
- WARNING: quiet_call is not thread-safe for concurrent calls (process-wide state) — lock or document before parallel convert (#2).
- WARNING: BaseException (KeyboardInterrupt/SystemExit) passes through quiet_call with its original message.
- WARNING: #7 test lacks a case where the RTSTRUCT's other study also holds other files.
- NOTE: _fds_quiet dups fds before creating the temp file (fd leak if TemporaryFile fails). NOTE: Python 3.14 context-aware warnings. NOTE: dicom_io imports SimpleITK at module level.

## task-04
- convert.py: convert(org_root, out_root, *, exclude_models) -> ConvertReport(rows); convert_plan(plan_dir, case_dir) -> PlanResult(files, ct_series_uid) is the seam for masks; write_atomic raises OSError(REASON_WRITE) so output failures are patient-level.
- The CLI clause of task-04 criterion 1 (exit 1 with the reason) lands in task-06 with the convert subcommand (__main__ is task-06's file).
- Two PLAN doses refuse even when a FRACTION exists (strict reading). Plan folders sorted numerically.
- Missing-tag refusal catches AttributeError/KeyError/IndexError/TypeError/ValueError around the per-plan path: a coding bug surfaces as REASON_MISSING_TAG, not a crash.
- sd.write_rtstruct writes no ContourImageSequence, so such plans are refused: organize->convert fixtures (A2.9) must use write_rtstruct_rois(contour_images=True).
- Rescale test needs mixed slopes to tell Cast from SetOutputPixelType. Mutation checks (5) all fail the suite.

## review-group-2 (task-04)
- BLOCKING fixed by orchestrator: a sheared/gantry-tilted CT passed _check_volume and was written with wrong geometry; slices must now lie on the normal through the first slice (test_sheared_ct_is_refused). Spec step 2.2 updated.
- Verified by probe: CT under 4 orientations + shuffled files exact; dose with relative/negative/absolute offsets identical; DoseGridScaling and PixelSpacing order right.
- WARNING (deferred): broad (AttributeError, KeyError, IndexError, TypeError, ValueError) catch turns coding bugs into REASON_MISSING_TAG.
- WARNING (deferred): no tests for non-identity CT orientation or negative/absolute dose offsets (writer is identity-only).
- NOTE: an RTDOSE not naming the plan is silently ignored; a non-DICOM file (.DS_Store) in a plan folder refuses the plan.
