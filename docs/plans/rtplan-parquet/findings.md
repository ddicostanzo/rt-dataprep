# Findings: photon RTPLAN plan.parquet

Spec: docs/specs/2026-11-11-rtplan-parquet.md · Plan: docs/plans/rtplan-parquet/plan.json

Summary: convert writes plan.parquet (one filled-forward row per control point, pyarrow schema) for photon EBRT plans;
non-photon and malformed photon plans get no parquet and still convert. preprocess copies it unchanged.
Tasks: task-01 plan_table module (+pyarrow, photon fixture) → task-02 convert integration → task-03 preprocess + docs.

Append findings under `## <task-id>` (append only).

## task-01
- Commit 75abb8e (squashed). Files: pyproject.toml, uv.lock (pyarrow>=26.0.1), src/rt_dataprep/plan_table.py, tests/synthetic_dicom.py, tests/test_plan_table.py (29 tests).
- API: plan_table.SCHEMA, REASON_PLAN_TABLE, plan_rows(plan, ct) -> list|None, to_table(rows), write_plan_table(table, path). Task-02 imports these names.
- Every tag read uses .get, so a missing tag is CaseRefused(REASON_PLAN_TABLE), not a pydicom AttributeError. No try/except, quiet_call or _pydicom_quiet (AST-checked test).
- Treatment beam = TreatmentDeliveryType TREATMENT/absent/empty; OPEN_TREATMENT, CONTINUATION dropped like SETUP. Missing RadiationType = not photon → None.
- Any device type starting with MLC counts; two MLC devices refuse. Extra refusals: jaw list != 2 values, IsocenterPosition != 3, beam with no control points.
- write_photon_rtplan(path, struct_sop_uid, sop_uid, frame_uid, *, beams=[photon_beam(...)], fractions, ...): call by keyword (write_rtplan's 3rd positional is positions_mm). BeamMeterset in a beam dict is moved into the fraction group.
- nan gantry triggers a pydicom DS warning; filtered in those tests.
- Not checked: NumberOfControlPoints vs ControlPointSequence length.

## review-1
PASS (no BLOCKING). Fixed in aeffd85: empty WedgeSequence item returned the Dataset as wedge_id (ArrowTypeError outside _header_call).
Deferred (file in 2.9): NumberOfControlPoints/ControlPointIndex not checked; duplicate BeamNumber/ReferencedBeamNumber last-wins;
wedge_orientation typed string though VR DS (spec choice); CONTINUATION beams dropped (MU under-count, owner question);
empty RadiationType silently drops the parquet. Task-02 fixtures must call write_photon_rtplan by keyword.

## task-02
- Commit 52b2aaf (squashed). convert.py: PLAN_NAME = 'plan.parquet'; _header_call(plan_rows) in try/except CaseRefused → log.warning REASON_PLAN_TABLE; to_table outside _header_call, before mkdir; parquet written last via _atomic; _series_rows has an explicit last branch for the plan row (series_description/UIDs/date from the plan header).
- make_plan(plan_writer=sd.write_rtplan); only the default writer gets positions_mm. 8 new tests (1617 passed total).
- write_photon_rtplan writes no FrameOfReferenceUID by default; tests pass frame_uid explicitly and cover the '' fallback.
