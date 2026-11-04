# Continuation prompt for next session: F2a PR 2 (other series, images/, REG, misc_files)

## Anchor state
- Repo `/home/dominic/Repositories/rt-dataprep`, target `main`, synced at `e789e5a` "F2a PR 1: plan_<n>, #7,
  native-grid convert (CT, dose, masks) (#19)" (squash merge).
- The F2a PR 1 worktrees (`rt-dataprep-f2-convert`, `worktrees/task-01..06`) and their local branches are removed. The
  remote branch `origin/feat/f2-convert` still exists (merged; safe to delete).

## What shipped
- PR #19: organize writes `plan_<n>` folders and fixes #7. `convert` writes per plan, on the native CT grid,
  `planning_ct.nii.gz`, `dose.nii.gz`, masks (DicomRTTool 6.2), applicator point files, `series.csv` and
  `convert_report.csv`, with atomic patient folders. Files: `src/rt_dataprep/convert.py`, `masks.py`, `dicom_io.py`,
  `organize.py`, `__main__.py`; tests in `tests/`.
- Review-round fixes: a non-axial planning CT is refused, each mask must sit on exactly the CT slices its contours
  map to, and a kept ROI on another frame of reference refuses the plan (closes #29).

## Open items / next stage
- **Next: PR 2** per `docs/specs/2026-11-03-f2a-convert.md` (Delivery: PR 2): other series, registration and
  `reg_matrix`, `images/`, reading `misc_files/`, a refused plan's CT and plan-less patients. Criteria A2.4, A2.4b
  (the `images/` part), A2.4c. Start a new `/dev` seeded with this file and the spec. `exclude_models` in
  `convert()` is already in place for it.
- Hardening issues relevant to PR 2: #18 (DicomRTTool links only the last referenced series), #1 (resample REG-linked
  MR/PET into the planning CT frame), #17 (imaging-only cases), #2 (parallel header reading; see #20, quiet_call
  is process-wide).
- Other open hardening issues from PR #19: #20-#28, #30-#33. #33 (the broad except in convert_plan hides coding bugs
  as REASON_MISSING_TAG) cost debugging time this session, so it is worth fixing early.

## Guardrails that carry over
- Commit gate (`~/.claude/hooks/pre-commit.sh`): never write a dotted name whose last part starts with an underscore
  on an added line, in any file type (docs and findings.md included). Lines `tests/test_organize.py:24` and
  `tests/test_dicom_io.py:85` trip the MRN rule if edited. See `docs/plans/f2a-convert/plan.json`
  `metadata.commit_gate_landmines`.
- At most 5 staged files per commit. PHI rules in `project-profile.yml` `compliance`: fixed REASON_ constants, no
  path, ID, UID or name in messages or logs; synthetic test data only.
- DicomRTTool 6.2 traps (z-only slice matching, last-series linking, silent all-zero masks, no frame check):
  vault note `Engineering/Patterns/medical-imaging/dicomrttool-6-api-and-mclahe-deps.md`.
- Tests: `uv run --group dev pytest -q --tb=short` (157 pass at `e789e5a`); lint `uv run --group dev ruff check .`.

## First ask
Start `/dev` for F2a PR 2 and run its interview. Don't push, merge or close anything until the user confirms in that
session.
