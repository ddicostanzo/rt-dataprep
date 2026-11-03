# Handoff: F2a PR 1 (convert), halted at the review round

**Why halted.** Escalation counter tripped: *a review finding in the same category returned twice on one PR*. The
category is planning-CT geometry that `_check_volume` accepts and a later step then handles silently wrong.
1. Wave-2 review: a sheared or gantry-tilted CT passed and was written with wrong geometry. Fixed in `5a316b2` (slices
   must lie on the normal through the first slice).
2. `/code-review` round (BLOCKING, unfixed): a **coronal or sagittal** planning CT passes `_check_volume`. DicomRTTool
   6.2 picks a contour's slice by `ImagePositionPatient[2]` only (`DicomRTTool/ReaderWriter.py` ~L975-981, ~L1427).
   With a slice normal other than ±z, every contour lands on slice 0. The mask is non-empty and its grid is copied
   from the CT, so the `_checked` helper in masks.py passes it and the plan reports **converted** with wrong labels. Reviewer probe:
   `/tmp/claude-1000/probe/coronal.py` (a 3×4×5 box gave 20 voxels on slice 0 instead of 60 on slices 1-3). Axial
   ±z and mirrored in-plane axes were placed correctly (`/tmp/claude-1000/probe/orient.py`). Dose is unaffected,
   because SimpleITK uses the full direction matrix.

## Goal

Merge PR #19 (`feat/f2-convert` → `main`): F2a PR 1 per `docs/specs/2026-11-03-f2a-convert.md` (Delivery: PR 1) and
`docs/plans/f2a-convert/plan.json`. The user chose full auto-merge after one clean review round. PR 2 (other series,
`images/`, REG, `misc_files`) is a later `/dev`.

## State

- Worktree `/home/dominic/Repositories/rt-dataprep-f2-convert`, branch `feat/f2-convert`, pushed. PR #19 is open.
- All six tasks are done. 153 tests pass, ruff is clean, and the commit gate passes.
- Per-task worktrees `../worktrees/task-01..06` and their `feat/task-0N-f2-convert` branches still exist; clean them
  up at merge (`.dev-worktrees.json` was never filled in, so remove them by hand).
- Review round done: the spec axis met 51 of 52 criteria (no blocking). The correctness review found 1 BLOCKING (above)
  and 3 deferred. Everything is recorded in `docs/plans/f2a-convert/findings.md`, sections `review-group-1..4`.

## Hypotheses not yet ruled out (fix approach)

1. **Narrow guard.** Refuse unless `abs(normal[2]) >= 1 - DIRECTION_ATOL` in `_check_volume` (convert.py), with a coronal
   CT test and one line in spec step 2.2. This is cheap, but the counter suggests guards are being added one case at a time.
2. **Close the class, preferred.** Add an independent check of mask placement after DicomRTTool: transform every
   CLOSED_PLANAR contour vertex (patient mm) to a continuous index with the CT image
   (`TransformPhysicalPointToContinuousIndex`) and require each contour's slice index to hold voxels in that ROI's mask.
   Alternatively, compare each mask's per-slice voxel presence against the slices its contours sit on. This catches
   z-only matching, out-of-range snapping to end slices (deferred finding 2 below), and any other DicomRTTool geometry
   assumption, not only orientation. Keep hypothesis 1 as well.
3. **Enumerate DicomRTTool's assumptions first.** Read `ReaderWriter.py` and `Services/` for every place it uses
   patient z, the last series, or a frame, and write each down as a guard or a test before coding. Known so far:
   z-only slice matching, last-referenced-series linking (#18), ROIs in another frame snapped onto the CT.

## Deferred findings to file as `hardening` issues (step 2.9.5, not yet done)

From the wave reviews (findings.md) and the review round:
- `quiet_call`: not thread-safe (process-wide state). BaseException passes through with its original message.
  `_fds_quiet` leaks fds if `TemporaryFile` fails. Python 3.14 context-aware warnings.
- #7 test lacks a case where the RTSTRUCT's own study holds other files.
- The broad `(AttributeError, KeyError, IndexError, TypeError, ValueError)` catch in `convert_plan` turns coding
  bugs into `REASON_MISSING_TAG`.
- No tests for non-identity CT orientation or negative/absolute dose offsets (the writer is identity-only). The BLOCKING
  fix should add the orientation one.
- A mixed POINT and CLOSED_PLANAR ROI is kept, and its POINT items reach DicomRTTool unchecked.
- Only the broken-RTSTRUCT half of A2.10 catches a regression in the wrapper; the truncated-CT error carries no path.
- `series.csv` and `convert_report.csv` are opened without `encoding='utf-8'`.
- Mask rows take `ReferencedFrameOfReferenceSequence[0]` as their frame. The CLI re-raises unexpected ValueErrors. A
  missing org root leaves an empty out root.
- `read_header` is not routed through `quiet_call` (spec axis, PARTIAL).
- `os.replace` in `_atomic` and `write_applicator` can raise a raw OSError with a path when `convert_plan` is called
  directly.
- Input-side OSErrors are also reported as `REASON_IO_ERROR`.
- An ROI in another frame of reference is rasterized onto the CT grid. This is closed if hypothesis 2 is done.
- A rerun overwrites `convert_report.csv` with every patient skipped, so earlier refusal reasons are lost.
- Spec should list the extra conservative checks (pixel-spacing and frame uniformity, ROINumber match).

## Next session

Load the `pickup` skill with this file, then `/dev continue` (the `dev` skill, Phase 3). Fix the BLOCKING finding
yourself. The model routing rules say not to send it back to Sonnet. Then run one verification pass on that fix only,
file the deferred issues above, update the PR body with their numbers, and merge (`/dev merge`) with the standard
cleanup. Then write the PR 2 continuation prompt (Step 3.9). Guardrails carry over: commit gate rules in
`plan.json` metadata.commit_gate_landmines, at most 5 staged files per commit, and PHI rules.
