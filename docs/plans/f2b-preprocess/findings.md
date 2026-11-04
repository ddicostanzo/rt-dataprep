# F2b findings (shared, append-only)

Spec: `docs/specs/2026-11-03-f2b-preprocess.md`. Plan: `docs/plans/f2b-preprocess/plan.json`.

Summary:
- New `rt_dataprep preprocess <converted> <out> --legacy-rnt [--legacy-linear-masks]` reads F2a output through
  `series.csv` and writes a separate tree.
- Steps run on the CT in legacy order: initial resample 1 mm, rescale, final spacing 1.25 mm, crop/pad 701x701x433.
  The grid follows torchio 1.2.1: `floor` size and centre-preserving origin; crop/pad puts `ceil(d/2)` at the start.
- Dose and masks are resampled once onto the final CT image, with nearest neighbour (uint8) for masks or linear
  (float32) under legacy_linear_masks. Names are prefixed with no ahe.
- Failure table follows convert: a partial folder per patient, and REASON_ constants.
- Every row is checked before the first write of a plan. REASON_CROP refuses label truncation.

Guardrails:
- PHI: fixed REASON_ constants. No paths, IDs or UIDs in logs or exceptions. Synthetic IDs have no digits or
  underscores.
- Commit gate: at most 5 staged files, and no dotted name whose last part starts with `_` in added lines (docs
  included). tests/test_organize.py:24 and tests/test_dicom_io.py:85 trip the MRN rule if edited.
- No test pads to 701x701x433: use dataclasses.replace(LEGACY_RNT, new_shape=<small>).
- Third-party calls go through dicom_io.quiet_call. A CaseRefused with REASON_INTERNAL re-raises to an abort.
