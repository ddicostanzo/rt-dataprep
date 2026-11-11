# Findings — beam volumes (#280)

Spec: docs/specs/2026-11-11-beam-volumes.md · Plan: docs/plans/beam-volumes/plan.json

Summary: convert writes beam_mu.nii.gz (float32, MU per fraction, pure geometric divergent MLC+jaw aperture per control
point, IEC 61217, HFS/HFP/FFS/FFP) on the planning-CT grid with a `beam` series.json row; not-modelled plans warn
(REASON_BEAM_VOLUME) and still convert. preprocess resamples it like dose but exempt from crop/retained checks, and a
geometric run rewrites plan.parquet isocenter_i/j/k for the final CT. manifest gains optional `plan`/`beam` keys.
Tasks (sequential): task-01 module → task-02 convert → task-03 preprocess → task-04 manifest.

Append findings under `## <task-id>` (append only).

## task-01
- `src/rt_dataprep/beam_volume.py`: `beam_mu(plan, rows, ct, patient_position) -> sitk.Image` (float32, CT geometry). One `raise`
  (`_refuse`), no try/except; the AST test mirrors plan_table's. Validation runs on every row, including zero-weight ones.
- Geometry: per control point `b = M (p - c)`, `M = Rz(coll)^T Ry(gantry)^T Rz(couch) P`; since `p = origin + D diag(spacing) index`,
  `b = t + A index` is precomputed in float64 and applied in float32 per z-chunk (2M voxels), so no (N, 3) point array is built.
  The MLC leaf lookup runs only on voxels already inside the jaws (the main cost lever).
- Gotcha (spec wording): a divergent beam is *narrower toward its source* and wider away from it. AC5a's "widens toward superior"
  is the source-inferior case; the sign tests compare lit footprint at planes +-100 mm on the source axis (smaller = source side)
  and pin the one-sided jaw sides by DICOM patient axes (left +x, posterior +y, superior +z for HFS), with a non-identity
  direction (rot90 about z) run for every position.
- Mutation check done by hand: dropping the transpose on gantry/couch/collimator, flipping an HFP/FFP map row, dropping the
  direction matrix, the half-MU split, or the leaf X test each fail at least one test.
- `photon_beam` gained `SourceAxisDistance=1000.0`; plan_table rows and series.json of existing callers are unchanged (full suite green).
- `pyproject.toml`: `slow` marker + `addopts = "-m 'not slow'"`. Selecting the slow test by node id alone deselects it; use `-m slow`.
- AC14 (`uv run --group dev pytest -q -m slow -s tests/test_beam_volume.py`): 512x512x150 CT (1x1x2 mm), 2 arcs x 178 control points,
  40-pair MLC, 200x200 mm jaws: `beam_mu` wall **352 s (5.9 min)**, process peak RSS **645 MB** (ru_maxrss of the whole test process,
  which also holds the CT image and the plan). Under the 15 min soft budget: no hardening issue needed.
