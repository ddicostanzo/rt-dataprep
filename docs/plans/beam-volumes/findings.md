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

## task-02
- `_convert_plan` calls `beam_volume.beam_mu` only when `rows is not None`, outside `_header_call`, before `case_dir.mkdir`; it catches
  `CaseRefused` (beam_mu raises only REASON_BEAM_VOLUME, so no reason check is needed) and logs the fixed reason. `beam_mu.nii.gz` is
  written after `plan.parquet`; the `beam` series row follows the plan row and carries the CT's FrameOfReferenceUID. BEAM_NAME is not in
  `masks.classify`'s `taken` (an ROI named 'Beam' gives `beam.nii.gz` beside it; tested).
- Tests: `photon_plan()` defaults the CT to `PatientPosition='HFS'` (ct_meta=dict(tags=...)); only the three named tests needed edits, no exact
  CT `dicom` dict asserted for photon plans. Not-modelled cases (blocks, MLCY, null boundaries, no SAD, unbounded axis, unsupported and absent
  position, two conditions) each log exactly one WARNING and pass assert_no_leak. Brachy/ion/mixed assert no beam line inside the existing test.
- Gotcha: the default synthetic photon beam's field barely touches the 4x5x6 convert-fixture CT, so beam_mu.nii.gz is all zeros there; geometry is
  task-01's tests' job, convert tests pin file/row/geometry/dtype only.
- Not covered in convert tests: a CONTINUATION/port-film plan (plan_rows returns None there, same path as brachy/ion).
- Full default suite: 1677 passed, 4 skipped, 1 deselected (slow); ruff and compileall clean.

## task-03
- `preprocess.py`: `KINDS` gains `beam`; `check_rows` same_grid-checks it (REASON_GRID now names the beam volume); `output_name` unchanged (beam gets
  the dose's prefixes); a geometric run resamples it with `process_dose` but skips check_crop/crosses_final, `cropped`, `retained` (cell stays ''),
  the has_voxels -> REASON_CROP check and (by kind) min_mask_retention. `manifest._retained` already counts only dose/mask; run-log counters read
  `cropped`/`retained` only, so they needed no change.
- `index_plan_table` (new): `pq.read_table`, catches only `pa.ArrowInvalid`; schema compared with `table.schema != plan_table.SCHEMA` -- pyarrow's `==`
  ignores schema metadata but does compare field nullability, so parquet round-trips it (the tests pin `after.schema == SCHEMA`). An unreadable or
  non-SCHEMA file is left out with its `plan` row and one WARNING `preprocess: <REASON_PLAN_INDEX>`; an OSError keeps the patient-level IO path.
  Written through `conv._atomic` + `plan_table.write_plan_table`. A zero-row table is handled (`indices or [[]] * 3`).
- Tests: `photon_converted` defaults the CT to HFS. The old parametrised byte-identical test now covers non-geometric runs only (rescale, ahe); new
  tests pin AC11 (isocenter index vs. the final CT file, 1e-6, other columns equal, null_count == 0), AC10 (cut, cut out entirely, padded; with and
  without allow_crop and min_mask_retention; float32, final grid, exact zeros outside the native CT) and the off-grid refusal. Beam tests use
  `write_plan` + a synthetic `beam` row rather than a converted photon plan, because the converted fixture's beam volume is all zeros.
  Mutation check by hand: dropping the beam exemption from the crop block, or from the has_voxels check, fails the AC10 tests.
- Beyond the task: regression test for the unreadable / non-SCHEMA `plan.parquet` case was added (the plan deferred it); it also pins the log text.
- `test_manifest_rows_are_unchanged_by_plan_parquet` still passes unchanged (both trees keep the beam row; task-04 owns the manifest `beam` key).
- Not done / deferred: a null in a SCHEMA-valid isocenter_x/y/z column is not guarded (the schema marks them non-nullable).

## task-04
- The manifest helper `_files(kept)` returns `{'plan': file, 'beam': file}` for the first `plan` / `beam` series.json rows, a key only when its row exists; merged as
  `_optional(kept[0]) | retained | _files(kept)`, so a plan without them writes the same bytes as before. `_retained` is untouched (dose/mask only);
  a test drops the beam row from a preprocessed tree and pins `retained` equal.
- `test_manifest.read` now allows `plan`/`beam` after `retained` in its key-order assert. New tests: converted photon (`plan.parquet`, `beam_mu.nii.gz`),
  geometric run (beam under its prefixed name, `retained` then `plan` then `beam`), each key alone, non-photon plan with neither.
- `test_manifest_rows_are_unchanged_by_plan_parquet`: the bare tree keeps its beam row, so the manifests differ by the `plan` key only (asserted).
- README: manifest key table gains `plan`/`beam`; plan.parquet section no longer says the manifest ignores it; `retained` is no longer called "the last key".
