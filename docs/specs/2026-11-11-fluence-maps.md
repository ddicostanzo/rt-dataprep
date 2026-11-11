# convert: per-beam photon fluence maps (fluence_mu.nii.gz) (#279)

Date: 2026-11-11 · Branch: `feat/fluence-maps` · Status: REVIEWED (stress-test rounds 1–2 and advisor applied, 2026-11-11)

## Background
PR #271 writes `plan.parquet` (one row per control point of a photon EBRT plan) and PR #286 writes `beam_mu.nii.gz` (the
MU-weighted divergent aperture on the CT grid). Issue #279 asks for per-beam 2D fluence images from the MLC/jaw apertures,
weighted by MU. The owner put this ahead of the first real photon cohort run (2026-11-11), which stays deferred.

## Problem
The owner (physicist / ML developer) wants a per-beam fluence channel for **both** dose-prediction models and plan
complexity / QA features (aperture area, modulation). beam_mu mixes all beams on the CT grid and samples apertures at voxel
centres, so it gives neither a per-beam view nor exact aperture areas.

## Decisions (owner, 2026-11-11 interview)
- **Content:** one float32 NIfTI per plan, **`fluence_mu.nii.gz`**, shape (beams, 400, 400) as `sitk.GetArrayFromImage`
  returns it (k, row, column); nibabel shows the same data as (column, row, k). Slice k is the fluence of the k-th
  treatment beam in plan_rows order (the order of `plan.parquet`). Accumulated in float64, cast to float32 at the end.
- **Frame:** the isocenter plane in the **beam-limiting-device (collimator) frame**, mm, viewed from the source. The
  collimator angle is **not** applied; gantry, couch, table-top, SAD and PatientPosition play no part. Column index
  increases toward IEC +X (the X2 jaw, the X2 leaf bank); row index increases toward IEC +Y (the Y2 jaw; leaf pair n+1 is at
  larger Y than pair n). (Owner: "BEV from source, +X/+Y increasing".)
- **Grid:** fixed 1 mm pixels, 400 × 400, pixel centres at -199.5 … 199.5 mm on both axes (pixel j covers
  [j - 200, j - 199] mm). NIfTI geometry **as SimpleITK reads it**: origin (-199.5, -199.5, 0), spacing (1, 1, 1),
  identity direction; the third image axis is the beam index, not a distance. SimpleITK stores NIfTI in LPS, so nibabel
  shows affine diag(-1, -1, 1) with origin (199.5, 199.5, 0). The README says to index the array (or read with
  SimpleITK), not to take ±X/±Y from the nibabel affine, and to plot with `origin='lower'` for the view from the source
  (owner, round 2: document, keep the geometry).
- **Edge pixels: exact area fraction.** Per control point the open region is a union of rectangles, one per open leaf
  pair intersected with the jaws (or the jaw rectangle alone without an MLC). A rectangle's contribution to a pixel is
  weight × (x-overlap in mm) × (y-overlap in mm), with the pixel area 1 mm², each overlap `max(0, …)`. A rectangle of zero
  or negative width or height (closed or crossed leaves, crossed jaws, a pair outside the jaws) is dropped before any
  other check: it is neither drawn nor refused. So Σ pixels = Σ weight × open area (mm²).
- **Leaf model: pure geometry.** Open = 1, closed = 0. No transmission, tongue-and-groove, rounded leaf ends or leaf-gap
  offsets.
- **Weighting and units:** as beam_mu: segment i's MU (`mu` of row i) goes half to control point i-1 and half to control
  point i, each with its own aperture; weight-0 control points add nothing. Values are MU per fraction: a pixel fully inside
  a static open field reads the beam's MU.
- **Aperture rules shared with beam_mu.** The per-row aperture parsing and not-modelled checks in
  `beam_volume._point` (jaws, MLCX leaves in IEC X1-then-X2 order, X1 ≥ X2 is a closed pair, blocks, non-MLCX MLC, null
  leaf boundaries, an axis bounded by neither jaws nor MLC) move into a public **`beam_volume.aperture(row)`** that
  **never raises**: it returns `(jaws, leaves)`, or None when the aperture is not modelled. beam_mu does
  `_check(result is not None)` (REASON_BEAM_VOLUME), fluence the same with REASON_FLUENCE (owner, stress-test round).
  The helper also treats leaf boundaries that are not strictly increasing, or fewer than two of them (added in implementation: zero pairs would mis-slice `mlc_positions`), as not modelled (owner): a **new beam_mu
  refusal**, for data beam_mu would otherwise assign to the wrong pair. All other beam_mu behaviour is unchanged.
- **Where:** a new module `src/rt_dataprep/fluence.py` with its own fixed reason **`REASON_FLUENCE`**, entry point
  `fluence_mu(plan, rows)` returning `(image, beam_numbers)` (the plan for the BeamNumber check, compared as
  `int(BeamNumber)` like `_sads`; `beam_numbers` in slice order). `aperture(row)` is applied to **every** row, weight 0
  included, as beam_mu does; only the off-grid check is limited to control points with non-zero weight. Called by convert from the plan rows it already has, in
  its own catch (outside `_header_call`, like beam_mu), after beam_mu and before `case_dir.mkdir` (everything that can
  refuse runs before the folder appears), written after `beam_mu.nii.gz`. Existing trees gain it only through
  `convert --overwrite`.
- **series.json:** a row of `kind` `fluence` (modality `RTPLAN`, plan columns filled, `frame_of_reference_uid` = the
  plan's, `dicom` = the plan's metadata as on the `plan` row **plus** `beam_numbers`: the slice order's BeamNumbers as a
  `;`-joined string, e.g. `"1;2;3"`). No SERIES_COLUMNS change; `dicom` is carried unread by preprocess.
  `_series_rows` gains a `beam_numbers` argument and appends the `fluence` row after the `beam` row, only when
  `FLUENCE_NAME in files`. `beam_numbers` is the last key of that row's `dicom` (after the plan keys). The catalog
  key-order test (tests/test_dicom_metadata_convert.py) covers no plan/beam row, so it is unaffected; AC7 pins the
  fluence row's `dicom`.
- **Name:** `fluence_mu.nii.gz` has an underscore, so no mask or line (alphanumeric `format_name`) or applicator
  (`*_channel`) can produce it, as for `beam_mu.nii.gz`.
- **Kind-dispatch sweep (2026-11-11):** only convert, preprocess and manifest dispatch on series.json kinds;
  `scripts/parity.py` reads only ct/dose/mask and ignores other kinds (no change).
- **beam-volumes spec:** the non-increasing leaf boundaries refusal is added to the "Not modelled" list of
  `docs/specs/2026-11-11-beam-volumes.md` (as #287 did).
- **preprocess:** `fluence` is a known kind, copied unchanged under its own name in every run (it is not on the CT
  grid): no resample, crop, retained or same-grid check. This takes three edits in preprocess.py: `KINDS`, the
  keep-the-name tuple (`kind in ('applicator', 'line', 'plan')`, ~line 767) and the copy tuple (~line 775). Without the
  last two, a geometric run would send it through `process_dose` and silently resample it onto the CT grid.
- **manifest:** a plan line gains an optional `fluence` key (the file name), after `plan` and `beam`, only when the
  `fluence` row exists (`_files` tuple).
- **README:** a fluence section next to beam_mu's (frame, orientation, units, slice order, `dicom.beam_numbers`, the
  nibabel/imshow notes), plus `fluence` in the folder tree, the per-kind `dicom` keys paragraph (noting `beam_numbers` is
  not a DICOM attribute), the manifest key table and the list of `kind` values.
- **Existing convert tests change on purpose:** the not-modelled photon test in tests/test_convert.py (its exact
  WARNING count, file list and row kinds) and the exact file lists of the photon convert tests gain fluence. AC8 reuses
  that test's `NOT_MODELLED` table rather than a new fixture.

## Refusals (REASON_FLUENCE: no fluence map, one WARNING, the plan still converts with plan.parquet and beam_mu)
- Any not-modelled aperture in the shared helper (the beam_mu list above).
- Any open rectangle of a control point with non-zero weight that reaches outside [-200, 200] mm on either axis
  (strictly: an edge at exactly ±200 is inside; refuse, never clip).
- Two treatment beams (non-SETUP, the plan_rows filter) in the RTPLAN `BeamSequence` with the same BeamNumber (#285),
  since slice ↔ beam number would be ambiguous. Checked on the plan, not the rows: grouping rows merges adjacent duplicates.
- Nothing else. A plan beam_mu refuses for SAD, PatientPosition or table-top angles still gets a fluence map, and a plan
  fluence refuses can still get beam_mu.
- No plan rows (brachy, ion, port film, REASON_PLAN_TABLE) → no fluence step, no new log line.

## Constraints
- Clinical output: a derived ML/QA input, never used for treatment. A misplaced or mis-scaled map is silent, so
  orientation and area are pinned by analytic tests with one-sided fields and exact integrals, and anything not modelled
  is refused loudly. Not a treatment-planning device; no IEC 62304 class beyond the repo's existing practice.
- PHI: no new surface. The file holds MU values; series.json gains beam numbers (not identifying); logs carry only the
  fixed reason. FHIR: n/a.
- Module rules (AST-tested like beam_volume and plan_table): headers and rows read with `.get`/keys, the only exception
  raised is `CaseRefused(REASON_FLUENCE)`, no try/except. It reads `plan['BeamSequence']` (BeamNumber, TreatmentDeliveryType) with `.get` exactly as `beam_volume._sads` does, after `plan_rows` has already accessed those tags; no pydicom quieting, the same precedent as beam_mu.

## Approaches considered
- **Chosen: new module in convert, shared aperture helper.** Reuses plan_rows already in memory; one place for the
  aperture rules (the owner's worry: divergence from beam_mu).
- Rejected: a second function inside beam_volume.py (mixes two outputs and two reasons in one AST-pinned module).
- Rejected: a separate pass over plan.parquet (a new CLI stage and a second reader of the parquet schema).
- Rejected: pixel-centre sampling (biases area-based QA metrics by up to 0.5 mm per edge); constant transmission (needs a
  per-machine value); clipping off-grid apertures; listing beam numbers in a new SERIES_COLUMNS column (schema change).

## Acceptance criteria
1. Static jaw-only field X [-50, 50], Y [-30, 30] mm, MU 100: pixels fully inside read 100 (float32 tolerance 1e-4),
   pixels fully outside read 0, shape (1, 400, 400), float32, NIfTI origin/spacing/direction as in Decisions.
2. Orientation, each with a one-sided field: jaws X [10, 60], Y [-30, 30] lights only columns ≥ 210 (X > 0);
   Y [10, 40], X [-50, 50] lights only rows ≥ 210 (Y > 0); an MLCX whose only open pair is the highest-Y pair lights only
   rows inside that pair's boundaries; an open X2 leaf bank shifted to +X lights only X > 0.
3. Collimator independence: the same beam at (collimator, gantry, couch) = (0, 0, 0), (90, 0, 0) and (30, 270, 45)
   yields identical maps.
4. Exact area: a field whose edges fall mid-pixel (jaws X [-10.3, 20.7], Y [-5.25, 7.5], MU M) has
   Σ pixels = M × 31.0 × 12.75 within 1e-3 relative, and an edge pixel holds M × its covered fraction.
5. MLC: leaf boundaries every 2.5 mm and 5 mm, sub-mm leaf ends, one closed pair (X1 = X2) and one crossed pair (X1 > X2):
   Σ pixels = Σ over open pairs of MU × width × height clipped to the jaws; closed and crossed pairs add 0, a crossed jaw
   pair (X1 > X2) gives an all-zero slice, and no geometry (crossed jaws or leaves) makes a pixel negative.
6. Weighting: a two-segment step-and-shoot beam (MU a then b, different apertures) gives a where only segment 1 is
   open, b where only segment 2, a+b where both; a multi-control-point arc-like beam has
   Σ pixels = Σᵢ weightᵢ × open areaᵢ with weightᵢ from the half-segment rule.
7. Multi-beam: a three-beam plan gives shape (3, 400, 400); slice k is beam k in plan_rows order; series.json
   `dicom.beam_numbers` lists them in that order.
8. Refusals: each not-modelled aperture case, an off-grid open rectangle, and duplicate BeamNumbers (including two
   adjacent beams sharing a number) write no `fluence_mu.nii.gz` and no `fluence` row, log exactly one WARNING with
   REASON_FLUENCE, and still write plan.parquet. (An aperture case also logs beam_mu's WARNING: two in total.)
   A plan with a non-zero table-top angle (beam_mu refused) still gets a fluence map. The off-grid rule is strict: an
   open edge at exactly ±200 mm (e.g. leaves parked there on a jaw-less-X machine) is accepted; ±200.01 mm is refused.
9. Shared rules: for each aperture case of beam_volume's test list (`_blocks`, `_mlc_not_mlcx`, `_null_boundaries`,
   `_no_y_limit`) and non-increasing leaf boundaries, both beam_mu and fluence refuse; `aperture(row)` returns None for
   each and raises nothing; existing beam_volume tests pass unchanged (convert tests change as stated in Decisions).
10. AST test: fluence.py raises only CaseRefused(REASON_FLUENCE), has no try/except and no quieting; REASON_FLUENCE is a
    fixed string.
11. preprocess (geometric and non-geometric runs) copies fluence_mu.nii.gz byte-identical under the same name and keeps
    its series.json row; a tree without one is unaffected.
12. manifest: a plan line has `fluence` exactly when the `fluence` row exists, after `plan` and `beam`.
13. No PHI: `assert_no_leak` runs over the fluence refusal cases with the patient identifiers.
14. Performance: the full-size synthetic VMAT plan of beam_volume's slow test (2 arcs × 178 control points) produces its
    fluence map in under 10 s (slow test). (Plan hint: per control point, pair×400 Y- and X-overlap matrices and
    `w · Yᵀ @ X` in float64, not a full 400×400 outer product per pair.)

## Failure handling
- Not-modelled or off-grid apertures and duplicate beam numbers: refused with REASON_FLUENCE (WARNING), surfaced in the
  log; the rest of the plan's outputs are written.
- A write failure of fluence_mu.nii.gz behaves as the other outputs' atomic writes (OSError → the plan fails as today).
- Anything else in fluence.py (a bug) propagates uncaught, as in beam_volume.

## Delivery and validation
- Analytic synthetic tests in this PR. Real photon exports have not been through convert; validation on the first cohort
  run (e.g. fluence Σ vs plan MU × aperture area, visual check against the TPS fluence) is a follow-up.

## Deferred
- Transmission, tongue-and-groove, rounded leaf ends, leaf-gap offsets.
- Per-control-point or per-segment maps; gantry-frame (collimator-rotated) maps.
- Blocks, MLCY, apertures beyond ±200 mm.
- A negative BeamMeterset or a decreasing CumulativeMetersetWeight (plan_table checks presence only): either gives a
  negative `mu` and so negative pixels, in beam_mu too.
- Real-data validation on the first photon cohort: fluence Σ vs MU × aperture area per beam, and a visual check against
  the TPS fluence (#293).
- Fluence for proton (#281) and brachy (#282).
