# beam_mu performance: per-row analytic intervals (#289)

- Date: 2026-11-13
- Branch: `feat/beam-mu-perf`
- Status: REVIEWED
- Builds on: `docs/specs/2026-11-11-beam-volumes.md` (geometry, "not modelled" list, AC14). Nothing in that spec's
  behaviour changes.

## Background
- #289 came out of the #286 review and was parked until a real cohort run proved slow. The owner is taking it now to
  get ready for the first real photon cohort (#283), where cohort throughput is the goal.
- Profile of AC14 (512×512×150 CT, 2 arcs × 178 control points, 356 weighted points) at 324e2b2: `_add` costs 0.90 s
  per point over the full grid, about 5.4 min per case, and that per-voxel loop is essentially all the time. On average
  each point lights 9.3% of the voxels, and the lit voxels' bounding box covers 47.5% of the grid. A box crop therefore
  caps out near 2×. Testing only the lit band cuts per-voxel work about 10×, which sets the target. Ragged per-row
  gathers in numpy carry overhead, so 10× is tight.
- The issue's ideas, as decided: (1) the z-chunk skip is replaced by per-row intervals, a superset of it. (2) Merging
  identical control points is dropped: it changes the float32 summation order, so the output is no longer
  bit-identical, and VMAT has no identical neighbours. (3) The weight check before `_point` is dropped: it would let a
  zero-weight control point with an unmodelled aperture pass, which changes refusals. (4) Beam-level checks already
  run once per plan.

## Problem
`beam_mu` evaluates every voxel of the CT for every weighted control point, although only about 9% are lit. A VMAT
case costs about 5.5 min per worker, and the cohort run multiplies that by every photon case.

## Constraints
- **Clinical output, full rigor.** beam_mu is MU per voxel. The output must be **bit-identical** to today's for every
  input (owner). Refusals (which plans raise `CaseRefused(REASON_BEAM_VOLUME)`, and when) do not change.
- **Bit-identity is a design constraint, not only a test outcome.** Intervals only prune work. Every voxel that is
  tested runs exactly today's float32 operation sequence: `_axis` (`f(t) + f(a0)*x`, then `+ f(a1)*y`, then
  `+ f(a2)*z`), `depth = sad - b_z`, `scale = np.divide(sad, depth, ..., where=depth > 0)`, `u`/`v = axis * scale`,
  the inclusive jaw comparisons, and the leaf lookup as in `_add` today. Each voxel receives its weight additions in
  float32, one per lit point, in point order. Nobody "optimises" those expressions. Interior voxels (Design 1) skip the
  test but receive the identical float32 add.
- **No try/except and no quieting** anywhere in `beam_volume.py` (`test_beam_volume_catches_and_quiets_nothing`
  AST-scans it). The fallback is a geometric check, not an exception path.
- **PHI:** not applicable. No new data flow, logs or messages.
- **FHIR:** not applicable.
- **Compatibility:** public signatures unchanged (`beam_mu(plan, rows, ct, patient_position)`, `aperture(row)`);
  float32 output on the CT geometry. No new dependency (numpy and SimpleITK only).

## Premises (confirmed by owner)
- The output stays bit-identical, so there is no clinical behaviour change and no change to the safety class.
- Refusal behaviour is untouched; ideas 2 and 3 are dropped, idea 4 is already done.
- The AC14 synthetic case stands in for real VMAT throughput.

## Approaches considered
- Per-point index-box crop: bit-identical but caps near 2× (lit bbox 47.5%). Rejected as too small for the target.
- Crop + merging identical control points: loses bit-identity. Rejected.
- **Chosen: per-row analytic intervals (owner).** For a voxel row (fixed z, y), each beam coordinate is linear in
  the x index. With `depth > 0`, each condition `low ≤ X ≤ high` is equivalent to a linear inequality in x
  (`sad·b_x − low·depth ≥ 0`), and so is each leaf-band and leaf-opening condition. The candidate lit set of a row is
  therefore an interval per leaf pair (one interval without an MLC), found in float64 at a cost of rows × pairs.

## Design
1. **Candidate set.** For each weighted point, work out each row's candidate x-range in float64. It is the
   intersection of these half-lines, each linear in x:
   - `depth > 0`, which the fallback (Design 3) guarantees for every voxel, so it is not solved per row;
   - the jaw bounds on each axis that has a jaw (`sad·b − low·depth ≥ 0`, `high·depth − sad·b ≥ 0`). An axis with no
     jaw but an MLC (legal on either axis) is bounded by the MLC alone: `B[0] ≤ Y ≤ B[-1]` for Y, the pair openings for X;
   - with an MLC, the union over pairs of (leaf band `B[n] ≤ Y < B[n+1]`, the last pair closed) ∩ (opening
     `x1 ≤ X ≤ x2`). A pair with `x1 ≥ x2` is closed and gives nothing.

   Each row is split into three disjoint classes using `EDGE_MARGIN_MM` = ε (0.01 mm at the isocenter plane, i.e.
   `ε·depth` in the linear form):
   - **interior**: clears every inequality by more than ε (tightened inward), minus a one-voxel pad at each end. These
     voxels are **credited** without the per-voxel test: the weight goes in through `np.add.at` over the runs' flat indices with the same
     float32 add as today (`float32 + float32(weight)`). The owner chose this to make 10× reachable: the scattered
     exact test alone measured 0.086 s per point against 0.90 s dense. It relies on the same error bound as exclusion.
   - **edge band**: inside the outward-relaxed (by ε) intervals, padded one voxel and clipped to the grid, but not
     interior. Each band voxel runs today's full test **once** (jaws plus the `searchsorted` leaf lookup, never a
     per-pair test). Band voxels an interior run already holds are dropped; a voxel in two adjacent pairs' bands may be tested
     twice, but `flat[lit] += weight` adds a repeated index once, so a voxel is never added twice.
   - **outside**: everything else gets nothing, as it does today.

   The oracle (AC1) is the proof, not a written error bound. Each voxel gets at most one add per point, in point
   order.
2. **Degenerate slopes.** A coordinate that barely varies along the row (gantry 90/270 computes cos ≈ 6e-17, not 0;
   collimator 90; rotated CT directions) is handled by the coordinate-space margin: the inequality's value over the
   row is compared against ±ε, so the row is fully in, fully out or fully band. Each relaxed inequality is evaluated at
   the two row ends: both ≥ 0 means in, both < 0 means out, and otherwise the crossing `−c/m` has `|m| ≥ 2ε/(N−1)` by
   construction. So there is no slope threshold, no divide-by-near-zero and no `RuntimeWarning` (`np.errstate` is not
   needed).
3. **Fallback to today's full per-voxel loop**, as a geometric check, never an exception. A point falls back if:
   - any of the grid's 8 corners has depth `< FALLBACK_DEPTH_FRACTION × SAD` (0.5). Depth is linear in the index, so
     checking the corners in float64 is exact;
   - any of `t`, `a`, the jaws, the leaf positions or the leaf boundaries is non-finite;
   - any grid corner's beam coordinate exceeds `MAX_COORD_MM` (5000) in magnitude. Within that bound the float32
     error at the isocenter plane is a few times 1e-3 mm
     (×2 for scale at depth ≥ SAD/2), under ε.

   Refusals are unchanged: the same `_check` calls run in the same order.
4. **Scalar types stay exactly as they are.** plan_rows gives Python floats for jaws (so jaw comparisons run in
   float32 under numpy 2.5's weak-scalar rule), and `aperture` builds float64 leaf arrays (so leaf comparisons run in
   float64). Neither may be converted.
5. **Chunking and memory.** Candidates are processed in z-chunks bounded by `CHUNK_VOXELS`, which stays
   monkeypatchable. The float64 interval arrays per chunk (rows × pairs) are bounded by the same chunk. Output does
   not depend on chunk size (AC3).
6. **Observable path.** The fallback is a named private function (`_add`, today's code) that tests can spy on to
   count fallbacks.
7. **Oracle.** `tests/beam_mu_reference.py` is a verbatim copy of today's computation (`_axis`, `_add`, the chunk
   loop), not an import, so the oracle shares no code with what it checks.

## Acceptance criteria
1. **Bit-identity oracle.** For a geometry matrix, `np.array_equal(new, reference)` holds, where `reference` is the
   frozen copy run on the same `_points` output. The matrix covers gantry {0, 45, 90, 180, 270, 315}, collimator
   {0, 30, 90}, couch {0, 90, 270}, all four patient positions, identity, ROT90 and an oblique CT direction matrix,
   anisotropic spacing, an off-centre isocenter, asymmetric jaws, MLC with closed, open and partly open pairs, jaw-only
   beams, an MLC with no Y jaw, an MLC with no X jaw, and field edges landing exactly on voxel centres (5 mm grid, jaws on multiples of 5),
   including gantry 90/270 rows that lie exactly on an edge. It also covers non-finite jaws or leaves that reach the
   computation, and a grid origin beyond `MAX_COORD_MM`. The matrix is one factor at a time around a base case, plus a
   seeded random sample of at least 50 geometries. It runs in the default suite.
2. **The interval path is exercised.** The oracle cases include geometries that take the interval path (e.g. SAD 1000
   on the 61³ grid), and a test spies on the fallback function to confirm which path ran: zero fallbacks on those cases. Both the
   all-interval and the all-fallback settings of `FALLBACK_DEPTH_FRACTION` are bit-identical to the reference on the
   same SAD 1000 case (an all-interval setting is valid only where no voxel is near the source).
3. **Chunk independence (#289).** With `CHUNK_VOXELS` monkeypatched small enough to force several z-chunks, the
   output is bit-identical to the single-chunk output and to the reference.
4. **Speed (AC14 replaced).** The slow test `test_ac14_full_size_vmat_case` runs the new `beam_mu` first, reads and
   prints its wall time and peak RSS, then runs the frozen reference on the same rows. It asserts bit-identity and
   `reference_wall / new_wall ≥ 10`. RSS is recorded, not asserted (`ru_maxrss` is a process high-water mark).
5. **Existing behaviour.** The full existing suite passes unchanged, including every `tests/test_beam_volume.py`
   geometry, refusal, PHI and no-catch test; no test is edited to pass, except AC14 as stated in criterion 4.

## Failure handling
- Unmodelled input: the same `_check` calls in the same order raise `CaseRefused(REASON_BEAM_VOLUME)`, unchanged.
- A float edge case the pad does not cover would make a voxel differ. AC1/AC2 detect it as an array_equal failure.
  The fix is a larger pad or a stricter fallback, never a tolerance.
- Speed between 5× and 10×: /dev marks criterion 4 FAILED and halts without merging; the owner decides (owner).

## Delivery
- One PR closing #289. /dev may auto-merge after a clean review round and a local full-suite pass (owner). There is
  no CI.
- Record the AC14 before/after numbers in `findings.md` and in the PR body.

## Deferred
- Output copy memory cut (owner wanted it as a bonus): `sitk.GetArrayViewFromImage` is read-only in SimpleITK 2.5.7,
  so writing into the image through a view is not possible without bypassing numpy's write protection. Filed as #340
  (hardening). Vault trap for any later attempt: `simpleitk-array-view-of-temporary-image-reads-freed-memory`.
- Further speedups beyond 10×: only if the cohort run asks.
