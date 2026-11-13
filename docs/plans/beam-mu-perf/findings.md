# Findings: beam_mu per-row intervals (#289)

Spec: docs/specs/2026-11-13-beam-mu-perf.md. Plan: docs/plans/beam-mu-perf/plan.json.

Summary: beam_mu must stay bit-identical while pruning per-voxel work. Per row, interior voxels (clear every edge by
> EDGE_MARGIN_MM) are credited with today's float32 add; the edge band runs today's exact test once; the rest gets
nothing. Points fall back to today's _add near the source (corner depth < 0.5 SAD), for non-finite values, or beyond
MAX_COORD_MM. tests/beam_mu_reference.py is a frozen copy of the 324e2b2 algorithm and is the oracle.

Baseline (324e2b2, this workstation): AC14 full-volume _add 0.90 s/point, ~5.4 min per case; lit fraction 9.3%,
lit-bbox 47.5%. Scattered exact test on 4.5M voxels: 0.086 s/point.

Append-only below, one section per task.

## task-01
- Files: tests/beam_mu_reference.py (324e2b2 `_axis`, `_add` and the chunk loop, extracted by script from `git show`, so
  verbatim; own CHUNK_VOXELS; no rt_dataprep import), tests/test_beam_mu_oracle.py (129 tests).
- Decisions: `assert_identical` runs beam_mu and `_points` + reference, maps CaseRefused to a sentinel, and compares
  dtype, shape and `tobytes()`; matrix cases also assert the reference lights something (`lit=False` for random/non-finite).
  Random sample = 60 parametrized seeds plus one test that >= 45 of them light voxels. Chunk test (AC3) uses a 3-beam plan
  (vmat_arc 178 points + two others) with CHUNK_VOXELS = 61*61*20 (4 chunks) and 1 (61 chunks). Mixed plan uses beam 2 at SAD 400 G45.
- Gotchas: all 5 non-finite mutations and the 6000 mm origin run to completion in today's code (no refusal), and are
  bit-identical; mutated rows are `rows[1]` (CP 1, the weighted one). Non-finite tests carry no RuntimeWarning mark
  (isocenter_x=inf warns in `_points` and in both `_add`s). task-02 must keep point keys sad, weight, jaws, leaves, t, a.
  The shared scratchpad dir is used by other agents; do not rely on its contents.
- Runtime: oracle file ~9.5 s; oracle + test_beam_volume 14 s. Full suite 2220 passed, 4 skipped, 3 deselected, 272 s.

## review-0
Wave review of task-01 (opus): no BLOCKING. Verbatim copy confirmed by diff; mutation check (jaw >= to >, leaf <= to <,
last-pair fix removed) fails 42 / 22 / 17 tests. WARNING: the oracle uses the live `_points`, so task-02 must not change
`_points` (it does not). NOTE: the non-finite cases are not refused today and some light nothing.

## task-02
- Written by the orchestrator. Files: src/rt_dataprep/beam_volume.py, tests/test_beam_mu_oracle.py (AC2 spy fixture,
  mixed-plan point-order test, FALLBACK_DEPTH_FRACTION 0 / 1e9 test, `fallback=` expectations on the AC1 cases).
- Engine: `_rows_ready` (finite values, every grid corner at depth >= 0.5 SAD and |b| <= 5000 mm), `_half` (end-value
  half-line solve, divides only where the signs differ), `_window`, `_jaw_bounds`, `_runs`, `_exact`, `_add_rows`.
  `_add` and `_points` are unchanged.
- Iterations on AC14 (new beam_mu only): 66.6 s (np.unique band dedupe, full jaw bounds per leaf entry) -> 42.3 s (row
  jaw windows once, no unique) -> 34.3 s (exact leaf-pair range, band exclusion by sorted search) -> 26.7 s (int32 run
  indices + np.add.at for the solid runs).
- Gotcha: int64 run-index building (np.repeat + arange) measured 8x slower than int32 on this machine; np.add.at beat
  `flat[idx] += w` about 2x. np.add.at adds twice on duplicate indices, so the band (where adjacent leaf pairs can
  repeat a voxel) keeps `flat[idx] += w`, which adds once.
- Decision: interior voxels go in via np.add.at over run indices, not literal slices; the float32 add is identical.
- Oracle: 133 passed (AC1, AC2, AC3). Full suite: 2224 passed, 4 skipped. ruff clean.
- AC14 once (task-02 criterion): new beam_mu 26.5 s, peak RSS 586 MB; against the 324 s baseline profile, ~12.2x.

## review-1
Wave review of task-02 (opus): 1 BLOCKING, fixed. `_window` started from scalar (0, last), so an MLC beam with no jaws
(aperture() allows it) raised TypeError in `_add_rows`. Class: window helpers assuming at least one bound; `_window` is
the only instance. Fix: per-row arrays from the start; new oracle test `test_ac1_mlc_with_no_jaws` (red before, green
after). Reviewer fuzzed ~18,000 random geometries (0.01-0.1 mm voxels, n=1 axes, oblique, edges on voxel centres,
grids out to 4900 mm, CHUNK_VOXELS 1/50) byte-identical to the reference. WARNING: `isocenter_x_inf` warns from the
unchanged `_points` matmul, so the "no RuntimeWarning" criterion applies to the marked oracle tests only (as task-01
planned). NOTE: np.cumsum promotes int32 to int64 inside `_runs`; the measured speedup stands.

## task-03
- test_ac14_full_size_vmat_case rewritten as the AC4 gate (still `@pytest.mark.slow`): new beam_mu x3 (wall and peak RSS
  printed after run 1, before the reference runs), then `reference_beam_mu` once on `beam_volume._points`; asserts byte
  identity (dtype, shape, tobytes), zero `_add` fallback calls (monkeypatch spy), and reference_wall / min(new) >= 10.
- Measured on this workstation (`pytest -m slow -s -k ac14`, 401.8 s total): new beam_mu 26.5 / 26.3 / 26.5 s
  (min 26.3 s, median 26.5 s); reference 321.1 s; ratio 321.1 / 26.3 = 12.2x; peak RSS after the first new run 602 MB
  (printed, not asserted). Baseline before the rewrite: 324 s.
- Result: PASSED (ratio >= 10). Byte-identical to the reference, zero fallback calls.

## review-loop-pass-1
One review round on PR #341 (opus code review + opus spec axis). BLOCKING: none. The code review fuzzed 12,000 more cases
(rotations, grids of size 1-40, negative weights, uneven leaf boundaries): bit-identical, no warnings; the measured
float32 drift was at most 1.6e-3 mm against the 0.01 mm margin.
- TRIVIAL, fixed: `_rows_ready` sends jaw/leaf values beyond MAX_COORD_MM to `_add`. Values near 1e306 used to overflow
  in `_window` and emit new RuntimeWarnings; the output was already identical.
- TRIVIAL, fixed: spec Design 1 wording (np.add.at instead of "contiguous slice"; band duplicates add once via fancy
  `+=`; depth > 0 guaranteed by the fallback rather than solved per row).
- DEFERRED: #342 (rows x pairs path ~20% slower near collimator 90/270 with an MLC), #343 (AC14 gate on the median).
No blocking fix was made, so no verification pass is needed.

## review-loop-pass-1 (Codex thread)
BLOCKING, fixed: Codex (PR #341, beam_volume.py:182) showed that the corner gates (depth >= 0.5 SAD, |b| <= 5000 mm) do
not bound float32 error when depth comes from cancellation and SAD is small: SAD 1, t_z = -1000.00007, depth from
-1000 + 1000.5. Float32 X is about 0.024 mm off, beyond EDGE_MARGIN_MM, and the interval path dropped a row `_add` lights.
Class: the fast-path gate admits a point whose float32 error exceeds the margin. `_rows_ready` is the only gate. Fix:
`_float32_error`, a first-order bound from the partial-sum magnitudes, the least depth, the largest edge value and the
float32 rounding of the jaw, which must be <= EDGE_MARGIN_MM / 2. Test `test_ac1_cancellation_in_depth_falls_back`
(red before, green after). A 3000-case gate fuzz (SAD 1-3000, depth near threshold, cancellation) found 0 mismatches.
Full suite 2227 passed; AC14 28.5 s min against a 374.5 s reference = 13.1x, zero fallbacks.
