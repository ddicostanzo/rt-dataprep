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
