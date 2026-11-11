# Findings: fluence maps (#279)

Spec: docs/specs/2026-11-11-fluence-maps.md (REVIEWED). Plan: docs/plans/fluence-maps/plan.json.

Summary: convert writes fluence_mu.nii.gz, float32 (beams, 400, 400), 1 mm, collimator (BEV) frame, exact area-fraction
apertures (pure geometry), MU per fraction with beam_mu's half-segment weights. New module fluence.py with REASON_FLUENCE;
aperture parsing shared through a non-raising beam_volume.aperture(row). series.json kind 'fluence' with
dicom.beam_numbers; preprocess copies it unchanged; manifest gains an optional 'fluence' key.

Order: task-01 (aperture helper) and task-03 (preprocess/manifest) are independent; then task-02 (fluence module); then
task-04 (convert + README, makes the suite see real fluence rows).

Repo notes for implementers:
- Commit gate: at most 5 files per commit; private attribute use needs CLAUDE_ALLOW_PRIVATE_API=1 with the use listed in the body.
- Full suite ~4 min: `uv run --group dev pytest -q --tb=short`; slow tests opt-in `-m slow`. No bare `python`.
- Synthetic DICOM only (tests/synthetic_dicom.py, save_as/dcmread). No PHI in logs; refusal reasons are fixed strings.
- Temp files: create only in your own scratch dir; never delete /tmp/claude-<uid> or any tasks/ dir (tmp-guard hook).
- Vault: Patterns/radiotherapy/iec-61217-dicom-patient-position-beam-projection.md (pin signs with one-sided fields).

Append below under `## <task-id>`: files changed, decisions, gotchas (append only).

## task-01
Files: src/rt_dataprep/beam_volume.py, tests/test_beam_volume.py, docs/specs/2026-11-11-beam-volumes.md (+ this file).
- `beam_volume.aperture(row)` -> `(jaws, leaves)` | None, never raises; `_point` does `_check(aperture(row) is not None)`.
  jaws = {'x': (lo, hi) | None, 'y': ...}; leaves = None or (boundaries, x1, x2, opening) numpy arrays.
- Decision: boundaries with fewer than 2 entries (zero pairs) are also not modelled (would index pair -1 in beam_mu).
- Strictly-increasing test is `np.all(np.diff(b) > 0)`, so NaN boundaries are also not modelled.
- Gotcha: jaws values are whatever plan_rows holds (floats); jaws-only rows return leaves None.
- Tests added: flat/reversed boundary cases in the ac8 refusal test; aperture None for the six cases; aperture shapes for a
  jaws-only and an MLC row. `vmat_arc(number, start, step)` is module-level (slow test uses it).
- Full suite 1708 passed, 4 skipped; ruff clean.

## task-03
Files: src/rt_dataprep/preprocess.py, src/rt_dataprep/manifest.py, tests/test_preprocess.py, tests/test_manifest.py (+ this file).
- preprocess: 'fluence' added to KINDS, the keep-the-name tuple and the copy tuple (so a geometric run copies it byte-identical too); not the same-grid tuple.
- manifest: `_files` tuple is now ('plan', 'beam', 'fluence'); module and `_files` docstrings updated.
- Tests: preprocess copy test (geometric + non-geometric) and a no-fluence-row test; manifest key test (absent, then present after plan and beam); test_manifest.read() ignores the new key.
- Gotcha: a series.json row injected by a test must carry the same key set as the others (`dicom: {}` when converted rows have it), else read_series_json calls the file malformed and manifest refuses the plan. The file must exist or REASON_FILE_MISSING.
- Gotcha: the fluence tests write placeholder bytes (preprocess never reads the file); `FLUENCE_NAME` is a literal in the tests until convert defines one (task-04).

## task-02
Files: src/rt_dataprep/fluence.py, tests/test_fluence.py (+ this file).
- `fluence_mu(plan, rows)` -> `(sitk image float32 (beams, 400, 400), [int BeamNumber])`; origin (-199.5, -199.5, 0), spacing 1, identity direction.
  The only beam_volume name used is `aperture` (`from rt_dataprep.beam_volume import aperture`).
- Per control point: rectangles = open pairs (opening mask) intersected with the jaws (or the jaw rectangle alone), kept only if
  `(xhi - xlo > 0) & (yhi - ylo > 0)` (so NaN is dropped too). Off-grid check (strict, `< -200` / `> 200`) runs on the kept
  rectangles, only for non-zero weight; `aperture(row)` runs for every row. Accumulate `w * Yov.T @ Xov` in float64.
- Duplicate-number check is on `plan['BeamSequence']` non-SETUP beams (`int(BeamNumber)`), before any row work.
- Weights are beam_mu's half-segment rule, recomputed in fluence.py (not shared: `_points` is beam_mu-private and the spec bars other beam_volume names).
- Tests import fixtures by name from test_beam_volume (static, set_mlc, written, ct_image, vmat_arc, _blocks, ...). They were written
  after the module was drafted, then checked with five hand mutations (transposed matmul, off-grid limit +1, pixel edges shifted
  0.5, weight-0 check removed, `keep` filter removed): each fails at least one test.
- Gotchas: a weight-0 control point needs a 3-point beam whose CP 1 re-states the jaws (forward fill otherwise carries the wide
  aperture into every point); the AC4 edge fractions are 0.3 (X, -10.3 to -10) and 0.25 (Y, -5.25 to -5).
- AC14 (2 arcs x 178 control points): 1.45 s wall in `pytest -m slow tests/test_fluence.py` (budget 10 s; other agents' jobs were
  sharing the machine, so not strictly idle).
- Full suite 1735 passed, 4 skipped, 2 deselected (slow); ruff clean.
