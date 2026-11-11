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
