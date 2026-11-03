# Continuation prompt for next session (rt-dataprep, after F1)

**Anchor state.** Repo `/home/dominic/Repositories/rt-dataprep` (private, github.com/ddicostanzo/rt-dataprep), target
`main`. Merged: `b194639 F1: package skeleton and organize (#6)` on top of `a0506c1` (plan). Worktree
`rt-dataprep-f1-organize` and branch `feat/f1-organize` removed; local main synced.

**What shipped.** PR #6: `pyproject.toml` (pydicom, numpy; dev group pytest, ruff; run with `uv run --group dev ...`),
`src/rt_dataprep/{__main__,dicom_io,organize}.py`, `tests/synthetic_dicom.py` (ported from rnt_transformer, plus MR,
REG, dates, model, image type), 35 tests. `organize` turns `<root>/<PatientID>/*.dcm` into `<out>/<patient>/fx_<n>/`,
one per RTPLAN, with `organize_report.csv`.

**Next stage.** F2 (convert) then F3 (manifest + split) per `docs/plans/2026-11-03-rt-dataprep/phase1.md`; decisions in
`proposal.md` beside it. Run each through `/dev`. F2 notes:
- F2 adds SimpleITK and `DicomRTTool==6.2.0`. Use the 6.2 call sequence in phase1.md F2 step 4: `contour_names=`, `set_index`,
  `arg_max=False`. Vault: `Patterns/medical-imaging/dicomrttool-6-api-and-mclahe-deps.md`.
- `mclahe` pins numpy 1.26 and drags in TensorFlow inside a universal uv lock. Decide how to ship `[ahe]` (uv `conflicts`, or a
  separate install) before adding it.
- An fx folder can hold more than one CT series and RTSTRUCT (A1.5b). Convert must follow the reference chain, never
  "the CT in the folder".
- Legacy pipeline facts (order, defaults, prefixes, text/pickle `.nii.gz`) are in proposal.md.
- Open hardening issues: #1–#5 (plan deferrals) and #7–#15 (F1 review). #7 (an RTSTRUCT in a different study is
  dropped) affects F2 case completeness, so consider it first.

**Guardrails that carry over.**
- The global pre-commit MRN rule has no SYNTHETIC_ whitelist. Test lines with `patient_id=` next to 6–10 digits trip
  it. The user approved `--no-verify` for PR #6 only; ask again for a new PR, or ask the user to patch
  `~/.claude/hooks/pre-commit.sh`.
- Never set `core.hooksPath` (blocked). The repo hook installs as a `.git/hooks/pre-commit` symlink.
- One review round per PR. Deferred findings become `hardening` issues.
- PHI rules: no identifiers in logs or exceptions; synthetic data only; reports and logs go under the output root.

**First ask.** Confirm with the user before starting F2. Do not push, merge or close issues until they confirm.
