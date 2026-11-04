# Hardening B8: test gaps (#98, #42 remainder, #51, #53)

Test-only batch, plus spec/plan wording. **Do not change `src/`.** If a new test exposes a real bug, stop: leave that test out and report the bug with the input that triggers it. Never weaken a test to make it pass.

Branch `fix/hardening-b8-tests`, in its own worktree off `main`. Run tests with `uv run --group dev pytest -q --tb=short`, lint with `uv run --group dev ruff check .`.

## Guardrails
Same as `docs/plans/hardening-b6-tests/plan.md` (synthetic data, commit gate, pinning by mutation, stop counter). Do not touch `tests/test_dicom_io.py`, `tests/test_split.py` or `tests/test_cli.py` (batch B7 edits them).

**Verify first.** Several items may already be pinned by later work (e.g. `test_the_ct_is_processed_first_whatever_its_series_csv_row`, `test_check_crop_refuses_exactly_outside_half_a_voxel_on_both_sides`, B6 T3). For each item, apply the mutant first; if an existing test fails, record "already pinned by <test>" and skip it.

## Tasks and acceptance

### T1 — parallel header reading (#98; context `docs/plans/parallel-header-reading/findings.md`)
- [ ] organize and convert pass `workers` through: patch `org.read_pooled` / `conv.read_pooled` with a recorder; `organize(..., workers=3)` and `convert(..., workers=3)` record 3. Mutant: hard-code `workers=1` at the call site → test fails.
- [ ] A convert test patches `dicom_io.read_header_core` (module attribute) and sees the patch take effect, so a `from`-import in convert would fail it.
- [ ] organize timing: the timing line comes right after the summary line; on Ctrl-C there is no timing line.
- [ ] organize Ctrl-C during the copy phase (not during reading) leaves no `.partial`.
- [ ] Spec `docs/specs/2026-11-04-parallel-header-reading.md`: "total" covers the report write and the summary line, not only the patient loop.

### T2 — series scenarios (#42 remainder)
- [ ] A series registered only to a refused plan goes where the placement rules say (assert the actual destination; report it if it surprises).
- [ ] A PT on the planning CT's frame (identity) of a patient whose plan is refused/absent lands in `images/`.
- [ ] A SOP resolved through an excluded-study or localizer header: the series is placed by the rules, no crash.
- [ ] B2 skip kinds counted end to end (convert report/summary): two-echo, multi-frame, no PixelSpacing, off-orthonormal — each adds one to the skip count.
- [ ] Rename the `frame_none` REG test label so it states only what it covers.

### T3 — preprocess (#51, minus A2.5i)
- [ ] A2.6 torchio equivalence also on a flipped (e.g. feet-first) and an oblique CT direction (skip if torchio absent, as now).
- [ ] CT-first sort of `series.csv` rows (verify-first).
- [ ] Exact-boundary `REASON_CROP` mutants (`>=`→`>`, `<`→`<=`) killed (verify-first).
- [ ] Memory release: after a plan's image is written, no reference to it remains (a `weakref` to the image object is dead after the step). Drop and report if SimpleITK objects can't be weakref'd.
- [ ] `docs/plans/2026-11-03-rt-dataprep/phase1.md` parity wording: preprocess runs on the converted tree, not the raw tree.

### T4 — convert out of memory (#53 bullet 1)
- [ ] Convert-level: the CT read (convert.py ~418) and the dose read (~487) raising ITK's "Failed to allocate memory" `RuntimeError` each abort with `ConvertAborted`, a `REASON_INTERNAL` row, no partial folder (verify-first against B6 T3 in `tests/test_series.py`).

## Done
Full suite and ruff green. Report per item: added / already pinned by X / dropped (why), with mutant results.

## Deferred
- #51 A2.5i (a mask that loses most voxels): look for it in the OSC run.
- #53 bullet 2 is already handled in `quiet_call` (str() failure is caught); bullet 3 needs real ITK text.
