# Hardening B6: test gaps

Test-only batch. **Do not change `src/`.** If a new test exposes a real bug, stop: leave that test out, and report the bug with the input that triggers it. Never weaken a test to make it pass.

Branch `fix/hardening-b6-tests`, in `/home/dominic/Repositories/rt-dataprep`. Run tests with `uv run --group dev pytest -q --tb=short`, and lint with `uv run --group dev ruff check .`.

## Guardrails
- **Test data:** synthetic only. Ids and names use a `SYNTHETIC`/`FAKE_` prefix, and synthetic patient ids hold no digits or underscores. Build trees in `tmp_path` with the existing helpers: `tests/synthetic_dicom.py`, the `make_plan`/`run_plan`/`planning_study`/`write_plan` helpers, and the `World` fixture in `test_series.py`.
- **Commit gate:**
  - No added line may contain a dotted name whose last part starts with `_`. Reach such attributes through `operator.attrgetter` or `getattr` with the name as a string.
  - At most 5 staged files per commit.
  - Don't edit `tests/test_organize.py:24` or `tests/test_dicom_io.py:85`, which trip the MRN rule.
- **Pinning a guard:** a test that pins a guard must fail when that guard is mutated. For each one, check it by temporarily mutating the src line, running the test, and restoring with `git checkout -- src/`. Report each mutant and its result.
- **Stop counter:** after 3 failed attempts at one test, drop it and report.

## Tasks and acceptance

### T1 — convert cleanup and empty frame (#57)
- [ ] A refused plan whose folder exists, where `shutil.rmtree` of that folder fails, gives the patient one `REASON_IO_ERROR` row and leaves no patient folder. The fake rmtree must honour `ignore_errors` the way the real one does. Use a refusal that happens after the plan folder exists.
- [ ] A convert-level test shows that an other series on an empty FrameOfReferenceUID, with the planning CT's frame also empty, lands in `images/`, not in the plan folder. If an empty CT frame is refused earlier in convert, test what happens and report it.

### T2 — organize root and dedupe (#61)
- [ ] `organize(src, src)` and `organize(src, link)`, where `link` is a symlink in tmp_path pointing inside `src`, both raise `ValueError(REASON_OUTSIDE_SRC)` and leave `src` byte-identical.
- [ ] In the dedupe test (plan and dose copied into a second sub-folder), `plan_1`'s RTPLAN comes from the first path. Check with a distinguishing byte, for example a different private tag or file name in the copy, so the test can tell which copy was used.

### T3 — out of memory and locale test (#53, #55)
- [ ] Convert-level: a planning CT read that raises ITK's out-of-memory `RuntimeError` ("...Failed to allocate memory for image.") aborts the run with `ConvertAborted`. The report has a `REASON_INTERNAL` patient row and there is no partial folder. Monkeypatch where convert/series reads the CT.
- [ ] `test_series_csv_is_utf8_whatever_the_locale` asserts `result.returncode == 0, result.stderr`.

### T4 — layout and dose offsets (#21, #22)
- [ ] #21: the RTSTRUCT's own study, which differs from the CT's, also holds another series. The plan still converts, and the other series goes where the placement rules say (registered → plan folder, else `images/`).
- [ ] #22: feet-first axial CT (slice normal −z) converts, and the CT, dose and mask stay aligned. Check that a named physical point maps to the same voxel value.
- [ ] #22: a CT with mirrored in-plane axes converts, with alignment checked the same way. Extend `synthetic_dicom` only if needed, keeping its current defaults.
- [ ] #22: an RTDOSE with a negative GridFrameOffsetVector, and one with an absolute offset (first value equals ImagePositionPatient z), both convert to the same dose grid as the relative-positive form.

### T5 — boundaries (#42, subset)
- [ ] `SERIES_ATOL_MM`: a series whose slice-position deviation sits just under it is kept, and one just over is skipped.
- [ ] `RIGID_ATOL` and `LAST_ROW_ATOL`: a REG matrix just inside each tolerance is usable, and one just outside gives `reg_matrix == ''`.
- [ ] The implied-identity REG mutation (`found or [...]` → `found + [...]`) is killed by a test.
- [ ] The single-slice guard in the series volume check is pinned directly, not only through a NaN.

## Deferred
- #42's other bullets: the tag-catch wrapper removals, the B2 skip kinds end to end, and the B9 / PT / localizer cases.
- #24: needs a CT read failure whose raw message names the path.
- #51's torchio cases.
