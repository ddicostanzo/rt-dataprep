# Hardening B11: imaging-cases review leftovers (#110-#116)

Branch `fix/hardening-b11-imaging` off `main` (`44429ff`). Tests: `uv run --group dev pytest -q --tb=short`; lint: `uv run --group dev ruff check .`. Baseline: 707 passed, 4 skipped. Red first: each new test fails on the old code, then passes.

Scope: `src/rt_dataprep/{convert,preprocess,manifest}.py`, `tests/test_{series,preprocess,manifest}.py`. `tests/test_plan_regression.py` is never re-pinned.

## Guardrails
Same as `docs/handoffs/hardening-b9-continuation.md`: at most 5 staged files per commit, no dotted private names in added lines (`getattr(module, '_name')  # noqa: B009`), no raise inside an except block, `SYNTHETIC_` test data, no push or merge until the owner confirms.

## Tasks and acceptance

### T1 — #110 whitespace-only StudyInstanceUID
pydicom strips trailing spaces and nulls from a UI value on read, so a StudyInstanceUID stored as spaces reads as `''` and `series.candidates` already drops the series, as it does an absent one (uncounted, `test_series.py` "nostudy"). No code change.
- [ ] Test: a misc series whose StudyInstanceUID is written as spaces is not written to any study folder, same as the absent-UID series.

### T2 — #111 blank StudyTime ranks last within its date
`convert.py` study rank key becomes `(date == '', date, time == '', time)`.
- [ ] Test: two studies on one StudyDate, one with StudyTime `090000` and one blank: the timed one is `study_1`.

### T3 — #112 ASCII digits and modality vs file name
`STUDY_FOLDER` and `STUDY_FILE` use `[0-9]`. `check_study_rows` refuses (REASON_STUDY_MODALITY, text widened) a row whose modality is not the MOD in its file name.
- [ ] Test: a study row with modality MR on `series_CT_1.nii.gz` is refused.
- [ ] Test: a file name with a non-ASCII digit is refused with REASON_STUDY_NAME.

### T4 — #113 one STUDY_FOLDER, one CSV reader
`STUDY_FOLDER` lives next to `REASON_NO_CASES` in preprocess and manifest imports it. `read_study_csv` becomes `read_series_csv` and `check_rows` uses it too. `study_refusal_count` is `max(1, len(rows))` over that reader, so the refused path parses with the same code (the second read of a refused study's file stays; it is a rare path).
- [ ] Existing `test_study_refusal_count*` and check_rows tests pass unchanged.

### T5 — #114 linear log count
`plan_rows` is `r.folder and not r.folder.startswith('study_')`.
- [ ] Existing run-log count test passes.

### T6 — #115 manifest OSError on a study series.csv
- [ ] Test: monkeypatch the reader manifest uses to raise OSError for a study; the study is counted 1 refused and the plan row is kept.

### T7 — #116 rescale-only run refuses MR/PT study series (owner decision)
In `preprocess_series`, after rescale is turned off for a non-CT series, no step left raises `CaseRefused(REASON_NO_STEPS)` before any read.
- [ ] Test: rescale-only steps on a study with MR: its row is refused with REASON_NO_STEPS and no MR file is written; a CT series in the same study is preprocessed.

## Deferred
- The double parse of a refused study's series.csv in manifest (T4 note).
