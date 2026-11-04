# Findings: imaging cases (#17)

Spec: docs/specs/2026-11-04-imaging-cases.md · Plan: docs/plans/imaging-cases/plan.json

Summary: convert retires images/; every CT/MR/PT series other than a converted plan's planning CT is written to
<patient>/study_<n>/series_<MOD>_<k>.nii.gz (registered ones also stay in their plan folder), one series.csv per study.
manifest gains imaging rows (case_type, modality, image columns); preprocess processes study folders per series (CT full
chain, MR/PT geometric only). Plan folders must stay identical (task-01 pin). Tasks append below under ## <task-id>.

## task-01

Files: tests/test_plan_regression.py (new, 3 tests; no src/ change). Commit 0a5c194 on feat/task-01-imaging-cases.
- Fixture: plan_1 (CT, dose, Liver mask, registered MR via a REG in plan_1) and plan_2 (CT, dose); unregistered MR and PT in misc_files (they land in images/ today, which is deliberately not pinned).
- Pinned per plan folder: sorted file list, exact series.csv text (CRLF, UIDs and dates as {PLACEHOLDER}s filled at test time), per volume sha256 of GetArrayFromImage bytes plus size/spacing/origin/direction literals.
- All UIDs are generate_uid(entropy_srcs=['SYNTHETIC_<name>']); identical across runs. test_convert.make_plan cannot fix the struct/dose series UIDs, so the test has its own build_plan.
- Gotcha: the test imports BOX, CT_DATE.., SHAPE, SPACING, ORIGIN, dose_spec from tests/test_convert.py and VALUES, rigid from tests/test_series.py (as test_series does). Changing those constants or helpers in task-02 moves the pinned hashes; keep them, or the pin goes red for the wrong reason.
- Gotcha: do not count .nii.gz files outside plan folders to prove the unregistered series exist: after task-02 the registered MR also appears in study_<n>/.

## task-07

Files: tests/synthetic_dicom.py, tests/test_synthetic_dicom.py (test_study_date_and_time_only_when_given).
- New keywords `study_date=` and `study_time=` (default None: tag omitted) on `_dataset`, hence on EVERY writer through `**meta`: write_ct_series (also modality='MR'/'PT', there is no separate MR/PT writer), write_rtstruct, write_rtstruct_rois, write_rtstruct_contours, write_rtplan, write_rtdose, write_reg.
- Strings as DICOM DA/TM ('YYYYMMDD', 'HHMMSS'). invalid_values=True still writes StudyDate=INVALID_DA; an explicit study_date overrides it. Defaults leave all existing fixtures unchanged.
- Gotcha: write_ct_series forwards `**meta` per slice, so all slices of a series get the same StudyDate/StudyTime; use one study_date per study (same study_uid), not per series.

## review-group-1
PASS, no BLOCKING. WARNING: tests/test_plan_regression.py imports SHAPE, SPACING, ORIGIN, BOX, dose_spec and dates from
tests/test_convert.py and VALUES, rigid from tests/test_series.py; task-02 must not change those or the pin moves. Never
edit tests/test_plan_regression.py to make it pass.

## task-02

Files: src/rt_dataprep/convert.py, tests/test_series.py, tests/test_convert.py (branch feat/task-02-imaging-cases). test_plan_regression.py untouched and green.
- New in convert.py: `STUDIES_NAME = 'studies'` (folder of the patient summary row), `REASON_NO_STUDIES`, `STUDY_SERIES = r'study_\d+/series_(CT|MR|PT)_\d+'` (folder of a per-series report row). `IMAGES_NAME` and `REASON_NO_IMAGES` are gone.
- `_place_series` now returns a LIST of ConvertRows ([] when no candidate series): the `studies` summary row (converted with skipped_series, or skipped with REASON_NO_STUDIES) then one row per written series, `folder = study_<n>/series_<MOD>_<k>` (no .nii.gz), `files = [volume]`, plus `series.csv` in the study's first row only. `_convert_patient` does `rows += _place_series(...)`.
- Layout: hidden `.study_<i>` (i = 0-based first-seen order of the StudyInstanceUID among series actually read and written) in the patient partial, renamed to `study_<n>` after the loop, then series.csv. The study target is appended after the plan targets in the single global loop; the study volume is the image read_series returned (one write_atomic call per destination).
- D7 key: `min((date == '', date, time))` over each written series' first header (stripped `_tag` text), then StudyInstanceUID. Gotcha: time is part of the key, so an undated study WITH a StudyTime sorts after an undated study without one.
- study series.csv rows: kind=series, reg_matrix and plan columns blank, k per modality per study in the global order; rows sorted by (IMAGE_MODALITIES index, k). Registered series appear in the plan folder (unchanged) and in a study folder; so a converted plan's `series.csv` and volumes are unchanged.
- Run log line is now: `convert: %d plan rows (...), %d imaging rows, %d studies rows, %d patient rows (...)`; plan rows = PLAN_FOLDER fullmatch.
- Tests: helpers in tests/test_series.py (`study`, `studies_row`, `series_rows_of`, `study_uids`, `set_study_date`, `STUDY_D1..D3`, `STUDY_T1/T2`, `study_rows`). Existing images/ tests moved to study_<n>/; assertions that a registered series left images/ empty now expect it in its study too.
- Gotcha: `put()` gives every series its own study (new study_uid), so tests that need a deterministic study number pin study_uid or study_date. Series UIDs from generate_uid sort before '1.9.x' text ('1.2.826...'), pin series_uid when a test depends on order.
- Not done here (other tasks): manifest/preprocess/CLI help/README still say images/ or ignore study folders; torchio-dependent tests skip locally (4 skipped, pre-existing).

## review-group-2
PASS, no BLOCKING. Deferred WARNING (file as hardening in 2.9): convert.py:291 a whitespace-only StudyInstanceUID may
group unrelated series into one study folder; treat a stripped-empty study as at_risk. NOTE: StudyTime compared as text
per spec.

## task-03

Files: src/rt_dataprep/preprocess.py, tests/test_preprocess.py (branch feat/task-03-imaging-cases). check_rows, preprocess_patient and preprocess_plan are unchanged (walking is task-05).
- `preprocess.check_study_rows(study_dir) -> list[dict]`: the series.csv rows (keys = conv.SERIES_COLUMNS, file order) or `CaseRefused(<reason>)`. Reads no NIfTI header; the caller reads headers with `read_info(study_dir / row['file'])` (a failure there is that series' own refusal, not the study's).
- `preprocess.study_refusal_count(study_dir) -> int`: data rows of series.csv, or 1 when the reader raises ValueError/csv.Error or there is no row. Needs series.csv to exist (OSError otherwise, like check_rows); call it only on folders that hold one. Meant for the manifest after `check_study_rows` raised.
- `preprocess.series_id(name) -> str | None`: D15 group 5 (`series_<MOD>_<k>`) of a file name, full match, None when the name is not a study volume. `preprocess.STUDY_FILE` is the compiled pattern. `case_dir = f'{patient}/{study_dir.name}/{series_id(file)}'`.
- Reasons: `REASON_STUDY_MODALITY`, `REASON_STUDY_NAME` (new); reused `REASON_SERIES_CSV` (unparseable, columns, None cells, zero rows, duplicate file, duplicate derived id), `REASON_KIND` (kind != series), `REASON_FILE_MISSING` (is_file and not symlink; text now "case folder"). `REASON_TOO_SMALL` now says "the image". Check order: csv shape/zero rows/duplicate file, kind, modality, name, duplicate id, presence.
- Not checked (not in the spec): that the modality column agrees with the MOD in the file name.
- `read_study_csv(study_dir) -> (rows, malformed)` is a small shared reader; `check_rows` still has its own inline copy.

## review-group-3
PASS, no BLOCKING. Deferred: STUDY_FILE `\d` matches non-ASCII digits (use [0-9]); study series.csv modality column is
not checked against the MOD in the file name (an edited MR label on a CT file would skip the HU rescale). Contract:
walkers call check_study_rows/study_refusal_count only on folders holding series.csv (absent -> OSError).

## task-04

Files: src/rt_dataprep/manifest.py, tests/test_manifest.py, tests/test_split.py (branch feat/task-04-imaging-cases, commit 23a6f8e).
- `manifest.COLUMNS` gains `case_type`, `modality`, `image` (13 columns). Plan rows: `plan`, `CT`, blank. Imaging row: case_dir `<patient>/study_<n>/series_<MOD>_<k>` (via `series_id`), fx blank, dose `0`, masks `[]`, rtplan_uid blank, ct_series_uid/study_uid from the series.csv row, image = the file name.
- Walk: after each patient's plan folders, folders fullmatching `STUDY_FOLDER = r'study_\d+'` that hold series.csv (hidden, `study_x`, `study_1b`, series.csv-less and `images/` are ignored, not counted). `_study_rows(patient, study)` -> (rows, refused): `check_study_rows` failure (CaseRefused/OSError) -> `study_refusal_count` (OSError there -> 1); then `read_info` per row, a CaseRefused/OSError there counts 1 and keeps siblings. No `raise` in any except (flag then act).
- `_geometry(size, spacing)` is the one formatter for plan and imaging ct_size/ct_spacing (np.float32 text). Log line unchanged: `manifest: %d written, %d refused`.
- manifest() still returns (written, refused); rows.sort() by case_dir as string (study_10 sorts before study_2, like plan_10).
- Tests: module fixture `master` in test_manifest.py converts a synthetic tree (plan_1 + CT study + 2-MR study, via test_series.put and test_convert.make_plan); `converted` copies it per test. 10 broken-study cases (counts 1 for unparseable/empty/header-only, 2 for the rest), 2 unreadable-header cases, walk-ignore test, caplog/capfd test. test_split: write_manifest writes 13 columns and takes `imaging=`; one new test (plan + imaging cases of a patient share a split, plus an imaging-only patient). split.py unchanged.
- Gotcha: the commit gate flags dunder setitem/delitem calls in test code as private-API access; use plain functions.
- Suite: 693 passed, 4 skipped (pre-existing torchio skips); ruff clean. Not done here: manifest on a preprocessed tree (task-05).

## review-group-4
PASS, no BLOCKING. Deferred: manifest.py:44-45 OSError on study series.csv counts 1 with no test. task-05 must cover
manifest on a preprocessed tree (same case_dir, prefixed image).

## task-05

Files: src/rt_dataprep/preprocess.py, tests/test_preprocess.py (branch feat/task-05-imaging-cases, commit 23de4bb). Suite 707 passed, 4 skipped (torchio, pre-existing); ruff clean.
- New in preprocess.py: `REASON_NO_CASES` (REASON_NO_PLANS removed), `STUDY_FOLDER = r'study_\d+'` (also defined in manifest.py; not merged to stay inside the task's files), `preprocess_study(study_dir, out_dir, steps) -> [PreprocessRow]`, `preprocess_series(study_dir, out_dir, row, steps) -> output name`.
- preprocess_patient: plan folders as before, then study folders (numeric order, fullmatch `study_\d+` holding series.csv). A patient with neither is skipped with REASON_NO_CASES. Plan code path unchanged.
- Per series: read_info -> too_small(info, steps) -> process_ct(read_image) -> write_atomic. MR/PT use `replace(steps, rescale=False)` and `output_name(..., ct=False)`. A refusal always happens before the write, and write_atomic is temp+replace, so a refused series never has an output file: no unlink code exists. The study folder is made on the first write (all refused = no folder; the patient folder is still renamed into place).
- Report: one row per series `study_<n>/series_<MOD>_<k>` (series_id of the converted name); study series.csv appended to the files of the first written row; check_study_rows refusal = one row `study_<n>`. REASON_INTERNAL is re-raised (outside any except) to the patient level; OSError propagates as for plans.
- Run log: `preprocess: %d plan rows (...), %d imaging rows (%d preprocessed, %d refused), %d patient rows (...)`; imaging rows = folder startswith `study_`.
- Tests (16 new + 3 edited): helper `write_study_tree(root, patient, study, series, n, spacing)` writes NIfTI + series.csv directly; one DICOM test (convert -> preprocess -> manifest on both trees, same case_dir, image = output_name of the converted image).
- Gotcha: the CLI help/README still say images/ (not this task). Under rescale-only steps an MR keeps its own name (no step applies).

## review-group-5
PASS, no BLOCKING. Deferred: STUDY_FOLDER defined in both preprocess.py:362 and manifest.py:26 (import it from
preprocess; move next to REASON_NO_CASES); preprocess.py:222 plan-row count uses `r not in imaging_rows` (quadratic).

## task-06

Files: src/rt_dataprep/__main__.py, README.md (branch feat/task-06-imaging-cases, commit 9a54585). Suite 707 passed, 4 skipped (torchio, pre-existing); ruff clean.
- convert help: "... one folder per plan and one per study"; manifest help: "... one row per plan folder or study series". No stale `images/` wording left in src/ or README (grep clean; manifest.py docstring was already updated in task-04).
- README: new "## Output layout" section before "PHI rules": tree sketch (SYNTHETIC_P1), plan folders vs study_<n>/series_<MOD>_<k>.nii.gz, registered series in both places, planning CT not repeated, study numbering, three manifest columns, case_dir as id with image as file name, MR/PT geometric steps only, different split for the same seed (existing splits.json refused), old images/ trees ignored and must be re-converted. Checked against convert.py (planning CT excluded at line ~264) and the findings above.
- No test change; tests/test_cli.py green.
