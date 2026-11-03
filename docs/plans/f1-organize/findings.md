# F1 organize: shared findings

Spec: `docs/plans/2026-11-03-rt-dataprep/phase1.md` (section F1) and `proposal.md`. Plan: `plan.json` here.
Design was settled interactively in the planning session (2026-11-03); no separate spec file.

- Source functions to port live in `/home/dominic/Repositories/rnt_transformer` and `/home/dominic/Repositories/DICOM_Transport` (read-only; never edit them).
- Vault trap: pydicom validates lazily on first tag access; quiet the whole access (see project-profile.yml review rules).
- PHI: tests use only synthetic data generated in tmp_path.

## Decisions (orchestrator)
- No separate fresh-context plan-review: phase1.md already had two advisor passes; plan.json had one more.
- Tasks run sequentially in the session worktree (task-02 depends on task-01), not per-task worktrees.
- DicomRTTool, SimpleITK and the mclahe extra are deferred to F2: mclahe pins numpy 1.26 in a universal uv lock.
- Implementer never sets CLAUDE_ALLOW_PRIVATE_API; any third-party private access is reported to the orchestrator.

## task-01
Files: `pyproject.toml`, `uv.lock`, `README.md`, `src/rt_dataprep/{__init__,__main__,dicom_io}.py`, `tests/synthetic_dicom.py`, `tests/test_dicom_io.py`.

Decisions
- `__main__.main(argv=None)`: argparse with required subcommand `organize SRC OUT`; after parsing it calls `parser.error('organize is not wired up yet')`. task-02 replaces that line with the import and call of `organize`.
- `dicom_io.read_header(path)` raises `OSError(REASON_UNREADABLE)` for any failure (non-DICOM, truncated, missing) with `from None`, so no path reaches the message. `_pydicom_quiet` also wraps `warnings.catch_warnings` + `simplefilter('ignore')`, then forces `_ = element.value` for every element.
- Runtime deps are `pydicom>=3` and `numpy`. pytest `pythonpath = ["tests"]` so tests do `import synthetic_dicom as sd`.
- `[tool.ruff.lint] select = ["E4","E7","E9","F","B"]` is pinned: ruff 0.17's built-in default is much wider (C408, BLE001, RUF059, I001 ...) and would fail the ported file. Ported code got `zip(..., strict=True)` for B905.

Gotchas
- pydicom validates lazily; the test asserts no warnings and no log records with the forced conversion, and fails (PN length warning) when the loop is removed (checked).
- `dcmread` raises `InvalidDicomError`, which is not an OSError, hence the broad catch in `read_header`.
- Test for an empty `__init__` reads the file text, since `vars(rt_dataprep)` gains submodules once imported.

Synthetic writers (tests/synthetic_dicom.py). Shared `**meta` keywords on every writer, forwarded to `_dataset`: `patient_id`, `study_uid`, `series_date`, `series_time`, `manufacturer_model_name`. Unset means: PatientID `SYNTHETIC_ID`, fresh StudyInstanceUID, date/time/model tags omitted. Each writer returns the SOPInstanceUID except the CT writer.
- `write_ct_series(folder, values, spacing=(1.0,1.0,1.0), origin=(0.0,0.0,0.0), series_uid=None, frame_uid=None, prefix='ct', image_type=None, **meta)` returns `{'series_uid','frame_uid','study_uid','sop_uids','paths'}`; all slices share one study; `image_type` is the ImageType list (None omits it); series_date/series_time/manufacturer_model_name apply to every slice.
- `write_rtstruct(path, ct, sop_uid=None, extra_series_uids=(), **meta)`
- `write_rtplan(path, struct_sop_uid, positions_mm, times=None, fractions=1, sop_uid=None, frame_uid=None, dose_references=None, channels=None, plan_date=None, plan_time=None, **meta)` plan_date/plan_time set RTPlanDate/RTPlanTime.
- `write_rtstruct_contours(path, contours, sop_uid=None, brachy=True, **meta)`
- `write_rtdose(path, stored, scaling=1e-4, position=(0,0,0), orientation=(1,0,0,0,1,0), pixel_spacing=(2.0,3.0), offsets=None, plan_sop_uid=None, frame_uid=None, units='GY', dose_type='PHYSICAL', summation='PLAN', number_of_frames=None, invalid_values=False, sop_uid=None, **meta)`
- `write_reg(path, study_uid, other_study_uid, sop_uid=None, **meta)`: Modality REG in `study_uid`, with `StudiesContainingOtherReferencedInstancesSequence` naming `other_study_uid` (a UID string or a list of them).
- The RTSTRUCT writer has no CT-study default: pass `study_uid=ct['study_uid']` (and the same for RTPLAN/RTDOSE/REG) to put objects in the CT's study; the default is a fresh study each call.

## review-group-1
- APPROVE, no BLOCKING. W1 (no MR writer for A1.2) fixed by orchestrator: `write_ct_series(..., modality='MR')`.
- W2 (distinct PatientIDs can sanitize to the same folder; step 6 would silently skip the second): this touches patient
  matching, so task-02 must detect a sanitized-name collision between different PatientIDs in one run and refuse the
  later patient with a fixed reason counted in organize_report.csv (no silent skip). An existing folder from a
  previous run is still skipped and counted as such.
- NOTE: README hook line fails in a worktree (`.git` is a file); NOTE: `raise ... from None` keeps context on the object.

## task-02
Files: `src/rt_dataprep/organize.py` (new), `src/rt_dataprep/__main__.py` (calls `organize`, logging at INFO), `tests/test_organize.py` (new, 33 tests in the whole suite).

Decisions
- `organize(...)` returns `OrganizeReport(rows)`; each `PatientRow` has patient, fx, misc, unreadable, outcome (`organized`/`skipped`/`refused`), reason. `organize_report.csv` is rewritten each run (columns patient, fx_count, misc_count, unreadable_count, outcome, reason).
- Collision guard (review W2): `claimed` holds the casefolded output names of every patient folder seen in the run, including skipped ones. A later folder with the same name is refused with `REASON_NAME_CLAIMED`, never merged. This also refuses a second source folder with the same PatientID. Casefold because a case-insensitive filesystem would merge the folders.
- Outside the plan, patient matching, decided here: a source folder whose readable files carry more than one PatientID is refused (`REASON_MIXED_IDS`); a folder with no readable DICOM is refused (`REASON_NO_DICOM`). The CSV id for those rows is the sanitized source folder name. No output folder is made.
- "Planning study" (step 3, the REG rule) = any study holding an RTPLAN, not only the CT study of a complete chain. Wider, so a REG never pulls another plan's study in as "diagnostic".
- Step 3 says "every file of a study named by a REG"; the copy of those studies also drops RTPLANs and plan-naming RTDOSEs, so every fx folder holds exactly one RTPLAN (F2 relies on this, phase1.md F2 step 1). Such files go to misc_files.
- A1.5b follows step 3 literally: each fx folder holds both CT series and both RTSTRUCTs of the shared study, and its own plan only.
- RTSTRUCT -> CT: the referenced series UIDs that match a CT series in the pool; exactly one must match, otherwise the plan is left over (ambiguous = incomplete).
- Pre-filter: localizer CT files and `exclude_models` studies (case-sensitive substring) are removed before grouping and so land in misc_files; a file with no StudyInstanceUID is left over too. Unreadable files are counted and not copied.
- A plan-bound RTDOSE (names a plan) goes to the folder of each plan it names; one naming no plan is an ordinary study file.
- Same file name from two source sub-folders into one destination: the second gets `_<n>` before the suffix (copy2 would overwrite silently).
- Containment: `target.resolve().is_relative_to(out_root.resolve())`, raising `ValueError(REASON_OUTSIDE_ROOT)` (not `assert`, which -O strips). sanitize makes it unreachable in tests.
- Sort key: strings of (CT SeriesDate, CT SeriesTime, RTPlanDate, RTPlanTime, plan SOPInstanceUID); a missing tag sorts first as ''. The CT is the series file with the lowest SOPInstanceUID.

Gotchas
- The synthetic writers do not create folders; tests use an `mk()` helper.
- `write_rtstruct`/`write_rtplan`/`write_rtdose` default to a fresh study; the test helper passes `study_uid` every time.
- Mutation-checked: removing the collision guard, the REG planning-study exclusion, or the plan-bound filter each fails tests.

Deferred (not fixed here)
- organize_report.csv is overwritten, so a rerun replaces the counts of an earlier run for skipped patients.
- An `out_root` inside `src_root` would be walked as a patient folder; `*.dcm` is matched case-sensitively.

## review-loop-pass-1
- BLOCKING fixed (orchestrator): fx ordering was unproven (removing the sort passed every test). A1.1 and A1.5 now have
  name order opposite to date order; removing the sort fails 2 tests.
- BLOCKING fixed (orchestrator): an OSError mid-copy leaked the output path (sanitized PatientID) and left a partial
  patient that a rerun skipped as done. Patients are now built under `.<name>.partial` and renamed at the end; an
  OSError refuses that patient with REASON_IO_ERROR, removes the partial, and the run continues; the CLI turns an
  OSError on the roots into a fixed message and exit 1. Removing the cleanup fails a test.
- Deferred to hardening issues: RTSTRUCT in a different study than its CT; missing SOPInstanceUID aborts the run;
  missing SeriesDate sorts first; only `*.dcm` is read; duplicate plan file in two sub-folders gives two identical fx
  folders; one blank PatientID refuses the whole patient; unreadable files not copied to misc_files; report written
  only at the end and overwritten on rerun; out_root inside src_root.
