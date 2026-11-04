# Findings: F2a PR 2 (other series, registration, images/)

Spec: `docs/specs/2026-11-03-f2a-pr2-other-series.md` (addendum to `docs/specs/2026-11-03-f2a-convert.md`).
Plan: `docs/plans/f2a-other-series/plan.json`. Read both before exploring. Append under `## <task-id>` when done.

## Summary
- task-01: #33 narrows convert_plan's tag catch to individually wrapped header-only calls; an unexpected
  `Exception` per patient gives a REASON_INTERNAL row, the report is written, then ConvertAborted (fixed text).
- task-02: fixtures (write_reg RegistrationSequence; write_ct_series orientation/positions/PT/slopes/frames).
- task-03: series.py volume check (any orientation, SERIES_ATOL_MM 0.01, ORTHO_ATOL 1e-4) and geometry-owning read.
- task-04: REG parsing and reg_matrix = inv(M_ct) . M_series.
- task-05: convert_plan takes its headers; series.csv rows move out of the plan path.
- task-06: candidates, placement into plan folders / images/, report rows, plan-less patients.

## Facts checked before planning (2026-11-03)
- SimpleITK 2.5.7 ImageSeriesReader forces an orthogonal direction from the cosines and takes slice spacing as
  |last - first|/(n - 1).
- Its NIfTI writer never rejects a non-orthonormal direction; it coerces it to orthogonal with only a warning.
- organize puts excluded-model studies and CT localizers in misc_files/ (it filters cases only).

## Guardrails
- Commit gate: no added line with a dotted name whose last part starts with an underscore (any file type); no
  mrn/patient_id next to 6-10 digits; at most 5 staged files per commit (this file counts).
- Do not edit tests/test_organize.py or tests/test_dicom_io.py.
- PHI: fixed REASON_ constants; no path, ID, UID, date or description in logs or exception text; synthetic IDs
  without digits or underscores.
- Tests: `uv run --group dev pytest -q --tb=short`; lint: `uv run --group dev ruff check .`

## task-01
- Files: convert.py (REASON_INTERNAL, ConvertAborted, _header_call, _crash_site, per-call tag catch, _read_dose_sum split into
  candidates + header checks + pixel sum), __main__.py (ConvertAborted branch), tests/test_series.py (new), tests/test_cli.py.
- convert_plan no longer has the broad except; no existing test needed changing (168 of 169 pass).
- ruff B009 flags getattr with a constant name; the gate bans the dotted form, so those lines carry noqa: B009.
- Criterion 7 was amended by the user (the CLI's basicConfig puts the crash-site log line and the summary line on stderr):
  stdout empty, stderr's last line exactly 'rt_dataprep: ' + REASON_INTERNAL, no stderr line holds the exception message,
  a UID or a path. The log level and the summary line are unchanged; the CLI test asserts the amended wording.
## task-02
Fixtures in `tests/synthetic_dicom.py` (all new keywords optional; defaults checked identical to the old output tag for tag):
- `write_reg(..., frame_uid=None, registrations=None)`: `frame_uid` is the REG's own FrameOfReferenceUID.
  `registrations` is a list of dicts `frame_uid` (item frame, omitted if absent), `images` (SOP UIDs, omitted if absent)
  and `matrices` (required) = `[(type, 16 values)]`, all in ONE MatrixRegistrationSequence item's MatrixSequence.
  Without it no RegistrationSequence is written.
- `transformation_matrix(rotation=None, translation=(0, 0, 0))`: the 16 row-major values (M11 M12 M13 M14 M21 ...) of
  x_reg = R . x_item + t, i.e. item frame -> REG frame. A2.4 fixture (MR frame at +10 mm x of the CT frame, so
  p_ct = p_mr + (10, 0, 0)): REG in the CT frame = CT item `transformation_matrix()`, MR item
  `transformation_matrix(translation=(10, 0, 0))`; REG in the MR frame = MR item identity, CT item `translation=(-10, 0, 0)`.
- `write_ct_series(..., orientation=None, positions=None, rescales=None, number_of_frames=None, pixel_spacing=True)`:
  `orientation` = 6 cosines (slices then step `spacing[2]` along row x column); `positions` = one (x, y, z) per slice
  replacing the computed ones (jitter, gap, duplicate; must match the slice count); `rescales` = one (slope, intercept)
  per slice; `number_of_frames` sets the tag only (pixel data stays single-frame); `pixel_spacing=False` omits the tag.
  `modality='PT'` writes PET Image Storage (only the SOP class changes: no Units or other PET tags).
- Gotchas: `spacing` is (between columns, between rows, between slices) whatever the orientation. Sagittal
  `(0, 1, 0, 0, 0, -1)` has normal (-1, 0, 0), so slices step toward -x. Matrix values are written as `%.10g` strings
  (DS limit 16 chars). `images` items always carry the CT SOPClass UID. `matrices=[]` writes an item with an empty
  MatrixSequence; an item with no MatrixRegistrationSequence at all is not expressible (add a keyword if task-04 needs it).
  Any `modality` other than CT/MR/PT now raises KeyError (it used to fall back to CT Image Storage).

## review-1
No BLOCKING. WARNING: tests do not pin the tag-catch wrappers of masks.classify, _check_contour_images and
_dose_candidates (mutation check: removing them leaves the suite green); _read_dose outside the catch untested.
NOTE: synthetic_dicom sop_class KeyError on other modalities; _crash_site matches top-level package modules only; a
report-write OSError replaces ConvertAborted (CLI message still path-free). Deferred to the review round.

## task-03
- Files: src/rt_dataprep/series.py (new), tests/test_series.py (29 tests appended), this file. convert.py untouched.
- Public API (task-06 calls these; a skip is `None`, never an exception):
  - `check_volume(slices)`: `slices` = `[(path, header Dataset)]` of ONE series; returns the slices sorted ascending along
    row x column, or None. Header-only; any of the five tag-error types becomes None.
  - `read_series(slices)`: runs `check_volume` itself, then reads; returns the float32 `sitk.Image` with origin = first
    position, spacing = (column, row, step), direction = (row, column, normal) columns, or None (check skip, failed
    read, size mismatch). Callers need not call `check_volume` first.
  - Constants `SERIES_ATOL_MM = 0.01`, `ORTHO_ATOL = 1e-4`, `DIRECTION_ATOL = 1e-6`, `REASON_SERIES_UNREADABLE`
    (the quiet_call reason; never surfaces, read_series swallows the CaseRefused).
- Decisions: position tolerance is the Euclidean distance from the k-th point of the line `first + k*step*normal`
  (not per axis), so sideways and along-normal jitter share one 0.01 mm budget. Row and column are normalized, the normal
  is cross(row, column) normalized. Sort key = position . normal of the first input slice's orientation (all slices share it
  once the check passes). PixelSpacing equality uses SERIES_ATOL_MM (spec says "within SERIES_ATOL_MM").
- Gotchas:
  - SimpleITK 2.5.7 with `SetOutputPixelType(float32)` before Execute applies EACH slice's own RescaleSlope/Intercept:
    no per-slice pydicom fallback was needed (B5 test uses four different slopes; dropping the float32 line fails it).
  - ITK's reader already sets origin/spacing/direction itself, so a "geometry set outside quiet_call" test cannot compare
    values; it spies on `sitk.Image.SetOrigin/SetSpacing/SetDirection` and quiet_call instead.
  - pydicom refuses to assign a non-numeric string to NumberOfFrames (IS); a malformed-tag test uses None.
  - A truncated pixel buffer makes `ImageSeriesReader.Execute` raise (no silent zero fill).
## task-05
- Signature: `convert_plan(headers, unreadable, case_dir)`; `headers` is the plan folder's own `[(path, Dataset)]` (not deduplicated),
  `unreadable` refuses with REASON_FILE_UNREADABLE before anything is read. `read_folder_headers(folder) -> (headers, unreadable)`
  (public; sorted by path, `read_header`, so quiet) is the one reader; tests wrap it as `run_plan(folder, case)` in tests/test_convert.py.
- Headers are read in `_convert_patient`, after the target-exists and no-plans checks and before the partial folder is made: every
  `plan_<n>/` into `plan_headers`, `misc_files/` into the local `misc_headers` (read and kept, nothing consumes it yet: task-06).
  An unreadable misc file only produces `convert: N unreadable files in misc_files` (info). A patient without plans is returned
  before any read; task-06 decides what plan-less patients read.
- `PlanResult(files, ct_series_uid, series_rows)`; `series_rows` are the dicts `_series_rows` built (CT, dose, masks, applicators; no
  series.csv row). `_convert_plan` no longer writes series.csv.
- `_convert_one(plan_dir, headers, unreadable, partial)` returns `(ConvertRow, PlanResult | None)`. `_convert_patient` writes each converted
  plan's series.csv through `write_series_csv` after all plans ran, before the partial is renamed: this is where task-06 inserts
  placement (its extra files and rows go before this loop; a row's `files` already ends with series.csv).
- Gotchas: convert_plan's case folder now holds no series.csv after a direct call. tests/test_series.py (task-01) also called
  convert_plan, so it was restated too (3 calls, same assertions). A failed series.csv write is still the patient-level REASON_IO_ERROR.

## review-2
BLOCKING (fixed by orchestrator): a zero or negative PixelSpacing passed check_volume and SetSpacing raised out of
read_series; now refused in the check, with zero/negative-spacing and all-duplicate-position tests (the latter also
pins the step guard). WARNING: the single-slice guard is only pinned via a NaN path. NOTE: task-05 edited
tests/test_series.py outside its declared files (needed by the convert_plan signature change; logged, not a DAG
change); misc unreadable count re-scans the folder.
## task-04
- Files: src/rt_dataprep/series.py (REG block appended), tests/test_series.py (22 tests appended), this file. convert.py untouched.
- API (task-06 calls it): `registration(ct_frame, series_frame, regs, headers_by_sop) -> (registered: bool, reg_matrix: str)`.
  - `regs`: the patient's REG headers (Datasets), ALREADY deduplicated by SOPInstanceUID and without those in an excluded
    study: the caller does both (series.py does neither). `headers_by_sop`: `{SOPInstanceUID str: header}` of every readable
    file of the patient (excluded studies and localizers too); used only for items without a FrameOfReferenceUID.
  - Returns: `(True, 'identity')` when `ct_frame == series_frame` (regs are not even iterated); `(True, '<16 values>')`
    row-major, space-separated, `%.10g`, = inv(M_ct) . M_series; `(True, '')` when the first qualifying REG has no single usable
    rigid matrix for a frame; `(False, '')` when no REG qualifies. Constants `IDENTITY`, `RIGID_ATOL = 1e-4`, `LAST_ROW_ATOL = 1e-6`.
  - Caller still decides when to call it (only for a series that is not the planning CT itself, and never for `images/`, where
    reg_matrix stays empty): the function only knows frames.
- Decisions: "exactly one entry" in a MatrixSequence is counted over ALL MatrixRegistrationSequence items of the RegistrationSequence
  item (two registration items with one matrix each give ''; nothing is picked silently). A malformed tag inside a MATRIX
  (missing type, non-numeric value, ...) leaves reg_matrix '' but the REG still qualifies ("whatever its matrices hold"); a malformed
  tag anywhere else in the REG (RegistrationSequence item shape, ReferencedImageSequence entries, FrameOfReferenceUID, a missing
  SOPInstanceUID) makes the whole REG not qualify. An empty-string item frame counts as absent. A header among `headers_by_sop`
  lacking FrameOfReferenceUID is ignored when resolving. Non-Dataset entries in `regs` do not qualify.
- Gotchas: translation-only fixtures cannot tell inv(M_ct).M_series from M_series.inv(M_ct) (translations commute), so the rotation test
  carries the order check (mutation-checked: swapping the order, dropping the sort, dropping det > 0 each fail a test). `%.10g` can
  print `-0` or `1e-17` for a computed zero; left as is. An implied identity item exists only when the REG has no real item for that frame.
