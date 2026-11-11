# convert: DICOM metadata in series.json and manifest.jsonl (PR 2 of 2)

Status: REVIEWED (implemented after PR 1, in its own /dev run)

Depends on PR 1 (docs/specs/2026-11-09-json-outputs.md): series.json and manifest.jsonl exist.

## Problem
`series.json` keeps only a few header fields per output file. Patient, study, scanner and acquisition metadata that ML work
needs (covariates, scanner stratification, time ordering, PET dose, RT context) survives only in the DICOM copies under
`organized/`, and is lost if that tree is removed.

## Decisions (owner, 2026-11-09)
- Each series.json object gains one key, `dicom`: a JSON object holding the metadata below. Keys are present only when the
  value is present, so new tags can be added later without any schema change. The existing fields are unchanged.
- Patient data: age, sex, size, weight. No birth date, name or ethnic group is written.
- Dates and times: raw DICOM DA/TM/DT strings.
- Site and device identifiers are kept: manufacturer, model, software, institution, station, device serial, accession
  number, study ID, treatment machine. Outputs were PHI already (folder = PatientID); values appear only in series.json and
  manifest.jsonl and never in a log line, exception message or report CSV.
- Values are the DICOM values as written. Derived: `patient_age` (a number) and `slices` (a count) only.
- preprocess carries `dicom` through unchanged; manifest.jsonl gains 8 flat keys.

## Value rules
- One header's value of an element: absent tag, `None` or empty → absent. VM 1 → a string. VM > 1 (pydicom MultiValue) →
  a list of strings (never `str()` of the MultiValue, which renders as a Python list repr). Strings are `str()` of each
  value as pydicom gives it, stripped of surrounding whitespace.
- Aggregation over the row's headers (every slice of an image series; every file of a multi-file dose): absent values
  ignored; no value → key omitted; one distinct value → that value (distinct by numeric value for a VM 1 numeric element
that parses, else by the rendered string/list);
  several distinct values →
  - for a VM 1 element whose VR is numeric (DS, IS, FL, FD, US, UL, SS, SL) or DA/TM/DT: `{"min": a, "max": b}` with the
    first-seen texts of the extremes (numeric order for numbers, string order for DA/TM/DT: standard (colon-free) TM fields are fixed width from the left,
    so string order is time order across precision, `0900` < `090000.5`); if a value does not parse as a
    number, the element falls back to the next rule;
  - otherwise `{"distinct": [v1, v2, ...]}` in first-seen order.
- The VR is read from the element (`ds[keyword].VR`), not from a keyword list. For a sequence-item source, the same rules
  apply to the item's element. A header whose sequence is empty where the path takes item `[0]` is skipped like a
  header without the tag (#252).
- Per-cell guard: each key is computed under a guard that catches `AttributeError`, `KeyError`, `IndexError`, `TypeError`
  and `ValueError` (like `series._or_none`); a failure omits that key. A metadata value never refuses a file, plan or
  series and never aborts the run. Headers are already fully converted inside the quiet block by
  `dicom_io.read_header_core`, so reading them raises no pydicom warning.

### patient_age
Computed per header, then aggregated like a number. From PatientAge (`nnnY` → n as an integer; `nnnM` → n/12, `nnnW` →
n·7/365.25, `nnnD` → n/365.25, as numbers rounded to 2 decimals); else completed years between PatientBirthDate and the
first present of StudyDate, SeriesDate, AcquisitionDate; an unparseable or impossible date (`20230229`), that reference
date before the birth date (a negative age) or no dates → absent. Written as a JSON number. The birth date is read, never written.

## Which keys go on which row
Header sources: ct row = every planning-CT slice header (`_series_rows` receives `ct_slices`, not `ct_slices[0][1]`);
`series` rows (registered plan series and study series) = every slice header of the series (`ordered`, not `first`),
aggregated once per series and reused for each target folder; dose row = every RTDOSE file of the plan; mask, applicator
and line rows = the RTSTRUCT.

| row kind | groups |
| --- | --- |
| ct, series | All, Images, plus RTPLAN when in a plan folder, plus REG on registered plan-folder rows |
| dose | All, RTDOSE, RTPLAN |
| mask, applicator, line | All, RTSTRUCT, RTPLAN |

The Images group is read on every image row whatever the modality (a vendor may put an MR tag on another modality); it is
not read on dose rows. The RTPLAN group reaches series rows the way today's plan fields do (convert's PLAN_COLUMNS
mechanism carries plan values into each target folder's rows); in study folders it is absent. Concretely: a plan-folder
series row's `dicom` is its own aggregated keys merged with the RTPLAN-group keys of that plan's ct-row `dicom` (the
RTPLAN keys live inside `dicom`, not as top-level fields).

## Catalog (key → source)
### All
patient_age (derived), patient_sex PatientSex, patient_size PatientSize, patient_weight PatientWeight, study_date StudyDate,
study_time StudyTime, study_description StudyDescription, accession_number AccessionNumber, study_id StudyID, series_number
SeriesNumber, series_time SeriesTime, acquisition_date AcquisitionDate, acquisition_time AcquisitionTime, content_date
ContentDate, content_time ContentTime, manufacturer Manufacturer, model ManufacturerModelName, software_versions
SoftwareVersions, institution_name InstitutionName, station_name StationName, device_serial_number DeviceSerialNumber,
body_part_examined BodyPartExamined, patient_position PatientPosition, protocol_name ProtocolName.

### Images
image_type ImageType, rows Rows, columns Columns, slices (count of the row's slice headers, a JSON number), pixel_spacing
PixelSpacing, slice_thickness SliceThickness, spacing_between_slices SpacingBetweenSlices, contrast_agent ContrastBolusAgent;
CT: kvp KVP, tube_current XRayTubeCurrent, exposure Exposure, exposure_time ExposureTime, convolution_kernel
ConvolutionKernel, recon_diameter ReconstructionDiameter, ctdi_vol CTDIvol;
MR: field_strength MagneticFieldStrength, scanning_sequence ScanningSequence, sequence_variant SequenceVariant,
sequence_name SequenceName, acquisition_type MRAcquisitionType, repetition_time RepetitionTime, echo_time EchoTime,
inversion_time InversionTime, flip_angle FlipAngle, echo_train_length EchoTrainLength, b_value DiffusionBValue,
receive_coil ReceiveCoilName;
PT (from RadiopharmaceuticalInformationSequence[0], "R"): radiopharmaceutical R.Radiopharmaceutical,
radiopharmaceutical_code R.RadiopharmaceuticalCodeSequence[0].CodeMeaning, radionuclide
R.RadionuclideCodeSequence[0].CodeMeaning, injected_dose R.RadionuclideTotalDose (Bq), injection_time
R.RadiopharmaceuticalStartDateTime if any header of the row has it, else R.RadiopharmaceuticalStartTime (chosen once per
row), half_life R.RadionuclideHalfLife (s), units Units, decay_correction DecayCorrection, corrected_image CorrectedImage,
recon_method ReconstructionMethod.

### RTPLAN
plan_time RTPlanTime, plan_intent PlanIntent, approval_status ApprovalStatus, plan_description RTPlanDescription;
from sequences, each the distinct values over the items (one → value, several → `{"distinct": [...]}`):
prescription_dose DoseReferenceSequence[*].TargetPrescriptionDose, radiation_type BeamSequence[*].RadiationType, beam_type
BeamSequence[*].BeamType, treatment_machine BeamSequence[*].TreatmentMachineName else
TreatmentMachineSequence[*].TreatmentMachineName, source_isotope SourceSequence[*].SourceIsotopeName, source_strength
SourceSequence[*].ReferenceAirKermaRate, source_strength_date SourceSequence[*].SourceStrengthReferenceDate,
source_strength_time SourceSequence[*].SourceStrengthReferenceTime;
number_of_beams FractionGroupSequence[0].NumberOfBeams, brachy_technique BrachyTreatmentTechnique, brachy_type
BrachyTreatmentType.

### RTSTRUCT
structure_set_label StructureSetLabel, structure_set_date StructureSetDate, structure_set_time StructureSetTime;
for the row's ROI (by ROINumber): roi_interpreted_type (RTROIObservationsSequence item with that ReferencedROINumber →
RTROIInterpretedType), roi_generation_algorithm (StructureSetROISequence item with that ROINumber →
ROIGenerationAlgorithm); an item whose number is empty or not an integer is skipped. masks.py's applicator and line records gain the ROI number (today only `rois.kept` carries it);
there is no fallback by name.

### RTDOSE
dose_units DoseUnits, dose_type DoseType.

### REG (registered plan-folder series rows)
From the REG that the registration chose (also when it qualified but gave no usable matrix, i.e. `reg_matrix` ''):
reg_type RegistrationSequence[*].MatrixRegistrationSequence[*].MatrixSequence[*].FrameOfReferenceTransformationMatrixType
(distinct), reg_date ContentDate, reg_time ContentTime, reg_manufacturer Manufacturer. A new series.py helper returns
(registered, text, reg header or None); `registration()` stays a 2-tuple wrapper over it so its callers and tests are
unchanged. Same-frame (`identity`) rows have no REG keys.

## manifest.jsonl
Each line gains flat keys `patient_age`, `patient_sex`, `patient_weight`, `study_date`, `series_date`, `manufacturer`,
`model`, `institution_name`: from the `dicom` object of the case's primary image row (a plan's ct row; each study line's
own series row), `series_date` from the existing field. A key absent from `dicom` is absent from the line. This relaxes PR 1's
same-key-set rule for manifest lines: these 8 keys are optional per line; split ignores them. If a value is a
`{"min","max"}` or `{"distinct"}` object it is copied as is.

## Preprocess
preprocess's series.json keeps `dicom` unchanged on every object (its rows are copied with `retained` added). This PR
changes PR 1's reader: `dicom` is an optional key (present on all items of a file or none) whose value must be an object,
exempt from the string-value check. Key order on write follows PR 1 (SERIES_COLUMNS, then other keys in insertion order).

## Test data
`tests/synthetic_dicom.py` gains the fixtures the tests need (PT RadiopharmaceuticalInformationSequence, REG
MatrixSequence types, brachy SourceSequence, RTROIObservationsSequence, per-slice varying tags) instead of hand-built
headers inside test files.

## Acceptance
- A synthetic CT+RTSTRUCT+RTPLAN+RTDOSE plan with a registered MR (rigid REG), a brachy-style plan with a source, and a PT
  study convert with every catalog key present on the right rows and absent where it does not apply (test asserts the full
  key set per row kind, which catches keyword typos the guard would hide).
- Per-slice variation: XRayTubeCurrent and AcquisitionTime differing over slices → `{"min","max"}`; ImageType differing →
  `{"distinct": [[...], [...]]}`; PixelSpacing (VM 2) differing → `{"distinct"}`; a tag missing on some slices → the value
  of the rest; slices of `3` and `3.0` → `"3"` (numerically equal values are one value, first-seen text).
- patient_age: `063Y` → 63; `006M` → 0.5; no PatientAge with birth date and StudyDate → completed years; StudyDate blank →
  SeriesDate used; `abc` or `20230229` → absent, and the plan converts.
- Guard: an empty RadiopharmaceuticalInformationSequence, an RTPLAN without FractionGroupSequence and a malformed
  PatientAge each convert with the affected keys absent, no refusal, no abort.
- An RTSTRUCT with two ROIs of different RTROIInterpretedType gives each mask row its own type; an applicator row gets its
  own ROI's type.
- A two-file RTDOSE whose files differ in a tag (e.g. Manufacturer) → `{"distinct"}` on the dose row. (DoseType itself cannot
  differ end to end: convert refuses a DoseType other than PHYSICAL; its distinct path is unit-tested past that check.)
- A registered MR's rows carry the chosen REG's keys; the CT row has none.
- preprocess keeps `dicom`; manifest.jsonl has the 8 keys from the primary image.
- No new log line; caplog of the full synthetic run and of the guard and refused-plan cases holds no synthetic patient
  value.

## Release check (manual)
- Release v0.2.0 after PR 1 and PR 2; rerun the 20-folder OSC imaging subset with `convert --overwrite` then `manifest`;
  CT and MR objects have `dicom` with manufacturer, model, study_date and acquisition keys (the subset has no PT/RT/REG).
- Release note: v0.1.x converted trees must be reconverted (preprocess and manifest find no series.json in them).

## Deferred
- Normalized values (ISO dates, floats, MBq); days-from-first-study; date shifting for sharing.
- Beam-level detail (gantry/collimator angles, MU), ROI observation labels and colours, per-slice listings.
- Birth date, name, ethnic group.
- min/max of acquisition_date and acquisition_time are independent (a series across midnight loses the pairing); a NaN DS
  orders unpredictably.
