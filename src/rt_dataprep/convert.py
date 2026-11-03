"""
Turn an organized tree (<patient>/plan_<n>/) into NIfTI on the native CT grid, one folder per plan (spec: f2a-convert, F2a).

Per plan: RTPLAN -> RTSTRUCT -> the one CT series it references -> planning_ct.nii.gz (float32 HU), and the plan's RTDOSE
as dose.nii.gz (Gy, linearly resampled onto the CT grid). Masks and applicators (task-05) and series.csv (task-06) extend
`convert_plan`. The organized tree is only read. Log lines and exception messages carry counts and the fixed REASON_
constants only, never a PatientID, folder name, date, path or UID; convert_report.csv in the output root is the one place the
folder names appear. Every third-party read runs inside dicom_io's quiet wrapper.
"""
import contextlib
import csv
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from rt_dataprep.dicom_io import CaseRefused, quiet_call, read_header

log = logging.getLogger(__name__)

REPORT_NAME = 'convert_report.csv'
CT_NAME, DOSE_NAME = 'planning_ct.nii.gz', 'dose.nii.gz'
OUTCOME_CONVERTED, OUTCOME_SKIPPED, OUTCOME_REFUSED = 'converted', 'skipped', 'refused'
ORIGIN_ATOL_MM = 1e-3  # slice-spacing uniformity and duplicate positions (rnt gauge_rr.py)
DIRECTION_ATOL = 1e-6  # image orientation equality (rnt gauge_rr.py)
OFFSET_TOL_MM = 1e-3  # uniformity of the RTDOSE frame offsets (rnt rtdose.py)
SUMMATIONS = ('PLAN', 'FRACTION', 'BEAM')

REASON_OUTSIDE_ORG = 'output root is the organized root or inside it'
REASON_EXISTS = 'output folder already exists'
REASON_NO_PLANS = 'patient folder holds no plan folder'
REASON_IO_ERROR = 'file system error while reading or writing; nothing kept for this patient'
REASON_WRITE = 'an output file could not be written'
REASON_FILE_UNREADABLE = 'a DICOM file in the plan folder is unreadable'
REASON_MISSING_TAG = 'a header in the plan folder lacks a tag the conversion reads'
REASON_PLAN_COUNT = 'the plan folder does not hold exactly one RTPLAN'
REASON_NO_STRUCT = 'the RTPLAN structure set is not exactly one RTSTRUCT in the plan folder'
REASON_CT_SERIES = 'the RTSTRUCT does not reference exactly one CT series in the plan folder'
REASON_CT_NOT_VOLUME = 'the planning CT is not one uniform volume of at least two slices'
REASON_NO_CONTOUR_IMAGES = 'the RTSTRUCT lists no contour images for the planning CT'
REASON_CT_SLICE_MISSING = 'a CT slice the RTSTRUCT lists is not in the plan folder'
REASON_CT_UNREADABLE = 'the planning CT series is unreadable'
REASON_DOSE_RULE = 'the RTDOSEs of the plan fit no dose rule'
REASON_DOSE_UNITS = 'DoseUnits is not GY'
REASON_DOSE_TYPE = 'DoseType is not PHYSICAL'
REASON_DOSE_GRID = 'the RTDOSE frame offsets are not uniform or differ from NumberOfFrames'
REASON_DOSE_FRAME = 'the RTDOSE FrameOfReferenceUID differs from the planning CT'
REASON_DOSE_UNREADABLE = 'an RTDOSE is unreadable or lacks a required tag'


@dataclass
class ConvertRow:
    patient: str  # the organized folder name; convert_report.csv only
    folder: str  # plan_<n>, or '' for a patient-level row
    outcome: str
    reason: str = ''
    files: list = field(default_factory=list)
    skipped_series: int = 0


@dataclass
class ConvertReport:
    rows: list


@dataclass
class PlanResult:
    files: list  # names written into the case folder, in write order
    ct_series_uid: str


def convert(org_root, out_root, *, exclude_models=('Acuity',)):
    """
    Convert every <patient>/plan_<n>/ under org_root into out_root/<patient>/plan_<n>/ and write convert_report.csv.

    exclude_models is for the other-series placement of the next PR; the per-plan path does not use it.
    """
    org_root, out_root = Path(org_root), Path(out_root)
    if out_root.resolve().is_relative_to(org_root.resolve()):  # before any mkdir: nothing is written into the input
        raise ValueError(REASON_OUTSIDE_ORG)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in sorted(p for p in org_root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        try:
            rows += _convert_patient(folder, out_root)
        except OSError:  # its message names a path, i.e. the patient: one fixed reason, and no plan row of this patient
            rows.append(ConvertRow(folder.name, '', OUTCOME_REFUSED, REASON_IO_ERROR))
            log.warning('convert: %s', REASON_IO_ERROR)
    with open(out_root / REPORT_NAME, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient', 'folder', 'outcome', 'reason', 'files', 'skipped_series'])
        writer.writerows([r.patient, r.folder, r.outcome, r.reason, ';'.join(r.files), r.skipped_series] for r in rows)
    log.info('convert: %d report rows, %d plans converted, %d refused, %d skipped', len(rows),
             sum(r.outcome == OUTCOME_CONVERTED for r in rows), sum(r.outcome == OUTCOME_REFUSED for r in rows),
             sum(r.outcome == OUTCOME_SKIPPED for r in rows))
    return ConvertReport(rows)


def _convert_patient(folder, out_root):
    target = out_root / folder.name
    if target.exists():
        return [ConvertRow(folder.name, '', OUTCOME_SKIPPED, REASON_EXISTS)]
    plans = sorted((p for p in folder.iterdir() if p.is_dir() and re.fullmatch(r'plan_\d+', p.name)),
                   key=lambda p: int(p.name[len('plan_'):]))
    if not plans:
        return [ConvertRow(folder.name, '', OUTCOME_SKIPPED, REASON_NO_PLANS)]
    # Built under a hidden name and renamed at the end: a run that dies part-way leaves no folder that a later run would
    # skip as done. The flag, not an except block, decides the cleanup, so any failure (and Ctrl-C) removes the partial.
    partial = out_root / f'.{folder.name}.partial'
    shutil.rmtree(partial, ignore_errors=True)
    rows, done = [], False
    try:
        partial.mkdir()
        rows = [_convert_one(plan_dir, partial) for plan_dir in plans]
        partial.rename(target)  # an empty partial (every plan refused) is renamed too: the patient has its outcome
        done = True
    finally:
        if not done:
            shutil.rmtree(partial, ignore_errors=True)
    return rows


def _convert_one(plan_dir, partial):
    reason = None
    try:
        result = convert_plan(plan_dir, partial / plan_dir.name)
    except CaseRefused as exc:
        reason = str(exc)
    if reason is not None:
        log.warning('convert: plan refused: %s', reason)
        return ConvertRow(plan_dir.parent.name, plan_dir.name, OUTCOME_REFUSED, reason)
    return ConvertRow(plan_dir.parent.name, plan_dir.name, OUTCOME_CONVERTED, files=result.files)


def convert_plan(plan_dir, case_dir):
    """
    Convert one plan folder into `case_dir`, which must not exist; returns a PlanResult.

    A refusal raises CaseRefused(fixed reason) and leaves no `case_dir` behind. A header that lacks a tag the path reads
    refuses with REASON_MISSING_TAG. An OSError on the output side propagates: it is the patient's, not the plan's.
    """
    plan_dir, case_dir = Path(plan_dir), Path(case_dir)
    existed = case_dir.exists()
    reason = None
    try:
        result = _convert_plan(plan_dir, case_dir)
    except CaseRefused as exc:
        reason = str(exc)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):  # a missing tag or an odd value; not quoted
        reason = REASON_MISSING_TAG
    if reason is not None:  # raised here, after the except blocks: nothing of the original is chained
        if not existed:
            shutil.rmtree(case_dir, ignore_errors=True)
        raise CaseRefused(reason)
    return result


def _convert_plan(plan_dir, case_dir):
    headers = _read_headers(plan_dir)
    plan, struct, ct_slices, series_uid = _assemble(headers)
    _check_volume(ct_slices)
    _check_contour_images(struct, series_uid, ct_slices)
    ct = _read_ct([p for p, _ in ct_slices], ct_slices[0][1])
    dose = _read_dose_sum(headers, plan, str(ct_slices[0][1].FrameOfReferenceUID), ct)
    # Everything that can refuse has run: only now does the plan folder appear.
    case_dir.mkdir(parents=True)
    write_atomic(ct, case_dir / CT_NAME)
    files = [CT_NAME]
    if dose is not None:
        write_atomic(dose, case_dir / DOSE_NAME)
        files.append(DOSE_NAME)
    return PlanResult(files, series_uid)


def _read_headers(plan_dir):
    """[(path, Dataset)] of every file in the plan folder; any unreadable file refuses the plan (it may be a CT slice)."""
    headers, unreadable = [], False
    for path in sorted(p for p in plan_dir.iterdir() if p.is_file()):
        try:
            headers.append((path, read_header(path)))
        except OSError:
            unreadable = True
    if unreadable:
        raise CaseRefused(REASON_FILE_UNREADABLE)
    return headers


def _of(headers, modality):
    return [(p, ds) for p, ds in headers if ds.get('Modality') == modality]


def _referenced_series(struct):
    return [series for frame in struct.get('ReferencedFrameOfReferenceSequence', [])
            for study in frame.get('RTReferencedStudySequence', [])
            for series in study.get('RTReferencedSeriesSequence', [])]


def _assemble(headers):
    """(RTPLAN, RTSTRUCT, [(path, ds)] of the planning CT sorted ascending along the slice normal, its SeriesInstanceUID)."""
    plans = _of(headers, 'RTPLAN')
    if len(plans) != 1:
        raise CaseRefused(REASON_PLAN_COUNT)
    plan = plans[0][1]
    struct_uid = str(plan.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID)
    structs = [ds for _, ds in _of(headers, 'RTSTRUCT') if str(ds.SOPInstanceUID) == struct_uid]
    if len(structs) != 1:
        raise CaseRefused(REASON_NO_STRUCT)
    referenced = {str(series.SeriesInstanceUID) for series in _referenced_series(structs[0])}
    ct = [(p, ds) for p, ds in _of(headers, 'CT') if str(ds.SeriesInstanceUID) in referenced]
    uids = {str(ds.SeriesInstanceUID) for _, ds in ct}
    if len(uids) != 1:
        raise CaseRefused(REASON_CT_SERIES)
    iop = np.asarray([float(v) for v in ct[0][1].ImageOrientationPatient])
    normal = np.cross(iop[:3], iop[3:])
    ct.sort(key=lambda item: float(np.dot(normal, [float(v) for v in item[1].ImagePositionPatient])))
    return plan, structs[0], ct, uids.pop()


def _check_volume(ct_slices):
    """One orientation, size, pixel spacing and frame; at least two slices; no duplicate position; uniform spacing."""
    heads = [ds for _, ds in ct_slices]
    iops = np.asarray([[float(v) for v in ds.ImageOrientationPatient] for ds in heads])
    pixel_spacing = np.asarray([[float(v) for v in ds.PixelSpacing] for ds in heads])
    ok = (len(heads) >= 2
          and np.allclose(iops, iops[0], rtol=0, atol=DIRECTION_ATOL)
          and np.allclose(pixel_spacing, pixel_spacing[0], rtol=0, atol=ORIGIN_ATOL_MM)
          and len({(int(ds.Rows), int(ds.Columns)) for ds in heads}) == 1
          and len({str(ds.FrameOfReferenceUID) for ds in heads}) == 1)
    if ok:
        normal = np.cross(iops[0][:3], iops[0][3:])
        along = np.asarray([float(np.dot(normal, [float(v) for v in ds.ImagePositionPatient])) for ds in heads])
        steps = np.diff(along)  # the slices are sorted, so a duplicate position is a step of ~0
        ok = bool(steps[0] > ORIGIN_ATOL_MM and np.all(np.abs(steps - steps[0]) <= ORIGIN_ATOL_MM))
    if not ok:
        raise CaseRefused(REASON_CT_NOT_VOLUME)


def _check_contour_images(struct, series_uid, ct_slices):
    """Every slice the RTSTRUCT lists for the planning CT is present: a lost end slice still passes the spacing check."""
    items = [s for s in _referenced_series(struct) if str(s.SeriesInstanceUID) == series_uid]
    if not items or any(not s.get('ContourImageSequence') for s in items):
        raise CaseRefused(REASON_NO_CONTOUR_IMAGES)
    listed = {str(image.ReferencedSOPInstanceUID) for s in items for image in s.ContourImageSequence}
    if not listed <= {str(ds.SOPInstanceUID) for _, ds in ct_slices}:
        raise CaseRefused(REASON_CT_SLICE_MISSING)


def _read_ct(paths, first):
    """The slices (ordered along the normal) as one float32 HU image; float32 is set before Execute, so rescale slopes apply."""
    expected = (int(first.Columns), int(first.Rows), len(paths))

    def read():
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(p) for p in paths])
        reader.SetOutputPixelType(sitk.sitkFloat32)
        image = reader.Execute()
        if image.GetSize() != expected:
            raise ValueError
        return image
    return quiet_call(read, REASON_CT_UNREADABLE)


def _dose_candidates(headers, plan):
    """The RTDOSE (path, ds) pairs to use for the plan, [] when it has none; CaseRefused(REASON_DOSE_RULE) otherwise."""
    plan_uid = str(plan.SOPInstanceUID)
    found = [(p, ds) for p, ds in _of(headers, 'RTDOSE')
             if any(str(item.ReferencedSOPInstanceUID) == plan_uid for item in ds.get('ReferencedRTPlanSequence', []))]
    kind = lambda name: [c for c in found if str(c[1].DoseSummationType).strip() == name]  # noqa: E731
    plans, fractions = kind('PLAN'), kind('FRACTION')
    if not found:
        return []
    if len(plans) > 1:
        raise CaseRefused(REASON_DOSE_RULE)
    if plans:
        return plans
    if len(fractions) == 1:
        return fractions
    if not fractions and len(kind('BEAM')) == len(found):
        return found  # beams only: their sum is the plan dose
    raise CaseRefused(REASON_DOSE_RULE)


def _check_dose_header(ds, ct_frame_uid):
    """The ported rtdose _convert checks that need no pixels; returns the frame offsets."""
    if str(ds.DoseUnits).strip() != 'GY':
        raise CaseRefused(REASON_DOSE_UNITS)
    if str(ds.DoseType).strip() != 'PHYSICAL':
        raise CaseRefused(REASON_DOSE_TYPE)
    if str(ds.DoseSummationType).strip() not in SUMMATIONS:
        raise CaseRefused(REASON_DOSE_RULE)
    frames = int(ds.NumberOfFrames)
    offsets = np.asarray([float(o) for o in ds.GridFrameOffsetVector])
    steps = np.diff(offsets)
    if (frames < 2 or len(offsets) != frames or np.any(np.abs(steps - steps[0]) > OFFSET_TOL_MM) or steps[0] == 0):
        raise CaseRefused(REASON_DOSE_GRID)
    if str(ds.FrameOfReferenceUID) != ct_frame_uid:
        raise CaseRefused(REASON_DOSE_FRAME)
    return offsets


def resample_dose(dose, ct):
    """`dose` (Gy) linearly resampled onto ct's grid as float32; 0 where ct's voxel is outside the dose grid."""
    return sitk.Resample(dose, ct, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def _read_dose(path, offsets, ct):
    """The RTDOSE at `path` in Gy (pixel x DoseGridScaling), linearly resampled onto ct's grid as float32, 0 outside."""
    def read():
        ds = pydicom.dcmread(path)
        values = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)  # (frames, rows, columns)
        if values.shape[0] != len(offsets):
            raise ValueError
        # Row cosine runs along the columns (x), column cosine along the rows (y); frame k sits at
        # ImagePositionPatient + (o_k - o_0) n, which holds for relative (o_0 = 0) and absolute offsets alike.
        iop = np.asarray([float(v) for v in ds.ImageOrientationPatient])
        r, c = iop[:3], iop[3:]
        n = np.cross(r, c)
        origin = np.asarray([float(v) for v in ds.ImagePositionPatient])
        step = offsets[1] - offsets[0]
        if step < 0:  # reorder so the image has a positive slice spacing, its origin at the last frame
            values = values[::-1]
            origin = origin + (offsets[-1] - offsets[0]) * n
        row_spacing, column_spacing = (float(v) for v in ds.PixelSpacing)
        dose = sitk.GetImageFromArray(values)
        dose.SetSpacing((column_spacing, row_spacing, abs(float(step))))
        dose.SetDirection(tuple(np.column_stack([r, c, n]).ravel()))
        dose.SetOrigin(tuple(origin))
        return resample_dose(dose, ct)
    return quiet_call(read, REASON_DOSE_UNREADABLE)


def _read_dose_sum(headers, plan, ct_frame_uid, ct):
    """The plan dose on the CT grid (Gy, float32), or None when the plan has no RTDOSE."""
    used = _dose_candidates(headers, plan)
    offsets = [_check_dose_header(ds, ct_frame_uid) for _, ds in used]  # every used dose is checked before any is read
    images = [_read_dose(path, off, ct) for (path, _), off in zip(used, offsets, strict=True)]
    if not images:
        return None
    total = images[0]
    for image in images[1:]:
        total = total + image
    return total


def write_atomic(image, path):
    """
    Write `image` to `path` through a temp file in the same folder (its name ends .nii.gz: SimpleITK picks the writer by
    extension) and os.replace. A failure of either step raises OSError(REASON_WRITE), and the temp file is removed.
    """
    path = Path(path)
    temp = path.with_name(f'.{os.getpid()}.tmp.{path.name}')
    failed = False
    try:
        try:
            quiet_call(lambda: sitk.WriteImage(image, str(temp)), REASON_WRITE)  # an ITK writer message would name the path
        except CaseRefused:
            failed = True
        if not failed:
            os.replace(temp, path)
    finally:
        with contextlib.suppress(OSError):
            temp.unlink()
    if failed:
        raise OSError(REASON_WRITE)
