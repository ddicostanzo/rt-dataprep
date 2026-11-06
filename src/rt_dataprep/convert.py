"""
Turn an organized tree (<patient>/plan_<n>/) into NIfTI on the native CT grid, one folder per plan and one per study (spec: f2a-convert, F2a).

Per plan: RTPLAN -> RTSTRUCT -> the one CT series it references -> planning_ct.nii.gz (float32 HU), and the plan's RTDOSE
as dose.nii.gz (Gy, linearly resampled onto the CT grid). Masks and applicators (masks.py) are written into the same
folder. Every other CT/MR/PT series of the patient (series.py) is read once and written into the plan folders it is registered
to and into its study folder <patient>/study_<n>/ (numbered by StudyDate, StudyTime, StudyInstanceUID). series.csv is written last, one row per file. The organized tree is only read. Log lines and exception messages
carry counts and the fixed REASON_ constants only, never a PatientID, folder name, date, path or UID. convert_report.csv in
the output root is the one place the folder names appear; the series.csv files hold the UIDs, dates and descriptions. Every third-party read runs inside dicom_io's quiet wrapper.
"""
import contextlib
import csv
import logging
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.datadict import tag_for_keyword

from rt_dataprep import dicom_io, masks, pool
from rt_dataprep import series as ser
from rt_dataprep.dicom_io import CaseRefused, quiet_call, write_report
from rt_dataprep.series import DIRECTION_ATOL

log = logging.getLogger(__name__)

PLAN_FOLDER = r'(plan|fx)_\d+'  # organize writes plan_<n>; a legacy tree (OSC raw/) has fx_<n>, kept as named
REPORT_NAME = 'convert_report.csv'
CT_NAME, DOSE_NAME, SERIES_NAME = 'planning_ct.nii.gz', 'dose.nii.gz', 'series.csv'
SERIES_COLUMNS = ['file', 'kind', 'modality', 'series_description', 'series_instance_uid', 'study_instance_uid',
                  'frame_of_reference_uid', 'series_date', 'roi_name', 'dose_summation_type', 'reg_matrix', 'rtplan_uid',
                  'rtplan_label', 'rtplan_date', 'fractions_planned']
OUTCOME_CONVERTED, OUTCOME_SKIPPED, OUTCOME_REFUSED = 'converted', 'skipped', 'refused'
ORIGIN_ATOL_MM = 1e-3  # slice-spacing uniformity and duplicate positions (rnt gauge_rr.py)
OFFSET_TOL_MM = 1e-3  # uniformity of the RTDOSE frame offsets (rnt rtdose.py)
SUMMATIONS = ('PLAN', 'FRACTION', 'BEAM')
PLAN_COLUMNS = ('rtplan_uid', 'rtplan_label', 'rtplan_date', 'fractions_planned')

REASON_OUTSIDE_ORG = 'output root is the organized root or inside it'
REASON_EXISTS = 'output folder already exists'
REASON_NAME_CLAIMED = 'another folder of this run already maps to this output name (case-insensitive)'
REASON_NO_PLANS = 'patient folder holds no plan folder'
REASON_NO_STUDIES = 'every other series was skipped; no study folder was written'
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
REASON_INTERNAL = 'an unexpected error stopped the conversion; the report lists what was finished'
STUDIES_NAME = 'studies'  # folder of the patient's summary row for the study folders
STUDY_SERIES = r'study_\d+/series_(CT|MR|PT)_\d+'  # folder of a per-series row
PACKAGE_DIR = Path(__file__).parent


class ConvertAborted(Exception):
    """A patient hit an unexpected error: the report is written, the run stops. str() is exactly REASON_INTERNAL."""


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
    files: list  # the data files written into the case folder, in write order; series.csv, written last, is not in it
    ct_series_uid: str
    series_rows: list  # dicts of SERIES_COLUMNS for the plan's series.csv, which the patient level writes last


def convert(org_root, out_root, *, exclude_models=('Acuity',), workers=1, overwrite=False, _patient_fn=None):
    """
    Convert every <patient>/plan_<n>/ under org_root into out_root/<patient>/plan_<n>/ and write convert_report.csv.

    exclude_models: a study that holds a file whose ManufacturerModelName contains one of these is left out of the other series.
    workers: patient processes, one patient per process (about 7 GB RAM each at 701x701x433); the output is the same for any N.
    overwrite: replace a patient's existing output folder once the new output holds a converted row; otherwise the old one is kept.
    _patient_fn: internal test seam, replaces the per-patient call only (module-level and picklable for workers > 1).
    """
    org_root, out_root = Path(org_root), Path(out_root)
    if out_root.resolve().is_relative_to(org_root.resolve()):  # before any mkdir: nothing is written into the input
        raise ValueError(REASON_OUTSIDE_ORG)
    if overwrite and org_root.resolve().is_relative_to(out_root.resolve()):  # a replaced patient folder could hold the input
        raise ValueError(dicom_io.REASON_INPUT_INSIDE_OUTPUT)
    folders = sorted(p for p in org_root.iterdir() if p.is_dir() and not p.name.startswith('.'))  # a missing root fails before the mkdir
    out_root.mkdir(parents=True, exist_ok=True)
    patient_fn = _patient_fn or _convert_patient
    # The parent owns the claims. Under overwrite a repeat of a casefolded name (a case-insensitive root would merge them) is refused here,
    # undispatched; otherwise the repeat is dispatched under the same hold key, so it starts after the first one ended.
    dispatched, refused, claimed = [], {}, set()
    for i, folder in enumerate(folders):
        if overwrite and folder.name.casefold() in claimed:
            refused[i] = [ConvertRow(folder.name, '', OUTCOME_REFUSED, REASON_NAME_CLAIMED)]
        else:
            dispatched.append(i)
        claimed.add(folder.name.casefold())
    tasks = [pool.Task(_convert_worker, (patient_fn, folders[i], out_root, exclude_models, overwrite), folders[i].name.casefold())
             for i in dispatched]
    results, aborted = {}, False  # folder index -> (rows, reading seconds, header files, status, seconds, internal)

    def handle(k, result):
        nonlocal aborted
        i = dispatched[k]
        if isinstance(result, pool.Failed):  # escaped the worker wrapper (a broken pool...): only the type name is known
            if result.type_name == 'SystemExit':  # as before: exits the run with no row; the report is still written
                raise SystemExit(result.code)  # its sanitized code, as quiet_call keeps it
            log.error('convert: %s: %s', REASON_INTERNAL, result.type_name)
            result = [ConvertRow(folders[i].name, '', OUTCOME_REFUSED, REASON_INTERNAL)], 0.0, 0, None, 0.0, True
        results[i] = result
        if result[-1]:
            aborted = True
            return pool.ABORT

    started = time.monotonic()
    try:
        pool.run(workers, tasks, handle)
    finally:  # Ctrl-C too: the report lists what was done
        rows = rows_in_order(len(folders), refused, {i: r[0] for i, r in results.items()})
        write_report(out_root, REPORT_NAME, ['patient', 'folder', 'outcome', 'reason', 'files', 'skipped_series'],
                     ([r.patient, r.folder, r.outcome, r.reason, ';'.join(r.files), r.skipped_series] for r in rows))
    # A refused or aborted patient adds no reading seconds or files; every patient adds its own time minus its reading.
    reading, files = sum(r[1] for r in results.values()), sum(r[2] for r in results.values())
    converting = sum(r[4] - r[1] for r in results.values())
    replaced, kept = (sum(r[3] == status for r in results.values()) for status in ('replaced', 'kept'))
    plan_rows = [r for r in rows if re.fullmatch(PLAN_FOLDER, r.folder)]
    patient_rows = [r for r in rows if r.folder == '']
    log.info('convert: %d plan rows (%d converted, %d refused, %d skipped), %d imaging rows, %d studies rows, %d patient rows (%d refused, %d skipped); %d replaced, %d kept',
             len(plan_rows), *(sum(r.outcome == o for r in plan_rows) for o in (OUTCOME_CONVERTED, OUTCOME_REFUSED, OUTCOME_SKIPPED)),
             sum(bool(re.fullmatch(STUDY_SERIES, r.folder)) for r in rows), sum(r.folder == STUDIES_NAME for r in rows), len(patient_rows),
             *(sum(r.outcome == o for r in patient_rows) for o in (OUTCOME_REFUSED, OUTCOME_SKIPPED)), replaced, kept)
    log.info('convert: %.1f s total, %.1f s reading (summed over patients), %.1f s converting (summed over patients), %d files, workers %d',
             time.monotonic() - started, reading, converting, files, workers)
    if aborted:  # raised here: nothing of the original is chained
        raise ConvertAborted(REASON_INTERNAL)
    return ConvertReport(rows)


def rows_in_order(count, refused, done):
    """
    The rows of folders 0..count-1 in order: `done` maps a folder index to its patient's rows, `refused` to the rows of a repeat that was
    never dispatched. A repeat has a row only if every dispatched folder before it has one: the run got that far.
    """
    rows, reached = [], True
    for i in range(count):
        if i in refused:
            rows += refused[i] if reached else []
        elif i in done:
            rows += done[i]
        else:
            reached = False
    return rows


def _convert_worker(patient_fn, folder, out_root, exclude_models, overwrite):
    """
    One patient in a worker (or inline): patient_fn's (rows, reading seconds, header files, status) plus the patient's seconds and an
    internal flag. A failure becomes data here, as a traceback does not cross the process boundary: an OSError is the patient's
    REASON_IO_ERROR row, any other Exception its REASON_INTERNAL row, with only the type and the crash site logged (no message).
    """
    started = time.monotonic()
    try:
        return (*patient_fn(folder, out_root, exclude_models, overwrite), time.monotonic() - started, False)
    except OSError:  # its message names a path, i.e. the patient: one fixed reason, and no plan row of this patient
        log.warning('convert: %s', REASON_IO_ERROR)
        return [ConvertRow(folder.name, '', OUTCOME_REFUSED, REASON_IO_ERROR)], 0.0, 0, None, time.monotonic() - started, False
    except Exception as exc:  # not KeyboardInterrupt/SystemExit: those keep the flag-based cleanup and add no row
        module, line = crash_site(sys.exc_info()[2])
        log.error('convert: %s: %s at %s:%d', REASON_INTERNAL, getattr(type(exc), '__name__'), module, line)  # noqa: B009 (no message)
        return pool.Stop(([ConvertRow(folder.name, '', OUTCOME_REFUSED, REASON_INTERNAL)], 0.0, 0, None, time.monotonic() - started, True))


def crash_site(tb):
    """
    (module name, line) of the innermost rt_dataprep frame of `tb` outside dicom_io (quiet_call raises every failure from its
    own line, so its caller is the site); the file name only selects it and is never formatted.
    """
    site = ('unknown', 0)
    for frame, lineno in traceback.walk_tb(tb):
        if Path(frame.f_code.co_filename).parent == PACKAGE_DIR and frame.f_globals['__name__'] != 'rt_dataprep.dicom_io':
            site = (frame.f_globals['__name__'], lineno)
    return site


def _convert_patient(folder, out_root, exclude_models, overwrite):
    """(rows, reading seconds, header files read, status None/'new'/'replaced'/'kept'); a raise adds neither of the middle two."""
    target = out_root / folder.name
    dicom_io.recover(out_root, folder.name)
    exists = os.path.lexists(target)  # lexists, as publish: a dangling link is an existing target
    if exists and not overwrite:
        return [ConvertRow(folder.name, '', OUTCOME_SKIPPED, REASON_EXISTS)], 0.0, 0, None
    plans = sorted((p for p in folder.iterdir() if p.is_dir() and re.fullmatch(PLAN_FOLDER, p.name)),
                   key=lambda p: (int(p.name.split('_')[1]), p.name))  # plan_1 and fx_1: a fixed order
    # Headers are read once per patient. An unreadable file in misc_files/ is not a series and is ignored: only its count
    # reaches the log.
    read_started = time.monotonic()
    plan_headers = [read_folder_headers(plan_dir) for plan_dir in plans]
    misc_dir = folder / 'misc_files'
    misc_headers, misc_unreadable = read_folder_headers(misc_dir) if misc_dir.is_dir() else ([], False)
    reading = time.monotonic() - read_started
    files = sum(len(found) for found, _ in plan_headers) + len(misc_headers)  # a value: misc_headers is filtered below
    # An unreadable misc file may be a lost slice of any series of its study: those are skipped (#36). Its study is read
    # with force=True; when even that fails, every misc-only series of the patient is skipped.
    misc_taint = set(), False
    if misc_unreadable:
        read = {p for p, _ in misc_headers}
        studies = [_salvage_study(p) for p in sorted(misc_dir.iterdir()) if p.is_file() and p not in read]
        log.info('convert: %d unreadable files in misc_files', len(studies))
        misc_taint = {s for s in studies if s}, None in studies
    blank = [ds for _, ds in misc_headers if not str(ds.get('PatientID', '')).strip()]
    misc_headers = [(p, ds) for p, ds in misc_headers if str(ds.get('PatientID', '')).strip()]  # blank id: kept, never placed (#12)
    headers = [item for found, _ in plan_headers for item in found] + misc_headers  # folder order: plan_<n> by n, then misc_files
    # A series with a blank-id slice in misc_files is short without it: known short, like a slice-missing planning CT. A blank-id
    # copy of a slice already present (same SOPInstanceUID) takes nothing away, and a blank or whitespace UID is no series (#71).
    pair = lambda ds: (_tag(ds, 'SOPInstanceUID'), _tag(ds, 'SeriesInstanceUID'))  # noqa: E731 (a SOP alone may collide, #81)
    present = {pair(ds) for _, ds in headers}
    series_of = lambda items: {u for ds in items if (u := _tag(ds, 'SeriesInstanceUID')).strip()}  # noqa: E731
    split = series_of(ds for ds in blank if pair(ds) not in present) & series_of(ds for _, ds in headers)
    if not plans and not ser.candidates(headers, exclude_models)[0]:
        return [ConvertRow(folder.name, '', OUTCOME_SKIPPED, REASON_NO_PLANS)], reading, files, 'kept' if exists else None
    # Built under a hidden name and renamed at the end: a run that dies part-way leaves no folder that a later run would
    # skip as done. The flag, not an except block, decides the cleanup, so any failure (and Ctrl-C) removes the partial.
    partial = out_root / f'.{folder.name}.partial'  # recover, above, already removed a leftover
    rows, done, status = [], False, None
    try:
        partial.mkdir()
        outcomes = [_convert_one(plan_dir, found, unreadable, partial)
                    for plan_dir, (found, unreadable) in zip(plans, plan_headers, strict=True)]
        # A slice-missing refusal names its planning CT: short wherever its copies are. Assembly already passed for that plan.
        lost = {_assemble(found)[3] for (found, _), (row, _) in zip(plan_headers, outcomes, strict=True)
                if row.reason == REASON_CT_SLICE_MISSING} | split
        outcomes = [_refuse_on_lost_ct(row, result, partial, lost) for row, result in outcomes]
        rows = [row for row, _ in outcomes]
        # After every plan has its outcome: which CTs are planning CTs, and which plans a series is registered to, depend on it.
        # A plan refused for a lost or unreadable file may hold a CT missing an end slice, which still passes the volume check.
        short = {plan_dir for plan_dir, (row, _) in zip(plans, outcomes, strict=True)
                 if row.reason in (REASON_FILE_UNREADABLE, REASON_CT_SLICE_MISSING)}
        # Organize copies a study whole into every folder that holds it, so a converted folder's copy is complete: the
        # SOP dedup keeps it over a short folder's copy (#35). Only a SOP found nowhere else stays in a short folder.
        ordered = sorted(headers, key=lambda item: item[0].parent in short)
        rows += _place_series(folder.name, ordered, exclude_models, outcomes, partial, short, lost, misc_taint, misc_dir)
        for row, result in outcomes:  # series.csv of each converted plan folder goes in last
            if result is not None:
                write_series_csv(result.series_rows, partial / row.folder / SERIES_NAME)
        # An empty partial (every plan refused) is renamed too when there is no old folder: the patient has its outcome.
        status = dicom_io.publish(out_root, folder.name, overwrite=overwrite,
                                  success=any(row.outcome == OUTCOME_CONVERTED for row in rows))
        done = True
    finally:
        if not done:
            shutil.rmtree(partial, ignore_errors=True)
    return rows, reading, files, status


def _salvage_study(path):
    """StudyInstanceUID of an unreadable file, read again with force=True and tags only; None when even that fails (#36)."""
    def read():  # the value is decoded on first access, so it is read inside the quiet block too: its warning names it
        ds = pydicom.dcmread(path, force=True, stop_before_pixels=True, specific_tags=['StudyInstanceUID'])
        return str(ds.get('StudyInstanceUID', ''))
    try:
        study = quiet_call(read, REASON_FILE_UNREADABLE)
    except CaseRefused:
        return None
    return study or None


def _refuse_on_lost_ct(row, result, partial, lost):
    """(row, result) of a plan, refused as REASON_CT_SLICE_MISSING when it converted on a planning CT a sibling found short (#44)."""
    if result is None or result.ct_series_uid not in lost:
        return row, result
    shutil.rmtree(partial / row.folder)
    log.warning('convert: plan refused: %s', REASON_CT_SLICE_MISSING)
    return ConvertRow(row.patient, row.folder, OUTCOME_REFUSED, REASON_CT_SLICE_MISSING), None


def _place_series(patient, headers, exclude_models, outcomes, partial, short=frozenset(), lost=frozenset(), misc_taint=(frozenset(), False),
                  misc_dir=None):
    """
    Write every other series of the patient into the converted plan folders it is registered to and into its study folder,
    each series read once (the study volume is the image already read). Adds the series files and rows to the plan rows and
    results; writes each study_<n>/series.csv. Returns the ConvertRows to append after the plan rows: the `studies` summary row
    and one row per written series; [] when the patient has no candidate series. Nothing here catches: a bad header is a None
    from series.py, anything else is the patient's REASON_INTERNAL. A series with a kept file in a folder of `short` (refused
    plan folders that may have lost a slice) or whose SeriesInstanceUID is in `lost` (known short) is skipped and counted,
    and so is a series of a tainted study (`misc_taint` and series.tainted_studies) or, when a taint has no study, a series
    held only in `misc_dir` (the patient's misc_files), or whose slices disagree on study or have a stripped-empty study UID.
    """
    found, regs, by_sop = ser.candidates(headers, exclude_models)
    converted = [(row, result, result.series_rows[0]) for row, result in outcomes if result is not None]  # row 0 is the CT row
    if not all(ct['kind'] == 'ct' for _, _, ct in converted):  # not an assert: python -O would strip it (#81)
        raise RuntimeError('series_rows[0] is not the CT row')
    planning = {result.ct_series_uid for _, result, _ in converted}
    found = [(uid, slices) for uid, slices in found if uid not in planning]
    if not found:
        return []
    dropped, dropped_blind = ser.tainted_studies(headers)
    tainted, blind = misc_taint[0] | dropped, misc_taint[1] or dropped_blind

    def at_risk(slices):
        studies = {_tag(ds, 'StudyInstanceUID') for _, ds in slices}  # slices that disagree on study are skipped too (#69)
        return (any(not study.strip() for study in studies) or len(studies) > 1 or bool(studies & tainted)
                or (blind and all(path.parent == misc_dir for path, _ in slices)))
    checked = [None if uid in lost or any(path.parent in short for path, _ in slices) or at_risk(slices) else ser.check_volume(slices)
               for uid, slices in found]
    skipped = checked.count(None)
    order = lambda ordered: tuple(_tag(ordered[0][1], tag) for tag in ('SeriesDate', 'SeriesTime', 'SeriesInstanceUID'))  # noqa: E731
    counts, written = {}, {}  # (folder, modality) -> last k; folder -> [(modality, k, series.csv row)]
    study_folders, study_dates = {}, {}  # StudyInstanceUID -> hidden folder name; -> (StudyDate, StudyTime) of each series written to it
    for ordered in sorted((o for o in checked if o is not None), key=order):
        first = ordered[0][1]
        modality, frame = str(first.Modality), str(first.FrameOfReferenceUID)
        targets = []
        for row, _, ct_row in converted:
            registered, text = ser.registration(ct_row['frame_of_reference_uid'], frame, regs, by_sop)
            if registered:
                targets.append((row.folder, text, {c: ct_row[c] for c in PLAN_COLUMNS}))
        image = ser.read_series(ordered)
        if image is None:
            skipped += 1
            continue
        study_uid = _tag(first, 'StudyInstanceUID')  # one study per series: at_risk skipped the rest
        hidden = study_folders.setdefault(study_uid, f'.study_{len(study_folders)}')  # an index, never the UID
        study_dates.setdefault(study_uid, []).append(tuple(_tag(first, tag).strip() for tag in ('StudyDate', 'StudyTime')))
        for folder_name, text, plan_columns in [*targets, (hidden, '', dict.fromkeys(PLAN_COLUMNS, ''))]:
            k = counts[folder_name, modality] = counts.get((folder_name, modality), 0) + 1
            name = f'series_{modality}_{k}.nii.gz'
            (partial / folder_name).mkdir(exist_ok=True)
            write_atomic(image, partial / folder_name / name)
            written.setdefault(folder_name, []).append((modality, k, dict(
                file=name, kind='series', modality=modality, roi_name='', dose_summation_type='', reg_matrix=text,
                frame_of_reference_uid=frame, **plan_columns,
                **{column: _tag(first, tag) for column, tag in (
                    ('series_description', 'SeriesDescription'), ('series_instance_uid', 'SeriesInstanceUID'),
                    ('study_instance_uid', 'StudyInstanceUID'), ('series_date', 'SeriesDate'))})))
    for folder_name in written:  # CT, MR, PT, then k
        written[folder_name].sort(key=lambda item: (ser.IMAGE_MODALITIES.index(item[0]), item[1]))
    for row, result, _ in converted:
        rows = [item[2] for item in written.get(row.folder, [])]
        result.series_rows.extend(rows)
        row.files = [*result.files, *(r['file'] for r in rows), SERIES_NAME]
    if not study_folders:
        return [ConvertRow(patient, STUDIES_NAME, OUTCOME_SKIPPED, REASON_NO_STUDIES, skipped_series=skipped)]
    rows = [ConvertRow(patient, STUDIES_NAME, OUTCOME_CONVERTED, skipped_series=skipped)]
    # study_<n> by the earliest (undated, then untimed, last) StudyDate and StudyTime over the series written to it, then the UID. The text is
    # compared as stored, stripped: an invalid DA sorts and never raises.
    rank = lambda uid: (min((date == '', date, time == '', time) for date, time in study_dates[uid]), uid)  # noqa: E731
    for n, uid in enumerate(sorted(study_folders, key=rank), 1):
        (partial / study_folders[uid]).rename(partial / f'study_{n}')
        series_rows = [item[2] for item in written[study_folders[uid]]]
        write_series_csv(series_rows, partial / f'study_{n}' / SERIES_NAME)
        rows += [ConvertRow(patient, f'study_{n}/{r["file"].removesuffix(".nii.gz")}', OUTCOME_CONVERTED,
                            files=[r['file'], *([SERIES_NAME] if i == 0 else [])]) for i, r in enumerate(series_rows)]
    return rows


def _convert_one(plan_dir, headers, unreadable, partial):
    """(ConvertRow, PlanResult) of a converted plan, (ConvertRow, None) of a refused one."""
    reason = None
    try:
        result = convert_plan(headers, unreadable, partial / plan_dir.name)
    except CaseRefused as exc:
        reason = str(exc)
    if reason is not None:
        log.warning('convert: plan refused: %s', reason)
        return ConvertRow(plan_dir.parent.name, plan_dir.name, OUTCOME_REFUSED, reason), None
    return ConvertRow(plan_dir.parent.name, plan_dir.name, OUTCOME_CONVERTED, files=[*result.files, SERIES_NAME]), result


def convert_plan(headers, unreadable, case_dir):
    """
    Convert one plan folder into `case_dir`, which must not exist; returns a PlanResult.

    `headers` is the plan folder's own [(path, Dataset)] list (not deduplicated by SOP) and `unreadable` says whether any
    file in it could not be read (see read_folder_headers); this function reads no headers and writes no series.csv: the
    rows are in the PlanResult. An unreadable file refuses the plan (it may be a CT slice). A refusal raises CaseRefused(fixed reason) and leaves no `case_dir` behind. A header that lacks a tag the path reads
    refuses with REASON_MISSING_TAG (_header_call). Any other error, an OSError on the output side included, propagates:
    it is the patient's, not the plan's.
    """
    case_dir = Path(case_dir)
    existed = case_dir.exists()
    reason = None
    try:
        result = _convert_plan(headers, unreadable, case_dir)
    except CaseRefused as exc:
        reason = str(exc)
    if reason is not None:  # raised here, after the except blocks: nothing of the original is chained
        if not existed and case_dir.exists():  # a failure is an OSError, the patient's: a refused plan's files must never be renamed into place
            shutil.rmtree(case_dir)
        raise CaseRefused(reason)
    return result


def _header_call(fn, *args, **kwargs):
    """
    fn(*args, **kwargs) for a header-only step: a missing tag or odd value (#33) is CaseRefused(REASON_MISSING_TAG), logged at
    INFO with the class name and site. An AttributeError/KeyError raised outside pydicom, or an AttributeError for a name that is no
    DICOM keyword (a typo), is ours: it propagates as itself, and the abort path formats only class and site.
    """
    failed, result, ours_exc = False, None, None
    try:
        result = fn(*args, **kwargs)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:  # not quoted
        tb = sys.exc_info()[2]
        origin = [frame.f_globals.get('__name__', '') for frame, _ in traceback.walk_tb(tb)][-1]
        ours = origin != 'pydicom' and not origin.startswith('pydicom.')
        if isinstance(exc, (AttributeError, KeyError)) and (ours or (isinstance(exc, AttributeError)
                                                                     and tag_for_keyword(getattr(exc, 'name', None)) is None)):
            ours_exc = exc
        else:
            module, line = crash_site(tb)
            log.info('convert: %s: %s at %s:%d', REASON_MISSING_TAG, getattr(type(exc), '__name__'), module, line)  # noqa: B009 (no message)
            failed = True
    if ours_exc is not None:  # raised here too (a test forbids a raise inside an except block): the same object, as itself
        raise ours_exc
    if failed:  # raised here, after the except block: nothing of the original is chained
        raise CaseRefused(REASON_MISSING_TAG)
    return result


def _convert_plan(headers, unreadable, case_dir):
    if unreadable:
        raise CaseRefused(REASON_FILE_UNREADABLE)
    plan, struct, ct_slices, series_uid = _header_call(_assemble, headers)
    _header_call(_check_volume, ct_slices)
    _header_call(_check_contour_images, struct, series_uid, ct_slices)
    ct_frame_uid, struct_series_uid = _header_call(lambda: (str(ct_slices[0][1].FrameOfReferenceUID), str(struct.SeriesInstanceUID)))
    ct = _read_ct([p for p, _ in ct_slices], ct_slices[0][1])
    used = _header_call(_dose_candidates, headers, plan)
    offsets = [_header_call(_check_dose_header, ds, ct_frame_uid) for _, ds in used]  # every used dose is checked before any is read
    dose, dose_heads = _read_dose_sum(used, offsets, ct)
    rois = _header_call(masks.classify, struct, plan, ct_frame_uid, taken=(CT_NAME, DOSE_NAME), ct=ct)
    struct_path = next(p for p, ds in headers if ds is struct)
    # The scratch folder of DicomRTTool goes in the patient's partial folder: a stale partial's cleanup removes it too.
    mask_images = masks.build_masks(rois.kept, [p for p, _ in ct_slices], struct_path, struct_series_uid,
                                    series_uid, ct, case_dir.parent, ORIGIN_ATOL_MM, DIRECTION_ATOL)
    # Everything that can refuse has run: only now does the plan folder appear.
    case_dir.mkdir(parents=True)
    write_atomic(ct, case_dir / CT_NAME)
    files = [CT_NAME]
    if dose is not None:
        write_atomic(dose, case_dir / DOSE_NAME)
        files.append(DOSE_NAME)
    for name, image in mask_images.items():
        write_atomic(image, case_dir / name)
        files.append(name)
    for name, points, _ in rois.applicators + rois.lines:
        _atomic(lambda temp, points=points: np.savetxt(temp, points, fmt='%.4f'), case_dir / name)  # gzip text: the name ends .nii.gz
        files.append(name)
    series_rows = _header_call(_series_rows, plan, struct, ct_slices[0][1], dose_heads, rois, files)
    return PlanResult(files, series_uid, series_rows)


def read_folder_headers(folder):
    """([(path, Dataset)] of every readable file in `folder`, sorted by path; whether any file was unreadable)."""
    with dicom_io._pydicom_quiet():  # the whole folder read: pydicom's warnings and log records quote values
        results = [_read_one(p) for p in sorted(p for p in Path(folder).iterdir() if p.is_file())]
    headers = [(path, ds) for path, ds in results if ds is not None]
    return headers, len(headers) < len(results)


def _read_one(path):
    """(path, header), or (path, None) when the file is unreadable: one bad file never ends the folder read."""
    try:
        return path, dicom_io.read_header_core(path)  # via the module: a test patches this seam
    except OSError:
        return path, None


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
    """One axial orientation, size, pixel spacing and frame; at least two slices; no duplicate position; uniform spacing."""
    heads = [ds for _, ds in ct_slices]
    iops = np.asarray([[float(v) for v in ds.ImageOrientationPatient] for ds in heads])
    pixel_spacing = np.asarray([[float(v) for v in ds.PixelSpacing] for ds in heads])
    rescale = np.asarray([[float(ds.get('RescaleSlope', 1)), float(ds.get('RescaleIntercept', 0))] for ds in heads])
    # GDCM reads an inf spacing as 1.0 and a non-finite rescale as 1 or 0; allclose takes inf == inf as close.
    ok = (len(heads) >= 2
          and np.isfinite(pixel_spacing).all() and np.isfinite(rescale).all()
          and np.allclose(iops, iops[0], rtol=0, atol=DIRECTION_ATOL)
          and np.allclose(pixel_spacing, pixel_spacing[0], rtol=0, atol=ORIGIN_ATOL_MM)
          and len({(int(ds.Rows), int(ds.Columns)) for ds in heads}) == 1
          and len({str(ds.FrameOfReferenceUID) for ds in heads}) == 1)
    if ok:
        normal = np.cross(iops[0][:3], iops[0][3:])
        positions = np.asarray([[float(v) for v in ds.ImagePositionPatient] for ds in heads])
        along = positions @ normal
        steps = np.diff(along)  # the slices are sorted, so a duplicate position is a step of ~0
        # A gantry tilt moves slices sideways too; the image grid would put them on the normal and misplace them.
        offset = positions - positions[0]
        sideways = offset - np.outer(offset @ normal, normal)
        # DicomRTTool places contours by patient z alone: only an axial CT (normal along z) keeps them on their slices.
        ok = bool(steps[0] > ORIGIN_ATOL_MM and np.all(np.abs(steps - steps[0]) <= ORIGIN_ATOL_MM)
                  and np.abs(sideways).max() <= ORIGIN_ATOL_MM and abs(normal[2]) >= 1 - DIRECTION_ATOL)
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
    if not np.all(np.isfinite(offsets)):  # before np.diff: inf - inf warns
        raise CaseRefused(REASON_DOSE_GRID)
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
    def extract():  # third-party access and value extraction only: no pydicom object leaves, `offsets` is used by len alone
        ds = pydicom.dcmread(path)
        scaling = float(ds.DoseGridScaling)
        values = ds.pixel_array.astype(np.float64) * scaling  # (frames, rows, columns)
        if values.shape[0] != len(offsets) or not np.isfinite(scaling):  # a NaN scaling would be written as a NaN dose
            raise ValueError
        geometry = []
        for name, count in (('ImageOrientationPatient', 6), ('ImagePositionPatient', 3), ('PixelSpacing', 2)):
            floats = [float(v) for v in getattr(ds, name)]
            if len(floats) != count or not all(np.isfinite(floats)):
                raise ValueError
            geometry.append(floats)
        return values, *geometry
    values, iop, origin, spacing = quiet_call(extract, REASON_DOSE_UNREADABLE)
    # Row cosine runs along the columns (x), column cosine along the rows (y); frame k sits at
    # ImagePositionPatient + (o_k - o_0) n, which holds for relative (o_0 = 0) and absolute offsets alike.
    iop, origin = np.asarray(iop), np.asarray(origin)
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    step = offsets[1] - offsets[0]
    if step < 0:  # reorder so the image has a positive slice spacing, its origin at the last frame
        values = values[::-1]
        origin = origin + (offsets[-1] - offsets[0]) * n
    row_spacing, column_spacing = spacing
    spacing = (column_spacing, row_spacing, abs(float(step)))
    direction = tuple(np.column_stack([r, c, n]).ravel())
    origin = tuple(origin)
    def build():
        dose = sitk.GetImageFromArray(values)
        dose.SetSpacing(spacing)
        dose.SetDirection(direction)
        dose.SetOrigin(origin)
        return resample_dose(dose, ct)
    return quiet_call(build, REASON_DOSE_UNREADABLE)


def _read_dose_sum(used, offsets, ct):
    """(the plan dose on the CT grid in Gy as float32, the headers of the RTDOSEs summed); (None, []) without an RTDOSE."""
    images = [_read_dose(path, off, ct) for (path, _), off in zip(used, offsets, strict=True)]
    if not images:
        return None, []
    total = images[0]
    for image in images[1:]:
        total = total + image
    return total, [ds for _, ds in used]


def _tag(ds, name):
    """The raw value of an optional tag as text ('' when absent or empty; dates stay as stored)."""
    value = ds.get(name)
    return '' if value is None else str(value)


def _joined(values):
    return ';'.join(dict.fromkeys(values))


def _series_rows(plan, struct, ct_head, dose_heads, rois, files):
    """One dict per mask, applicator, CT and dose file in `files` (series.csv itself has none), from headers already read."""
    groups = plan.get('FractionGroupSequence', [])
    plan_columns = dict(rtplan_uid=str(plan.SOPInstanceUID), rtplan_label=_tag(plan, 'RTPlanLabel'),
                        rtplan_date=_tag(plan, 'RTPlanDate'),
                        fractions_planned=_tag(groups[0], 'NumberOfFractionsPlanned') if groups else '')
    frames = struct.get('ReferencedFrameOfReferenceSequence', [])
    struct_frame = _tag(struct, 'FrameOfReferenceUID') or (_tag(frames[0], 'FrameOfReferenceUID') if frames else '')

    def row(file, kind, heads, modality, frame, roi_name='', summation='', reg_matrix=''):
        return dict(file=file, kind=kind, modality=modality, roi_name=roi_name, dose_summation_type=summation,
                    reg_matrix=reg_matrix, frame_of_reference_uid=frame, **plan_columns,
                    **{column: _joined(_tag(ds, tag) for ds in heads) for column, tag in (
                        ('series_description', 'SeriesDescription'), ('series_instance_uid', 'SeriesInstanceUID'),
                        ('study_instance_uid', 'StudyInstanceUID'), ('series_date', 'SeriesDate'))})
    rows = [row(CT_NAME, 'ct', [ct_head], 'CT', _tag(ct_head, 'FrameOfReferenceUID'), reg_matrix='identity')]
    if dose_heads:
        rows.append(row(DOSE_NAME, 'dose', dose_heads, 'RTDOSE', _joined(_tag(ds, 'FrameOfReferenceUID') for ds in dose_heads),
                        summation=_joined(_tag(ds, 'DoseSummationType').strip() for ds in dose_heads)))
    roi_names = {file: ('mask', name) for _, name, file, _ in rois.kept} | {file: ('applicator', name) for file, _, name in rois.applicators}
    roi_names |= {file: ('line', name) for file, _, name in rois.lines}
    rows += [row(file, roi_names[file][0], [struct], 'RTSTRUCT', struct_frame, roi_name=roi_names[file][1])
             for file in files if file in roi_names]
    return rows


def write_series_csv(rows, path):
    """`rows` (dicts of SERIES_COLUMNS) at `path`, written atomically; a failure raises OSError(REASON_WRITE)."""
    def write(temp):
        with open(temp, 'w', newline='', encoding='utf-8') as handle:  # a description the locale cannot encode
            writer = csv.DictWriter(handle, SERIES_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    _atomic(write, path)


def write_atomic(image, path):
    """
    Write `image` to `path` through a temp file in the same folder (its name ends .nii.gz: SimpleITK picks the writer by
    extension) and os.replace. A failure of either step raises OSError(REASON_WRITE), and the temp file is removed.
    """
    _atomic(lambda temp: quiet_call(lambda: sitk.WriteImage(image, str(temp)), REASON_WRITE), path)  # ITK names the path


def _atomic(write, path):
    """write(temp) then os.replace(temp, path); any OSError or CaseRefused of either step is OSError(REASON_WRITE)."""
    path = Path(path)
    temp = path.with_name(f'.{os.getpid()}.tmp.{path.name}')
    failed = False
    try:
        try:
            write(temp)
            os.replace(temp, path)
        except (CaseRefused, OSError):  # both name the path
            failed = True
    finally:
        with contextlib.suppress(OSError):
            temp.unlink()
    if failed:
        raise OSError(REASON_WRITE)
