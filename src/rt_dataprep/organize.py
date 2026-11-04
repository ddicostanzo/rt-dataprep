"""
Sort an export tree into <out>/<patient>/plan_<n>/, one folder per RTPLAN (spec: phase1.md, F1).

Files are copied, never moved. Log lines and exception messages carry counts and the fixed REASON_ constants only, never a
PatientID, folder name, date or path; organize_report.csv in the output root is the one place the sanitized id appears.
"""
import csv
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rt_dataprep.dicom_io import read_header, sanitize_path_component

log = logging.getLogger(__name__)

REPORT_NAME = 'organize_report.csv'
OUTCOME_ORGANIZED, OUTCOME_SKIPPED, OUTCOME_REFUSED = 'organized', 'skipped', 'refused'
REASON_EXISTS = 'output folder already exists'
REASON_NAME_CLAIMED = 'output folder name already used by an earlier patient folder in this run'
REASON_MIXED_IDS = 'source folder holds more than one PatientID'
REASON_NO_DICOM = 'source folder holds no readable DICOM file'
REASON_OUTSIDE_ROOT = 'output folder resolves outside the output root'
REASON_IO_ERROR = 'file system error while reading or copying; nothing kept for this patient'
REASON_OUTSIDE_SRC = 'output root is the source root or inside it'


@dataclass
class PatientRow:
    patient: str  # sanitized id; the sanitized source folder name when no single PatientID could be read
    plans: int
    misc: int
    unreadable: int
    outcome: str
    reason: str = ''


@dataclass
class OrganizeReport:
    rows: list


def organize(src_root, out_root, *, exclude_models=('Acuity',)):
    src_root, out_root = Path(src_root), Path(out_root)
    if out_root.resolve().is_relative_to(src_root.resolve()):  # before any mkdir: it would be walked as a patient folder
        raise ValueError(REASON_OUTSIDE_SRC)
    out_root.mkdir(parents=True, exist_ok=True)
    claimed = set()  # casefolded output names taken in this run: a case-insensitive filesystem would merge them
    rows = []
    for folder in sorted(p for p in src_root.iterdir() if p.is_dir()):
        try:
            rows.append(_organize_patient(folder, out_root, claimed, exclude_models))
        except OSError:  # its message names a path, i.e. the patient: report a fixed reason and go on
            rows.append(PatientRow(sanitize_path_component(folder.name), 0, 0, 0, OUTCOME_REFUSED, REASON_IO_ERROR))
            log.warning('organize: %s', REASON_IO_ERROR)
    with open(out_root / REPORT_NAME, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient', 'plan_count', 'misc_count', 'unreadable_count', 'outcome', 'reason'])
        writer.writerows([r.patient, r.plans, r.misc, r.unreadable, r.outcome, r.reason] for r in rows)
    log.info('organize: %d patient folders, %d organized, %d plan folders, %d misc files, %d unreadable files',
             len(rows), sum(r.outcome == OUTCOME_ORGANIZED for r in rows), sum(r.plans for r in rows),
             sum(r.misc for r in rows), sum(r.unreadable for r in rows))
    return OrganizeReport(rows)


def _organize_patient(folder, out_root, claimed, exclude_models):
    headers, unreadable = [], 0
    for path in sorted(folder.rglob('*.dcm')):
        try:
            ds = read_header(path)
        except OSError:
            unreadable += 1
            continue
        if 'SOPInstanceUID' not in ds:  # type 1, and every grouping step keys on it
            unreadable += 1
            continue
        headers.append((path, ds))
    ids = {str(ds.get('PatientID', '')) for _, ds in headers}
    if len(ids) != 1:  # never pick one of several ids: that would file one patient's data under another
        reason = REASON_MIXED_IDS if ids else REASON_NO_DICOM
        return PatientRow(sanitize_path_component(folder.name), 0, 0, unreadable, OUTCOME_REFUSED, reason)
    name = sanitize_path_component(ids.pop())
    if name.casefold() in claimed:  # two folders, one output name: refuse, never merge
        return PatientRow(name, 0, 0, unreadable, OUTCOME_REFUSED, REASON_NAME_CLAIMED)
    claimed.add(name.casefold())
    target = out_root / name
    if not target.resolve().is_relative_to(out_root.resolve()):
        raise ValueError(REASON_OUTSIDE_ROOT)
    if target.exists():
        return PatientRow(name, 0, 0, unreadable, OUTCOME_SKIPPED, REASON_EXISTS)
    cases, misc = _group(headers, exclude_models)
    # Build under a hidden name and rename at the end: a run that dies part-way leaves no folder that a later run
    # would skip as already done.
    partial = out_root / f'.{name}.partial'
    shutil.rmtree(partial, ignore_errors=True)
    try:
        for n, paths in enumerate(cases, start=1):
            _copy_all(paths, partial / f'plan_{n}')
        _copy_all(misc, partial / 'misc_files')
        partial.mkdir(exist_ok=True)
        partial.rename(target)
    except OSError:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return PatientRow(name, len(cases), len(misc), unreadable, OUTCOME_ORGANIZED)


def _uids(ds, sequence, tag):
    return [getattr(item, tag, None) for item in ds.get(sequence, [])]


def _plan_bound(ds):
    """An RTPLAN, or an RTDOSE that names a plan: these only ever sit in the folder of their own plan."""
    return ds.get('Modality') == 'RTPLAN' or (ds.get('Modality') == 'RTDOSE' and bool(ds.get('ReferencedRTPlanSequence')))


def _planning_ct_series(plan, structs, ct_series):
    """The one CT series UID reached by RTPLAN -> RTSTRUCT -> referenced series, or None if the chain is not complete."""
    struct_uids = _uids(plan, 'ReferencedStructureSetSequence', 'ReferencedSOPInstanceUID')
    struct = structs.get(struct_uids[0]) if struct_uids else None
    if struct is None:
        return None
    found = {series.get('SeriesInstanceUID')
             for frame in struct.get('ReferencedFrameOfReferenceSequence', [])
             for study in frame.get('RTReferencedStudySequence', [])
             for series in study.get('RTReferencedSeriesSequence', [])} & ct_series.keys()
    return found.pop() if len(found) == 1 else None


def _group(headers, exclude_models):
    """([file lists, one per plan folder in plan order], [left-over files]) for one patient's readable headers."""
    study_of = lambda ds: str(ds.get('StudyInstanceUID', ''))  # noqa: E731
    excluded = {study_of(ds) for _, ds in headers if any(m in str(ds.get('ManufacturerModelName', '')) for m in exclude_models)}
    excluded.add('')  # a file with no study cannot be matched to a case
    pool = [(p, ds) for p, ds in headers
            if study_of(ds) not in excluded and not (ds.get('Modality') == 'CT' and 'LOCALIZER' in ds.get('ImageType', []))]

    by_study, ct_series = defaultdict(list), defaultdict(list)
    for path, ds in pool:
        by_study[study_of(ds)].append((path, ds))
        if ds.get('Modality') == 'CT':
            ct_series[ds.get('SeriesInstanceUID')].append((path, ds))
    structs = {ds.SOPInstanceUID: ds for _, ds in pool if ds.get('Modality') == 'RTSTRUCT'}
    struct_paths = {ds.SOPInstanceUID: p for p, ds in pool if ds.get('Modality') == 'RTSTRUCT'}
    # shortcut: a "planning study" is any study that holds an RTPLAN; a diagnostic study with a plan is not copied in
    planning_studies = {study_of(ds) for _, ds in pool if ds.get('Modality') == 'RTPLAN'}

    cases, seen = [], set()
    for plan_path, plan in pool:
        series = _planning_ct_series(plan, structs, ct_series) if plan.get('Modality') == 'RTPLAN' else None
        if series is None or plan.SOPInstanceUID in seen:  # one RTPLAN in two sub-folders: the first by path, once
            continue
        seen.add(plan.SOPInstanceUID)
        ct = min((ds for _, ds in ct_series[series]), key=lambda ds: ds.SOPInstanceUID)
        key = tuple(str(v) for v in (ct.get('SeriesDate', ''), ct.get('SeriesTime', ''), plan.get('RTPlanDate', ''),
                                     plan.get('RTPlanTime', ''), plan.SOPInstanceUID))
        ct_study = study_of(ct)
        # The resolved RTSTRUCT always comes along, even when it was exported in a study other than the CT's.
        files = [plan_path, struct_paths[_uids(plan, 'ReferencedStructureSetSequence', 'ReferencedSOPInstanceUID')[0]]]
        files += [p for p, ds in pool if ds.get('Modality') == 'RTDOSE'
                               and plan.SOPInstanceUID in _uids(ds, 'ReferencedRTPlanSequence', 'ReferencedSOPInstanceUID')]
        studies = [ct_study]
        for _, reg in by_study[ct_study]:
            if reg.get('Modality') == 'REG':
                studies += [u for u in _uids(reg, 'StudiesContainingOtherReferencedInstancesSequence', 'StudyInstanceUID')
                            if u not in planning_studies]
        # Other plans and their doses never come along, so every plan folder holds exactly one RTPLAN.
        files += [p for study in dict.fromkeys(studies) for p, ds in by_study[study] if not _plan_bound(ds)]
        cases.append((key, list(dict.fromkeys(files))))

    cases = [files for _, files in sorted(cases)]
    placed = {p for files in cases for p in files}
    return cases, [p for p, _ in headers if p not in placed]


def _copy_all(paths, folder):
    if not paths:
        return
    folder.mkdir(parents=True)
    for source in paths:
        target, n = folder / source.name, 1
        while target.exists():  # the same file name from two source sub-folders: keep both
            target, n = folder / f'{source.stem}_{n}{source.suffix}', n + 1
        shutil.copy2(source, target)
