"""
Sort an export tree into <out>/<patient>/plan_<n>/, one folder per RTPLAN (spec: phase1.md, F1).

Files are copied, never moved. Log lines and exception messages carry counts and the fixed REASON_ constants only, never a
PatientID, folder name, date or path; organize_report.csv in the output root is the one place the sanitized id appears.
"""
import logging
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rt_dataprep import dicom_io
from rt_dataprep.dicom_io import read_pooled, sanitize_path_component, write_report

log = logging.getLogger(__name__)

REPORT_NAME = 'organize_report.csv'
OUTCOME_ORGANIZED, OUTCOME_SKIPPED, OUTCOME_REFUSED = 'organized', 'skipped', 'refused'
REASON_EXISTS = 'output folder already exists'
REASON_NAME_CLAIMED = 'output folder name already used by an earlier patient folder in this run'
REASON_MIXED_IDS = 'source folder holds more than one PatientID'
REASON_NO_DICOM = 'source folder holds no readable DICOM file'
REASON_NO_ID = 'source folder holds no file with a PatientID'
REASON_SPLIT_SERIES = 'a series holds files with and without a PatientID'
REASON_OUTSIDE_ROOT = 'output folder resolves outside the output root'
REASON_IO_ERROR = 'file system error while reading or copying; nothing kept for this patient'
REASON_OUTSIDE_SRC = 'output root is the source root or inside it'
REASON_NO_SOP_UID = 'a DICOM file has no or an empty SOPInstanceUID'
_CT_DATES = (('SeriesDate', 'SeriesTime'), ('StudyDate', 'StudyTime'), ('AcquisitionDate', 'AcquisitionTime'),
             ('ContentDate', 'ContentTime'))  # fx order: the first non-blank date wins, with its own time as the tiebreak (#68)


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


def organize(src_root, out_root, *, exclude_models=('Acuity',), workers=1, overwrite=False):
    src_root, out_root = Path(src_root), Path(out_root)
    if out_root.resolve().is_relative_to(src_root.resolve()):  # before any mkdir: it would be walked as a patient folder
        raise ValueError(REASON_OUTSIDE_SRC)
    out_root.mkdir(parents=True, exist_ok=True)
    claimed = set()  # casefolded output names taken in this run: a case-insensitive filesystem would merge them
    rows, replaced, kept = [], 0, 0
    reading = copying = files = 0  # seconds, seconds, files walked; a refused-by-OSError patient adds none
    started = time.monotonic()
    try:
        for folder in sorted(p for p in src_root.iterdir() if p.is_dir() and not p.name.startswith('.')):  # .X.partial too
            try:
                row, read_s, copy_s, walked, status = _organize_patient(folder, out_root, claimed, exclude_models, workers, overwrite)
                rows.append(row)
                reading, copying, files = reading + read_s, copying + copy_s, files + walked
                replaced, kept = replaced + (status == 'replaced'), kept + (status == 'kept')
            except OSError:  # its message names a path, i.e. the patient: report a fixed reason and go on
                rows.append(PatientRow(sanitize_path_component(folder.name), 0, 0, 0, OUTCOME_REFUSED, REASON_IO_ERROR))
                log.warning('organize: %s', REASON_IO_ERROR)
    finally:  # Ctrl-C too: the report lists what was done
        write_report(out_root, REPORT_NAME, ['patient', 'plan_count', 'misc_count', 'unreadable_count', 'outcome', 'reason'],
                     ([r.patient, r.plans, r.misc, r.unreadable, r.outcome, r.reason] for r in rows))
    log.info('organize: %d patient folders, %d organized, %d plan folders, %d misc files, %d unreadable files; %d replaced, %d kept',
             len(rows), sum(r.outcome == OUTCOME_ORGANIZED for r in rows), sum(r.plans for r in rows),
             sum(r.misc for r in rows), sum(r.unreadable for r in rows), replaced, kept)
    log.info('organize: %.1f s total, %.1f s reading, %.1f s copying, %d files, workers %d',
             time.monotonic() - started, reading, copying, files, workers)
    return OrganizeReport(rows)


def _organize_patient(folder, out_root, claimed, exclude_models, workers, overwrite):
    """(row, reading seconds, copying seconds, files walked, status None/'new'/'replaced'/'kept'); a raise (OSError) adds none."""
    started = time.monotonic()
    paths = sorted(_walk(folder))  # serial and fully listed first, so the pooled step sees the sorted order
    results = [r for r in read_pooled(_read_one, paths, workers) if r]
    headers = [(path, ds) for path, ds in results if ds is not None]
    bad = [path for path, ds in results if ds is None]  # has the preamble but the read fails; copied to misc_files (#13)
    unreadable = len(bad)
    reading, walked = time.monotonic() - started, len(paths)  # `paths` is rebound by the copy loop below
    done = lambda row, copy_s=0.0, status=None: (row, reading, copy_s, walked, status)  # noqa: E731
    # A file with a blank PatientID is the folder's but never placed in a plan: it is copied to misc_files (#12).
    blank = [(path, ds) for path, ds in headers if not str(ds.get('PatientID', '')).strip()]
    headers = [(path, ds) for path, ds in headers if str(ds.get('PatientID', '')).strip()]
    ids = {str(ds.get('PatientID', '')) for _, ds in headers}
    if len(ids) != 1:  # never pick one of several ids: that would file one patient's data under another
        reason = REASON_MIXED_IDS if ids else REASON_NO_ID if blank else REASON_NO_DICOM
        return done(PatientRow(sanitize_path_component(folder.name), 0, 0, unreadable, OUTCOME_REFUSED, reason))
    name = sanitize_path_component(ids.pop())
    # read-only: a refusal before the claimed check never recovers or touches the root; it only reports an old folder kept
    kept = lambda: 'kept' if overwrite and name.casefold() not in claimed and os.path.lexists(out_root / name) else None  # noqa: E731
    if any(not str(ds.get('SOPInstanceUID', '')).strip() for _, ds in headers):  # type 1; grouping and dedup key on it
        return done(PatientRow(name, 0, 0, unreadable, OUTCOME_REFUSED, REASON_NO_SOP_UID), status=kept())
    # A blank or whitespace UID is no series, and a blank-id copy of a slice already present takes nothing away (#71).
    series_of = lambda items: {u for _, ds in items if (u := str(ds.get('SeriesInstanceUID', ''))).strip()}  # noqa: E731
    pair = lambda ds: (str(ds.get('SOPInstanceUID', '')), str(ds.get('SeriesInstanceUID', '')))  # noqa: E731 (a SOP alone may collide, #81)
    present = {pair(ds) for _, ds in headers}
    extra = [(path, ds) for path, ds in blank if pair(ds) not in present]
    if series_of(extra) & series_of(headers):  # a series split between blank and real ids would be placed short
        return done(PatientRow(name, 0, 0, unreadable, OUTCOME_REFUSED, REASON_SPLIT_SERIES), status=kept())
    if name.casefold() in claimed:  # two folders, one output name: refuse, never merge
        return done(PatientRow(name, 0, 0, unreadable, OUTCOME_REFUSED, REASON_NAME_CLAIMED))
    claimed.add(name.casefold())
    target = out_root / name
    if not target.resolve().is_relative_to(out_root.resolve()):
        raise ValueError(REASON_OUTSIDE_ROOT)
    dicom_io.recover(out_root, name)
    if not overwrite and os.path.lexists(target):  # lexists, as publish: a dangling link is an existing target
        return done(PatientRow(name, 0, 0, unreadable, OUTCOME_SKIPPED, REASON_EXISTS))
    cases, misc = _group(headers, exclude_models)
    misc += [path for path, _ in blank] + bad
    # Build under a hidden name and rename at the end: a run that dies part-way leaves no folder that a later run
    # would skip as already done.
    partial = out_root / f'.{name}.partial'
    copy_started = time.monotonic()
    shutil.rmtree(partial, ignore_errors=True)
    status = None
    try:
        for n, paths in enumerate(cases, start=1):
            _copy_all(paths, partial / f'plan_{n}')
        _copy_all(misc, partial / 'misc_files')
        partial.mkdir(exist_ok=True)
        status = dicom_io.publish(out_root, name, overwrite=overwrite, success=True)  # the row below is organized
    except BaseException:  # Ctrl-C too: no hidden half-copied folder left behind
        # A second Ctrl-C during this rmtree leaves a partly deleted .partial: the source scan skips it and the next run removes it before copying.
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return done(PatientRow(name, len(cases), len(misc), unreadable, OUTCOME_ORGANIZED), time.monotonic() - copy_started, status)


def _read_one(path):
    """(path, header) for a DICOM file, (path, None) when it has the preamble but its read fails, None for any other file."""
    if not _has_preamble(path):  # an OSError here propagates and refuses the patient
        return None
    try:
        return path, dicom_io.read_header_core(path)  # via the module: a test patches this seam
    except OSError:
        return path, None


def _walk(folder):
    """
    Every file under `folder`, hidden folders skipped (a hidden file is kept: a slice may be named so, and _has_preamble
    sorts it). An unlistable folder raises OSError, so no patient is organized with files missing (#68).
    """
    def fail(error):
        raise error
    for top, dirs, files in os.walk(folder, onerror=fail):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        yield from (Path(top) / name for name in files)


def _has_preamble(path):
    """A regular file whose bytes 128-131 are DICM, whatever its name (#10)."""
    if not path.is_file():
        return False
    with open(path, 'rb') as handle:
        return handle.read(132)[128:] == b'DICM'


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
    first = {}  # one instance in two sub-folders is used once, from the first path; the copy goes to misc_files
    for path, ds in headers:
        first.setdefault(ds.SOPInstanceUID, path)
    pool = [(p, ds) for p, ds in headers if first[ds.SOPInstanceUID] == p
            and study_of(ds) not in excluded and not (ds.get('Modality') == 'CT' and 'LOCALIZER' in ds.get('ImageType', []))]

    by_study, ct_series = defaultdict(list), defaultdict(list)
    for path, ds in pool:
        by_study[study_of(ds)].append((path, ds))
        if ds.get('Modality') == 'CT':
            ct_series[ds.get('SeriesInstanceUID')].append((path, ds))
    structs = {ds.SOPInstanceUID: ds for _, ds in pool if ds.get('Modality') == 'RTSTRUCT'}
    struct_paths = {ds.SOPInstanceUID: p for p, ds in pool if ds.get('Modality') == 'RTSTRUCT'}
    # shortcut: a "planning study" is any study that holds an RTPLAN; a diagnostic study with a plan is not copied in
    planning_studies = {study_of(ds) for _, ds in pool if ds.get('Modality') == 'RTPLAN'}

    cases = []
    for plan_path, plan in pool:
        series = _planning_ct_series(plan, structs, ct_series) if plan.get('Modality') == 'RTPLAN' else None
        if series is None:
            continue
        ct = min((ds for _, ds in ct_series[series]), key=lambda ds: ds.SOPInstanceUID)
        date, time = next(((d, str(ct.get(t, '')).strip()) for tag, t in _CT_DATES if (d := str(ct.get(tag, '')).strip())), ('', ''))
        key = (date == '', date, time) + tuple(str(v) for v in (plan.get('RTPlanDate', ''), plan.get('RTPlanTime', ''),
                                                                plan.SOPInstanceUID))  # undated last (#9)
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
