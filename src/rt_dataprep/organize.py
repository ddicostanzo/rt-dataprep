"""
Sort an export tree into <out>/<patient>/plan_<n>/, one folder per RTPLAN (spec: phase1.md, F1).

Files are copied, never moved. Log lines and exception messages carry counts and the fixed REASON_ constants only, never a
PatientID, folder name, date or path; organize_report.csv in the output root is the one place the sanitized id appears.
"""
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rt_dataprep import convert, dicom_io, pool
from rt_dataprep.dicom_io import sanitize_path_component, write_report

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
REASON_INTERNAL = 'an unexpected error stopped the organizing; the report lists what was finished'
_CT_DATES = (('SeriesDate', 'SeriesTime'), ('StudyDate', 'StudyTime'), ('AcquisitionDate', 'AcquisitionTime'),
             ('ContentDate', 'ContentTime'))  # fx order: the first non-blank date wins, with its own time as the tiebreak (#68)


class OrganizeAborted(Exception):
    """A patient hit an unexpected error: the report is written, the run stops. str() is exactly REASON_INTERNAL."""


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


def organize(src_root, out_root, *, exclude_models=('Acuity',), workers=1, overwrite=False, _read_fn=None, _copy_fn=None):
    """
    Sort every patient folder under src_root into out_root and write organize_report.csv.

    workers: patient processes, one patient per process; the output is the same for any N.
    _read_fn, _copy_fn: internal test seams, replace the per-patient read (task A) and copy (task B) only (module-level, picklable).
    """
    src_root, out_root = Path(src_root), Path(out_root)
    if out_root.resolve().is_relative_to(src_root.resolve()):  # before any mkdir: it would be walked as a patient folder
        raise ValueError(REASON_OUTSIDE_SRC)
    if overwrite and src_root.resolve().is_relative_to(out_root.resolve()):  # a replaced patient folder could hold the source
        raise ValueError(dicom_io.REASON_INPUT_INSIDE_OUTPUT)
    out_root.mkdir(parents=True, exist_ok=True)
    read_fn, copy_fn = _read_fn or _read_patient, _copy_fn or _copy_patient
    claimed = set()  # casefolded output names taken in this run: a case-insensitive filesystem would merge them
    done, held = {}, {}  # folder index -> its row; index -> (reading s, files walked, row) of a patient whose copy is under way
    replaced = kept = files = 0
    reading = copying = 0.0  # seconds; a patient refused by an error adds none
    aborted = False
    folders = []

    def finish(i, row, read_s, walked, copy_s=0.0, status=None):
        nonlocal reading, copying, files, replaced, kept
        done[i] = row
        reading, copying, files = reading + read_s, copying + copy_s, files + walked
        replaced, kept = replaced + (status == 'replaced'), kept + (status == 'kept')

    def handle(i, result):
        nonlocal aborted
        folder = folders[i]
        if isinstance(result, pool.Failed):  # escaped the worker wrapper (a broken pool...): only the type name is known
            if result.type_name == 'SystemExit':  # as before: exits the run with no row; the report is still written
                raise SystemExit(result.code)  # its sanitized code, as quiet_call keeps it
            log.error('organize: %s: %s', REASON_INTERNAL, result.type_name)
            result = (None, _failure_row(folder, REASON_INTERNAL))
        result, failure = result
        if failure is not None:  # an OSError (go on) or an unexpected error (stop), already logged by the worker
            held.pop(i, None)
            done[i] = failure
            if failure.reason == REASON_INTERNAL:
                aborted = True
                return pool.ABORT
            return None
        if i in held:  # task B finished: the copy is published
            read_s, walked, row = held.pop(i)
            finish(i, row, read_s, walked, *result)
            return None
        if aborted:  # a read delivered after the abort: that patient is not started (no claim, recover, copy or row)
            return None
        name, refusal, cases, misc, unreadable, walked, read_s = result
        if refusal is not None:
            row, reads_kept = refusal
            # read-only: a refusal before the claim never recovers or touches the root; it only reports an old folder kept
            status = 'kept' if reads_kept and overwrite and name.casefold() not in claimed and os.path.lexists(out_root / name) else None
            finish(i, row, read_s, walked, status=status)
            return None
        if name.casefold() in claimed:  # two folders, one output name: refuse, never merge
            finish(i, PatientRow(name, 0, 0, unreadable, OUTCOME_REFUSED, REASON_NAME_CLAIMED), read_s, walked)
            return None
        claimed.add(name.casefold())
        target = out_root / name
        try:
            if not target.resolve().is_relative_to(out_root.resolve()):
                raise ValueError(REASON_OUTSIDE_ROOT)
            dicom_io.recover(out_root, name)
        except OSError:  # its message names a path, i.e. the patient: report a fixed reason and go on
            done[i] = _failure_row(folder, REASON_IO_ERROR)
            log.warning('organize: %s', REASON_IO_ERROR)
            return None
        if not overwrite and os.path.lexists(target):  # lexists, as publish: a dangling link is an existing target
            finish(i, PatientRow(name, 0, 0, unreadable, OUTCOME_SKIPPED, REASON_EXISTS), read_s, walked)
            return None
        held[i] = (read_s, walked, PatientRow(name, len(cases), len(misc), unreadable, OUTCOME_ORGANIZED))
        return pool.Task(_worker, (copy_fn, folder, out_root, name, cases, misc, overwrite))

    started = time.monotonic()
    try:
        folders = sorted(p for p in src_root.iterdir() if p.is_dir() and not p.name.startswith('.'))  # .X.partial too
        pool.run(workers, [pool.Task(_worker, (read_fn, folder, exclude_models)) for folder in folders], handle)
    finally:  # Ctrl-C too: the report lists what was done
        rows = [done[i] for i in sorted(done)]
        write_report(out_root, REPORT_NAME, ['patient', 'plan_count', 'misc_count', 'unreadable_count', 'outcome', 'reason'],
                     ([r.patient, r.plans, r.misc, r.unreadable, r.outcome, r.reason] for r in rows))
    log.info('organize: %d patient folders, %d organized, %d plan folders, %d misc files, %d unreadable files; %d replaced, %d kept',
             len(rows), sum(r.outcome == OUTCOME_ORGANIZED for r in rows), sum(r.plans for r in rows),
             sum(r.misc for r in rows), sum(r.unreadable for r in rows), replaced, kept)
    log.info('organize: %.1f s total, %.1f s reading (summed over patients), %.1f s copying (summed over patients), %d files, workers %d',
             time.monotonic() - started, reading, copying, files, workers)
    if aborted:  # raised here: nothing of the original is chained
        raise OrganizeAborted(REASON_INTERNAL)
    return OrganizeReport(rows)


def _failure_row(folder, reason):
    return PatientRow(sanitize_path_component(folder.name), 0, 0, 0, OUTCOME_REFUSED, reason)


def _worker(fn, folder, *args):
    """
    One task in a worker (or inline): (fn(folder, *args), None). A failure becomes data here, as a traceback does not cross the
    process boundary: an OSError is the patient's REASON_IO_ERROR row, any other Exception its REASON_INTERNAL row, with only the
    type and the crash site logged (no message).
    """
    try:
        return fn(folder, *args), None
    except OSError:  # its message names a path, i.e. the patient: one fixed reason
        log.warning('organize: %s', REASON_IO_ERROR)
        return None, _failure_row(folder, REASON_IO_ERROR)
    except Exception as exc:  # not KeyboardInterrupt/SystemExit: those keep the flag-based cleanup and add no row
        module, line = convert.crash_site(sys.exc_info()[2])
        log.error('organize: %s: %s at %s:%d', REASON_INTERNAL, getattr(type(exc), '__name__'), module, line)  # noqa: B009 (no message)
        return pool.Stop((None, _failure_row(folder, REASON_INTERNAL)))


def _read_patient(folder, exclude_models):
    """
    Task A: (sanitized name, refusal, per-plan path lists, misc paths, unreadable count, files walked, reading seconds). The refusal is
    None or (row, whether its status may be 'kept'); the parent decides that and the claim. Paths only, no datasets. A raise (OSError) adds none.
    """
    started = time.monotonic()
    paths = sorted(_walk(folder))
    with dicom_io._pydicom_quiet():  # the whole folder read: pydicom's warnings and log records quote values
        results = [r for r in map(_read_one, paths) if r]
    headers = [(path, ds) for path, ds in results if ds is not None]
    bad = [path for path, ds in results if ds is None]  # has the preamble but the read fails; copied to misc_files (#13)
    unreadable = len(bad)
    reading, walked = time.monotonic() - started, len(paths)
    # A file with a blank PatientID is the folder's but never placed in a plan: it is copied to misc_files (#12).
    blank = [(path, ds) for path, ds in headers if not str(ds.get('PatientID', '')).strip()]
    headers = [(path, ds) for path, ds in headers if str(ds.get('PatientID', '')).strip()]
    ids = {str(ds.get('PatientID', '')) for _, ds in headers}
    if len(ids) != 1:  # never pick one of several ids: that would file one patient's data under another
        reason = REASON_MIXED_IDS if ids else REASON_NO_ID if blank else REASON_NO_DICOM
        name = sanitize_path_component(folder.name)
        return name, (PatientRow(name, 0, 0, unreadable, OUTCOME_REFUSED, reason), False), [], [], unreadable, walked, reading
    name = sanitize_path_component(ids.pop())
    refused = lambda reason: (name, (PatientRow(name, 0, 0, unreadable, OUTCOME_REFUSED, reason), True), [], [], unreadable, walked, reading)  # noqa: E731
    if any(not str(ds.get('SOPInstanceUID', '')).strip() for _, ds in headers):  # type 1; grouping and dedup key on it
        return refused(REASON_NO_SOP_UID)
    # A blank or whitespace UID is no series, and a blank-id copy of a slice already present takes nothing away (#71).
    series_of = lambda items: {u for _, ds in items if (u := str(ds.get('SeriesInstanceUID', ''))).strip()}  # noqa: E731
    pair = lambda ds: (str(ds.get('SOPInstanceUID', '')), str(ds.get('SeriesInstanceUID', '')))  # noqa: E731 (a SOP alone may collide, #81)
    present = {pair(ds) for _, ds in headers}
    extra = [(path, ds) for path, ds in blank if pair(ds) not in present]
    if series_of(extra) & series_of(headers):  # a series split between blank and real ids would be placed short
        return refused(REASON_SPLIT_SERIES)
    cases, misc = _group(headers, exclude_models)
    misc += [path for path, _ in blank] + bad
    return name, None, cases, misc, unreadable, walked, reading


def _copy_patient(folder, out_root, name, cases, misc, overwrite):
    """Task B: (copying seconds, status None/'new'/'replaced'/'kept'). `folder` only names the patient for the failure row."""
    # Build under a hidden name and rename at the end: a run that dies part-way leaves no folder that a later run
    # would skip as already done.
    partial = out_root / f'.{name}.partial'
    copy_started = time.monotonic()
    shutil.rmtree(partial, ignore_errors=True)
    try:
        for n, paths in enumerate(cases, start=1):
            _copy_all(paths, partial / f'plan_{n}')
        _copy_all(misc, partial / 'misc_files')
        partial.mkdir(exist_ok=True)
        status = dicom_io.publish(out_root, name, overwrite=overwrite, success=bool(cases))  # no plan (e.g. an unreadable RTPLAN) keeps the old folder
    except BaseException:  # Ctrl-C too: no hidden half-copied folder left behind
        # A second Ctrl-C during this rmtree leaves a partly deleted .partial: the source scan skips it and the next run removes it before copying.
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return time.monotonic() - copy_started, status


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
