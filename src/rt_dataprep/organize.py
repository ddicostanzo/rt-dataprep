"""
Sort an export tree into <out>/<patient>/plan_<n>/, one folder per RTPLAN (spec: phase1.md, F1).

Files are copied, never moved. Log lines and exception messages carry counts and the fixed REASON_ constants only, never a
PatientID, folder name, date or path; organize_report.csv in the output root is the one place the sanitized id appears.
"""
import functools
import logging
import lzma
import os
import shutil
import sys
import tempfile
import time
import zipfile
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rt_dataprep import convert, dicom_io, pool
from rt_dataprep.dicom_io import sanitize_path_component, write_report

log = logging.getLogger(__name__)

REPORT_NAME = 'organize_report.csv'
OUTCOME_ORGANIZED, OUTCOME_SKIPPED, OUTCOME_REFUSED = 'organized', 'skipped', 'refused'
REASON_EXISTS = 'output folder already exists'
REASON_NAME_CLAIMED = 'another folder of this run already maps to this output name (case-insensitive)'  # same text as convert's and preprocess's
REASON_MIXED_IDS = 'source folder holds more than one PatientID'
REASON_NO_DICOM = 'source folder holds no readable DICOM file'
REASON_NO_ID = 'source folder holds no file with a PatientID'
REASON_SPLIT_SERIES = 'a series holds files with and without a PatientID'
REASON_OUTSIDE_ROOT = 'output folder resolves outside the output root'
REASON_IO_ERROR = 'file system error while reading or copying; nothing kept for this patient'
REASON_OUTSIDE_SRC = 'output root is the source root or inside it'
REASON_NO_SOP_UID = 'a DICOM file has no or an empty SOPInstanceUID'
REASON_INTERNAL = 'an unexpected error stopped the organizing; the report lists what was finished'
REASON_ZIP_TOO_LARGE = 'zip archives in the source folder exceed the size or member cap; nothing kept'
REASON_ZIP_UNREADABLE = 'a zip archive in the source folder cannot be read safely; nothing kept'
REASON_LOCKED = 'another organize run holds the output root lock'
LOCK_NAME = '.organize.lock'
WARN_NO_LOCK = 'organize: the output root cannot be locked here; running without a lock and without removing leftover staging'
WARN_STAGING_LEFT = 'organize: %d item(s) of the staging folder under the output root could not be removed; the next run removes them'
MAX_UNZIP_BYTES = 16 * 2**30  # declared bytes of all zips in one source folder
MAX_ZIP_MEMBERS = 200_000  # non-directory members of all zips in one source folder
_ZIP_METHODS = {0, 8}  # stored, deflate: zipfile bounds bzip2/lzma output only on recent patch releases (#243)
_ZIP_ERRORS = (zipfile.BadZipFile, EOFError, zlib.error, lzma.LZMAError, UnicodeDecodeError, NotImplementedError, RuntimeError)
_CT_DATES = (('SeriesDate', 'SeriesTime'), ('StudyDate', 'StudyTime'), ('AcquisitionDate', 'AcquisitionTime'),
             ('ContentDate', 'ContentTime'))  # fx order: the first non-blank date wins, with its own time as the tiebreak (#68)


class ZipRefused(Exception):
    """A folder's zips cannot be used and it has no loose DICOM. str() is exactly the fixed reason; nothing is chained."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


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

    workers: patient processes, one patient per process, or 'auto' to size N from the first organized patient's peak memory; the output is the same for any N.
    _read_fn, _copy_fn: internal test seams, replace the per-patient read (task A) and copy (task B) only (module-level, picklable).
    """
    src_root, out_root = Path(src_root), Path(out_root)
    if out_root.resolve().is_relative_to(src_root.resolve()):  # before any mkdir: it would be walked as a patient folder
        raise ValueError(REASON_OUTSIDE_SRC)
    if overwrite and src_root.resolve().is_relative_to(out_root.resolve()):  # a replaced patient folder could hold the source
        raise ValueError(dicom_io.REASON_INPUT_INSIDE_OUTPUT)
    out_root.mkdir(parents=True, exist_ok=True)
    lock = _lock_root(out_root)
    try:
        if lock is not None:  # only the lock holder may sweep: another run's staging is not a leftover
            for leftover in out_root.glob('.unzip-*'):
                if not leftover.name.endswith(('.old', '.partial')):  # a patient named unzip-*: recover owns those
                    dicom_io.remove_tree(leftover)
        staging_parent = Path(tempfile.mkdtemp(prefix='.unzip-', dir=out_root))
    except BaseException:
        _unlock(lock)
        raise
    read_fn, copy_fn = _read_fn or _read_patient, _copy_fn or _copy_patient
    claimed = set()  # casefolded output names taken in this run: a case-insensitive filesystem would merge them
    done, held = {}, {}  # folder index -> its row; index -> (reading s, files walked, row, output name) of a patient whose copy is under way
    statuses = {}  # casefolded output name -> 'replaced'/'kept': one count per output folder, whatever number of source folders map to it
    files = 0
    zip_counts = [0, 0, 0]  # zips staged, zip members staged, zips skipped
    reading = copying = 0.0  # seconds; a patient refused by an error adds none
    aborted = False
    folders = []

    def finish(i, row, read_s, walked, name, copy_s=0.0, status=None):
        nonlocal reading, copying, files
        done[i] = row
        reading, copying, files = reading + read_s, copying + copy_s, files + walked
        if status is not None:
            statuses[name.casefold()] = status  # a later status for the same name overwrites an earlier one

    def handle(i, result):
        nonlocal aborted
        folder = folders[i]
        if isinstance(result, pool.Failed):  # escaped the worker wrapper (a broken pool...): only the type name is known
            if result.type_name == 'SystemExit':  # as before: exits the run with no row; the report is still written
                aborted = True  # #182: a read delivered after the raise is not claimed or recovered
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
            read_s, walked, row, name = held.pop(i)
            finish(i, row, read_s, walked, name, *result)
            return None
        name, refusal, cases, misc, unreadable, walked, read_s, staging, *counts = result
        if aborted:  # a read delivered after the abort: that patient is not started (no claim, recover, copy or row)
            _drop(staging)
            return None
        zip_counts[:] = [a + b for a, b in zip(zip_counts, counts, strict=True)]
        if refusal is not None:
            row, reads_kept = refusal
            # read-only: a refusal before the claim never recovers or touches the root; it only reports an old folder kept
            status = 'kept' if reads_kept and overwrite and name.casefold() not in claimed and os.path.lexists(out_root / name) else None
            finish(i, row, read_s, walked, name, status=status)
            return None
        if name.casefold() in claimed:  # two folders, one output name: refuse, never merge
            _drop(staging)
            finish(i, PatientRow(name, 0, 0, unreadable, OUTCOME_REFUSED, REASON_NAME_CLAIMED), read_s, walked, name)
            return None
        claimed.add(name.casefold())
        target = out_root / name
        try:
            if not target.resolve().is_relative_to(out_root.resolve()):
                aborted = True  # #182: as for SystemExit above
                raise ValueError(REASON_OUTSIDE_ROOT)
            dicom_io.recover(out_root, name)
        except OSError:  # its message names a path, i.e. the patient: report a fixed reason and go on
            _drop(staging)
            done[i] = _failure_row(folder, REASON_IO_ERROR)
            log.warning('organize: %s', REASON_IO_ERROR)
            return None
        # cases None: task A saw the target and did not group; recover never removes a target, so it is still a skip
        if cases is None or (not overwrite and os.path.lexists(target)):  # lexists, as publish: a dangling link is an existing target
            _drop(staging)
            finish(i, PatientRow(name, 0, 0, unreadable, OUTCOME_SKIPPED, REASON_EXISTS), read_s, walked, name)
            return None
        held[i] = (read_s, walked, PatientRow(name, len(cases), len(misc), unreadable, OUTCOME_ORGANIZED), name)
        return pool.Task(_worker, (copy_fn, folder, out_root, name, cases, misc, overwrite, staging))

    started = time.monotonic()
    try:
        try:
            folders = sorted(p for p in src_root.iterdir() if p.is_dir() and not p.name.startswith('.'))  # .X.partial too
            tasks = [pool.Task(_worker, (read_fn, folder, exclude_models, None if overwrite else out_root, staging_parent)) for folder in folders]
            workers = pool.run_auto(workers, tasks, handle, lambda k: k in done and done[k].outcome == OUTCOME_ORGANIZED)  # an int from here on
        finally:  # Ctrl-C too: the report lists what was done
            dicom_io.remove_tree(staging_parent)
            if os.path.lexists(staging_parent):  # remove_tree swallows its error; the next run's sweep retries
                try:
                    left = 1 + sum(1 for _ in staging_parent.rglob('*'))
                except OSError:  # an unreadable leftover: the report below must still be written
                    left = 1
                log.warning(WARN_STAGING_LEFT, left)
            rows = [done[i] for i in sorted(done)]
            write_report(out_root, REPORT_NAME, ['patient', 'plan_count', 'misc_count', 'unreadable_count', 'outcome', 'reason'],
                         ([r.patient, r.plans, r.misc, r.unreadable, r.outcome, r.reason] for r in rows))
        log.info('organize: %d patient folders, %d organized, %d plan folders, %d misc files, %d unreadable files, %d zips, %d zip members, '
                 '%d zips skipped; %d replaced, %d kept',
                 len(rows), sum(r.outcome == OUTCOME_ORGANIZED for r in rows), sum(r.plans for r in rows),
                 sum(r.misc for r in rows), sum(r.unreadable for r in rows), *zip_counts,
                 *(sum(v == status for v in statuses.values()) for status in ('replaced', 'kept')))
        log.info('organize: %.1f s total, %.1f s reading (summed over patients), %.1f s copying (summed over patients), %d files, workers %d',
                 time.monotonic() - started, reading, copying, files, workers)
        if aborted:  # raised here: nothing of the original is chained
            raise OrganizeAborted(REASON_INTERNAL)
        return OrganizeReport(rows)
    finally:
        _unlock(lock)


def _lock_root(out_root):
    """
    The open lock file, holding flock(LOCK_EX) while it stays open, or None where there is no flock (Windows, or a file system
    that refuses it: one fixed warning, and the caller must not sweep). ValueError(REASON_LOCKED) when another run holds it.
    """
    if sys.platform == 'win32':
        return None
    import fcntl  # POSIX only
    handle = None
    try:
        handle = open(out_root / LOCK_NAME, 'a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:  # before OSError: it is a subclass
        _unlock(handle)
        raise ValueError(REASON_LOCKED) from None
    except OSError:
        _unlock(handle)
        log.warning(WARN_NO_LOCK)
        return None
    return handle


def _unlock(handle):
    if handle is not None:
        handle.close()


def _drop(path):
    if path is not None:
        dicom_io.remove_tree(path)


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


def _read_patient(folder, exclude_models, skip_root, staging_parent):
    """
    Task A, skip_root the out_root without --overwrite: (sanitized name, refusal, per-plan path lists
    (None: the output exists, not grouped), misc paths, unreadable count, files walked, reading seconds, staging, zips,
    zip members, zips skipped). The refusal is None or (row, whether its status may be 'kept'); the parent decides that and
    the claim. Paths only, no datasets. A raise (OSError) adds none.
    The folder's zips are extracted into a fresh sub-folder of `staging_parent`; `staging` names it when the result has
    files to copy (task B removes it), else it is removed here, a raise included.
    """
    started = time.monotonic()
    paths = sorted(_walk(folder))
    kinds = [(path, _has_preamble(path)) for path in paths]  # the one classification read of each walked file
    staging = Path(tempfile.mkdtemp(dir=staging_parent)) if any(kind == 'zip' for _, kind in kinds) else None
    keep = False
    try:
        try:
            staged, zips, members, skipped = _stage_zips(paths, staging, kinds)
        except ZipRefused as refused:  # a fixed reason: this folder is refused, the run goes on
            name = sanitize_path_component(folder.name)
            row = PatientRow(name, 0, 0, 0, OUTCOME_REFUSED, refused.reason)
            return name, (row, False), [], [], 0, len(paths), time.monotonic() - started, None, 0, 0, 0
        result = _read_folder(exclude_models, skip_root, folder, started, [p for p, kind in kinds if kind == 'dicom'], staged, len(paths) + len(staged))
        keep = staging is not None and result[1] is None and result[2] is not None  # else: a refusal or a skip
    finally:  # Ctrl-C too
        if not keep:
            _drop(staging)
    return (*result, staging if keep else None, zips, members, skipped)


def _read_folder(exclude_models, skip_root, folder, started, loose, staged, walked):
    """The first seven fields of _read_patient's result, from the loose DICOM paths then the staged paths."""
    with dicom_io._pydicom_quiet():  # the whole folder read: pydicom's warnings and log records quote values
        results = [r for r in [*map(_read_header, loose), *map(_read_one, staged)] if r]
    headers = [(path, ds) for path, ds in results if ds is not None]
    bad = [path for path, ds in results if ds is None]  # has the preamble but the read fails; copied to misc_files (#13)
    unreadable = len(bad)
    reading = time.monotonic() - started
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
    if skip_root is not None and os.path.lexists(skip_root / name):  # #180: a patient the parent will skip is not grouped
        return name, None, None, [], unreadable, walked, reading
    cases, misc = _group(headers, exclude_models)
    misc += [path for path, _ in blank] + bad
    return name, None, cases, misc, unreadable, walked, reading


def _copy_patient(folder, out_root, name, cases, misc, overwrite, staging=None):
    """
    Task B: (copying seconds, status None/'new'/'replaced'/'kept'). `folder` only names the patient for the failure row.
    Files under `staging` are hard-linked (a copy where that fails), the others copied; `staging` is removed in every outcome.
    """
    # Build under a hidden name and rename at the end: a run that dies part-way leaves no folder that a later run
    # would skip as already done.
    partial = out_root / f'.{name}.partial'
    copy_started = time.monotonic()  # the parent's recover already removed a leftover partial
    copy_all = _copy_all if staging is None else functools.partial(_copy_all, staging=staging)
    try:
        partial.mkdir()  # a leftover (recover failed to remove it) raises: its stale plan folders must never be published
        for n, paths in enumerate(cases, start=1):
            copy_all(paths, partial / f'plan_{n}')
        copy_all(misc, partial / 'misc_files')
        status = dicom_io.publish(out_root, name, overwrite=overwrite, success=bool(cases))  # no plan (e.g. an unreadable RTPLAN) keeps the old folder
    except BaseException:  # Ctrl-C too: no hidden half-copied folder left behind
        # A second Ctrl-C during this rmtree leaves a partly deleted .partial: the source scan skips it and the next run removes it before copying.
        dicom_io.remove_tree(partial)
        raise
    finally:
        _drop(staging)
    return time.monotonic() - copy_started, status


def _read_one(path):
    """(path, header) for a DICOM file, (path, None) when it has the preamble but its read fails, None for any other file."""
    if _has_preamble(path) != 'dicom':  # an OSError here propagates and refuses the patient; a zip is never read as DICOM
        return None
    return _read_header(path)


def _read_header(path):
    """(path, header) for a file already classified 'dicom', (path, None) when its read fails."""
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
    """'dicom' for a regular file whose bytes 128-131 are DICM, whatever its name (#10); 'zip' for PK\\x03\\x04 first; else None."""
    if not path.is_file():
        return None
    with open(path, 'rb') as handle:
        head = handle.read(132)
    return 'dicom' if head[128:] == b'DICM' else 'zip' if head[:4] == b'PK\x03\x04' else None


def _zip_scan(path):
    """(declared bytes, non-directory members, infolist) of a zip that opens and passes the allow-list, else None. Reads the central directory only."""
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
    except _ZIP_ERRORS:  # no `as`: its text can quote a member name
        return None
    if any(i.flag_bits & 0x61 or i.compress_type not in _ZIP_METHODS or i.header_offset < 0 for i in infos):  # encrypted, strong-encrypted, patched; odd methods; a corrupt central directory (#242)
        return None
    files = [i for i in infos if not i.is_dir()]
    return sum(i.file_size for i in files), len(files), infos


def _unzip(path, dest, infos):
    """Stage every non-directory member of the scanned `infos` as dest/<infolist position>; the paths, or None if the zip fails while read."""
    try:
        with zipfile.ZipFile(path) as zf:
            staged = []
            for n, info in enumerate(infos, start=1):  # the scanned list, so the caps checked are the sizes read (#243)
                if info.is_dir():
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, open(dest / str(n), 'wb') as target:  # the name is never a path; zipfile bounds the read by file_size
                    shutil.copyfileobj(source, target)
                staged.append(dest / str(n))
            return staged
    except _ZIP_ERRORS:
        return None


def _stage_zips(paths, staging, kinds=None):
    """
    Extract the zips among a folder's walked `paths` into `staging` as <zip #>/<member #> (both from 1).
    Returns (staged paths, zips staged, zip members staged, zips skipped). With a loose DICOM file (preamble) a bad zip (over the
    folder's caps, off the allow-list, or failing while read) is skipped and counted; without one the folder gets ZipRefused.
    Whatever this staged is removed before a ZipRefused; `staging` itself stays the caller's.
    `kinds`: [(path, _has_preamble(path))] when the caller has classified `paths` already.
    """
    kinds = kinds or [(path, _has_preamble(path)) for path in paths]  # one read each; an OSError propagates
    zips = sorted(path for path, kind in kinds if kind == 'zip')
    if not zips:
        return [], 0, 0, 0
    loose = any(kind == 'dicom' for _, kind in kinds)
    scans = [_zip_scan(path) for path in zips]
    good = [scan for scan in scans if scan is not None]
    if sum(s[0] for s in good) > MAX_UNZIP_BYTES or sum(s[1] for s in good) > MAX_ZIP_MEMBERS:  # before anything is written
        if loose:
            return [], 0, 0, len(zips)
        raise ZipRefused(REASON_ZIP_TOO_LARGE) from None
    if len(good) < len(zips) and not loose:
        raise ZipRefused(REASON_ZIP_UNREADABLE) from None
    staged, extracted = [], 0
    for n, (path, scan) in enumerate(zip(zips, scans, strict=True), start=1):
        here = None if scan is None else _unzip(path, staging / str(n), scan[2])
        if here is None:  # a scan failure (loose DICOM only: else refused above) or a failure while read
            dicom_io.remove_tree(staging / str(n))
            if not loose:
                for k in range(1, len(zips) + 1):
                    dicom_io.remove_tree(staging / str(k))
                raise ZipRefused(REASON_ZIP_UNREADABLE) from None  # outside any except: nothing is chained
            continue
        staged += here
        extracted += 1
    return staged, extracted, len(staged), len(zips) - extracted


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


def _copy_all(paths, folder, staging=None):
    """Copy `paths` into a new `folder`; those under `staging` (ours, removed later) are hard-linked, never a file of the source tree."""
    if not paths:
        return
    folder.mkdir(parents=True)
    for source in paths:
        target, n = folder / source.name, 1
        while target.exists():  # the same file name from two source sub-folders: keep both
            target, n = folder / f'{source.stem}_{n}{source.suffix}', n + 1
        if staging is not None and source.is_relative_to(staging):
            linked = True
            try:
                os.link(source, target)
            except OSError:  # no hard links here (another file system...): copy, as masks._link_or_copy
                linked = False
            if linked:
                continue
        shutil.copy2(source, target)
