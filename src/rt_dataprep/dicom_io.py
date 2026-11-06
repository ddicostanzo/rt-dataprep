"""DICOM header reading and path sanitising. Nothing here puts a path, tag value or UID in a message, warning or log."""
import contextlib
import csv
import errno
import logging
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import warnings
from pathlib import Path

import pydicom
import SimpleITK as sitk

REASON_UNREADABLE = 'DICOM header is unreadable'
REASON_INPUT_INSIDE_OUTPUT = 'with --overwrite, the input root must not be the output root or inside it'
ALLOCATION_FAILURE = re.compile(
    r'\b(?:failed to allocate\b[^\n]*\bmemory|cannot allocate memory|out of memory|'
    r'std::bad_alloc|bad allocation|memory allocation (?:has )?failed)\b', re.IGNORECASE)


def allocation_failure(exc):
    """Typed allocation errors, or recognized ITK/C++ RuntimeError wording. Never return or emit the original text."""
    if isinstance(exc, MemoryError) or (isinstance(exc, OSError) and exc.errno == errno.ENOMEM):
        return True
    return isinstance(exc, RuntimeError) and ALLOCATION_FAILURE.search(str(exc)) is not None


@contextlib.contextmanager
def _pydicom_quiet():
    """No validation, no warnings, no pydicom log records, for one read; all back as they were. Their text quotes values."""
    settings = pydicom.config.settings
    mode = settings.reading_validation_mode
    logger = logging.getLogger('pydicom')
    disabled = logger.disabled
    settings.reading_validation_mode = pydicom.config.IGNORE
    logger.disabled = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield
    finally:
        settings.reading_validation_mode = mode
        logger.disabled = disabled


class CaseRefused(Exception):
    """A case is refused for one fixed reason. str() is exactly the reason: no path, UID, date or tag value."""


@contextlib.contextmanager
def _fds_quiet():
    """fd 1 and 2 point at a temp file (discarded) for one block, then back. C++ writers (ITK, GDCM) bypass sys.stderr."""
    sys.stdout.flush()
    sys.stderr.flush()
    sink = tempfile.TemporaryFile()  # before the dups: a failure here leaks no fd
    saved = []  # filled one dup at a time: the finally closes whichever were made
    try:
        saved.append(os.dup(1))
        saved.append(os.dup(2))
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        yield
    finally:
        with contextlib.ExitStack() as restore:  # every step runs even when one raises (#85); callbacks run last-in first
            restore.callback(sink.close)
            for fd in saved:
                restore.callback(os.close, fd)
            if len(saved) == 2:  # both dups made: fd 1 and 2 may have been redirected
                restore.callback(os.dup2, saved[1], 2)
                restore.callback(os.dup2, saved[0], 1)
            sys.stdout.flush()  # buffered text must land in the sink, not in the restored fd
            sys.stderr.flush()


def quiet_call(fn, reason):
    """
    fn() with every output channel shut, for any third-party call that may name a path (pydicom, SimpleITK, DicomRTTool).

    Silenced and discarded, all restored afterwards: pydicom validation and log, Python warnings, every logger
    (logging.disable, global, so DicomRTTool's worker threads too), ITK warning display, fd 1 and 2, sys.stdout/stderr.

    Contract for the closure: it holds third-party access, value extraction and deliberate raises only. No arithmetic or
    logic of ours: a bug there would read as a per-case refusal, so it belongs outside, where a bug aborts the run (#40).
    Every Exception from fn becomes CaseRefused(reason), raised after the except block so nothing of the original is chained.
    Running out of memory is the machine's, not the case's: it becomes a bare MemoryError() instead (#47).
    Any other BaseException keeps its kind and loses its message, raised bare after the except block: SystemExit keeps an
    int or None code (else 1); a builtin kind whose constructor needs arguments, and any kind from another module (its
    constructor would run outside the quiet block), becomes KeyboardInterrupt().

    Not thread-safe: it swaps process-wide state (fd 1 and 2, logging.disable, warning filters), so parallel callers (#2)
    must be process-based or serialize it.
    """
    failed = out_of_memory = False
    interrupt = result = None
    prior_disable = logging.root.manager.disable
    shown = sitk.ProcessObject.GetGlobalWarningDisplay()
    with open(os.devnull, 'w') as sink:
        try:
            logging.disable(logging.CRITICAL)
            sitk.ProcessObject.SetGlobalWarningDisplay(False)
            with (_fds_quiet(), contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink),
                  warnings.catch_warnings(record=True), _pydicom_quiet()):
                warnings.simplefilter('always')
                try:
                    result = fn()
                except Exception as exc:  # no message, no traceback kept
                    failed = True
                    try:
                        out_of_memory = allocation_failure(exc)
                    except Exception:  # an exception whose str() fails is a refusal like any other
                        out_of_memory = False
                except BaseException as exc:  # interrupt or exit: keep the kind (and an exit code), drop the message
                    interrupt = type(exc), exc.code if isinstance(exc, SystemExit) else None
        finally:
            sitk.ProcessObject.SetGlobalWarningDisplay(shown)
            logging.disable(prior_disable)
    if interrupt:
        kind, code = interrupt
        if issubclass(kind, SystemExit):
            if code is None:
                raise SystemExit()
            raise SystemExit(code if isinstance(code, int) else 1)
        try:
            bare = kind() if getattr(kind, '__module__') == 'builtins' else KeyboardInterrupt()  # noqa: B009 (a third-party __init__ may write)
        except Exception:  # BaseExceptionGroup, a class with required arguments
            bare = KeyboardInterrupt()
        raise bare
    if out_of_memory:
        raise MemoryError()
    if failed:
        raise CaseRefused(reason)
    return result


def read_header_core(path):
    """
    The Dataset at `path` without pixel data; any failure raises OSError(REASON_UNREADABLE) with no path or value in it.

    pydicom converts (and validates) a tag's value on first access, not at dcmread, so every element is converted here:
    the caller holds the quiet block, and later reads of the returned dataset hit cached values and stay silent.
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        for element in ds.iterall():
            _ = element.value  # the access is the point: it converts and caches
        return ds
    except Exception:  # InvalidDicomError, truncated file, missing file: all one fixed reason
        pass
    # Raised here, after the except block: inside it, __context__ would carry the original (path-bearing) exception.
    raise OSError(REASON_UNREADABLE)


def read_header(path):
    """read_header_core inside its own quiet block."""
    with _pydicom_quiet():
        return read_header_core(path)


# Ported from DICOM_Transport src/common.py. Everything outside this set becomes "_".
_UNSAFE_PATH_CHARS = re.compile(r'[^A-Za-z0-9._-]')
_WINDOWS_RESERVED_NAMES = frozenset(
    ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
)


def sanitize_path_component(value, fallback='UNKNOWN'):
    """A single safe path component from an untrusted value (a PatientID): no separator, no traversal, <= 64 chars."""
    if value is None:
        return fallback
    # Only the final component, so "../../x" and "C:\\Windows\\x" both reduce to "x".
    text = str(value).replace('\\', '/').split('/')[-1].strip()
    text = _UNSAFE_PATH_CHARS.sub('_', text)
    text = text.strip('.')  # "." and ".." carry traversal meaning even with no separator left
    if not text or set(text) <= {'_'}:
        return fallback
    if text.upper() in _WINDOWS_RESERVED_NAMES or text.split('.')[0].upper() in _WINDOWS_RESERVED_NAMES:
        text = f'_{text}'
    return text[:64]


def write_report(out_root, name, header, rows):
    """
    `rows` under `header` at out_root/<stem>_<UTC time>.csv, one file per run so a rerun loses nothing (#50), then copied
    to out_root/`name`, the latest run's report.
    """
    out_root = Path(out_root)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())  # shortcut: two runs in one second share a file; add a counter if that happens
    dated = out_root / f'{Path(name).stem}_{stamp}.csv'
    with open(dated, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    latest = out_root / f'.{os.getpid()}.tmp.{name}'
    try:
        shutil.copyfile(dated, latest)
        os.replace(latest, out_root / name)  # atomic: a second Ctrl-C mid-copy cannot truncate the latest report (#65)
    finally:
        with contextlib.suppress(OSError):  # gone after a replace; a failed copy leaves no hidden temp (#81)
            latest.unlink()


def rename_dir(src, dst):
    """The one place the swap renames a folder, so a test can fail a single step."""
    Path(src).rename(dst)


def _clear_readonly(func, path, exc):
    """rmtree onexc: clear a read-only bit (Windows) and retry the failed call once; a second failure propagates to remove_tree."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(path):
    """
    Delete the tree at `path`; a symlink (dangling or not) is unlinked, never followed. Read-only files are cleared first.
    Any OSError, a missing path included, is swallowed: a leftover is retried by the next recover.
    """
    try:
        if os.path.islink(path):
            os.unlink(path)
        else:
            shutil.rmtree(path, onexc=_clear_readonly)
    except OSError:
        pass


def recover(out_root, name):
    """
    Settle what an interrupted --overwrite swap left for `name` under out_root: `.<name>.old` goes back to `<name>` when
    that is missing, else it is deleted (the swap had finished); a leftover `.<name>.partial` is deleted. Only those three
    direct children are touched. A failed restore raises OSError. Logs nothing.
    """
    out_root = Path(out_root)
    target, old, partial = out_root / name, out_root / f'.{name}.old', out_root / f'.{name}.partial'
    if os.path.lexists(old):
        if os.path.lexists(target):
            remove_tree(old)  # a leftover is retried by the next recover
        else:
            rename_dir(old, target)
    remove_tree(partial)


def publish(out_root, name, *, overwrite, success):
    """
    Move the finished `.<name>.partial` into place as `<name>`; returns 'new', 'replaced' or 'kept'.

    No target: the partial becomes the target ('new'), whatever `overwrite` and `success` say. A target with overwrite=False
    is misuse: ValueError, nothing touched. A target with overwrite=True: when `success` (the build had a successful case)
    the target moves to `.<name>.old`, the partial moves in, `.<name>.old` is deleted ('replaced'); otherwise the target
    stays and the partial is deleted ('kept'). If the partial cannot move in, the target is moved back and that OSError
    is raised; if the move back fails too, `.<name>.old` stays for recover and the same OSError is raised. Logs nothing.
    """
    out_root = Path(out_root)
    target, old, partial = out_root / name, out_root / f'.{name}.old', out_root / f'.{name}.partial'
    if not os.path.lexists(target):
        rename_dir(partial, target)
        return 'new'
    if not overwrite:
        raise ValueError('output folder exists and overwrite is off')
    if not success:
        remove_tree(partial)
        return 'kept'
    rename_dir(target, old)  # a failure here moved nothing
    failure = None
    try:
        rename_dir(partial, target)
    except OSError as exc:
        failure = exc
    if failure is not None:  # raised after the except block, as elsewhere in this module
        try:
            rename_dir(old, target)
        except OSError:
            pass  # .<name>.old stays; recover restores it on the next run
        raise failure
    remove_tree(old)  # the new output is complete: a leftover is removed by the next recover
    return 'replaced'
