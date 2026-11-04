"""DICOM header reading and path sanitising. Nothing here puts a path, tag value or UID in a message, warning or log."""
import contextlib
import csv
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import pydicom
import SimpleITK as sitk

REASON_UNREADABLE = 'DICOM header is unreadable'


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
    saved = os.dup(1), os.dup(2)
    try:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()  # buffered text must land in the sink, not in the restored fd
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        sink.close()


def quiet_call(fn, reason):
    """
    fn() with every output channel shut, for any third-party call that may name a path (pydicom, SimpleITK, DicomRTTool).

    Silenced and discarded, all restored afterwards: pydicom validation and log, Python warnings, every logger
    (logging.disable, global, so DicomRTTool's worker threads too), ITK warning display, fd 1 and 2, sys.stdout/stderr.
    Any Exception becomes CaseRefused(reason), raised after the except block so nothing of the original is chained.
    Running out of memory is the machine's, not the case's: it becomes a bare MemoryError() instead (#47).
    """
    failed = out_of_memory = False
    result = None
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
                        out_of_memory = isinstance(exc, MemoryError) or 'Failed to allocate memory' in str(exc)  # ITK's is a RuntimeError
                    except Exception:  # an exception whose str() fails is a refusal like any other
                        out_of_memory = False
        finally:
            sitk.ProcessObject.SetGlobalWarningDisplay(shown)
            logging.disable(prior_disable)
    if out_of_memory:
        raise MemoryError()
    if failed:
        raise CaseRefused(reason)
    return result


def read_header(path):
    """
    The Dataset at `path` without pixel data; any failure raises OSError(REASON_UNREADABLE) with no path or value in it.

    pydicom converts (and validates) a tag's value on first access, not at dcmread, so every element is converted here,
    inside the quiet block: later reads of the returned dataset hit cached values and stay silent.
    """
    try:
        with _pydicom_quiet():
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            for element in ds.iterall():
                _ = element.value  # the access is the point: it converts and caches
            return ds
    except Exception:  # InvalidDicomError, truncated file, missing file: all one fixed reason
        pass
    # Raised here, after the except block: inside it, __context__ would carry the original (path-bearing) exception.
    raise OSError(REASON_UNREADABLE)


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
    shutil.copyfile(dated, out_root / name)
