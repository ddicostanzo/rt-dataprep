"""DICOM header reading and path sanitising. Nothing here puts a path, tag value or UID in a message, warning or log."""
import contextlib
import logging
import re
import warnings

import pydicom

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
        raise OSError(REASON_UNREADABLE) from None  # from None: the chained message could quote the path


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
