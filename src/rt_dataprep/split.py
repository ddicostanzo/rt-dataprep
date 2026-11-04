"""
splits.json from manifest.csv (spec: f3-manifest-split): whole patients per split, in rnt's {split: {patient: [case ids]}} shape.

Only manifest.csv is read, with csv.DictReader so every value stays a string (a patient '00123' stays '00123'). The patients
are ordered by sha256(f'{seed}\\n{patient}'), the same on every numpy and Python version (#75); the first
ceil(round(n*test_size, 9)) go to test, the next ceil(round(n*val_size, 9)) to val, the rest to train. Every refusal is a ValueError with a fixed text, raised before anything is written; nothing here
logs or raises a patient name, case id or path.
"""
import contextlib
import csv
import hashlib
import json
import logging
import math
import os
import uuid
from pathlib import Path

from rt_dataprep import convert as conv

log = logging.getLogger(__name__)

SPLITS_NAME = 'splits.json'
SPLIT_NAMES = ('train', 'val', 'test')
REASON_SIZE = 'test_size and val_size must each be at least 0 and below 1'
REASON_NO_TRAIN = 'the test and val sizes leave no patient for train'
REASON_OVERLAP = 'a patient of the external manifest is in a split'
REASON_EXISTS = 'splits.json already exists with different content; nothing written'
REASON_COLUMNS = 'a manifest has no patient or case_dir column'
REASON_ROWS = 'a manifest is not UTF-8 or has a row without a patient or case_dir'


def _read(manifest_path):
    rows, fields = None, set()
    with open(manifest_path, newline='', encoding='utf-8-sig') as handle:  # -sig: a BOM, as Excel saves it, is not a header
        try:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            rows = list(reader) if {'patient', 'case_dir'} <= fields else []
        except UnicodeDecodeError:  # a ValueError whose text quotes bytes of the file: refused below, nothing chained (#78)
            pass
    if rows is None or any(not (row.get(k) or '').strip() for row in rows for k in ('patient', 'case_dir')):  # short row: None
        raise ValueError(REASON_ROWS)
    if not {'patient', 'case_dir'} <= fields:
        raise ValueError(REASON_COLUMNS)
    return rows


def _write_new(path, text):
    """Create `path` holding `text` atomically, keeping a `path` that appeared meanwhile: os.link, unlike os.replace, refuses
    an existing target, so of two concurrent runs only the first writes (#73). Without hard links, an O_EXCL reservation of
    `path` gives the same guard (#79)."""
    temp = path.with_name(f'.{os.getpid()}.{uuid.uuid4().hex}.tmp.{path.name}')  # unique per call, threads included (#79)
    failed = False
    try:
        try:
            temp.write_text(text, encoding='utf-8')
            try:
                os.link(temp, path)
            except FileExistsError:
                pass  # the caller compares it
            except OSError:  # no hard links on this file system: reserve path with O_EXCL, then rename over the reservation
                try:
                    os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    pass  # another run got there first: the caller compares it
                else:
                    try:
                        os.replace(temp, path)
                    except OSError:  # names the path
                        failed = True
                        with contextlib.suppress(OSError):
                            path.unlink()  # this run holds the reservation: no empty splits.json may block later runs
        except OSError:  # names the path
            failed = True
    finally:
        with contextlib.suppress(OSError):
            temp.unlink()
    if failed:
        raise OSError(conv.REASON_WRITE)


def check_disjoint(splits, external_manifest):
    """Raise ValueError(REASON_OVERLAP) when a patient of the external manifest.csv is in any split of `splits`."""
    placed = {patient for name in SPLIT_NAMES for patient in splits.get(name, {})}
    if any(row['patient'] in placed for row in _read(external_manifest)):
        raise ValueError(REASON_OVERLAP)


def split(manifest_path, *, test_size, val_size, seed, external=None):
    """Write splits.json next to `manifest_path` and return its dict; whole patients per split, reproducible from `seed`."""
    manifest_path = Path(manifest_path)
    if not all(0 <= size < 1 for size in (test_size, val_size)):
        raise ValueError(REASON_SIZE)
    cases = {}
    for row in _read(manifest_path):
        cases.setdefault(row['patient'], []).append(row['case_dir'])
    patients = list(cases)
    n_test, n_val = (math.ceil(round(len(patients) * size, 9)) for size in (test_size, val_size))
    if n_test + n_val >= len(patients):
        raise ValueError(REASON_NO_TRAIN)
    order = sorted(patients, key=lambda patient: hashlib.sha256(f'{seed}\n{patient}'.encode()).hexdigest())
    chosen = {'test': order[:n_test], 'val': order[n_test:n_test + n_val], 'train': order[n_test + n_val:]}
    obj = {name: {patient: sorted(cases[patient]) for patient in chosen[name]} for name in SPLIT_NAMES}
    obj['case_id_hash'] = {name: hashlib.sha256('\n'.join(sorted(c for p in obj[name].values() for c in p)).encode()).hexdigest()
                           for name in SPLIT_NAMES}
    if external is not None:
        check_disjoint(obj, external)
    path, text = manifest_path.with_name(SPLITS_NAME), json.dumps(obj, indent=1, sort_keys=True) + '\n'
    if not path.exists():
        _write_new(path, text)
    if path.read_text(encoding='utf-8') != text:
        raise ValueError(REASON_EXISTS)
    log.info('split: %s patients, %s cases', '/'.join(str(len(obj[name])) for name in SPLIT_NAMES),
             '/'.join(str(sum(map(len, obj[name].values()))) for name in SPLIT_NAMES))
    return obj
