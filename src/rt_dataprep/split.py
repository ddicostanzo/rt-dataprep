"""
splits.json from manifest.csv (spec: f3-manifest-split): whole patients per split, in rnt's {split: {patient: [case ids]}} shape.

Only manifest.csv is read, with csv.DictReader so every value stays a string (a patient '00123' stays '00123'). The patients
are shuffled with numpy default_rng(seed); the first ceil(round(n*test_size, 9)) go to test, the next ceil(round(n*val_size, 9))
to val, the rest to train. Every refusal is a ValueError with a fixed text, raised before anything is written; nothing here
logs or raises a patient name, case id or path.
"""
import csv
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np

from rt_dataprep import convert as conv

log = logging.getLogger(__name__)

SPLITS_NAME = 'splits.json'
SPLIT_NAMES = ('train', 'val', 'test')
REASON_SIZE = 'test_size and val_size must each be at least 0 and below 1'
REASON_NO_TRAIN = 'the test and val sizes leave no patient for train'
REASON_OVERLAP = 'a patient of the external manifest is in a split'
REASON_EXISTS = 'splits.json already exists with different content; nothing written'


def _read(manifest_path):
    with open(manifest_path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


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
    patients = sorted(cases)
    n_test, n_val = (math.ceil(round(len(patients) * size, 9)) for size in (test_size, val_size))
    if n_test + n_val >= len(patients):
        raise ValueError(REASON_NO_TRAIN)
    order = [str(patient) for patient in np.random.default_rng(seed).permutation(patients)]
    chosen = {'test': order[:n_test], 'val': order[n_test:n_test + n_val], 'train': order[n_test + n_val:]}
    obj = {name: {patient: sorted(cases[patient]) for patient in chosen[name]} for name in SPLIT_NAMES}
    obj['case_id_hash'] = {name: hashlib.sha256('\n'.join(sorted(c for p in obj[name].values() for c in p)).encode()).hexdigest()
                           for name in SPLIT_NAMES}
    if external is not None:
        check_disjoint(obj, external)
    path, text = manifest_path.with_name(SPLITS_NAME), json.dumps(obj, indent=1, sort_keys=True) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError(REASON_EXISTS)
    conv._atomic(lambda temp: temp.write_text(text, encoding='utf-8'), path)
    log.info('split: %s patients, %s cases', '/'.join(str(len(obj[name])) for name in SPLIT_NAMES),
             '/'.join(str(sum(map(len, obj[name].values()))) for name in SPLIT_NAMES))
    return obj
