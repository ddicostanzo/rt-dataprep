"""
manifest.csv of a converted or preprocessed tree (spec: f3-manifest-split): one row per <patient>/<plan folder>/ that holds a series.csv.

Each plan goes through preprocess.check_rows, which validates series.csv and reads the CT header only. A plan it refuses is left
out and counted. The tree is only read; manifest.csv (and its dated copy) in the root is the one place the patient folder names
appear. The log line carries counts only.
"""
import logging
import re
from pathlib import Path

from rt_dataprep import convert as conv
from rt_dataprep.dicom_io import CaseRefused, write_report
from rt_dataprep.preprocess import check_rows

log = logging.getLogger(__name__)

MANIFEST_NAME = 'manifest.csv'
COLUMNS = ['case_dir', 'patient', 'fx', 'ct_size', 'ct_spacing', 'dose', 'masks', 'rtplan_uid', 'ct_series_uid', 'study_uid']


def manifest(root):
    """Write root/manifest.csv, one row per plan, sorted by case_dir; returns (written, refused) plan counts."""
    root = Path(root)
    rows, refused = [], 0
    for patient in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        for plan in (p for p in patient.iterdir() if p.is_dir() and re.fullmatch(conv.PLAN_FOLDER, p.name) and (p / conv.SERIES_NAME).is_file()):
            try:
                kept, ct = check_rows(plan)
            except CaseRefused:
                refused += 1
                continue
            masks = sorted(row['roi_name'] for row in kept if row['kind'] == 'mask')
            rows.append([f'{patient.name}/{plan.name}', patient.name, plan.name,
                         ';'.join(map(str, ct[0])), ';'.join(map(str, ct[2])),
                         str(int(any(row['kind'] == 'dose' for row in kept))), ';'.join(masks),
                         kept[0]['rtplan_uid'], kept[0]['series_instance_uid'], kept[0]['study_instance_uid']])  # kept[0] is the ct row
    rows.sort()  # case_dir is the first column
    write_report(root, MANIFEST_NAME, COLUMNS, rows)
    log.info('manifest: %d written, %d refused', len(rows), refused)
    return len(rows), refused
