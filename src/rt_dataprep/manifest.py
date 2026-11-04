"""
manifest.csv of a converted or preprocessed tree (spec: f3-manifest-split): one row per <patient>/<plan folder>/ that holds a series.csv,
and one per series of each <patient>/study_<n>/ that holds one (imaging-cases spec D9, D14-D16).

Each plan goes through preprocess.check_rows, which validates series.csv and reads the CT header only. A plan it refuses is left
out and counted. A study goes through preprocess.check_study_rows; one it refuses is left out and counts its series.csv rows, and a
series whose header cannot be read is left out and counted on its own. The tree is only read; manifest.csv (and its dated copy) in the root is the one place the patient folder names
appear. The log line carries counts only.
"""
import json
import logging
import re
from pathlib import Path

import numpy as np

from rt_dataprep import convert as conv
from rt_dataprep.dicom_io import CaseRefused, write_report
from rt_dataprep.preprocess import STUDY_FOLDER, check_rows, check_study_rows, read_info, series_id, study_refusal_count

log = logging.getLogger(__name__)

MANIFEST_NAME = 'manifest.csv'
COLUMNS = ['case_dir', 'patient', 'fx', 'ct_size', 'ct_spacing', 'dose', 'masks', 'rtplan_uid', 'ct_series_uid', 'study_uid',
           'case_type', 'modality', 'image']


def _geometry(size, spacing):
    """ct_size and ct_spacing text; spacing is the header's float32, 0.8 rather than 0.800000011920929 (#74)."""
    return ';'.join(map(str, size)), ';'.join(str(np.float32(v)) for v in spacing)


def _study_rows(patient, study):
    """(rows, refused) of one study folder: every series of a valid series.csv whose header reads, else the folder's refusal count."""
    failed = False
    try:
        series = check_study_rows(study)
    except (CaseRefused, OSError):  # an unreadable series.csv refuses the study, not the run
        failed = True
    if failed:
        try:
            return [], study_refusal_count(study)
        except OSError:
            return [], 1
    rows, refused = [], 0
    for row in series:
        try:
            size, _, spacing, _ = read_info(study / row['file'])
        except (CaseRefused, OSError):  # this series only: its siblings keep their rows
            refused += 1
            continue
        rows.append([f'{patient.name}/{study.name}/{series_id(row["file"])}', patient.name, '', *_geometry(size, spacing),
                     '0', json.dumps([]), '', row['series_instance_uid'], row['study_instance_uid'], 'imaging', row['modality'], row['file']])
    return rows, refused


def manifest(root):
    """Write root/manifest.csv, one row per plan and per study series, sorted by case_dir; returns (written, refused) counts."""
    root = Path(root)
    rows, refused = [], 0
    for patient in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        for plan in (p for p in patient.iterdir() if p.is_dir() and re.fullmatch(conv.PLAN_FOLDER, p.name) and (p / conv.SERIES_NAME).is_file()):
            try:
                kept, ct, _ = check_rows(plan)
            except (CaseRefused, OSError):  # an unreadable series.csv is one refused plan, not the end of the run (#74)
                refused += 1
                continue
            masks = sorted(row['roi_name'] for row in kept if row['kind'] == 'mask')
            # masks is a JSON list, as a name may hold ';'
            rows.append([f'{patient.name}/{plan.name}', patient.name, plan.name, *_geometry(ct[0], ct[2]),
                         str(int(any(row['kind'] == 'dose' for row in kept))), json.dumps(masks),
                         kept[0]['rtplan_uid'], kept[0]['series_instance_uid'], kept[0]['study_instance_uid'],  # kept[0] is the ct row
                         'plan', 'CT', ''])
        for study in (p for p in patient.iterdir() if p.is_dir() and re.fullmatch(STUDY_FOLDER, p.name) and (p / conv.SERIES_NAME).is_file()):
            study_rows, study_refused = _study_rows(patient, study)
            rows += study_rows
            refused += study_refused
    rows.sort()  # case_dir is the first column
    write_report(root, MANIFEST_NAME, COLUMNS, rows)
    log.info('manifest: %d written, %d refused', len(rows), refused)
    return len(rows), refused
