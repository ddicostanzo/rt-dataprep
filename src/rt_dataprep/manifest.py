"""
manifest.csv of a converted or preprocessed tree (spec: f3-manifest-split): one row per <patient>/<plan folder>/ that holds a series.csv,
and one per series of each <patient>/study_<n>/ that holds one (imaging-cases spec D9, D14-D16).

Each plan goes through preprocess.check_rows, which validates series.csv and reads the CT header only; the `series` column then
reads the header of each identity series; an entry gains `fov_mask` when fovmask_<file> beside it is on the CT grid. A plan check_rows refuses is left out and counted. A study goes through preprocess.check_study_rows; one it refuses is left out and counts its series.csv rows, and a
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
from rt_dataprep.preprocess import STUDY_FOLDER, check_rows, check_study_rows, read_info, read_series_csv, same_grid, series_id, study_refusal_count

log = logging.getLogger(__name__)

MANIFEST_NAME = 'manifest.csv'
COLUMNS = ['case_dir', 'patient', 'fx', 'ct_size', 'ct_spacing', 'dose', 'masks', 'rtplan_uid', 'ct_series_uid', 'study_uid',
           'case_type', 'modality', 'image', 'series']
FOV_COLUMNS = [*COLUMNS, 'fov_mask']  # original schema is retained when there are no primary-image support masks


def _geometry(size, spacing):
    """ct_size and ct_spacing text; spacing is the header's float32, 0.8 rather than 0.800000011920929 (#74)."""
    return ';'.join(map(str, size)), ';'.join(str(np.float32(v)) for v in spacing)


def _study_rows(patient, study):
    """(rows, refused) of one study folder: every series of a valid series.csv whose header reads, else the folder's refusal count."""
    try:
        parsed = read_series_csv(study)
    except OSError:  # an unreadable series.csv refuses the study, not the run
        return [], 1
    try:
        series = check_study_rows(study, parsed)
    except (CaseRefused, OSError):
        return [], study_refusal_count(parsed[0])
    rows, refused = [], 0
    for row in series:
        try:
            info = read_info(study / row['file'])
        except (CaseRefused, OSError):  # this series only: its siblings keep their rows
            refused += 1
            continue
        rows.append([f'{patient.name}/{study.name}/{series_id(row["file"])}', patient.name, '', *_geometry(info[0], info[2]),
                     '0', json.dumps([]), '', row['series_instance_uid'], row['study_instance_uid'], 'imaging', row['modality'],
                     row['file'], json.dumps([]), image_fov_mask(study, row['file'], info)])
    return rows, refused


def image_fov_mask(folder, file, info):
    """Name of a readable support mask on the image's grid, else ''; no voxel reads or symlink following."""
    mask = folder / f'fovmask_{file}'
    try:
        if mask.is_file() and not mask.is_symlink() and same_grid(read_info(mask), info):
            return mask.name
    except (CaseRefused, OSError):
        pass
    return ''


def _aligned(plan, series, ct):
    """The plan's series rows that are in the CT frame (identity) and on the CT grid, as {file, modality}, in series.csv order (D10);
    plus fov_mask when fovmask_<file> beside it has a header on the CT grid (#129, #146: a stray mask file is not listed)."""
    listed = []
    for row in series:
        if row['reg_matrix'] != 'identity':
            continue
        try:
            on_grid = same_grid(read_info(plan / row['file']), ct)
        except (CaseRefused, OSError):  # this series only: it is not listed, the plan keeps its row
            continue
        if on_grid:
            entry = {'file': row['file'], 'modality': row['modality']}
            mask = plan / f'fovmask_{row["file"]}'
            try:
                if mask.is_file() and same_grid(read_info(mask), ct):
                    entry['fov_mask'] = mask.name
            except (CaseRefused, OSError):  # an unreadable mask is not listed; the series keeps its entry
                pass
            listed.append(entry)
    return listed


def manifest(root):
    """Write root/manifest.csv, one row per plan and per study series, sorted by case_dir; returns (written, refused) counts."""
    root = Path(root)
    rows, refused = [], 0
    for patient in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        for plan in (p for p in patient.iterdir() if p.is_dir() and re.fullmatch(conv.PLAN_FOLDER, p.name) and (p / conv.SERIES_NAME).is_file()):
            try:
                kept, ct, series = check_rows(plan)
            except (CaseRefused, OSError):  # an unreadable series.csv is one refused plan, not the end of the run (#74)
                refused += 1
                continue
            masks = sorted(row['roi_name'] for row in kept if row['kind'] == 'mask')
            # masks is a JSON list, as a name may hold ';'
            rows.append([f'{patient.name}/{plan.name}', patient.name, plan.name, *_geometry(ct[0], ct[2]),
                         str(int(any(row['kind'] == 'dose' for row in kept))), json.dumps(masks),
                         kept[0]['rtplan_uid'], kept[0]['series_instance_uid'], kept[0]['study_instance_uid'],  # kept[0] is the ct row
                         'plan', 'CT', '', json.dumps(_aligned(plan, series, ct)), image_fov_mask(plan, kept[0]['file'], ct)])
        for study in (p for p in patient.iterdir() if p.is_dir() and re.fullmatch(STUDY_FOLDER, p.name) and (p / conv.SERIES_NAME).is_file()):
            study_rows, study_refused = _study_rows(patient, study)
            rows += study_rows
            refused += study_refused
    rows.sort()  # case_dir is the first column
    primary_masks = any(row[-1] for row in rows)
    write_report(root, MANIFEST_NAME, FOV_COLUMNS if primary_masks else COLUMNS,
                 rows if primary_masks else [row[:-1] for row in rows])
    log.info('manifest: %d written, %d refused', len(rows), refused)
    return len(rows), refused
