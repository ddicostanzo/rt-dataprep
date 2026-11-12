"""
manifest.jsonl of a converted or preprocessed tree (spec: f3-manifest-split; JSON Lines, one case object per line): one line per <patient>/<plan folder>/ that holds a series.json,
and one per series of each <patient>/study_<n>/ that holds one (imaging-cases spec D9, D14-D16).

Each plan goes through preprocess.check_rows, which validates series.json and reads the CT header only; the `series` key (a list of objects) then
reads the header of each identity series; an entry gains `fov_mask` when fovmask_<file> beside it is on the CT grid. A plan line ends with `retained`, the lowest dose/mask `retained` of its series.json, leaving out EXTERNAL masks, when a preprocess run set one (#225, #305), then `plan`, `beam`, `fluence`, `ion_plan` and `dwell`, the file names of its `plan`, `beam`, `fluence`, `ion_plan` and `dwell` series.json rows, each only when that row exists (#280, #279, #281, #282). A plan check_rows refuses is left out and counted. A patient folder whose name is not UTF-8 is left out and counts 1 (#260). A study goes through preprocess.check_study_rows; one it refuses is left out and counts its series.json items, and a
series whose header cannot be read is left out and counted on its own. The tree is only read; manifest.jsonl (and its dated copy, manifest_<UTC time>.jsonl) in the root is the one place the patient folder names
appear. The log line carries counts only.
"""
import logging
import re
from pathlib import Path

import numpy as np

from rt_dataprep import convert as conv
from rt_dataprep.dicom_io import CaseRefused, write_report
from rt_dataprep.preprocess import REASON_SERIES_CSV, STUDY_FOLDER, check_rows, check_study_rows, holds_series, is_external, is_legacy, read_info, read_series_json, same_grid, series_id, study_refusal_count

log = logging.getLogger(__name__)

MANIFEST_NAME = 'manifest.jsonl'
COLUMNS = ['case_dir', 'patient', 'fx', 'ct_size', 'ct_spacing', 'dose', 'masks', 'rtplan_uid', 'ct_series_uid', 'study_uid',
           'case_type', 'modality', 'image', 'series']
FOV_COLUMNS = [*COLUMNS, 'fov_mask']  # original schema is retained when there are no primary-image support masks
OPTIONAL_KEYS = ['patient_age', 'patient_sex', 'patient_weight', 'study_date', 'series_date', 'manufacturer', 'model', 'institution_name']


def _optional(row):
    """The optional keys of a case's primary image row, in order: from its `dicom` object (absent in a PR 1 series.json), series_date from its own
    field; a key with no value is left out of the line. A {min,max} or {distinct} value is copied as is."""
    dicom = row.get('dicom', {})
    values = {key: row['series_date'] if key == 'series_date' else dicom.get(key) for key in OPTIONAL_KEYS}
    return {key: value for key, value in values.items() if value not in (None, '')}


def _retained(kept):
    """{'retained': the lowest dose/mask `retained` of a plan as a float} (#225), or {} when no row has one (a converted tree, a non-geometric run) or only EXTERNAL masks do (#305).
    Every non-blank cell is parsed and validated first, an EXTERNAL one too; the minimum then leaves out is_external rows."""
    rows = [row for row in kept if row['kind'] in ('dose', 'mask') and row.get('retained')]
    try:
        values = [float(row['retained']) for row in rows]
    except ValueError:
        values = [float('nan')]
    if not all(np.isfinite(values)):  # a hand-edited cell: the plan is refused like any other malformed series.json
        raise CaseRefused(REASON_SERIES_CSV)
    counted = [value for row, value in zip(rows, values, strict=True) if not is_external(row)]
    return {'retained': min(counted)} if counted else {}


def _files(kept):
    """{'plan': file, 'beam': file, 'fluence': file, 'ion_plan': file, 'dwell': file} for the plan.parquet, beam volume, fluence map, ion_plan.parquet and dwell.parquet rows of a plan, in that order; a key is left out when its row is absent (#280, #279, #282)."""
    files = {kind: next((row['file'] for row in kept if row['kind'] == kind), None) for kind in ('plan', 'beam', 'fluence', 'ion_plan', 'dwell')}
    return {kind: file for kind, file in files.items() if file}


def _geometry(size, spacing):
    """ct_size and ct_spacing text; spacing is the header's float32, 0.8 rather than 0.800000011920929 (#74)."""
    return ';'.join(map(str, size)), ';'.join(str(np.float32(v)) for v in spacing)


def _study_rows(patient, study):
    """(rows, refused) of one study folder: every series of a valid series.json whose header reads, else the folder's refusal count."""
    if is_legacy(study):  # before any read: the CSV is never parsed, the study counts 1
        return [], 1
    try:
        parsed = read_series_json(study)
    except OSError:  # an unreadable series.json refuses the study, not the run
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
        rows.append(([f'{patient.name}/{study.name}/{series_id(row["file"])}', patient.name, '', *_geometry(info[0], info[2]),
                      '0', [], '', row['series_instance_uid'], row['study_instance_uid'], 'imaging', row['modality'],
                      row['file'], [], image_fov_mask(study, row['file'], info)], _optional(row)))
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
    """The plan's series rows that are in the CT frame (identity) and on the CT grid, as {file, modality}, in series.json order (D10);
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
    """Write root/manifest.jsonl (0 bytes when there are no cases), one row per plan and per study series, sorted by case_dir; returns (written, refused) counts."""
    root = Path(root)
    rows, refused = [], 0
    for patient in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        try:
            patient.name.encode('utf-8')
        except UnicodeEncodeError:  # a non-UTF-8 folder name (a surrogate escape) would make split refuse the whole manifest (#260)
            refused += 1
            continue
        for plan in (p for p in patient.iterdir() if p.is_dir() and re.fullmatch(conv.PLAN_FOLDER, p.name) and holds_series(p)):
            try:
                kept, ct, series = check_rows(plan)
                retained = _retained(kept)
            except (CaseRefused, OSError):  # an unreadable series.json is one refused plan, not the end of the run (#74)
                refused += 1
                continue
            masks = sorted(row['roi_name'] for row in kept if row['kind'] == 'mask')
            rows.append(([f'{patient.name}/{plan.name}', patient.name, plan.name, *_geometry(ct[0], ct[2]),
                          str(int(any(row['kind'] == 'dose' for row in kept))), masks,
                          kept[0]['rtplan_uid'], kept[0]['series_instance_uid'], kept[0]['study_instance_uid'],  # kept[0] is the ct row
                          'plan', 'CT', '', _aligned(plan, series, ct), image_fov_mask(plan, kept[0]['file'], ct)], _optional(kept[0]) | retained | _files(kept)))
        for study in (p for p in patient.iterdir() if p.is_dir() and re.fullmatch(STUDY_FOLDER, p.name) and holds_series(p)):
            study_rows, study_refused = _study_rows(patient, study)
            rows += study_rows
            refused += study_refused
    rows.sort(key=lambda pair: pair[0][0])  # case_dir; whole rows hold lists of dicts, which do not compare
    primary_masks = any(row[-1] for row, _ in rows)
    header = FOV_COLUMNS if primary_masks else COLUMNS
    write_report(root, MANIFEST_NAME, header,
                 [dict(zip(header, row if primary_masks else row[:-1], strict=True)) | optional for row, optional in rows])
    log.info('manifest: %d written, %d refused', len(rows), refused)
    return len(rows), refused
