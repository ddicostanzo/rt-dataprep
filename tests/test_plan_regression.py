"""
Plan folders are pinned (task-01, #17): the converted plan folders of one synthetic patient must keep the exact file list,
series rows (pinned as the CSV text they were) and voxel/geometry of every volume while images/ is replaced by study folders. A characterization test,
written green against the code before that change. The pinned values are literals, taken once from that run.
Every tree is synthetic.
"""
import csv
import hashlib
import io
import re

import numpy as np
import pytest
import series_files as sf
import SimpleITK as sitk
import synthetic_dicom as sd
from pydicom.uid import generate_uid
from test_convert import BOX, CT_DATE, DOSE_DATE, ORIGIN, PLAN_DATE, SHAPE, SPACING, STRUCT_DATE, dose_spec
from test_series import VALUES, rigid

from rt_dataprep import convert as conv

PID = 'SYNTHETICPTREG'
MR_SPACING, MR_ORIGIN = (1.5, 2.0, 3.0), (7.0, -4.0, 12.0)
PARTS = ('CT_SERIES', 'STUDY', 'FRAME', 'DOSE_SERIES', 'STRUCT_SERIES', 'PLAN_SOP', 'STRUCT_SOP', 'MR_SERIES', 'MR_STUDY', 'MR_FRAME')


def uid(name):
    """The same UID in every run, so the pinned series.csv text can name it by placeholder."""
    return generate_uid(entropy_srcs=[f'SYNTHETIC_{name}'])


def uids(n):
    return {part: uid(f'plan{n}_{part}') for part in PARTS}


def build_plan(folder, n, values, dose_value, rois, **plan_meta):
    """A plan folder as test_convert.make_plan writes it, with a fixed UID for everything that reaches its series.csv."""
    u = uids(n)
    folder.mkdir(parents=True)
    ct = sd.write_ct_series(folder, values, spacing=SPACING, origin=ORIGIN, patient_id=PID, series_uid=u['CT_SERIES'],
                            frame_uid=u['FRAME'], study_uid=u['STUDY'], series_date=CT_DATE)
    study = dict(patient_id=PID, study_uid=u['STUDY'])
    struct = sd.write_rtstruct_rois(folder / 'struct.dcm', ct, rois, sop_uid=u['STRUCT_SOP'], series_uid=u['STRUCT_SERIES'],
                                    series_date=STRUCT_DATE, **study)
    plan = sd.write_rtplan(folder / 'plan.dcm', struct, np.zeros((2, 3)), sop_uid=u['PLAN_SOP'], **study, **plan_meta)
    sd.write_rtdose(folder / 'dose_0.dcm', plan_sop_uid=plan, series_uid=u['DOSE_SERIES'], series_date=DOSE_DATE, **study,
                    **dose_spec(dose_value, frame_uid=u['FRAME']))
    return ct


def build(org):
    """plan_1 (CT, dose, mask, a registered MR) and plan_2 (CT, dose); an unregistered MR and PT in misc_files."""
    values = (np.arange(np.prod(SHAPE)) % 200 - 100).reshape(SHAPE)
    ct1 = build_plan(org / PID / 'plan_1', 1, values, 10, [BOX], label='Plan A', plan_date=PLAN_DATE, fractions=5)
    build_plan(org / PID / 'plan_2', 2, values[::-1], 20, [], label='Plan B', fractions=3)
    u = uids(1)
    mr = sd.write_ct_series(org / PID / 'plan_1', VALUES, spacing=MR_SPACING, origin=MR_ORIGIN, patient_id=PID, prefix='mr',
                            modality='MR', series_uid=u['MR_SERIES'], frame_uid=u['MR_FRAME'], study_uid=u['MR_STUDY'])
    sd.write_reg(org / PID / 'plan_1' / 'reg.dcm', ct1['study_uid'], mr['study_uid'], sop_uid=uid('reg'), patient_id=PID,
                 registrations=[dict(frame_uid=ct1['frame_uid'], matrices=rigid()),
                                dict(frame_uid=mr['frame_uid'], matrices=rigid(translation=(10, 0, 0)))])
    misc = org / PID / 'misc_files'
    misc.mkdir()
    for prefix, modality in (('free_mr', 'MR'), ('free_pt', 'PT')):  # no REG names their frame: not registered to any plan
        sd.write_ct_series(misc, VALUES, patient_id=PID, prefix=prefix, modality=modality, series_uid=uid(f'{prefix}_series'),
                           frame_uid=uid(f'{prefix}_frame'), study_uid=uid(f'{prefix}_study'))


@pytest.fixture
def converted(tmp_path):
    org, out = tmp_path / 'org', tmp_path / 'out'
    org.mkdir()
    build(org)
    conv.convert(org, out)
    return out / PID


def pinned_volume(path):
    image = sitk.ReadImage(str(path))
    digest = hashlib.sha256(sitk.GetArrayFromImage(image).tobytes()).hexdigest()
    return digest, image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()


EYE = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
HEADER = ('file,kind,modality,series_description,series_instance_uid,study_instance_uid,frame_of_reference_uid,series_date,'
          'roi_name,dose_summation_type,reg_matrix,rtplan_uid,rtplan_label,rtplan_date,fractions_planned')

def csv_text(*rows):
    return '\r\n'.join((HEADER, *rows)) + '\r\n'  # csv.DictWriter's line ending


PLAN_1_COLUMNS = '{PLAN_SOP},Plan A,{PLAN_DATE},5'
FILES = {
    'plan_1': ['dose.nii.gz', 'liver.nii.gz', 'planning_ct.nii.gz', 'series.json', 'series_MR_1.nii.gz'],
    'plan_2': ['dose.nii.gz', 'planning_ct.nii.gz', 'series.json'],
}
SERIES_CSV = {
    'plan_1': csv_text(
        'planning_ct.nii.gz,ct,CT,,{CT_SERIES},{STUDY},{FRAME},{CT_DATE},,,identity,' + PLAN_1_COLUMNS,
        'dose.nii.gz,dose,RTDOSE,,{DOSE_SERIES},{STUDY},{FRAME},{DOSE_DATE},,PLAN,,' + PLAN_1_COLUMNS,
        'liver.nii.gz,mask,RTSTRUCT,,{STRUCT_SERIES},{STUDY},{FRAME},{STRUCT_DATE},Liver,,,' + PLAN_1_COLUMNS,
        'series_MR_1.nii.gz,series,MR,,{MR_SERIES},{MR_STUDY},{MR_FRAME},,,,1 0 0 10 0 1 0 0 0 0 1 0 0 0 0 1,' + PLAN_1_COLUMNS),
    'plan_2': csv_text(
        'planning_ct.nii.gz,ct,CT,,{CT_SERIES},{STUDY},{FRAME},{CT_DATE},,,identity,{PLAN_SOP},Plan B,,3',
        'dose.nii.gz,dose,RTDOSE,,{DOSE_SERIES},{STUDY},{FRAME},{DOSE_DATE},,PLAN,,{PLAN_SOP},Plan B,,3'),
}
# per volume: sha256 of the voxel bytes, size (x, y, z), spacing, origin, direction
VOLUMES = {
    'plan_1': {
        'dose.nii.gz': ('a8f8807505d398b513259356d388cd31969e1223f4e1262fe95572f9d21121db', (6, 5, 4), (1.0, 1.5, 2.0), (-3.0, 5.0, 7.0), EYE),
        'liver.nii.gz': ('58b2378865700d2c9c9dea3dfd98293016c9d6bc35367539cc1fc5cdd8a1857d', (6, 5, 4), (1.0, 1.5, 2.0), (-3.0, 5.0, 7.0), EYE),
        'planning_ct.nii.gz': ('cd124b4cbb627956882f1cfe16ad64475474bc6a868db6d108902fa9284c56c7', (6, 5, 4), (1.0, 1.5, 2.0), (-3.0, 5.0, 7.0), EYE),
        'series_MR_1.nii.gz': ('c1ff115770d0609cbdccb867162367b37a17ef0d8ff9d5f1f059b43d74ac1048', (5, 3, 4), (1.5, 2.0, 3.0), (7.0, -4.0, 12.0), EYE),
    },
    'plan_2': {
        'dose.nii.gz': ('5569720abcaac2cf8a535a915dcee1760147ad22e8cde484e7f11afe7830f209', (6, 5, 4), (1.0, 1.5, 2.0), (-3.0, 5.0, 7.0), EYE),
        'planning_ct.nii.gz': ('ed83d83c79e306fe9f14cdb5704283db28de0457310ba1aee24b893552c62e84', (6, 5, 4), (1.0, 1.5, 2.0), (-3.0, 5.0, 7.0), EYE),
    },
}


@pytest.mark.parametrize('plan', ['plan_1', 'plan_2'])
def test_the_plan_folder_is_what_it_was_before_study_folders(converted, plan):
    folder = converted / plan
    assert sorted(p.name for p in folder.iterdir()) == FILES[plan]
    pinned = SERIES_CSV[plan].format(**uids(int(plan[-1])), CT_DATE=CT_DATE, STRUCT_DATE=STRUCT_DATE,
                                     DOSE_DATE=DOSE_DATE, PLAN_DATE=PLAN_DATE)
    # the rows (and their key order) v0.1.x wrote as CSV text are the objects of series.json now
    assert [list(r.items()) for r in csv.DictReader(io.StringIO(pinned, newline=''))] == [list(sf.columns(o).items()) for o in sf.read(folder)]
    assert {p.name: pinned_volume(p) for p in sorted(folder.glob('*.nii.gz'))} == VOLUMES[plan]


def test_the_patient_holds_exactly_the_two_plan_folders(converted):
    assert sorted(p.name for p in converted.iterdir() if re.fullmatch(conv.PLAN_FOLDER, p.name)) == ['plan_1', 'plan_2']
