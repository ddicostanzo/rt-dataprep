"""convert (task-04): assemble, planning CT, dose, atomic plan folders, report. Every tree is synthetic and built in tmp_path."""
import ast
import csv
import errno
import hashlib
import inspect
import logging
import re
import operator
import os
import subprocess
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pydicom
import pytest
import series_files as sf
import SimpleITK as sitk
import synthetic_dicom as sd
from test_dicom_io import fail_python_open as fail_open  # workers=1 where used: a worker process would not see the patch

from rt_dataprep import convert as conv
from rt_dataprep import dicom_io
from rt_dataprep import masks
from rt_dataprep.organize import organize
from rt_dataprep.dicom_io import CaseRefused

PID = 'SYNTHETICPTA'
SHAPE = (4, 5, 6)  # (z, y, x)
SPACING = (1.0, 1.5, 2.0)  # (x, y, z): anisotropic, so an axis swap cannot pass
ORIGIN = (-3.0, 5.0, 7.0)
SCALING = 0.5

def listing(root):
    """Sorted names under root, without the dated per-run copies of the report."""
    return sorted(p.name for p in root.iterdir() if not re.fullmatch(r'\w+_report_\d{8}T\d{6}Z\.csv', p.name))



def edit(path, change):
    ds = pydicom.dcmread(path)
    change(ds)
    pydicom.dcmwrite(path, ds)


def dose_spec(value, summation='PLAN', **kwargs):
    """A write_rtdose argument dict: a constant dose on the CT grid itself, so resampling is the identity."""
    spec = dict(stored=np.full(SHAPE, value), scaling=SCALING, summation=summation, position=ORIGIN,
                pixel_spacing=(SPACING[1], SPACING[0]), offsets=[k * SPACING[2] for k in range(SHAPE[0])])
    spec.update(kwargs)
    return spec


def make_plan(folder, doses=(), rois=(), values=None, contour_images=True, patient=PID, ct_meta=None, plan_meta=None):
    """One plan folder: CT, RTSTRUCT, RTPLAN and an RTDOSE per entry of `doses` (write_rtdose kwargs)."""
    folder.mkdir(parents=True, exist_ok=True)
    if values is None:
        values = (np.arange(np.prod(SHAPE)) % 200 - 100).reshape(SHAPE)
    ct = sd.write_ct_series(folder, values, spacing=SPACING, origin=ORIGIN, patient_id=patient, **(ct_meta or {}))
    study = dict(patient_id=patient, study_uid=ct['study_uid'])
    struct = sd.write_rtstruct_rois(folder / 'struct.dcm', ct, list(rois), contour_images=contour_images, **study)
    plan = sd.write_rtplan(folder / 'plan.dcm', struct, np.zeros((2, 3)), **study, **(plan_meta or {}))
    for k, spec in enumerate(doses):
        spec = dict(spec)
        spec.setdefault('frame_uid', ct['frame_uid'])
        sd.write_rtdose(folder / f'dose_{k}.dcm', plan_sop_uid=plan, **study, **spec)
    return dict(ct=ct, struct=struct, plan=plan, values=np.asarray(values), folder=folder)


@pytest.fixture
def tree(tmp_path):
    org, out = tmp_path / 'org', tmp_path / 'out'
    org.mkdir()
    return org, out


def run_plan(folder, case):
    """convert_plan on the plan folder `folder`: its headers read the way the patient level reads them."""
    headers, unreadable = conv.read_folder_headers(folder)
    return conv.convert_plan(headers, unreadable, case)


def one_plan(tree, **kwargs):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', **kwargs)
    return plan, conv.convert(org, out).rows, out


def row_of(rows, folder='plan_1'):
    return next(r for r in rows if r.folder == folder)


def read(path):
    return sitk.ReadImage(str(path))


def arr(path):
    return sitk.GetArrayFromImage(read(path))


# ---------------------------------------------------------------- A2.1: planning CT and dose

def test_a2_1_ct_equals_source_and_dose_is_scaled_and_resampled(tree):
    # dose grid: coarser, offset, covering part of the CT; stored value linear in the indices, so linear resampling is exact
    stored = 100 + 7 * np.arange(2)[:, None, None] + 5 * np.arange(2)[None, :, None] + 3 * np.arange(3)[None, None, :]
    dose = dict(stored=stored, scaling=1e-2, summation='PLAN', position=(-1.0, 6.5, 9.0), pixel_spacing=(3.0, 2.0),
                offsets=[0.0, 4.0])
    plan, rows, out = one_plan(tree, doses=[dose])
    row = row_of(rows)
    assert (row.outcome, row.reason) == (conv.OUTCOME_CONVERTED, '')
    folder = out / PID / 'plan_1'
    ct = read(folder / 'planning_ct.nii.gz')
    assert ct.GetPixelID() == sitk.sitkFloat32
    assert ct.GetSize() == (SHAPE[2], SHAPE[1], SHAPE[0])
    np.testing.assert_allclose(ct.GetSpacing(), SPACING)
    np.testing.assert_allclose(ct.GetOrigin(), ORIGIN)
    np.testing.assert_allclose(ct.GetDirection(), np.eye(3).ravel())
    np.testing.assert_array_equal(sitk.GetArrayFromImage(ct), plan['values'].astype(np.float32))

    got = arr(folder / 'dose.nii.gz')
    dose_img = read(folder / 'dose.nii.gz')
    assert dose_img.GetSize() == ct.GetSize() and dose_img.GetOrigin() == ct.GetOrigin()
    assert dose_img.GetSpacing() == ct.GetSpacing() and dose_img.GetDirection() == ct.GetDirection()
    x = ORIGIN[0] + SPACING[0] * np.arange(SHAPE[2])[None, None, :]
    y = ORIGIN[1] + SPACING[1] * np.arange(SHAPE[1])[None, :, None]
    z = ORIGIN[2] + SPACING[2] * np.arange(SHAPE[0])[:, None, None]
    ci, cj, ck = (x + 1.0) / 2.0, (y - 6.5) / 3.0, (z - 9.0) / 4.0  # continuous dose indices of every CT voxel
    expected = 1e-2 * (100 + 7 * ck + 5 * cj + 3 * ci)
    inside = (ci >= 0) & (ci <= 2) & (cj >= 0) & (cj <= 1) & (ck >= 0) & (ck <= 1)
    outside = (ci < -0.5) | (ci > 2.5) | (cj < -0.5) | (cj > 1.5) | (ck < -0.5) | (ck > 1.5)
    assert inside.sum() == 36 and outside.any()
    np.testing.assert_allclose(got[inside], expected[inside], atol=1e-4)
    assert (got[outside] == 0).all()
    assert row.files == ['planning_ct.nii.gz', 'dose.nii.gz', 'series.json']
    assert sorted(p.name for p in folder.iterdir()) == ['dose.nii.gz', 'planning_ct.nii.gz', 'series.json']  # no temp left


def test_ct_with_different_rescale_slopes_is_not_quantized(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    slopes = [1, 0.5, 1, 0.5]  # the first slice is integral: a Cast after reading would keep the others at integers
    for path, slope in zip(plan['ct']['paths'], slopes, strict=True):
        edit(path, lambda ds, slope=slope: setattr(ds, 'RescaleSlope', slope))
    conv.convert(org, out)
    expected = plan['values'] * np.asarray(slopes)[:, None, None]
    np.testing.assert_allclose(arr(out / PID / 'plan_1' / 'planning_ct.nii.gz'), expected, atol=1e-6)


# ---------------------------------------------------------------- A2.1b: the dose rule

def written_dose(rows, out):
    assert row_of(rows).outcome == conv.OUTCOME_CONVERTED
    return arr(out / PID / 'plan_1' / 'dose.nii.gz')


def test_dose_plan_wins_over_beam(tree):
    _, rows, out = one_plan(tree, doses=[dose_spec(10, 'BEAM'), dose_spec(20, 'PLAN'), dose_spec(30, 'BEAM')])
    assert np.allclose(written_dose(rows, out), 20 * SCALING)


def test_dose_plan_wins_over_fraction(tree):
    _, rows, out = one_plan(tree, doses=[dose_spec(10, 'FRACTION'), dose_spec(20, 'PLAN'), dose_spec(30, 'FRACTION')])
    assert np.allclose(written_dose(rows, out), 20 * SCALING)


def test_dose_one_fraction_is_used(tree):
    _, rows, out = one_plan(tree, doses=[dose_spec(10, 'FRACTION')])
    assert np.allclose(written_dose(rows, out), 10 * SCALING)


def test_dose_two_beams_are_summed(tree):
    _, rows, out = one_plan(tree, doses=[dose_spec(10, 'BEAM'), dose_spec(30, 'BEAM')])
    assert np.allclose(written_dose(rows, out), 40 * SCALING)


def test_dose_none_converts_without_dose_file(tree):
    _, rows, out = one_plan(tree, doses=[])
    assert row_of(rows).outcome == conv.OUTCOME_CONVERTED
    assert not (out / PID / 'plan_1' / 'dose.nii.gz').exists()
    assert row_of(rows).files == ['planning_ct.nii.gz', 'series.json']


@pytest.mark.parametrize('doses, reason', [
    ([dose_spec(10, 'PLAN'), dose_spec(20, 'PLAN')], conv.REASON_DOSE_RULE),
    ([dose_spec(10, 'FRACTION'), dose_spec(20, 'FRACTION')], conv.REASON_DOSE_RULE),
    ([dose_spec(10, 'BRACHY')], conv.REASON_DOSE_RULE),
    ([dose_spec(10, 'BEAM'), dose_spec(20, 'BRACHY')], conv.REASON_DOSE_RULE),
    ([dose_spec(10, 'PLAN', units='RELATIVE')], conv.REASON_DOSE_UNITS),
    ([dose_spec(10, 'PLAN', dose_type='EFFECTIVE')], conv.REASON_DOSE_TYPE),
    ([dose_spec(10, 'PLAN', frame_uid=pydicom.uid.generate_uid())], conv.REASON_DOSE_FRAME),
    ([dose_spec(10, 'PLAN', offsets=[0.0, 2.0, 4.0, 7.0])], conv.REASON_DOSE_GRID),
])
def test_dose_refusals(tree, doses, reason):
    _, rows, out = one_plan(tree, doses=doses)
    row = row_of(rows)
    assert (row.outcome, row.reason, row.files) == (conv.OUTCOME_REFUSED, reason, [])
    assert not (out / PID / 'plan_1').exists()


def test_unused_dose_is_not_checked(tree):
    # the BEAM is not used when a PLAN exists, so its unit is irrelevant
    _, rows, out = one_plan(tree, doses=[dose_spec(10, 'BEAM', units='RELATIVE'), dose_spec(20, 'PLAN')])
    assert np.allclose(written_dose(rows, out), 20 * SCALING)


# ---------------------------------------------------------------- A2.1c: the CT must be one volume

def test_ct_missing_middle_slice_is_refused(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    # the RTSTRUCT is rewritten to list only the slices that remain, so the spacing check is the one that must catch the gap
    kept = [s for k, s in enumerate(plan['ct']['sop_uids']) if k != 1]
    sd.write_rtstruct_rois(org / PID / 'plan_1' / 'struct.dcm', dict(plan['ct'], sop_uids=kept), [],
                           sop_uid=plan['struct'], patient_id=PID, study_uid=plan['ct']['study_uid'])
    Path(plan['ct']['paths'][1]).unlink()
    row = row_of(conv.convert(org, out).rows)
    assert (row.outcome, row.reason) == (conv.OUTCOME_REFUSED, conv.REASON_CT_NOT_VOLUME)
    assert not (out / PID / 'plan_1').exists()


def test_ct_missing_middle_slice_listed_by_rtstruct_is_refused(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    Path(plan['ct']['paths'][2]).unlink()
    row = row_of(conv.convert(org, out).rows)
    assert (row.outcome, row.reason) == (conv.OUTCOME_REFUSED, conv.REASON_CT_NOT_VOLUME)


def test_ct_missing_end_slice_listed_by_rtstruct_is_refused(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    Path(plan['ct']['paths'][-1]).unlink()  # the rest is still one uniform volume: only the RTSTRUCT's list shows the loss
    row = row_of(conv.convert(org, out).rows)
    assert (row.outcome, row.reason) == (conv.OUTCOME_REFUSED, conv.REASON_CT_SLICE_MISSING)


def test_ct_with_one_slice_is_refused(tree):
    _, rows, _ = one_plan(tree, values=np.zeros((1, 5, 6)))
    assert row_of(rows).reason == conv.REASON_CT_NOT_VOLUME


def test_ct_with_two_orientations_is_refused(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    edit(plan['ct']['paths'][1], lambda ds: setattr(ds, 'ImageOrientationPatient', [0, 1, 0, 1, 0, 0]))
    assert row_of(conv.convert(org, out).rows).reason == conv.REASON_CT_NOT_VOLUME


def test_ct_with_a_duplicate_slice_position_is_refused(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    paths = plan['ct']['paths']
    position = pydicom.dcmread(paths[1], stop_before_pixels=True).ImagePositionPatient
    edit(paths[2], lambda ds: setattr(ds, 'ImagePositionPatient', position))
    assert row_of(conv.convert(org, out).rows).reason == conv.REASON_CT_NOT_VOLUME


def test_sheared_ct_is_refused(tree):
    """A gantry tilt shifts each slice sideways: a SimpleITK volume cannot hold that grid, so the plan is refused."""
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    for k, path in enumerate(plan['ct']['paths']):
        edit(path, lambda ds, k=k: setattr(ds, 'ImagePositionPatient',
                                           [float(ds.ImagePositionPatient[0]) + 0.5 * k, *ds.ImagePositionPatient[1:]]))
    assert row_of(conv.convert(org, out).rows).reason == conv.REASON_CT_NOT_VOLUME


NON_FINITE_CT = [(tag, x, scope) for tag in ('PixelSpacing', 'ImagePositionPatient', 'RescaleSlope', 'RescaleIntercept')
                 for x in (float('nan'), float('inf')) for scope in ('all', 'one')]


@pytest.mark.parametrize('tag, x, scope', NON_FINITE_CT, ids=[f'{t}-{x}-{s}' for t, x, s in NON_FINITE_CT])
def test_ct_with_a_non_finite_header_value_is_refused(tree, tag, x, scope):
    """GDCM reads an inf PixelSpacing as 1.0 and a non-finite rescale as 1 or 0: the CT would convert with a wrong grid or HU."""
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    paths = plan['ct']['paths'] if scope == 'all' else plan['ct']['paths'][1:2]
    for path in paths:
        edit(path, lambda ds: setattr(ds, tag, [x, *ds.get(tag)[1:]] if tag in ('PixelSpacing', 'ImagePositionPatient') else x))
    assert not np.isfinite(np.ravel(np.asarray(pydicom.dcmread(paths[0], stop_before_pixels=True).get(tag), dtype=float))).all()
    row = row_of(conv.convert(org, out).rows)
    assert (row.outcome, row.reason) == (conv.OUTCOME_REFUSED, conv.REASON_CT_NOT_VOLUME)
    assert not (out / PID / 'plan_1').exists()


@pytest.mark.parametrize('tag', ['RescaleSlope', 'RescaleIntercept'])
@pytest.mark.parametrize('value', [None, [1, 2]], ids=['empty', 'two-values'])
def test_ct_with_an_empty_or_multi_valued_rescale_is_refused_as_missing_tag(tree, tag, value):
    """Both tags are Type 1, single-valued; GDCM would fall back to slope 1 / intercept 0."""
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    path = plan['ct']['paths'][1]
    edit(path, lambda ds: setattr(ds, tag, value))
    assert len(pydicom.dcmread(path, stop_before_pixels=True)[tag].value or []) != 1
    row = row_of(conv.convert(org, out).rows)
    assert (row.outcome, row.reason) == (conv.OUTCOME_REFUSED, conv.REASON_MISSING_TAG)
    assert not (out / PID / 'plan_1').exists()


def test_rtstruct_without_contour_images_refuses_the_plan(tree):
    _, rows, _ = one_plan(tree, contour_images=False)
    assert row_of(rows).reason == conv.REASON_NO_CONTOUR_IMAGES


def test_unreadable_file_in_plan_folder_refuses_the_plan(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    (org / PID / 'plan_1' / 'junk.dcm').write_bytes(b'not a dicom file')
    row = row_of(conv.convert(org, out).rows)
    assert (row.outcome, row.reason) == (conv.OUTCOME_REFUSED, conv.REASON_FILE_UNREADABLE)


def test_ct_with_unreadable_pixel_data_is_refused(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    path = Path(plan['ct']['paths'][1])
    path.write_bytes(path.read_bytes()[:-40])  # header intact, pixel data truncated
    row = row_of(conv.convert(org, out).rows)
    assert (row.outcome, row.reason) == (conv.OUTCOME_REFUSED, conv.REASON_CT_UNREADABLE)


# ---------------------------------------------------------------- assemble

def test_two_ct_series_the_rtstruct_reference_is_planning_ct(tree):
    org, out = tree
    folder = org / PID / 'plan_1'
    plan = make_plan(folder)
    other = np.full(SHAPE, 7)
    sd.write_ct_series(folder, other, spacing=SPACING, origin=ORIGIN, patient_id=PID, prefix='other')
    row = row_of(conv.convert(org, out).rows)
    assert row.outcome == conv.OUTCOME_CONVERTED
    np.testing.assert_array_equal(arr(out / PID / 'plan_1' / 'planning_ct.nii.gz'), plan['values'].astype(np.float32))
    assert sorted(p.name for p in (out / PID).iterdir()) == ['plan_1', 'study_1']  # the other CT goes to a study folder (A2.4b)


def test_a2_8_missing_rtstruct_refuses_and_leaves_nothing(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    (org / PID / 'plan_1' / 'struct.dcm').unlink()
    rows = conv.convert(org, out).rows
    row = row_of(rows)
    assert (row.outcome, row.reason) == (conv.OUTCOME_REFUSED, conv.REASON_NO_STRUCT)
    assert not (out / PID / 'plan_1').exists()
    assert listing(out) == [PID, conv.REPORT_NAME]  # no partial, no temp
    assert [p.name for p in (out / PID).iterdir()] == ['study_1']  # the refused plan's CT
    with open(out / conv.REPORT_NAME, newline='') as handle:
        assert next(r for r in csv.DictReader(handle) if r['folder'] == 'plan_1')['reason'] == conv.REASON_NO_STRUCT


def test_two_rtplans_in_a_plan_folder_refuse(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    sd.write_rtplan(org / PID / 'plan_1' / 'plan_b.dcm', plan['struct'], np.zeros((2, 3)), patient_id=PID,
                    study_uid=plan['ct']['study_uid'])
    assert row_of(conv.convert(org, out).rows).reason == conv.REASON_PLAN_COUNT


def test_rtstruct_referencing_two_ct_series_in_the_folder_refuses(tree):
    org, out = tree
    folder = org / PID / 'plan_1'
    plan = make_plan(folder)
    other = sd.write_ct_series(folder, np.zeros(SHAPE), spacing=SPACING, origin=ORIGIN, patient_id=PID, prefix='other')
    sd.write_rtstruct_rois(folder / 'struct.dcm', plan['ct'], [], sop_uid=plan['struct'], patient_id=PID,
                           study_uid=plan['ct']['study_uid'])

    def reference_other(ds):
        series = pydicom.Dataset()
        series.SeriesInstanceUID = other['series_uid']
        study = ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
        study.RTReferencedSeriesSequence.append(series)
    edit(folder / 'struct.dcm', reference_other)
    assert row_of(conv.convert(org, out).rows).reason == conv.REASON_CT_SERIES


@pytest.mark.parametrize('what', ['ct_position', 'plan_struct_reference'])
def test_header_missing_a_tag_the_path_reads_refuses_that_plan_and_the_run_continues(tree, what):
    org, out = tree
    broken = make_plan(org / PID / 'plan_1')
    make_plan(org / PID / 'plan_2')
    if what == 'ct_position':
        edit(broken['ct']['paths'][1], lambda ds: delattr(ds, 'ImagePositionPatient'))
    else:
        edit(broken['folder'] / 'plan.dcm', lambda ds: delattr(ds, 'ReferencedStructureSetSequence'))
    rows = conv.convert(org, out).rows
    assert (row_of(rows, 'plan_1').outcome, row_of(rows, 'plan_1').reason) == (conv.OUTCOME_REFUSED, conv.REASON_MISSING_TAG)
    assert row_of(rows, 'plan_2').outcome == conv.OUTCOME_CONVERTED
    assert sorted(p.name for p in (out / PID).iterdir()) == (['plan_2', 'study_1'] if what == 'plan_struct_reference' else ['plan_2'])


# ---------------------------------------------------------------- layout, atomicity, report

def test_refused_plan_leaves_no_folder_while_the_other_plans_are_written(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    make_plan(org / PID / 'plan_2', doses=[dose_spec(1, 'PLAN'), dose_spec(2, 'PLAN')])
    make_plan(org / PID / 'plan_3')
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome) for r in rows] == [('plan_1', 'converted'), ('plan_2', 'refused'), ('plan_3', 'converted'),
                                                      ('studies', 'converted'), ('study_1/series_CT_1', 'converted')]  # the refused plan's CT
    assert sorted(p.name for p in (out / PID).iterdir()) == ['plan_1', 'plan_3', 'study_1']
    assert [p.name for p in out.iterdir() if p.name.startswith('.')] == []


def test_plan_folders_are_read_in_numeric_order(tree):
    org, out = tree
    for n in (10, 2):
        make_plan(org / PID / f'plan_{n}')
    assert [r.folder for r in conv.convert(org, out).rows] == ['plan_2', 'plan_10']


def test_patient_whose_plans_are_all_refused_is_renamed_into_place(tree):
    org, out = tree
    for n in (1, 2):
        make_plan(org / PID / f'plan_{n}', doses=[dose_spec(1, 'BRACHY')])
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome) for r in rows] == [
        ('plan_1', 'refused'), ('plan_2', 'refused'), ('studies', 'converted'), ('study_1/series_CT_1', 'converted'), ('study_2/series_CT_1', 'converted')]
    assert (out / PID).is_dir() and sorted(p.name for p in (out / PID).iterdir()) == ['study_1', 'study_2']  # only the refused plans' CTs


def test_patient_without_plan_folders_is_skipped_with_no_output_folder(tree):
    org, out = tree
    misc = org / PID / 'misc_files'
    misc.mkdir(parents=True)
    sd.write_ct_series(misc, np.zeros(SHAPE), patient_id=PID, image_type=['ORIGINAL', 'PRIMARY', 'LOCALIZER'])  # not a candidate
    rows = conv.convert(org, out).rows
    assert [(r.patient, r.folder, r.outcome, r.reason) for r in rows] == [(PID, '', 'skipped', conv.REASON_NO_PLANS)]
    assert not (out / PID).exists()


def test_dot_folders_and_misc_files_are_never_converted(tree):
    org, out = tree
    make_plan(org / '.SYNTHETICPTB.partial' / 'plan_1')
    make_plan(org / PID / 'plan_1')
    (org / PID / 'misc_files').mkdir()
    rows = conv.convert(org, out).rows
    assert [(r.patient, r.folder) for r in rows] == [(PID, 'plan_1')]
    assert listing(out) == [PID, conv.REPORT_NAME]


def test_stale_partial_is_removed_first(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    stale = out / f'.{PID}.partial' / 'plan_9'
    stale.mkdir(parents=True)
    (stale / 'leftover').write_text('x')
    conv.convert(org, out)
    assert listing(out) == [PID, conv.REPORT_NAME]
    assert sorted(p.name for p in (out / PID).iterdir()) == ['plan_1']


def test_existing_patient_output_is_skipped_on_rerun(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    conv.convert(org, out)
    rows = conv.convert(org, out).rows
    assert [(r.patient, r.folder, r.outcome, r.reason) for r in rows] == [(PID, '', 'skipped', conv.REASON_EXISTS)]


def test_oserror_while_writing_gives_a_patient_row_and_removes_the_partial(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    make_plan(org / PID / 'plan_2')

    def fail(*args, **kwargs):
        raise OSError(f'disk full writing under {out} for {PID}')
    monkeypatch.setattr(conv.sitk, 'WriteImage', fail)
    rows = conv.convert(org, out).rows
    assert [(r.patient, r.folder, r.outcome, r.reason, r.files) for r in rows] == [(PID, '', 'refused', conv.REASON_IO_ERROR, [])]
    assert listing(out) == [conv.REPORT_NAME]


def test_report_columns_and_joined_files(tree):
    _, rows, out = one_plan(tree, doses=[dose_spec(10, 'PLAN')])
    with open(out / conv.REPORT_NAME, newline='') as handle:
        reader = csv.DictReader(handle)
        written = list(reader)
        assert reader.fieldnames == ['patient', 'folder', 'outcome', 'reason', 'files', 'skipped_series']
    assert written == [dict(patient=PID, folder='plan_1', outcome='converted', reason='',
                            files='planning_ct.nii.gz;dose.nii.gz;series.json', skipped_series='0')]


@pytest.mark.parametrize('where', ['same', 'inside'])
def test_out_root_equal_to_or_inside_org_root_raises_before_any_mkdir(tree, where):
    org, _ = tree
    make_plan(org / PID / 'plan_1')
    out = org if where == 'same' else org / 'deeper' / 'out'
    with pytest.raises(ValueError) as caught:
        conv.convert(org, out)
    assert str(caught.value) == conv.REASON_OUTSIDE_ORG
    assert not (org / 'deeper').exists() and not (org / conv.REPORT_NAME).exists()


def test_the_organized_tree_is_only_read(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')])
    before = {p: p.read_bytes() for p in org.rglob('*') if p.is_file()}
    conv.convert(org, out)
    assert {p: p.read_bytes() for p in org.rglob('*') if p.is_file()} == before


# ---------------------------------------------------------------- the per-plan seam

def test_convert_plan_writes_into_case_dir_and_returns_the_files(tree):
    org, _ = tree
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')])
    case = org.parent / 'case'
    result = run_plan(org / PID / 'plan_1', case)
    assert result.files == ['planning_ct.nii.gz', 'dose.nii.gz']  # series.json is the report's to add
    assert sorted(p.name for p in case.iterdir()) == ['dose.nii.gz', 'planning_ct.nii.gz']  # series.json is the patient level's, last
    assert [r['file'] for r in result.series_rows] == result.files  # one series.json row per data file, in write order
    assert result.series_rows[0]['kind'] == 'ct' and result.series_rows[1]['kind'] == 'dose'


def test_convert_plan_with_an_unreadable_flag_refuses_the_plan_and_leaves_no_case_dir(tree):
    org, _ = tree
    make_plan(org / PID / 'plan_1')
    headers, _ = conv.read_folder_headers(org / PID / 'plan_1')
    case = org.parent / 'case'
    with pytest.raises(CaseRefused) as caught:
        conv.convert_plan(headers, True, case)
    assert str(caught.value) == conv.REASON_FILE_UNREADABLE and not case.exists()


def test_an_unreadable_file_in_misc_files_is_ignored_and_only_its_count_is_logged(tree, caplog):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    misc = org / PID / 'misc_files'
    misc.mkdir()
    (misc / 'SYNTHETIC_junk.dcm').write_bytes(b'not a dicom file')
    caplog.set_level(logging.DEBUG)
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.files, r.skipped_series) for r in rows] == [
        ('plan_1', conv.OUTCOME_CONVERTED, ['planning_ct.nii.gz', 'series.json'], 0)]
    assert 'convert: 1 malformed or vanished files in misc_files' in caplog.messages
    assert all(secret not in text for text in caplog.messages for secret in (PID, 'SYNTHETIC_junk', str(org)))


def test_convert_plan_refusal_raises_the_reason_and_leaves_no_case_dir(tree):
    org, _ = tree
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'BRACHY')])
    case = org.parent / 'case'
    with pytest.raises(CaseRefused) as caught:
        run_plan(org / PID / 'plan_1', case)
    assert str(caught.value) == conv.REASON_DOSE_RULE
    assert not case.exists()


# ---------------------------------------------------------------- series.json (A2.4d)

SERIES_COLUMNS = ['file', 'kind', 'modality', 'series_description', 'series_instance_uid', 'study_instance_uid',
                  'frame_of_reference_uid', 'series_date', 'roi_name', 'dose_summation_type', 'reg_matrix', 'rtplan_uid',
                  'rtplan_label', 'rtplan_date', 'fractions_planned']
CT_DATE, STRUCT_DATE, DOSE_DATE, PLAN_DATE = '20240105', '20240106', '20240107', '20240301'
BOX = dict(number=1, name='Liver', kind='box', box=((0, 2), (0, 2), (0, 2)))
RING = dict(number=2, name='ROI A', kind='channel', points=np.arange(15.0).reshape(5, 3), interpreted='BRACHY_CHANNEL')
RING_CHANNEL = [dict(applicator_id='Ring-1', dwells_mm=np.zeros((2, 3)), roi=2)]


def series_rows(out, folder='plan_1'):
    rows = sf.read(out / PID / folder)
    assert all(list(row) == SERIES_COLUMNS for row in rows)  # the keys and their order are what series.csv's header was
    return rows


def describe(path, text, date):
    def change(ds):
        ds.SeriesDescription, ds.SeriesDate = text, date
    edit(path, change)


def test_series_json_has_one_row_per_written_file_with_the_values_of_each_kind(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN', series_date=DOSE_DATE)], rois=[BOX, RING],
                     ct_meta=dict(series_date=CT_DATE),
                     plan_meta=dict(label='Plan A', plan_date=PLAN_DATE, fractions=5, channels=RING_CHANNEL))
    folder = plan['folder']
    for path in plan['ct']['paths']:
        describe(path, 'Planning CT', CT_DATE)
    describe(folder / 'struct.dcm', 'Structures', STRUCT_DATE)
    describe(folder / 'dose_0.dcm', 'Dose', DOSE_DATE)
    struct, dose = pydicom.dcmread(folder / 'struct.dcm'), pydicom.dcmread(folder / 'dose_0.dcm')
    row = row_of(conv.convert(org, out).rows)
    assert row.files == ['planning_ct.nii.gz', 'dose.nii.gz', 'liver.nii.gz', 'Ring1.nii.gz', 'series.json']
    rows = series_rows(out)
    assert [r['file'] for r in rows] == row.files[:-1]  # no row for series.json itself
    ct_row, dose_row, mask_row, applicator_row = rows
    plan_columns = dict(rtplan_uid=plan['plan'], rtplan_label='Plan A', rtplan_date=PLAN_DATE, fractions_planned='5')
    assert ct_row == dict(file='planning_ct.nii.gz', kind='ct', modality='CT', series_description='Planning CT',
                          series_instance_uid=plan['ct']['series_uid'], study_instance_uid=plan['ct']['study_uid'],
                          frame_of_reference_uid=plan['ct']['frame_uid'], series_date=CT_DATE, roi_name='',
                          dose_summation_type='', reg_matrix='identity', **plan_columns)
    assert dose_row == dict(file='dose.nii.gz', kind='dose', modality='RTDOSE', series_description='Dose',
                            series_instance_uid=str(dose.SeriesInstanceUID), study_instance_uid=str(dose.StudyInstanceUID),
                            frame_of_reference_uid=plan['ct']['frame_uid'], series_date=DOSE_DATE, roi_name='',
                            dose_summation_type='PLAN', reg_matrix='', **plan_columns)
    for mask_kind, got, name, file in (('mask', mask_row, 'Liver', 'liver.nii.gz'), ('applicator', applicator_row, 'ROI A', 'Ring1.nii.gz')):
        assert got == dict(file=file, kind=mask_kind, modality='RTSTRUCT', series_description='Structures',
                           series_instance_uid=str(struct.SeriesInstanceUID), study_instance_uid=str(struct.StudyInstanceUID),
                           frame_of_reference_uid=plan['ct']['frame_uid'], series_date=STRUCT_DATE, roi_name=name,
                           dose_summation_type='', reg_matrix='', **plan_columns)


def test_series_json_of_a_summed_beam_dose_joins_the_series_uids(tree):
    org, out = tree
    uids = [pydicom.uid.generate_uid(), pydicom.uid.generate_uid()]
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'BEAM', series_uid=uid) for uid in uids])
    conv.convert(org, out)
    dose_row = series_rows(out)[1]
    assert (dose_row['file'], dose_row['modality'], dose_row['dose_summation_type']) == ('dose.nii.gz', 'RTDOSE', 'BEAM')
    assert dose_row['series_instance_uid'] == ';'.join(uids)


def test_series_json_leaves_absent_plan_tags_empty(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1', plan_meta=dict(fractions=None))
    conv.convert(org, out)
    [ct_row] = series_rows(out)
    assert {k: ct_row[k] for k in ('rtplan_label', 'rtplan_date', 'fractions_planned', 'series_description', 'series_date')} == {
        'rtplan_label': '', 'rtplan_date': '', 'fractions_planned': '', 'series_description': '', 'series_date': ''}


def test_series_json_is_the_last_file_and_goes_through_a_temp_name(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1', rois=[BOX])
    replaced = []
    real = conv.os.replace

    def spy(source, target):
        replaced.append((Path(source).name, Path(target).name, Path(target).parent.name))
        real(source, target)
    monkeypatch.setattr(conv.os, 'replace', spy)
    conv.convert(org, out)
    source, target, _ = [r for r in replaced if r[2] == 'plan_1'][-1]  # the run report is replaced after it, in out/
    assert target == 'series.json' and source != target and source.endswith('series.json')
    assert sorted(p.name for p in (out / PID / 'plan_1').iterdir()) == ['liver.nii.gz', 'planning_ct.nii.gz', 'series.json']


def test_a_failed_series_json_write_is_a_patient_level_io_error_and_leaves_nothing(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    real = conv.os.replace

    def fail(source, target):
        if Path(target).name == 'series.json':
            raise OSError(f'cannot write {target} for {PID}')
        real(source, target)
    monkeypatch.setattr(conv.os, 'replace', fail)
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'refused', conv.REASON_IO_ERROR)]
    assert listing(out) == [conv.REPORT_NAME]


# ---------------------------------------------------------------- A2.9: organize -> convert

def test_a2_9_an_rtstruct_in_another_study_is_organized_and_converts(tmp_path):
    src, org, out = tmp_path / 'src', tmp_path / 'org', tmp_path / 'out'
    folder = src / PID / 'p'
    folder.mkdir(parents=True)
    ct_time = '100000'
    ct = sd.write_ct_series(folder, np.zeros((4, 5, 6), int), spacing=SPACING, origin=ORIGIN, patient_id=PID,
                            series_date=CT_DATE, series_time=ct_time)
    other_study = pydicom.uid.generate_uid()  # the RTSTRUCT was exported in a study of its own, not the CT's
    struct = sd.write_rtstruct_rois(folder / 'struct.dcm', ct, [BOX], contour_images=True, patient_id=PID, study_uid=other_study)
    plan_time = '110000'
    sd.write_rtplan(folder / 'plan.dcm', struct, np.zeros((2, 3)), plan_date=CT_DATE, plan_time=plan_time,
                    patient_id=PID, study_uid=ct['study_uid'])
    organize(src, org)
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.files) for r in rows] == [
        ('plan_1', 'converted', ['planning_ct.nii.gz', 'liver.nii.gz', 'series.json'])]
    assert arr(out / PID / 'plan_1' / 'liver.nii.gz').sum() == 27


def test_organize_then_convert_is_the_same_for_one_worker_and_four(tmp_path):
    src = tmp_path / 'src'
    for patient in (PID, 'SYNTHETICPTB'):
        make_plan(src / patient / 'p', doses=[dose_spec(10, 'PLAN')], rois=[BOX], patient=patient)
    runs = []
    for workers in (1, 4):
        org, out = tmp_path / f'org{workers}', tmp_path / f'out{workers}'
        organized = organize(src, org, workers=workers).rows
        converted = conv.convert(org, out, workers=workers).rows
        runs.append((organized, converted, tree_hashes(org), tree_hashes(out), org, out))
    (org_rows, conv_rows, org_tree, out_tree, *_), (org_rows4, conv_rows4, org_tree4, out_tree4, *_) = runs

    def stable(hashes):  # the dated report copies carry the run's UTC time in their names
        return {k: v for k, v in hashes.items() if not re.fullmatch(r'\w+_report_\d{8}T\d{6}Z\.csv', k)}

    assert len(org_rows) == 2 and [r.outcome for r in conv_rows] == ['converted'] * 2  # the comparison has something to compare
    assert sum(k.endswith('.nii.gz') for k in out_tree) == 6 and any(k.endswith('.dcm') for k in org_tree)
    assert stable(org_tree) == stable(org_tree4) and stable(out_tree) == stable(out_tree4)
    assert org_rows == org_rows4 and conv_rows == conv_rows4
    assert 'organize_report.csv' in stable(org_tree4) and 'convert_report.csv' in stable(out_tree4)


# ---------------------------------------------------------------- A2.10: PHI

def chain(exc):
    """`exc` and everything on its __cause__/__context__ links, recursively."""
    seen, todo = [], [exc]
    while todo:
        item = todo.pop()
        if item is not None and all(item is not other for other in seen):
            seen.append(item)
            todo += [operator.attrgetter('__cause__')(item), operator.attrgetter('__context__')(item)]
    return seen


def every_uid(folder):
    found = set()
    for path in folder.glob('*.dcm'):
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        for dataset in (ds, ds.file_meta):
            found |= {str(e.value) for e in dataset.iterall() if e.VR == 'UI'}
    return found


def truncate_pixels(plan):
    path = Path(plan['ct']['paths'][1])
    path.write_bytes(path.read_bytes()[:-40])  # header intact, pixel data truncated


def break_rtstruct(plan):
    def change(ds):  # DicomRTTool reads [0] of this sequence, convert never does
        ds.RTROIObservationsSequence[0].RTROIIdentificationCodeSequence = []
    edit(plan['folder'] / 'struct.dcm', change)


@pytest.mark.parametrize('damage, reason', [(truncate_pixels, conv.REASON_CT_UNREADABLE),
                                            (break_rtstruct, masks.REASON_MASK_LINK)])
def test_a2_10_no_output_channel_or_exception_carries_the_patient_id_the_path_or_a_uid(tree, tmp_path, capfd, caplog, recwarn,
                                                                                         damage, reason):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', rois=[BOX])
    secrets = [PID, str(tmp_path), *every_uid(plan['folder'])]
    assert len(secrets) > 10
    damage(plan)
    caplog.set_level(logging.DEBUG)
    capfd.readouterr()
    rows = conv.convert(org, out).rows
    with pytest.raises(CaseRefused) as caught:
        run_plan(plan['folder'], tmp_path / 'case')
    assert [(r.outcome, r.reason) for r in rows if r.folder == 'plan_1'] == [(conv.OUTCOME_REFUSED, reason)] and str(caught.value) == reason
    assert chain(caught.value) == [caught.value]
    captured = capfd.readouterr()
    texts = [captured.out, captured.err, caplog.text, *(r.getMessage() for r in caplog.records),
             *(str(w.message) for w in recwarn.list), *(f'{e!r} {e}' for e in chain(caught.value))]
    assert [s for s in secrets for t in texts if s in t] == []


def test_a2_10_the_broken_rtstruct_really_makes_dicomrttool_log_the_path(tree, caplog):
    """Guards the guard: unwrapped, DicomRTTool logs the path of the broken RTSTRUCT."""
    from DicomRTTool import DicomReaderWriter
    org, _ = tree
    plan = make_plan(org / PID / 'plan_1', rois=[BOX])
    break_rtstruct(plan)
    caplog.set_level(logging.WARNING)
    DicomReaderWriter(arg_max=False, verbose=False).walk_through_folders(str(plan['folder']))
    assert str(plan['folder']) in caplog.text


def test_a2_10_a_slice_lost_after_the_header_read_refuses_without_the_path_ITK_names(tree, tmp_path, capfd, caplog, recwarn,
                                                                                         monkeypatch):
    """ITK's own read error says 'Could not open file: <path>'; none of that may reach a row, a log record or fd 1/2."""
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', rois=[BOX])
    lost = Path(plan['ct']['paths'][1])
    secrets = [PID, str(tmp_path), lost.name, *every_uid(plan['folder'])]
    assert len(secrets) > 10
    real = conv.read_folder_headers

    def read_then_lose_a_slice(folder, *args, **kwargs):
        found = real(folder, *args, **kwargs)
        if Path(folder).name == 'plan_1':
            lost.unlink()  # setup only: a slice lost between the header read and ITK's read; the code under test deletes nothing
        return found
    monkeypatch.setattr(conv, 'read_folder_headers', read_then_lose_a_slice)
    caplog.set_level(logging.DEBUG)
    capfd.readouterr()
    rows = conv.convert(org, out).rows
    assert not lost.exists()
    assert [(r.outcome, r.reason) for r in rows if r.folder == 'plan_1'] == [(conv.OUTCOME_REFUSED, conv.REASON_CT_UNREADABLE)]
    captured = capfd.readouterr()
    texts = [captured.out, captured.err, caplog.text, *(r.getMessage() for r in caplog.records),
             *(str(w.message) for w in recwarn.list),
             *(repr((r.folder, r.outcome, r.reason, r.files)) for r in rows)]  # the patient column is the folder name by design
    assert [s for s in secrets for t in texts if s in t] == []


# ---------------------------------------------------------------- A2.11: rerun safety

def tree_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}


def test_a2_11_the_organized_tree_keeps_its_hashes_and_a_rerun_skips_every_patient(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')], rois=[BOX])
    make_plan(org / 'SYNTHETICPTB' / 'plan_1', contour_images=False, patient='SYNTHETICPTB')  # refused: still has its outcome
    before = tree_hashes(org)
    first = conv.convert(org, out).rows
    assert sorted((r.patient, r.outcome) for r in first if r.folder == 'plan_1') == [(PID, 'converted'), ('SYNTHETICPTB', 'refused')]
    assert tree_hashes(org) == before
    after_first = tree_hashes(out)
    rows = conv.convert(org, out).rows
    assert sorted((r.patient, r.folder, r.outcome, r.reason) for r in rows) == [
        (PID, '', 'skipped', conv.REASON_EXISTS), ('SYNTHETICPTB', '', 'skipped', conv.REASON_EXISTS)]
    assert tree_hashes(org) == before
    assert {k: v for k, v in tree_hashes(out).items() if not str(k).startswith('convert_report')} == {
        k: v for k, v in after_first.items() if not str(k).startswith('convert_report')}


# ---------------------------------------------------------------- --overwrite

PAST = 1_000_000_000  # an mtime long before any run of these tests


def stale_output(tree):
    """One converted patient, then a marker added and every file aged: the state a rerun after a code fix finds."""
    org, out = tree
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')], rois=[BOX])
    conv.convert(org, out)
    (out / PID / 'marker.txt').write_text('stale')
    for p in (out / PID).rglob('*'):
        os.utime(p, (PAST, PAST))
    return org, out


def no_hidden(out):
    return [n for n in listing(out) if n.startswith('.')]


def test_overwrite_without_the_flag_skips_and_treats_names_differing_in_case_as_today(tree):
    org, out = stale_output(tree)
    before = tree_hashes(out / PID)
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'skipped', conv.REASON_EXISTS)]
    assert tree_hashes(out / PID) == before
    other = tree[0].parent / 'org2'
    for name in (PID, PID.lower()):
        make_plan(other / name / 'plan_1', patient=name)
    rows = conv.convert(other, tree[0].parent / 'out2').rows
    assert [(r.patient, r.outcome) for r in rows if r.folder == 'plan_1'] == [(PID, 'converted'), (PID.lower(), 'converted')]


def test_without_overwrite_a_dangling_link_at_the_output_name_is_an_existing_output(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    out.mkdir()
    (out / PID).symlink_to(out / 'SYNTHETIC_missing', target_is_directory=True)
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'skipped', conv.REASON_EXISTS)]
    assert (out / PID).is_symlink() and no_hidden(out) == []


def test_overwrite_replaces_a_stale_folder_with_a_fresh_run(tree, caplog):
    org, out = stale_output(tree)
    started = time.time()
    caplog.set_level(logging.INFO)
    rows = conv.convert(org, out, overwrite=True).rows
    assert row_of(rows).outcome == 'converted'
    fresh = tree[0].parent / 'fresh'
    conv.convert(org, fresh)
    assert sorted(tree_hashes(out / PID)) == sorted(tree_hashes(fresh / PID))
    assert not (out / PID / 'marker.txt').exists()
    written = [p for p in (out / PID).rglob('*') if p.is_file()]
    assert written and all(p.stat().st_mtime >= started - 1 for p in written)  # nothing of the old folder was merged in
    assert no_hidden(out) == []
    assert any(m.endswith('; 1 replaced, 0 kept') for m in caplog.messages)


def test_overwrite_with_a_failed_build_keeps_the_old_folder(tree, monkeypatch, caplog):
    org, out = stale_output(tree)
    before = tree_hashes(out / PID)

    def fail(*args, **kwargs):
        raise OSError(f'disk full writing under {out} for {PID}')
    monkeypatch.setattr(conv.sitk, 'WriteImage', fail)
    caplog.set_level(logging.INFO)
    rows = conv.convert(org, out, overwrite=True).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'refused', conv.REASON_IO_ERROR)]
    assert tree_hashes(out / PID) == before
    assert no_hidden(out) == []
    assert any(m.endswith('; 0 replaced, 0 kept') for m in caplog.messages)
    assert caplog.records and [r.getMessage() for r in caplog.records if PID in r.getMessage() or str(out) in r.getMessage()] == []


def test_overwrite_with_a_failed_partial_rename_restores_the_old_folder(tree, monkeypatch, caplog):
    org, out = stale_output(tree)
    before = tree_hashes(out / PID)
    real = dicom_io.rename_dir

    def rename(src, dst):
        if Path(src).name == f'.{PID}.partial':
            raise OSError(f'cannot move {src} to {dst} for {PID}')
        real(src, dst)
    monkeypatch.setattr(dicom_io, 'rename_dir', rename)
    caplog.set_level(logging.DEBUG)
    rows = conv.convert(org, out, overwrite=True).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'refused', conv.REASON_IO_ERROR)]
    assert tree_hashes(out / PID) == before
    assert no_hidden(out) == []
    assert [m for m in caplog.messages if PID in m or str(out.parent) in m] == []


def test_recovery_restores_or_deletes_the_old_folder_with_or_without_overwrite(tree):
    org, out = stale_output(tree)
    before = tree_hashes(out / PID)
    (out / PID).rename(out / f'.{PID}.old')
    (out / f'.{PID}.partial').mkdir()
    rows = conv.convert(org, out).rows
    assert [(r.outcome, r.reason) for r in rows] == [('skipped', conv.REASON_EXISTS)]
    assert tree_hashes(out / PID) == before and no_hidden(out) == []
    (out / f'.{PID}.old').mkdir()
    (out / f'.{PID}.old' / 'x').write_text('x')
    conv.convert(org, out)
    assert tree_hashes(out / PID) == before and no_hidden(out) == []


def unreadable(monkeypatch):
    def fail(path):
        raise OSError('unreadable')
    monkeypatch.setattr(dicom_io, 'read_header_core', fail)


def test_overwrite_keeps_the_old_folder_when_every_plan_is_refused(tree, monkeypatch, caplog):
    org, out = stale_output(tree)
    before = tree_hashes(out / PID)
    unreadable(monkeypatch)
    caplog.set_level(logging.INFO)
    rows = conv.convert(org, out, overwrite=True).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('plan_1', 'refused', conv.REASON_FILE_UNREADABLE)]
    assert tree_hashes(out / PID) == before and no_hidden(out) == []
    assert any(m.endswith('; 0 replaced, 1 kept') for m in caplog.messages)


def test_overwrite_keeps_the_old_folder_of_a_patient_without_plan_folders(tree, caplog):
    org, out = stale_output(tree)
    before = tree_hashes(out / PID)
    shutil.rmtree(org / PID)
    (org / PID).mkdir()
    caplog.set_level(logging.INFO)
    rows = conv.convert(org, out, overwrite=True).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'skipped', conv.REASON_NO_PLANS)]
    assert tree_hashes(out / PID) == before and no_hidden(out) == []
    assert any(m.endswith('; 0 replaced, 1 kept') for m in caplog.messages)


def test_overwrite_refuses_a_second_folder_whose_name_differs_only_in_case(tree, tmp_path):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    make_plan(org / PID.lower() / 'plan_1', patient=PID.lower())
    conv.convert(org, tmp_path / 'ref', overwrite=True)  # the same source into an empty root: what the first folder's output must equal
    canary = out / f'.{PID.lower()}.old'  # named for the refused (second) folder: a recover run for it would consume it
    canary.mkdir(parents=True)
    (canary / 'keep.txt').write_text('keep')
    rows = conv.convert(org, out, overwrite=True).rows
    first, second = sorted(rows, key=lambda r: r.patient)
    assert (first.patient, first.outcome) == (PID, 'converted')
    assert (second.patient, second.folder, second.outcome, second.reason) == (PID.lower(), '', 'refused', conv.REASON_NAME_CLAIMED)
    assert (out / PID / 'plan_1').is_dir() and tree_hashes(out / PID) == tree_hashes(tmp_path / 'ref' / PID)
    assert no_hidden(out) == [canary.name] and (canary / 'keep.txt').read_text() == 'keep' and not (out / PID.lower()).exists()


def test_overwrite_leaves_an_unrelated_output_folder_alone(tree):
    org, out = stale_output(tree)
    (out / 'SYNTHETICOTHER').mkdir()
    (out / 'SYNTHETICOTHER' / 'keep.txt').write_text('keep')
    (out / '.SYNTHETICOTHER.old').mkdir()  # a hidden sibling: no recover or publish for PID may touch it
    (out / '.SYNTHETICOTHER.old' / 'keep.txt').write_text('keep')
    conv.convert(org, out, overwrite=True)
    assert (out / '.SYNTHETICOTHER.old' / 'keep.txt').read_text() == 'keep'
    assert (out / 'SYNTHETICOTHER' / 'keep.txt').read_text() == 'keep'


# ---------------------------------------------------------------- guardrails on the source

def test_no_case_refused_is_raised_inside_an_except_block():
    tree = ast.parse(inspect.getsource(conv))
    inside = [node for handler in ast.walk(tree) if isinstance(handler, ast.ExceptHandler)
              for node in ast.walk(handler) if isinstance(node, ast.Raise)]
    assert inside == []


def test_no_array_view_of_a_temporary_image():
    assert 'GetArrayViewFromImage' not in inspect.getsource(conv)


def test_a_failed_rename_of_an_output_file_names_no_path(tmp_path, monkeypatch):
    def refuse_rename(source, target):
        raise OSError(f'cannot rename {target}')
    monkeypatch.setattr(conv.os, 'replace', refuse_rename)
    with pytest.raises(OSError) as err:
        conv.write_series_json([], tmp_path / 'series.json')
    assert str(err.value) == conv.REASON_WRITE
    assert operator.attrgetter('__cause__', '__context__')(err.value) == (None, None)
    assert list(tmp_path.rglob('*')) == []


def test_crash_site_is_the_caller_of_quiet_call(monkeypatch):
    from rt_dataprep import preprocess as pre

    def boom(image, spacing):
        raise RuntimeError('SYNTHETICPTA')
    monkeypatch.setattr(pre, 'resample_to', boom)
    with pytest.raises(CaseRefused) as err:
        pre.process_ct(sitk.Image(2, 2, 2, sitk.sitkFloat32), pre.PreprocessSteps(resample=True, rescale=True, harmonize_spacing=True, reshape=True))
    module, line = conv.crash_site(err.tb)
    assert module == 'rt_dataprep.preprocess'
    assert 'quiet_call(run' in inspect.getsource(pre).splitlines()[line - 1]


def test_series_json_is_utf8_whatever_the_locale(tmp_path):
    script = ("import sys; from rt_dataprep import convert as conv; "
              "row = dict.fromkeys(conv.SERIES_COLUMNS, '') | dict(file='SYNTHETIC.nii.gz', kind='ct', series_description='Thorax \\u00e9'); "
              "conv.write_series_json([row], sys.argv[1])")
    env = {**os.environ, 'LC_ALL': 'C', 'PYTHONCOERCECLOCALE': '0', 'PYTHONUTF8': '0'}  # an ASCII locale
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path / 'series.json')], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'Thorax \u00e9' in (tmp_path / 'series.json').read_bytes().decode('utf-8')


def test_a_missing_organized_root_writes_no_output_root(tmp_path):
    with pytest.raises(OSError):
        conv.convert(tmp_path / 'missing', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


# ---------------------------------------------------------------- hardening B6 (#57)

def test_a_refused_plan_whose_folder_cannot_be_removed_is_the_patients_io_error_and_leaves_no_patient_folder(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    real = conv.shutil.rmtree

    def rmtree(path, ignore_errors=False, **kwargs):
        if Path(path).name == 'plan_1':  # the plan folder only: the patient's partial folder is removed for real
            if ignore_errors:  # as the real one: errors are swallowed and the folder stays
                return
            raise OSError(f'cannot remove {path} of {PID}')
        real(path, ignore_errors=ignore_errors, **kwargs)

    def refuse_after_the_plan_folder_exists(*args):
        raise ValueError  # _series_rows runs after the plan folder is written: a refusal (REASON_MISSING_TAG) from here
    monkeypatch.setattr(conv.shutil, 'rmtree', rmtree)
    monkeypatch.setattr(conv, '_series_rows', refuse_after_the_plan_folder_exists)
    rows = conv.convert(org, out).rows
    assert [(r.patient, r.folder, r.outcome, r.reason, r.files) for r in rows] == [(PID, '', 'refused', conv.REASON_IO_ERROR, [])]
    assert listing(out) == [conv.REPORT_NAME]  # no patient folder, no partial


def test_another_series_on_an_empty_frame_is_not_registered_to_a_planning_ct_on_an_empty_frame(tree):
    org, out = tree
    folder = org / PID / 'plan_1'
    plan = make_plan(folder)
    other = sd.write_ct_series(folder, np.full((3, 5, 6), 7), spacing=SPACING, origin=ORIGIN, patient_id=PID, prefix='other',
                               modality='MR')
    for path in [*plan['ct']['paths'], *other['paths']]:
        edit(path, lambda ds: setattr(ds, 'FrameOfReferenceUID', ''))
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [
        ('plan_1', 'converted', ''), ('studies', 'converted', ''), ('study_1/series_MR_1', 'converted', '')]
    assert sorted(p.name for p in (out / PID / 'plan_1').iterdir()) == ['planning_ct.nii.gz', 'series.json']
    assert sorted(p.name for p in (out / PID / 'study_1').iterdir()) == ['series.json', 'series_MR_1.nii.gz']
    assert series_rows(out, 'study_1')[0]['reg_matrix'] == ''


# ---------------------------------------------------------------- hardening B6 (#22): orientation and dose offsets

STORED = 1 + np.arange(np.prod(SHAPE)).reshape(SHAPE)  # a different dose value in every voxel
INSIDE = ((1, 2), (1, 3), (2, 4))  # box of voxel indices (z, y, x), inclusive
NAMED = (2, 3, 4)  # (k, j, i): a voxel in the box
OTHER = (0, 0, 0)  # and one outside it


def physical_point(orientation, voxel):
    """Where voxel (k, j, i) of the synthetic CT sits, from the DICOM geometry alone (not from any converted file)."""
    k, j, i = voxel
    row, column = np.asarray(orientation[:3], float), np.asarray(orientation[3:], float)
    return tuple(np.asarray(ORIGIN) + i * SPACING[0] * row + j * SPACING[1] * column + k * SPACING[2] * np.cross(row, column))


def value_at(image, point):
    return float(image[image.TransformPhysicalPointToIndex(point)])


def aligned_dose(orientation, **kwargs):
    """A dose on the CT's own grid, in the CT's orientation."""
    return dose_spec(0, stored=STORED, orientation=orientation, **kwargs)


@pytest.mark.parametrize('orientation', [(-1, 0, 0, 0, 1, 0), (1, 0, 0, 0, -1, 0), (-1, 0, 0, 0, -1, 0)],
                         ids=['feet-first-supine', 'feet-first-prone', 'flipped-in-plane'])
def test_a_ct_with_flipped_axes_or_a_slice_normal_along_minus_z_keeps_ct_dose_and_mask_aligned(tree, orientation):
    box = dict(number=1, name='Liver', kind='box', box=INSIDE)
    _, rows, out = one_plan(tree, doses=[aligned_dose(orientation)], rois=[box], ct_meta=dict(orientation=orientation))
    assert (row_of(rows).outcome, row_of(rows).reason) == (conv.OUTCOME_CONVERTED, '')
    ct, dose, mask = (read(out / PID / 'plan_1' / name) for name in ('planning_ct.nii.gz', 'dose.nii.gz', 'liver.nii.gz'))
    values = (np.arange(np.prod(SHAPE)) % 200 - 100).reshape(SHAPE)  # make_plan's default
    expected_mask = np.zeros(SHAPE)
    expected_mask[1:3, 1:4, 2:5] = 1
    for image, expected in ((ct, values), (dose, STORED * SCALING), (mask, expected_mask)):
        np.testing.assert_allclose(sitk.GetArrayFromImage(image), expected, atol=1e-5)  # one grid for all three
    for voxel, in_box in ((NAMED, 1), (OTHER, 0)):
        point = physical_point(orientation, voxel)
        assert value_at(ct, point) == values[voxel]
        assert value_at(dose, point) == pytest.approx(STORED[voxel] * SCALING, abs=1e-5)
        assert value_at(mask, point) == in_box


@pytest.mark.parametrize('form', ['relative', 'negative', 'absolute'])  # relative-positive is the control
def test_a_dose_with_negative_or_absolute_frame_offsets_converts_to_the_grid_of_the_relative_positive_one(tree, form):
    top = ORIGIN[2] + (SHAPE[0] - 1) * SPACING[2]
    if form == 'negative':  # the first frame is the highest slice and the offsets count down from it
        dose = dose_spec(0, stored=STORED[::-1], position=(ORIGIN[0], ORIGIN[1], top),
                         offsets=[-k * SPACING[2] for k in range(SHAPE[0])])
    elif form == 'absolute':  # the first value is the first frame's own z
        dose = dose_spec(0, stored=STORED, offsets=[ORIGIN[2] + k * SPACING[2] for k in range(SHAPE[0])])
    else:
        dose = dose_spec(0, stored=STORED)
    _, rows, out = one_plan(tree, doses=[dose])
    assert row_of(rows).outcome == conv.OUTCOME_CONVERTED
    ct, got = read(out / PID / 'plan_1' / 'planning_ct.nii.gz'), read(out / PID / 'plan_1' / 'dose.nii.gz')
    assert got.GetOrigin() == ct.GetOrigin() and got.GetSize() == ct.GetSize()
    np.testing.assert_allclose(sitk.GetArrayFromImage(got), STORED * SCALING, atol=1e-5)  # the relative-positive form's dose
    assert value_at(got, physical_point((1, 0, 0, 0, 1, 0), NAMED)) == pytest.approx(STORED[NAMED] * SCALING, abs=1e-5)


def test_a_kept_roi_whose_only_points_are_point_items_refuses_the_plan(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', rois=[dict(number=1, name='Liver', kind='box', box=INSIDE)])
    struct = pydicom.dcmread(plan['folder'] / 'struct.dcm')
    for item in struct.ROIContourSequence[0].ContourSequence:  # planar items left with no points
        item.NumberOfContourPoints, item.ContourData = 0, []
    point = pydicom.Dataset()
    point.ContourGeometricType, point.NumberOfContourPoints, point.ContourData = 'POINT', 1, list(ORIGIN)
    struct.ROIContourSequence[0].ContourSequence.append(point)
    struct.save_as(plan['folder'] / 'struct.dcm')
    rows = conv.convert(org, out).rows
    # an empty ContourData reads back as no value: refused at the tag read, before DicomRTTool (#69)
    assert (row_of(rows).outcome, row_of(rows).reason) == (conv.OUTCOME_REFUSED, conv.REASON_MISSING_TAG)


def test_point_items_of_a_kept_roi_are_dropped_and_the_mask_is_its_planar_contours(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', rois=[dict(number=1, name='Liver', kind='box', box=INSIDE)])
    struct = pydicom.dcmread(plan['folder'] / 'struct.dcm')
    point = pydicom.Dataset()
    point.ContourGeometricType, point.NumberOfContourPoints, point.ContourData = 'POINT', 1, list(ORIGIN)  # voxel (0, 0, 0)
    struct.ROIContourSequence[0].ContourSequence.append(point)
    struct.save_as(plan['folder'] / 'struct.dcm')
    rows = conv.convert(org, out).rows
    assert (row_of(rows).outcome, row_of(rows).reason) == (conv.OUTCOME_CONVERTED, '')
    expected = np.zeros(SHAPE)
    expected[1:3, 1:4, 2:5] = 1
    np.testing.assert_array_equal(sitk.GetArrayFromImage(read(out / PID / 'plan_1' / 'liver.nii.gz')), expected)


# ---------------------------------------------------------------- quiet_call redesign: _read_dose's three steps (B1, B6)

class NoNumber:
    """An offset that is no number: no __float__ or __array__, and it cannot be subtracted."""
    def __sub__(self, other):
        raise TypeError('SYNTHETIC_no subtraction')

    def __rsub__(self, other):
        raise TypeError('SYNTHETIC_no subtraction')


def test_a_type_error_in_the_dose_arithmetic_aborts_the_run_and_is_no_unreadable_dose(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')])
    monkeypatch.setattr(conv, '_check_dose_header', lambda ds, ct_frame_uid: [NoNumber() for _ in range(SHAPE[0])])  # len and index work
    with pytest.raises(conv.ConvertAborted):
        conv.convert(org, out)
    with open(out / conv.REPORT_NAME, newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [(r['patient'], r['outcome'], r['reason']) for r in rows] == [(PID, conv.OUTCOME_REFUSED, conv.REASON_INTERNAL)]
    assert not any(r['reason'] == conv.REASON_DOSE_UNREADABLE for r in rows)


def resave_dose(folder, change):
    """The plan's first RTDOSE changed through save_as, then read back so the test sees what the code will."""
    path = folder / 'dose_0.dcm'
    ds = pydicom.dcmread(path)
    change(ds)
    ds.save_as(path)
    return pydicom.dcmread(path)


@pytest.mark.parametrize('tag, value', [('ImageOrientationPatient', [1.0, 0.0, 0.0, 0.0, 1.0]), ('PixelSpacing', [1.5]),
                                        ('ImagePositionPatient', [1.0, 2.0])], ids=['iop-5', 'spacing-1', 'ipp-2'])
def test_a_dose_with_a_wrong_count_of_geometry_values_refuses_the_plan_and_the_run_continues(tree, tag, value):
    org, out = tree
    bad = make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')])
    make_plan(org / PID / 'plan_2', doses=[dose_spec(20, 'PLAN')])
    held = resave_dose(bad['folder'], lambda ds: setattr(ds, tag, value))
    assert np.ravel(np.asarray(getattr(held, tag), dtype=float)).tolist() == value  # one value reads back as a scalar, not a list
    rows = conv.convert(org, out).rows
    assert (row_of(rows, 'plan_1').outcome, row_of(rows, 'plan_1').reason) == (conv.OUTCOME_REFUSED, conv.REASON_DOSE_UNREADABLE)
    assert (row_of(rows, 'plan_2').outcome, row_of(rows, 'plan_2').reason) == (conv.OUTCOME_CONVERTED, '')


def test_a_dose_with_a_nan_frame_offset_refuses_with_the_grid_reason(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')])
    held = resave_dose(plan['folder'], lambda ds: setattr(ds, 'GridFrameOffsetVector', [0.0, float('nan'), 2.0, 3.0]))
    assert np.isnan(float(held.GridFrameOffsetVector[1]))
    rows = conv.convert(org, out).rows
    assert (row_of(rows).outcome, row_of(rows).reason) == (conv.OUTCOME_REFUSED, conv.REASON_DOSE_GRID)


def test_two_adjacent_inf_frame_offsets_refuse_with_the_grid_reason_and_no_numpy_warning(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')])
    held = resave_dose(plan['folder'], lambda ds: setattr(ds, 'GridFrameOffsetVector', [0.0, float('inf'), float('inf'), 3.0]))
    assert np.isinf(float(held.GridFrameOffsetVector[1])) and np.isinf(float(held.GridFrameOffsetVector[2]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rows = conv.convert(org, out).rows
    assert (row_of(rows).outcome, row_of(rows).reason) == (conv.OUTCOME_REFUSED, conv.REASON_DOSE_GRID)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize('tag, value', [('ImagePositionPatient', [float('nan'), 5.0, 7.0]),
                                        ('ImageOrientationPatient', [1.0, 0.0, 0.0, 0.0, float('nan'), 0.0]),
                                        ('PixelSpacing', [1.5, float('nan')])], ids=['ipp', 'iop', 'spacing'])
def test_a_dose_with_a_nan_geometry_value_refuses_as_unreadable(tree, tag, value):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')])
    held = resave_dose(plan['folder'], lambda ds: setattr(ds, tag, value))
    assert any(np.isnan(float(v)) for v in getattr(held, tag))
    rows = conv.convert(org, out).rows
    assert (row_of(rows).outcome, row_of(rows).reason) == (conv.OUTCOME_REFUSED, conv.REASON_DOSE_UNREADABLE)


@pytest.mark.parametrize('scaling', [float('nan'), float('inf')], ids=['nan', 'inf'])
def test_a_dose_with_a_non_finite_grid_scaling_refuses_as_unreadable(tree, scaling):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')])
    held = resave_dose(plan['folder'], lambda ds: setattr(ds, 'DoseGridScaling', scaling))
    assert not np.isfinite(float(held.DoseGridScaling))
    rows = conv.convert(org, out).rows
    assert (row_of(rows).outcome, row_of(rows).reason) == (conv.OUTCOME_REFUSED, conv.REASON_DOSE_UNREADABLE)


# ---------------------------------------------------------------- quiet_call redesign: _header_call refuses only pydicom's own misses (B5)

def header_call(fn):
    return getattr(conv, '_header_call')(fn)  # noqa: B009


def empty_dataset():
    return pydicom.Dataset()


def test_an_attribute_error_from_our_own_frame_propagates_out_of_header_call():
    with pytest.raises(AttributeError):
        header_call(lambda: object().SYNTHETIC_missing)


def test_a_key_error_from_our_own_dict_propagates_out_of_header_call():
    with pytest.raises(KeyError):
        header_call(lambda: {}['SYNTHETIC_key'])


def test_a_misspelled_keyword_on_a_dataset_propagates_out_of_header_call():
    with pytest.raises(AttributeError):
        header_call(lambda: empty_dataset().SYNTHETICNotAKeyword)


@pytest.mark.parametrize('read', [lambda ds: ds.PatientID, lambda ds: ds['PatientName']], ids=['attribute', 'item'])
def test_a_missing_tag_refuses_with_one_info_record_naming_class_and_site(read, caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(CaseRefused) as caught:
        header_call(lambda: read(empty_dataset()))
    assert str(caught.value) == conv.REASON_MISSING_TAG
    assert getattr(caught.value, '__cause__') is None and getattr(caught.value, '__context__') is None  # noqa: B009
    records = [r for r in caplog.records if r.name == 'rt_dataprep.convert']
    assert len(records) == 1 and records[0].levelno == logging.INFO
    assert re.fullmatch(rf'convert: {conv.REASON_MISSING_TAG}: (AttributeError|KeyError) at rt_dataprep\.\w+:\d+', records[0].getMessage())
    assert 'SYNTHETIC_' not in records[0].getMessage() and '/' not in records[0].getMessage()


def test_the_missing_tag_site_is_the_line_in_the_wrapped_function_not_header_call(caplog):
    caplog.set_level(logging.INFO)
    check = getattr(conv, '_check_dose_header')  # noqa: B009
    with pytest.raises(CaseRefused):
        header_call(lambda: check(empty_dataset(), 'SYNTHETIC_frame'))
    lines, start = inspect.getsourcelines(check)
    line = start + next(i for i, text in enumerate(lines) if 'ds.DoseUnits' in text)
    records = [r for r in caplog.records if r.name == 'rt_dataprep.convert']
    assert records[0].getMessage().endswith(f' at rt_dataprep.convert:{line}')


def test_an_attribute_error_from_our_own_frame_in_the_classifier_aborts_the_run(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1')

    def classify(*args, **kwargs):
        raise AttributeError('SYNTHETIC_missing')
    monkeypatch.setattr(masks, 'classify', classify)
    with pytest.raises(conv.ConvertAborted):
        conv.convert(org, out)
    with open(out / conv.REPORT_NAME, newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [(r['patient'], r['outcome'], r['reason']) for r in rows] == [(PID, conv.OUTCOME_REFUSED, conv.REASON_INTERNAL)]


# ---------------------------------------------------------------- pooled header reads, workers, run timing line

TIMING = re.compile(r'convert: (\d+\.\d) s total, (\d+\.\d) s reading \(summed over patients\), (\d+\.\d) s converting \(summed over patients\), (\d+) files, workers (\d+)')


def add_bad_files(folder):
    """Two files that fail the read (pydicom tolerates a cut-off DICOM, so: an empty .dcm and a text file); neither is a header."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'SYNTHETIC_empty.dcm').write_bytes(b'')
    (folder / 'SYNTHETIC_text.txt').write_bytes(b'not a dicom file')


def timing_records(caplog):
    return [r.getMessage() for r in caplog.records if r.name == 'rt_dataprep.convert' and ' s total, ' in r.getMessage()]


def rows_of(report):
    return [(r.patient, r.folder, r.outcome, r.reason, r.files, r.skipped_series) for r in report.rows]


def test_read_folder_headers_sorted_by_path_with_the_unreadable_flag(tree):
    org, _ = tree
    make_plan(org / PID / 'plan_1')
    add_bad_files(org / PID / 'plan_1')
    headers, unreadable = conv.read_folder_headers(org / PID / 'plan_1')
    assert unreadable and [p for p, _ in headers] == sorted(p for p, _ in headers)
    assert sorted(p.name for p, _ in headers) == sorted(p.name for p in (org / PID / 'plan_1').glob('*.dcm') if 'empty' not in p.name)


def test_the_rows_are_identical_for_workers_1_and_4_with_bad_files_in_a_plan_folder_and_in_misc_files(tree, tmp_path):
    org, out = tree
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')])
    add_bad_files(org / PID / 'plan_1')
    make_plan(org / PID / 'plan_2', doses=[dose_spec(20, 'PLAN')])
    misc = org / PID / 'misc_files'
    misc.mkdir()
    sd.write_ct_series(misc, np.zeros(SHAPE), patient_id=PID)
    add_bad_files(misc)
    serial = rows_of(conv.convert(org, out))
    pooled = rows_of(conv.convert(org, tmp_path / 'out4', workers=4))
    assert pooled == serial
    assert [(r[1], r[2], r[3]) for r in serial if r[1].startswith('plan_')] == [
        ('plan_1', conv.OUTCOME_REFUSED, conv.REASON_FILE_UNREADABLE), ('plan_2', conv.OUTCOME_CONVERTED, '')]  # not a vacuous equality


def test_one_timing_line_after_the_summary_counts_the_headers_read_and_nothing_of_a_refused_patient(tree, caplog, monkeypatch):
    org, out = tree
    other, refused = 'SYNTHETICPTB', 'SYNTHETICPTC'
    make_plan(org / PID / 'plan_1')  # 6 headers
    add_bad_files(org / PID / 'plan_1')
    misc = org / PID / 'misc_files'
    misc.mkdir()
    sd.write_ct_series(misc, np.zeros(SHAPE), patient_id=PID)  # 4 headers
    add_bad_files(misc)
    make_plan(org.parent / 'scratch', patient=other)
    (org / other / 'misc_files').mkdir(parents=True)
    for name in ('struct.dcm', 'plan.dcm'):  # 2 headers, no image series and no plan folder: the patient is skipped
        (org / other / 'misc_files' / name).write_bytes((org.parent / 'scratch' / name).read_bytes())
    make_plan(org / refused / 'plan_1', patient=refused)  # read, then its misc_files read fails: none of it counts
    add_bad_files(org / refused / 'misc_files')
    real = conv.read_folder_headers

    def fail_misc_of_refused(folder, *args, **kwargs):
        if Path(folder).name == 'misc_files' and Path(folder).parent.name == refused:
            raise OSError('SYNTHETIC_unlistable')
        return real(folder, *args, **kwargs)
    monkeypatch.setattr(conv, 'read_folder_headers', fail_misc_of_refused)
    caplog.set_level(logging.INFO)
    rows = conv.convert(org, out).rows  # a monkeypatch does not reach a worker process: workers=1
    assert (row_of(rows, '').reason, [r.reason for r in rows if r.patient == refused]) == (
        conv.REASON_NO_PLANS, [conv.REASON_IO_ERROR])
    texts = [r.getMessage() for r in caplog.records if r.name == 'rt_dataprep.convert']
    line = timing_records(caplog)
    assert len(line) == 1
    total, reading, converting, files, workers = TIMING.fullmatch(line[0]).groups()
    assert (int(files), int(workers)) == (6 + 4 + 2, 1)  # 16 files on disk in the counted patients, 12 are headers; the refused patient adds nothing
    assert float(total) >= float(reading) >= 0 and float(converting) >= 0
    assert texts.index(line[0]) == next(i for i, t in enumerate(texts) if t.startswith('convert: ') and 'plan rows' in t) + 1
    assert 'SYNTHETIC' not in line[0] and '/' not in line[0]


def test_the_timing_line_is_logged_before_convert_aborted_and_counts_nothing_of_the_aborted_patient(tree, caplog, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1')

    def classify(*args, **kwargs):
        raise AttributeError('SYNTHETIC_missing')
    monkeypatch.setattr(masks, 'classify', classify)
    caplog.set_level(logging.INFO)
    with pytest.raises(conv.ConvertAborted):
        conv.convert(org, out)
    line = timing_records(caplog)
    assert len(line) == 1
    assert TIMING.fullmatch(line[0]).group(4) == '0'  # its 6 headers were read, but an aborted patient contributes none


def test_no_timing_line_on_ctrl_c(tree, caplog, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1')

    def classify(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(masks, 'classify', classify)
    caplog.set_level(logging.INFO)
    with pytest.raises(KeyboardInterrupt):
        conv.convert(org, out)
    assert timing_records(caplog) == []


# ---------------------------------------------------------------- hardening B8 (#98): workers reach every read, the dicom_io seam

def test_a_folder_header_read_that_makes_pydicom_warn_prints_nothing_and_restores_state(tmp_path, capfd, caplog):
    sd.write_rtdose(tmp_path / 'dose.dcm', [[[1]]], invalid_values=True)
    state = lambda: (pydicom.config.settings.reading_validation_mode, logging.getLogger('pydicom').disabled)  # noqa: E731
    with warnings.catch_warnings(record=True) as control, caplog.at_level(logging.DEBUG):  # control: it warns unquieted
        warnings.simplefilter('always')
        for element in pydicom.dcmread(tmp_path / 'dose.dcm', stop_before_pixels=True).iterall():
            _ = element.value
    assert control or caplog.records
    caplog.clear()
    capfd.readouterr()
    before, filters = state(), list(warnings.filters)
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.DEBUG):
        warnings.simplefilter('always')
        headers, unreadable = conv.read_folder_headers(tmp_path)
    out = capfd.readouterr()
    assert len(headers) == 1 and not unreadable
    assert caught == [] and caplog.records == [] and (out.out, out.err) == ('', '')
    assert state() == before and list(warnings.filters) == filters


def test_the_header_read_goes_through_the_dicom_io_module_attribute(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    real = getattr(dicom_io, 'read_header_core')  # noqa: B009

    def fail_plan_file(path):
        if Path(path).name == 'plan.dcm':
            raise OSError('SYNTHETIC_unreadable')
        return real(path)

    monkeypatch.setattr(dicom_io, 'read_header_core', fail_plan_file)
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows if r.folder == 'plan_1'] == [
        ('plan_1', conv.OUTCOME_REFUSED, conv.REASON_FILE_UNREADABLE)]


# ---------------------------------------------------------------- patient processes (#171): N=2 equals N=1, parent-side claims, abort

def three_patients(org, names=('SYNTHETICPTA', 'SYNTHETICPTB', 'SYNTHETICPTC')):
    for k, name in enumerate(names):
        make_plan(org / name / 'plan_1', doses=[dose_spec(10 + k, 'PLAN')], patient=name)


def written_bytes(root):
    """{relative path: bytes} of every file under root except the report copies (their names carry a timestamp)."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file() and 'report' not in p.name}


def call_guarded(fn):
    """Run fn in a daemon thread and fail instead of hanging; return (result, raised exception)."""
    import threading
    outcome = {}

    def target():
        try:
            outcome['result'] = fn()
        except BaseException as exc:  # noqa: BLE001
            outcome['error'] = exc
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(120)
    assert not thread.is_alive(), 'the call did not return'
    return outcome.get('result'), outcome.get('error')


def test_two_patient_processes_write_the_same_files_and_rows_as_one(tree, tmp_path):
    org, _ = tree
    three_patients(org)
    one = conv.convert(org, tmp_path / 'out1')
    two = conv.convert(org, tmp_path / 'out2', workers=2)
    assert rows_of(two) == rows_of(one) and len(one.rows) >= 3
    assert all(r.outcome == conv.OUTCOME_CONVERTED for r in one.rows if r.folder.startswith('plan_'))  # not a vacuous equality
    assert written_bytes(tmp_path / 'out2') == written_bytes(tmp_path / 'out1') and written_bytes(tmp_path / 'out1')


@pytest.mark.parametrize('workers', [1, 2])
def test_under_overwrite_a_case_repeat_is_refused_by_the_parent_and_the_first_in_sorted_order_wins(tree, tmp_path, workers):
    org, _ = tree
    three_patients(org, ('SYNTHETICPTA', 'syntheticpta', 'SYNTHETICPTC'))  # sorted: ...PTA, ...PTC, syntheticpta
    rows = rows_of(conv.convert(org, tmp_path / 'out', workers=workers, overwrite=True))
    claimed = [r for r in rows if r[3] == conv.REASON_NAME_CLAIMED]
    assert claimed == [('syntheticpta', '', conv.OUTCOME_REFUSED, conv.REASON_NAME_CLAIMED, [], 0)]
    assert [r[0] for r in rows if r[1] == 'plan_1'] == ['SYNTHETICPTA', 'SYNTHETICPTC']
    assert not (tmp_path / 'out' / 'syntheticpta').exists()


def test_without_overwrite_a_case_repeat_is_held_and_the_rows_equal_those_of_one_process(tree, tmp_path):
    org, _ = tree
    three_patients(org, ('SYNTHETICPTA', 'syntheticpta', 'SYNTHETICPTC'))
    one = rows_of(conv.convert(org, tmp_path / 'out1'))
    two = rows_of(conv.convert(org, tmp_path / 'out2', workers=2))
    assert two == one and not any(r[3] == conv.REASON_NAME_CLAIMED for r in one)
    assert [r[0] for r in one if r[1] == 'plan_1'] == ['SYNTHETICPTA', 'SYNTHETICPTC', 'syntheticpta']  # the repeat ran, as today


def test_a_case_repeat_starts_only_after_its_twin_ends(tmp_path, tree):
    import pool_workers
    org, _ = tree
    three_patients(org, ('SYNTHETICPTA', 'syntheticpta', 'SYNTHETICPTC'))
    conv.convert(org, tmp_path / 'out3', workers=3, _patient_fn=pool_workers.convert_record)  # three slots: only the hold key keeps the twins apart
    times = {n: [float((org / n / f).read_text()) for f in ('start', 'end')] for n in ('SYNTHETICPTA', 'syntheticpta')}
    assert times['syntheticpta'][0] >= times['SYNTHETICPTA'][1]


def test_an_internal_error_in_one_patient_aborts_a_two_process_run_with_the_report_and_no_message(tree, capfd):
    import pool_workers
    org, out = tree
    for name in ('a', 'boom', 'c', 'd', 'e', 'f'):
        (org / name).mkdir()
    _, error = call_guarded(lambda: conv.convert(org, out, workers=2, _patient_fn=pool_workers.convert_patient))
    assert isinstance(error, conv.ConvertAborted) and str(error) == conv.REASON_INTERNAL
    with open(out / conv.REPORT_NAME, newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [(r['patient'], r['outcome'], r['reason']) for r in rows] == [('a', conv.OUTCOME_SKIPPED, ''), ('boom', conv.OUTCOME_REFUSED, conv.REASON_INTERNAL)]  # two slots: 'a' and 'boom' only
    err = capfd.readouterr().err
    assert pool_workers.SECRET not in err and 'RuntimeError' in err  # the worker logged the type and site, never the message


@pytest.mark.parametrize('workers', [1, 2])
def test_a_patient_system_exit_keeps_its_code(tmp_path, workers):
    # Codex P2 on #176: the scheduler must not turn SystemExit(3) into a successful exit
    import pool_workers
    (tmp_path / 'org' / 'SYNTHETIC_P').mkdir(parents=True)
    with pytest.raises(SystemExit) as raised:
        conv.convert(tmp_path / 'org', tmp_path / 'out', workers=workers, _patient_fn=pool_workers.patient_exits_3)
    assert raised.value.code == 3 and (tmp_path / 'out' / 'convert_report.csv').is_file()


# ---------------------------------------------------------------- a failed input read: a file-system error is the patient's I/O error (#210)

IO_ERRORS = [lambda path: PermissionError(errno.EACCES, 'denied', str(path)), lambda path: OSError(errno.EIO, 'input/output', str(path))]
POSIX_USER = pytest.mark.skipif(os.name != 'posix' or os.geteuid() == 0, reason='mode bits do not stop root')


def spy_read_failure(monkeypatch):
    """The (calling function, path) pairs dicom_io.read_failure is asked about; the real helper still runs."""
    seen, real = [], dicom_io.read_failure
    monkeypatch.setattr(dicom_io, 'read_failure',
                        lambda path: (seen.append((inspect.currentframe().f_back.f_code.co_name, str(path))), real(path))[1])
    return seen


def site_header(org):
    plan = make_plan(org / PID / 'plan_1')
    bad = plan['folder'] / 'junk.dcm'
    bad.write_bytes(b'not a dicom file')  # no header: it never gets past the first read
    return [bad], bad, conv.REASON_FILE_UNREADABLE


def site_ct(org):
    plan = make_plan(org / PID / 'plan_1')
    truncate_pixels(plan)  # header intact: only ITK's pixel read fails
    return [Path(p) for p in plan['ct']['paths']], Path(plan['ct']['paths'][1]), conv.REASON_CT_UNREADABLE


def site_dose(org):
    plan = make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')])
    resave_dose(plan['folder'], lambda ds: setattr(ds, 'DoseGridScaling', float('nan')))
    bad = plan['folder'] / 'dose_0.dcm'
    return [bad], bad, conv.REASON_DOSE_UNREADABLE


def misc_series(org, cut=False):
    """A non-planning MR series in misc_files; with `cut`, one slice loses pixel data (header intact): its pixel read fails."""
    values = (np.arange(np.prod(SHAPE)) % 50).reshape(SHAPE)
    mr = sd.write_ct_series(_mkdir(org / PID / 'misc_files'), values, patient_id=PID, prefix='mr', modality='MR')
    paths = [Path(p) for p in mr['paths']]
    if cut:
        paths[1].write_bytes(paths[1].read_bytes()[:-40])
    return paths


def _mkdir(folder):
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def site_series(org):
    make_plan(org / PID / 'plan_1')
    paths = misc_series(org, cut=True)
    return paths, paths[1], None  # the series is skipped: no refusal reason


SITES = {'header': site_header, 'ct': site_ct, 'dose': site_dose, 'series': site_series}
SITE_CALLERS = {'header': '_read_one', 'ct': '_read_ct', 'dose': '_read_dose', 'series': 'read_series'}  # the reader that re-reads


def report_cells(out):
    """Every cell of convert_report.csv but the patient column's, and the header line."""
    with open(out / conv.REPORT_NAME, newline='') as handle:
        rows = list(csv.DictReader(handle))
    return [rows[0].keys() if rows else [], *(v for row in rows for k, v in row.items() if k != 'patient')], rows


def only_io_error_row(out, rows):
    assert [(r.patient, r.folder, r.outcome, r.reason) for r in rows] == [(PID, '', conv.OUTCOME_REFUSED, conv.REASON_IO_ERROR)]
    assert not (out / PID).exists()


@pytest.mark.parametrize('site', SITES)
def test_a4_the_failing_fixture_of_each_site_reaches_the_reread_of_its_files_and_keeps_its_refusal(tree, monkeypatch, site):
    org, out = tree
    paths, _, reason = SITES[site](org)
    seen = spy_read_failure(monkeypatch)
    rows = conv.convert(org, out).rows
    # each once, by the site's own reader: a refused plan's CT is not read again as another series (#216)
    assert sorted(seen) == sorted((SITE_CALLERS[site], str(p)) for p in paths)
    plan_row = row_of(rows)
    if reason is None:
        assert (plan_row.outcome, plan_row.skipped_series) == (conv.OUTCOME_CONVERTED, 0)
        assert [(r.outcome, r.skipped_series) for r in rows if r.folder == 'studies'] == [(conv.OUTCOME_SKIPPED, 1)]
    else:
        assert (plan_row.outcome, plan_row.reason) == (conv.OUTCOME_REFUSED, reason)
    if site == 'ct':  # unread, the CT is still counted as a skipped series, as when its second read failed
        assert [(r.outcome, r.skipped_series) for r in rows if r.folder == 'studies'] == [(conv.OUTCOME_SKIPPED, 1)]


@pytest.mark.parametrize('make_error', IO_ERRORS, ids=['EACCES', 'EIO'])
@pytest.mark.parametrize('site', SITES)
def test_a3_a_file_system_error_from_the_reread_is_the_patients_io_error(tree, monkeypatch, site, make_error):
    org, out = tree
    _, bad, _ = SITES[site](org)
    modes = fail_open(monkeypatch, bad, make_error(bad))
    only_io_error_row(out, conv.convert(org, out).rows)
    assert modes == ['rb']  # the patch was reached, once, in binary


def test_a3_the_ct_read_itself_reports_the_io_error(tree, monkeypatch):
    org, out = tree
    _, bad, _ = site_ct(org)
    monkeypatch.setattr(conv.ser, 'read_series', lambda slices: None)  # the CT is read again as another series: only _read_ct's re-read is left
    modes = fail_open(monkeypatch, bad, PermissionError(errno.EACCES, 'denied', str(bad)))
    only_io_error_row(out, conv.convert(org, out).rows)
    assert modes == ['rb']


@pytest.mark.parametrize('site', SITES)
def test_a5_a_forced_enomem_from_the_reread_is_the_internal_error_and_stops_the_run(tree, monkeypatch, site):
    org, out = tree
    _, bad, _ = SITES[site](org)
    fail_open(monkeypatch, bad, OSError(errno.ENOMEM, 'no memory', str(bad)))
    with pytest.raises(conv.ConvertAborted):
        conv.convert(org, out)
    with open(out / conv.REPORT_NAME, newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert [(r['patient'], r['outcome'], r['reason']) for r in rows] == [(PID, conv.OUTCOME_REFUSED, conv.REASON_INTERNAL)]


def deny_header_read(monkeypatch, bad, error):
    """Root-safe twin of chmod 0 on `bad`: the header read fails as dcmread would at open, and so does the Python re-read."""
    real = dicom_io.read_header_core

    def core(path):
        if str(path) == str(bad):
            raise OSError(dicom_io.REASON_UNREADABLE)
        return real(path)
    monkeypatch.setattr(dicom_io, 'read_header_core', core)
    return fail_open(monkeypatch, bad, error)


def test_a1_a_plan_ct_slice_without_read_permission_is_the_patients_only_row_twin(tree, monkeypatch):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    bad = Path(plan['ct']['paths'][1])
    deny_header_read(monkeypatch, bad, PermissionError(errno.EACCES, 'denied', str(bad)))
    only_io_error_row(out, conv.convert(org, out).rows)


def test_a2_a_misc_files_dicom_without_read_permission_is_the_patients_io_error_twin(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    bad = misc_series(org)[1]
    deny_header_read(monkeypatch, bad, PermissionError(errno.EACCES, 'denied', str(bad)))
    only_io_error_row(out, conv.convert(org, out).rows)


@POSIX_USER
@pytest.mark.parametrize('where', ['plan', 'misc'])
def test_a1_a2_a_dicom_without_read_permission_is_the_patients_io_error(tree, where):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    bad = Path(plan['ct']['paths'][1]) if where == 'plan' else misc_series(org)[1]
    bad.chmod(0)
    try:
        rows = conv.convert(org, out).rows
    finally:
        bad.chmod(0o644)
    only_io_error_row(out, rows)


def assert_no_leak(caplog, out, secrets):
    cells, _ = report_cells(out)
    texts = [caplog.text, *(r.getMessage() for r in caplog.records), *map(str, cells)]
    assert [s for s in secrets for t in texts if s in t] == []


@pytest.mark.parametrize('site', SITES)
def test_a9_a_forced_io_error_at_each_site_leaves_the_path_and_the_patient_folder_out_of_the_log_and_the_report(tree, monkeypatch, caplog, site):
    org, out = tree
    _, bad, _ = SITES[site](org)
    fail_open(monkeypatch, bad, PermissionError(errno.EACCES, 'denied', str(bad)))
    caplog.set_level(logging.DEBUG)
    only_io_error_row(out, conv.convert(org, out).rows)
    assert caplog.records
    assert_no_leak(caplog, out, [str(bad), bad.name, str(org), PID])


def test_a9_a_denied_plan_ct_slice_leaves_the_path_and_the_patient_folder_out_of_the_log_and_the_report(tree, monkeypatch, caplog):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    bad = Path(plan['ct']['paths'][1])
    deny_header_read(monkeypatch, bad, PermissionError(errno.EACCES, 'denied', str(bad)))
    caplog.set_level(logging.DEBUG)
    only_io_error_row(out, conv.convert(org, out).rows)
    assert_no_leak(caplog, out, [str(bad), bad.name, str(org), PID])


def test_a10_a_happy_path_with_a_dose_and_a_misc_series_re_reads_nothing(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')])
    misc_series(org)
    seen = spy_read_failure(monkeypatch)
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome) for r in rows] == [('plan_1', 'converted'), ('studies', 'converted'), ('study_1/series_MR_1', 'converted')]
    assert 'dose.nii.gz' in row_of(rows).files and (out / PID / 'study_1' / 'series_MR_1.nii.gz').is_file()  # all four readers ran
    assert seen == []


# ---------------------------------------------------------------- A6: a folder that lists but cannot be searched

def hide_entries(monkeypatch, parent, deny=False):
    """Python 3.14's is_file/is_dir answer False for an entry of `parent` whose stat fails; so do these, for that folder's entries only.
    With `deny` the stat itself raises PermissionError there (the root-safe stand-in for mode 0o444). Every other path is untouched."""
    real = {name: getattr(Path, name) for name in ('is_file', 'is_dir', 'stat')}
    monkeypatch.setattr(Path, 'is_file', lambda self, **kw: False if self.parent == parent else real['is_file'](self, **kw))
    monkeypatch.setattr(Path, 'is_dir', lambda self, **kw: False if self.parent == parent else real['is_dir'](self, **kw))
    if deny:
        def stat(self, *, follow_symlinks=True):
            if self.parent == parent:
                raise PermissionError(errno.EACCES, 'denied')
            return real['stat'](self, follow_symlinks=follow_symlinks)
        monkeypatch.setattr(Path, 'stat', stat)


def test_a6_a_plan_folder_whose_entries_cannot_be_searched_is_the_patients_io_error_twin(tree, monkeypatch):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    hide_entries(monkeypatch, plan['folder'], deny=True)
    only_io_error_row(out, conv.convert(org, out).rows)


@POSIX_USER
def test_a6_a_plan_folder_with_mode_0o444_is_the_patients_io_error(tree, monkeypatch):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    hide_entries(monkeypatch, plan['folder'])  # what 3.14 does with the EACCES; 3.13 raises it already
    plan['folder'].chmod(0o444)
    try:
        rows = conv.convert(org, out).rows
    finally:
        plan['folder'].chmod(0o755)
    only_io_error_row(out, rows)


def test_a6_a_misc_files_folder_whose_stat_fails_is_the_patients_io_error_twin(tree, monkeypatch):
    """The misc_files check itself (#215): a failed stat is not "no misc_files"."""
    org, out = tree
    misc_series(org)  # no plan folder: no other entry of the patient is stat'ed first
    hide_entries(monkeypatch, org / PID, deny=True)
    only_io_error_row(out, conv.convert(org, out).rows)


def test_a6_a_misc_files_entry_whose_stat_fails_at_the_salvage_listing_is_the_patients_io_error_twin(tree, monkeypatch):
    """The second listing of misc_files, for the study of a malformed file (#215): a failed stat is not "no such file"."""
    org, out = tree
    make_plan(org / PID / 'plan_1')
    misc = misc_series(org)[0].parent
    (misc / 'junk.dcm').write_bytes(b'not a dicom file')
    real = conv.read_folder_headers

    def read_then_hide(folder):
        found = real(folder)
        if Path(folder) == misc:
            hide_entries(monkeypatch, misc, deny=True)  # setup: the entries fail their stat only after the header read
        return found
    monkeypatch.setattr(conv, 'read_folder_headers', read_then_hide)
    only_io_error_row(out, conv.convert(org, out).rows)


def test_an_unstatable_entry_the_name_filters_drop_is_never_statted(tree, monkeypatch):
    """#214: a hidden root entry (.snapshot on NFS) or a non-plan patient entry whose stat fails is filtered by name, unread."""
    org, out = tree
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'PLAN')], rois=[BOX])
    (org / '.snapshot').mkdir()
    (org / PID / 'notes').mkdir()
    real = Path.stat

    def stat(self, *, follow_symlinks=True):
        if self.name in ('.snapshot', 'notes'):
            raise PermissionError(errno.EACCES, 'denied')
        return real(self, follow_symlinks=follow_symlinks)
    monkeypatch.setattr(Path, 'stat', stat)
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome) for r in rows] == [('plan_1', conv.OUTCOME_CONVERTED)]


# ---------------------------------------------------------------- workers='auto' (#175)

def auto_pool_sizes(monkeypatch, n=2):
    """Size every auto run as n and record the worker count of each pool.run call."""
    from rt_dataprep import pool
    sizes, real = [], pool.run
    monkeypatch.setattr(pool, 'size_workers', lambda *args: n)
    monkeypatch.setattr(pool, 'run', lambda workers, tasks, handle: (sizes.append(workers), real(workers, tasks, handle))[1])
    return sizes


def test_workers_auto_gives_the_same_rows_and_files_as_one_worker(tree, tmp_path, monkeypatch, caplog):
    org, _ = tree
    three_patients(org)
    sizes = auto_pool_sizes(monkeypatch)
    caplog.set_level(logging.INFO)
    one = conv.convert(org, tmp_path / 'out1')
    sizes.clear()  # the workers=1 run's pool.run call
    auto = conv.convert(org, tmp_path / 'out_auto', workers='auto')
    assert sizes == [1, 2]  # the probe patient in-process, then a real pool of 2 for the other two
    assert rows_of(auto) == rows_of(one) and len(one.rows) >= 3
    assert all(r.outcome == conv.OUTCOME_CONVERTED for r in one.rows if r.folder.startswith('plan_'))
    assert written_bytes(tmp_path / 'out_auto') == written_bytes(tmp_path / 'out1') and written_bytes(tmp_path / 'out1')
    lines = [r.getMessage() for r in caplog.records if TIMING.fullmatch(r.getMessage())]
    assert [TIMING.fullmatch(line).groups()[-1] for line in lines] == ['1', '2']  # the count run_auto returned, never 'auto'


def test_workers_auto_on_a_rerun_without_overwrite_starts_no_pool(tree, monkeypatch):
    org, out = tree
    three_patients(org)
    conv.convert(org, out)
    sizes = auto_pool_sizes(monkeypatch)
    rows = conv.convert(org, out, workers='auto').rows
    assert {r.outcome for r in rows} == {conv.OUTCOME_SKIPPED} and len(rows) >= 3
    assert sizes == [1, 1, 1]  # every patient was skipped in the probe: no converted patient, no sized pool
