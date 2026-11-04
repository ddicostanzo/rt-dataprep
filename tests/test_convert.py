"""convert (task-04): assemble, planning CT, dose, atomic plan folders, report. Every tree is synthetic and built in tmp_path."""
import ast
import csv
import hashlib
import inspect
import logging
import operator
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
import synthetic_dicom as sd

from rt_dataprep import convert as conv
from rt_dataprep import masks
from rt_dataprep.organize import organize
from rt_dataprep.dicom_io import CaseRefused

PID = 'SYNTHETICPTA'
SHAPE = (4, 5, 6)  # (z, y, x)
SPACING = (1.0, 1.5, 2.0)  # (x, y, z): anisotropic, so an axis swap cannot pass
ORIGIN = (-3.0, 5.0, 7.0)
SCALING = 0.5


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
    assert row.files == ['planning_ct.nii.gz', 'dose.nii.gz', 'series.csv']
    assert sorted(p.name for p in folder.iterdir()) == ['dose.nii.gz', 'planning_ct.nii.gz', 'series.csv']  # no temp left


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
    assert row_of(rows).files == ['planning_ct.nii.gz', 'series.csv']


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
    assert sorted(p.name for p in (out / PID).iterdir()) == ['images', 'plan_1']  # the other CT goes to images/ (A2.4b)


def test_a2_8_missing_rtstruct_refuses_and_leaves_nothing(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    (org / PID / 'plan_1' / 'struct.dcm').unlink()
    rows = conv.convert(org, out).rows
    row = row_of(rows)
    assert (row.outcome, row.reason) == (conv.OUTCOME_REFUSED, conv.REASON_NO_STRUCT)
    assert not (out / PID / 'plan_1').exists()
    assert sorted(p.name for p in out.iterdir()) == [PID, conv.REPORT_NAME]  # no partial, no temp
    assert [p.name for p in (out / PID).iterdir()] == ['images']  # the refused plan's CT
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
    assert sorted(p.name for p in (out / PID).iterdir()) == (['images', 'plan_2'] if what == 'plan_struct_reference' else ['plan_2'])


# ---------------------------------------------------------------- layout, atomicity, report

def test_refused_plan_leaves_no_folder_while_the_other_plans_are_written(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    make_plan(org / PID / 'plan_2', doses=[dose_spec(1, 'PLAN'), dose_spec(2, 'PLAN')])
    make_plan(org / PID / 'plan_3')
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome) for r in rows] == [('plan_1', 'converted'), ('plan_2', 'refused'), ('plan_3', 'converted'),
                                                      ('images', 'converted')]  # the refused plan's CT
    assert sorted(p.name for p in (out / PID).iterdir()) == ['images', 'plan_1', 'plan_3']
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
    assert [(r.folder, r.outcome) for r in rows] == [('plan_1', 'refused'), ('plan_2', 'refused'), ('images', 'converted')]
    assert (out / PID).is_dir() and sorted(p.name for p in (out / PID).iterdir()) == ['images']  # only the refused plans' CTs


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
    assert sorted(p.name for p in out.iterdir()) == [PID, conv.REPORT_NAME]


def test_stale_partial_is_removed_first(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    stale = out / f'.{PID}.partial' / 'plan_9'
    stale.mkdir(parents=True)
    (stale / 'leftover').write_text('x')
    conv.convert(org, out)
    assert sorted(p.name for p in out.iterdir()) == [PID, conv.REPORT_NAME]
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
    assert sorted(p.name for p in out.iterdir()) == [conv.REPORT_NAME]


def test_report_columns_and_joined_files(tree):
    _, rows, out = one_plan(tree, doses=[dose_spec(10, 'PLAN')])
    with open(out / conv.REPORT_NAME, newline='') as handle:
        reader = csv.DictReader(handle)
        written = list(reader)
        assert reader.fieldnames == ['patient', 'folder', 'outcome', 'reason', 'files', 'skipped_series']
    assert written == [dict(patient=PID, folder='plan_1', outcome='converted', reason='',
                            files='planning_ct.nii.gz;dose.nii.gz;series.csv', skipped_series='0')]


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
    assert result.files == ['planning_ct.nii.gz', 'dose.nii.gz']  # series.csv is the report's to add
    assert sorted(p.name for p in case.iterdir()) == ['dose.nii.gz', 'planning_ct.nii.gz']  # series.csv is the patient level's, last
    assert [r['file'] for r in result.series_rows] == result.files  # one series.csv row per data file, in write order
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
        ('plan_1', conv.OUTCOME_CONVERTED, ['planning_ct.nii.gz', 'series.csv'], 0)]
    assert 'convert: 1 unreadable files in misc_files' in caplog.messages
    assert all(secret not in text for text in caplog.messages for secret in (PID, 'SYNTHETIC_junk', str(org)))


def test_convert_plan_refusal_raises_the_reason_and_leaves_no_case_dir(tree):
    org, _ = tree
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'BRACHY')])
    case = org.parent / 'case'
    with pytest.raises(CaseRefused) as caught:
        run_plan(org / PID / 'plan_1', case)
    assert str(caught.value) == conv.REASON_DOSE_RULE
    assert not case.exists()


# ---------------------------------------------------------------- series.csv (A2.4d)

SERIES_COLUMNS = ['file', 'kind', 'modality', 'series_description', 'series_instance_uid', 'study_instance_uid',
                  'frame_of_reference_uid', 'series_date', 'roi_name', 'dose_summation_type', 'reg_matrix', 'rtplan_uid',
                  'rtplan_label', 'rtplan_date', 'fractions_planned']
CT_DATE, STRUCT_DATE, DOSE_DATE, PLAN_DATE = '20240105', '20240106', '20240107', '20240301'
BOX = dict(number=1, name='Liver', kind='box', box=((0, 2), (0, 2), (0, 2)))
RING = dict(number=2, name='ROI A', kind='channel', points=np.arange(15.0).reshape(5, 3), interpreted='BRACHY_CHANNEL')
RING_CHANNEL = [dict(applicator_id='Ring-1', dwells_mm=np.zeros((2, 3)), roi=2)]


def series_rows(out, folder='plan_1'):
    with open(out / PID / folder / 'series.csv', newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == SERIES_COLUMNS
        return list(reader)


def describe(path, text, date):
    def change(ds):
        ds.SeriesDescription, ds.SeriesDate = text, date
    edit(path, change)


def test_series_csv_has_one_row_per_written_file_with_the_values_of_each_kind(tree):
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
    assert row.files == ['planning_ct.nii.gz', 'dose.nii.gz', 'liver.nii.gz', 'Ring1.nii.gz', 'series.csv']
    rows = series_rows(out)
    assert [r['file'] for r in rows] == row.files[:-1]  # no row for series.csv itself
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


def test_series_csv_of_a_summed_beam_dose_joins_the_series_uids(tree):
    org, out = tree
    uids = [pydicom.uid.generate_uid(), pydicom.uid.generate_uid()]
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'BEAM', series_uid=uid) for uid in uids])
    conv.convert(org, out)
    dose_row = series_rows(out)[1]
    assert (dose_row['file'], dose_row['modality'], dose_row['dose_summation_type']) == ('dose.nii.gz', 'RTDOSE', 'BEAM')
    assert dose_row['series_instance_uid'] == ';'.join(uids)


def test_series_csv_leaves_absent_plan_tags_empty(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1', plan_meta=dict(fractions=None))
    conv.convert(org, out)
    [ct_row] = series_rows(out)
    assert {k: ct_row[k] for k in ('rtplan_label', 'rtplan_date', 'fractions_planned', 'series_description', 'series_date')} == {
        'rtplan_label': '', 'rtplan_date': '', 'fractions_planned': '', 'series_description': '', 'series_date': ''}


def test_series_csv_is_the_last_file_and_goes_through_a_temp_name(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1', rois=[BOX])
    replaced = []
    real = conv.os.replace

    def spy(source, target):
        replaced.append((Path(source).name, Path(target).name, Path(target).parent.name))
        real(source, target)
    monkeypatch.setattr(conv.os, 'replace', spy)
    conv.convert(org, out)
    source, target, _ = replaced[-1]
    assert target == 'series.csv' and source != target and source.endswith('series.csv')
    assert sorted(p.name for p in (out / PID / 'plan_1').iterdir()) == ['liver.nii.gz', 'planning_ct.nii.gz', 'series.csv']


def test_a_failed_series_csv_write_is_a_patient_level_io_error_and_leaves_nothing(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    real = conv.os.replace

    def fail(source, target):
        if Path(target).name == 'series.csv':
            raise OSError(f'cannot write {target} for {PID}')
        real(source, target)
    monkeypatch.setattr(conv.os, 'replace', fail)
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'refused', conv.REASON_IO_ERROR)]
    assert sorted(p.name for p in out.iterdir()) == [conv.REPORT_NAME]


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
        ('plan_1', 'converted', ['planning_ct.nii.gz', 'liver.nii.gz', 'series.csv'])]
    assert arr(out / PID / 'plan_1' / 'liver.nii.gz').sum() == 27


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
    assert {k: v for k, v in tree_hashes(out).items() if k != conv.REPORT_NAME} == {
        k: v for k, v in after_first.items() if k != conv.REPORT_NAME}


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
        conv.write_series_csv([], tmp_path / 'series.csv')
    assert str(err.value) == conv.REASON_WRITE
    assert operator.attrgetter('__cause__', '__context__')(err.value) == (None, None)
    assert list(tmp_path.rglob('*')) == []


def test_crash_site_is_the_caller_of_quiet_call(monkeypatch):
    from rt_dataprep import preprocess as pre

    def boom(image, spacing):
        raise RuntimeError('SYNTHETICPTA')
    monkeypatch.setattr(pre, 'resample_to', boom)
    with pytest.raises(CaseRefused) as err:
        pre.process_ct(sitk.Image(2, 2, 2, sitk.sitkFloat32), pre.LEGACY_RNT)
    module, line = conv.crash_site(err.tb)
    assert module == 'rt_dataprep.preprocess'
    assert 'quiet_call(run' in inspect.getsource(pre).splitlines()[line - 1]


def test_series_csv_is_utf8_whatever_the_locale(tmp_path):
    script = ("import sys; from rt_dataprep import convert as conv; "
              "row = dict.fromkeys(conv.SERIES_COLUMNS, '') | dict(file='SYNTHETIC.nii.gz', kind='ct', series_description='Thorax \\u00e9'); "
              "conv.write_series_csv([row], sys.argv[1])")
    env = {**os.environ, 'LC_ALL': 'C', 'PYTHONCOERCECLOCALE': '0', 'PYTHONUTF8': '0'}  # an ASCII locale
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path / 'series.csv')], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'Thorax \u00e9' in (tmp_path / 'series.csv').read_bytes().decode('utf-8')


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
        raise KeyError  # _series_rows runs after the plan folder is written: a refusal (REASON_MISSING_TAG) from here
    monkeypatch.setattr(conv.shutil, 'rmtree', rmtree)
    monkeypatch.setattr(conv, '_series_rows', refuse_after_the_plan_folder_exists)
    rows = conv.convert(org, out).rows
    assert [(r.patient, r.folder, r.outcome, r.reason, r.files) for r in rows] == [(PID, '', 'refused', conv.REASON_IO_ERROR, [])]
    assert sorted(p.name for p in out.iterdir()) == [conv.REPORT_NAME]  # no patient folder, no partial


def test_another_series_on_an_empty_frame_is_not_registered_to_a_planning_ct_on_an_empty_frame(tree):
    org, out = tree
    folder = org / PID / 'plan_1'
    plan = make_plan(folder)
    other = sd.write_ct_series(folder, np.full((3, 5, 6), 7), spacing=SPACING, origin=ORIGIN, patient_id=PID, prefix='other',
                               modality='MR')
    for path in [*plan['ct']['paths'], *other['paths']]:
        edit(path, lambda ds: setattr(ds, 'FrameOfReferenceUID', ''))
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('plan_1', 'converted', ''), ('images', 'converted', '')]
    assert sorted(p.name for p in (out / PID / 'plan_1').iterdir()) == ['planning_ct.nii.gz', 'series.csv']
    assert sorted(p.name for p in (out / PID / 'images').iterdir()) == ['series.csv', 'series_MR_1.nii.gz']
    assert series_rows(out, 'images')[0]['reg_matrix'] == ''


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
