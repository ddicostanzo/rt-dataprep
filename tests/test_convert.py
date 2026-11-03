"""convert (task-04): assemble, planning CT, dose, atomic plan folders, report. Every tree is synthetic and built in tmp_path."""
import ast
import csv
import inspect
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
import synthetic_dicom as sd

from rt_dataprep import convert as conv
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


def make_plan(folder, doses=(), rois=(), values=None, contour_images=True, patient=PID):
    """One plan folder: CT, RTSTRUCT, RTPLAN and an RTDOSE per entry of `doses` (write_rtdose kwargs)."""
    folder.mkdir(parents=True, exist_ok=True)
    if values is None:
        values = (np.arange(np.prod(SHAPE)) % 200 - 100).reshape(SHAPE)
    ct = sd.write_ct_series(folder, values, spacing=SPACING, origin=ORIGIN, patient_id=patient)
    study = dict(patient_id=patient, study_uid=ct['study_uid'])
    struct = sd.write_rtstruct_rois(folder / 'struct.dcm', ct, list(rois), contour_images=contour_images, **study)
    plan = sd.write_rtplan(folder / 'plan.dcm', struct, np.zeros((2, 3)), **study)
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
    assert row.files == ['planning_ct.nii.gz', 'dose.nii.gz']
    assert sorted(p.name for p in folder.iterdir()) == ['dose.nii.gz', 'planning_ct.nii.gz']  # no temp left


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
    assert row_of(rows).files == ['planning_ct.nii.gz']


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
    assert sorted(p.name for p in (out / PID).iterdir()) == ['plan_1']  # the other series is not written in this PR


def test_a2_8_missing_rtstruct_refuses_and_leaves_nothing(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    (org / PID / 'plan_1' / 'struct.dcm').unlink()
    rows = conv.convert(org, out).rows
    row = row_of(rows)
    assert (row.outcome, row.reason) == (conv.OUTCOME_REFUSED, conv.REASON_NO_STRUCT)
    assert not (out / PID / 'plan_1').exists()
    assert sorted(p.name for p in out.iterdir()) == [PID, conv.REPORT_NAME]  # no partial, no temp
    assert list((out / PID).iterdir()) == []
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
    assert sorted(p.name for p in (out / PID).iterdir()) == ['plan_2']


# ---------------------------------------------------------------- layout, atomicity, report

def test_refused_plan_leaves_no_folder_while_the_other_plans_are_written(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    make_plan(org / PID / 'plan_2', doses=[dose_spec(1, 'PLAN'), dose_spec(2, 'PLAN')])
    make_plan(org / PID / 'plan_3')
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome) for r in rows] == [('plan_1', 'converted'), ('plan_2', 'refused'), ('plan_3', 'converted')]
    assert sorted(p.name for p in (out / PID).iterdir()) == ['plan_1', 'plan_3']
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
    assert [(r.folder, r.outcome) for r in rows] == [('plan_1', 'refused'), ('plan_2', 'refused')]
    assert (out / PID).is_dir() and list((out / PID).iterdir()) == []


def test_patient_without_plan_folders_is_skipped_with_no_output_folder(tree):
    org, out = tree
    misc = org / PID / 'misc_files'
    misc.mkdir(parents=True)
    sd.write_ct_series(misc, np.zeros(SHAPE), patient_id=PID)
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
                            files='planning_ct.nii.gz;dose.nii.gz', skipped_series='0')]


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
    result = conv.convert_plan(org / PID / 'plan_1', case)
    assert result.files == ['planning_ct.nii.gz', 'dose.nii.gz']
    assert sorted(p.name for p in case.iterdir()) == ['dose.nii.gz', 'planning_ct.nii.gz']


def test_convert_plan_refusal_raises_the_reason_and_leaves_no_case_dir(tree):
    org, _ = tree
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10, 'BRACHY')])
    case = org.parent / 'case'
    with pytest.raises(CaseRefused) as caught:
        conv.convert_plan(org / PID / 'plan_1', case)
    assert str(caught.value) == conv.REASON_DOSE_RULE
    assert not case.exists()


# ---------------------------------------------------------------- guardrails on the source

def test_no_case_refused_is_raised_inside_an_except_block():
    tree = ast.parse(inspect.getsource(conv))
    inside = [node for handler in ast.walk(tree) if isinstance(handler, ast.ExceptHandler)
              for node in ast.walk(handler) if isinstance(node, ast.Raise)]
    assert inside == []


def test_no_array_view_of_a_temporary_image():
    assert 'GetArrayViewFromImage' not in inspect.getsource(conv)
