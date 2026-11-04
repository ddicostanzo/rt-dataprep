"""preprocess: grid, crop/pad, rescale, names, CT pipeline, dose and mask resampling (task-01); the tree walk (task-02)."""
import ast
import csv
import dataclasses
import inspect
import re
import traceback

import numpy as np
import pytest
import SimpleITK as sitk

from rt_dataprep import convert as conv
from rt_dataprep import preprocess as pre
from rt_dataprep.dicom_io import CaseRefused
from rt_dataprep.preprocess import LEGACY_RNT, PreprocessSteps

ROTATED = (0.8660254037844387, -0.5, 0.0, 0.5, 0.8660254037844387, 0.0, 0.0, 0.0, 1.0)  # 30 degrees about z
QUARTER = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # 90 degrees about z: D @ (a, b, c) = (-b, a, c)


def make_image(array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)):
    """A float32 image from a (z, y, x) array; spacing, origin and direction are in (x, y, z) order."""
    image = sitk.GetImageFromArray(np.asarray(array, np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    return image


def blank(size, **geometry):
    return make_image(np.zeros(size[::-1]), **geometry)


def centroid_mm(mask):
    index = np.argwhere(sitk.GetArrayFromImage(mask) > 0)[:, ::-1].mean(axis=0)  # (x, y, z)
    return np.array(mask.TransformContinuousIndexToPhysicalPoint(index.tolist()))


def test_steps_defaults_and_preset():
    steps = PreprocessSteps()
    assert (steps.resample, steps.rescale, steps.harmonize_spacing, steps.reshape, steps.legacy_linear_masks) == (False,) * 5
    assert steps.new_spacing == (1, 1, 1) and steps.final_spacing == (1.25, 1.25, 1.25) and steps.new_shape == (701, 701, 433)
    with pytest.raises(dataclasses.FrozenInstanceError):
        steps.resample = True
    assert (LEGACY_RNT.resample, LEGACY_RNT.rescale, LEGACY_RNT.harmonize_spacing, LEGACY_RNT.reshape) == (True,) * 4
    assert not LEGACY_RNT.legacy_linear_masks
    assert dataclasses.replace(LEGACY_RNT, resample=False, rescale=False, harmonize_spacing=False, reshape=False) == steps


def test_reference_grid_hand_computed():
    # x: floor(10 * 1 / 1.5) = 6, y: floor(7 * 2 / 1.5) = 9, z: N = 1 stays 1; c = (4.5, 6, 0)
    # origin = c - (n - 1) / 2 * 1.5 = (4.5 - 3.75, 6 - 6, 0 - 0)
    image = blank((10, 7, 1), spacing=(1.0, 2.0, 3.0))
    size, origin, spacing, direction = pre.reference_grid(image, (1.5, 1.5, 1.5))
    assert size == [6, 9, 1]
    assert origin == pytest.approx((0.75, 0.0, 0.0))
    assert spacing == (1.5, 1.5, 1.5) and direction == image.GetDirection()


def test_reference_grid_non_identity_direction():
    # N = (5, 4, 3), s = 2, s' = 3 -> n = (3, 2, 2). D @ (a, b, c) = (-b, a, c).
    # c = (10, 20, 30) + D @ (4, 3, 2) = (7, 24, 32); origin = c - D @ (3, 1.5, 1.5) = c - (-1.5, 3, 1.5) = (8.5, 21, 30.5)
    image = blank((5, 4, 3), spacing=(2.0, 2.0, 2.0), origin=(10.0, 20.0, 30.0), direction=QUARTER)
    size, origin, spacing, direction = pre.reference_grid(image, (3.0, 3.0, 3.0))
    assert size == [3, 2, 2]
    assert origin == pytest.approx((8.5, 21.0, 30.5))
    assert direction == QUARTER


def test_reference_grid_uses_the_stored_float32_spacing(tmp_path):
    path = tmp_path / 'ct.nii.gz'
    sitk.WriteImage(blank((5, 5, 5), spacing=(0.9, 0.9, 0.9)), str(path))
    stored = pre.read_image(path)
    assert stored.GetSpacing()[0] < 0.9  # 0.89999998
    assert np.floor(5 * 0.9 / 1.5) == 3  # the nominal spacing would give 3
    assert pre.reference_grid(stored, (1.5, 1.5, 1.5))[0] == [2, 2, 2]


def test_crop_or_pad_odd_difference_puts_the_larger_half_first():
    # x 6 -> 9 pads 2 + 1; y 5 -> 2 crops 2 + 1; z 4 -> 8 pads 2 + 2
    array = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    image = make_image(array, spacing=(0.5, 1.5, 2.0), origin=(10.0, 20.0, 30.0), direction=ROTATED)
    out = pre.crop_or_pad(image, (9, 2, 8), -7.0)
    expected = np.full((8, 2, 9), -7.0)
    expected[2:6, :, 2:8] = array[:, 2:4, :]
    assert np.array_equal(sitk.GetArrayFromImage(out), expected)
    assert out.GetSpacing() == image.GetSpacing() and out.GetDirection() == image.GetDirection()
    shift = np.array(ROTATED).reshape(3, 3) @ np.array([-2 * 0.5, 2 * 1.5, -2 * 2.0])  # start voxels: pad x, crop y, pad z
    assert out.GetOrigin() == pytest.approx(tuple(np.array(image.GetOrigin()) + shift))


STEPS_SMALL = dataclasses.replace(LEGACY_RNT, new_spacing=(1.5, 1.5, 1.5), new_shape=(9, 9, 9))


def small_ct():
    """5^3 at 1 mm; HU linear in the voxel index, so the linear resamples are exact."""
    return make_image((np.linspace(-1000, 200, 125)).reshape(5, 5, 5))


def test_rescale_only():
    hu = np.array([[[-1024.0, -1000.0, -500.0], [0.0, 1.5, 3071.0]]])
    out = pre.process_ct(make_image(hu), PreprocessSteps(rescale=True))
    assert out.GetPixelID() == sitk.sitkFloat32
    assert np.abs(sitk.GetArrayFromImage(out) - (hu + 1024) / 4096).max() < 1e-6


def test_ct_steps_run_in_legacy_order():
    # N = 5, s = 1: 1.5 mm gives n = 3 (c = 2, origin 2 - 1.5 = 0.5), then 1.25 mm gives n = floor(3 * 1.5 / 1.25) = 3,
    # origin = 2 - 1.25 = 0.75. Skipping the first step would give floor(5 / 1.25) = 4.
    out = pre.process_ct(small_ct(), dataclasses.replace(STEPS_SMALL, reshape=False))
    assert out.GetSize() == (3, 3, 3) and out.GetSpacing() == (1.25, 1.25, 1.25)
    assert out.GetOrigin() == pytest.approx((0.75, 0.75, 0.75))


def test_padding_is_the_ct_minimum_for_the_ct_and_zero_for_dose_and_masks():
    ct = small_ct()
    dose = make_image(np.full((5, 5, 5), 5.0))
    mask = sitk.Cast(make_image(np.ones((5, 5, 5))), sitk.sitkUInt8)
    final = pre.process_ct(ct, STEPS_SMALL)
    assert final.GetSize() == (9, 9, 9)
    # interior voxel (0.75, 0.75, 0.75): HU = -1000 + 1200 / 124 * 31 * 0.75 = -775, rescaled (-775 + 1024) / 4096
    pad = 249 / 4096
    array = sitk.GetArrayFromImage(final)
    edge = np.ones((9, 9, 9), bool)
    edge[3:6, 3:6, 3:6] = False  # d = 6 per axis: three voxels padded at each end
    assert np.abs(array[edge] - pad).max() < 1e-6 and np.abs(array[3, 3, 3] - pad) < 1e-6
    dose_out = sitk.GetArrayFromImage(pre.process_dose(dose, final))
    mask_out = sitk.GetArrayFromImage(pre.process_mask(mask, final, STEPS_SMALL))
    # final centres are -3, -1.75, -0.5, 0.75, 2, 3.25, 4.5, 5.75, 7 mm; the native grid ends at -0.5 and 4.5
    outside = np.ones((9, 9, 9), bool)
    outside[2:7, 2:7, 2:7] = False
    assert pad != 0 and np.array_equal(dose_out[outside], np.zeros(outside.sum()))
    assert np.array_equal(mask_out[outside], np.zeros(outside.sum()))
    assert np.allclose(dose_out[3:6, 3:6, 3:6], 5.0) and (mask_out[3:6, 3:6, 3:6] == 1).all()


def test_dose_matches_the_analytic_dose_and_is_zero_outside():
    spacing, origin = (0.8, 0.9, 1.0), (-10.0, 5.0, 7.0)
    lo, hi = np.array([8, 6, 5]), np.array([19, 15, 12])  # box voxel index range (x, y, z), inclusive
    x_mm = origin[0] + spacing[0] * np.arange(30)
    native = np.zeros((20, 24, 30))
    native[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = (0.5 + 0.1 * x_mm)[lo[0]:hi[0] + 1]
    dose = make_image(native, spacing=spacing, origin=origin)
    final = pre.process_ct(make_image(np.zeros((20, 24, 30)), spacing=spacing, origin=origin),
                           dataclasses.replace(LEGACY_RNT, new_shape=(28, 28, 28)))
    out = sitk.GetArrayFromImage(pre.process_dose(dose, final))
    assert out.dtype == np.float32
    z, y, x = (np.array(final.GetOrigin())[::-1, None, None, None] + np.array(final.GetSpacing())[::-1, None, None, None]
               * np.indices(out.shape))
    hull_lo = np.array(origin) + np.array(spacing) * lo
    hull_hi = np.array(origin) + np.array(spacing) * hi
    inside = ((x >= hull_lo[0]) & (x <= hull_hi[0]) & (y >= hull_lo[1]) & (y <= hull_hi[1])
              & (z >= hull_lo[2]) & (z <= hull_hi[2]))
    far = ((hull_lo[0] - x >= spacing[0]) | (x - hull_hi[0] >= spacing[0]) | (hull_lo[1] - y >= spacing[1])
           | (y - hull_hi[1] >= spacing[1]) | (hull_lo[2] - z >= spacing[2]) | (z - hull_hi[2] >= spacing[2]))
    assert inside.sum() > 100 and far.sum() > 100
    assert np.abs(out[inside] - (0.5 + 0.1 * x[inside])).max() < 1e-4
    assert np.array_equal(out[far], np.zeros(far.sum()))


def cube_fixture(side=12):
    """A 24^3 CT at 0.9 mm and a cube mask of `side` voxels on its grid."""
    geometry = dict(spacing=(0.9, 0.9, 0.9), origin=(3.0, -2.0, 5.0))
    array = np.zeros((24, 24, 24))
    array[5:5 + side, 5:5 + side, 5:5 + side] = 1
    return make_image(np.zeros((24, 24, 24)), **geometry), sitk.Cast(make_image(array, **geometry), sitk.sitkUInt8)


def test_masks_are_binary_uint8_by_default_and_fractional_float32_when_legacy_linear():
    ct, mask = cube_fixture(side=8)
    steps = dataclasses.replace(LEGACY_RNT, new_shape=(20, 20, 20))
    final = pre.process_ct(ct, steps)
    out = pre.process_mask(mask, final, steps)
    assert out.GetPixelID() == sitk.sitkUInt8 and set(np.unique(sitk.GetArrayFromImage(out))) == {0, 1}
    linear = dataclasses.replace(steps, legacy_linear_masks=True)
    soft = pre.process_mask(mask, final, linear)
    values = sitk.GetArrayFromImage(soft)
    assert soft.GetPixelID() == sitk.sitkFloat32 and ((values > 0) & (values < 1)).any()


def test_default_masks_take_the_nearest_native_voxel():
    ct, mask = cube_fixture(side=8)
    steps = dataclasses.replace(LEGACY_RNT, new_shape=(20, 20, 20))
    final = pre.process_ct(ct, steps)
    out = sitk.GetArrayFromImage(pre.process_mask(mask, final, steps))
    native = sitk.GetArrayFromImage(mask)
    expected = np.zeros_like(out)
    for z, y, x in np.ndindex(out.shape):  # independent of sitk.Resample: round each final centre's native index
        index = np.rint(mask.TransformPhysicalPointToContinuousIndex(final.TransformIndexToPhysicalPoint((x, y, z)))).astype(int)
        if (index >= 0).all() and (index < native.shape[::-1]).all():
            expected[z, y, x] = native[index[2], index[1], index[0]]
    assert out.sum() > 0 and (out == expected).all()


def test_cube_keeps_its_centroid_and_volume_and_a_shifted_copy_fails():
    ct, mask = cube_fixture()
    steps = dataclasses.replace(LEGACY_RNT, new_shape=(20, 20, 20))
    final = pre.process_ct(ct, steps)
    out = pre.process_mask(mask, final, steps)
    s = max(steps.final_spacing)
    side = 12 * 0.9
    assert side >= 3 * s and all(f != n for f, n in zip(final.GetSpacing(), ct.GetSpacing(), strict=True))

    def holds(image):
        centroid_ok = (np.abs(centroid_mm(image) - centroid_mm(mask)) < s / 2).all()
        volume = (sitk.GetArrayFromImage(image) > 0).sum() * np.prod(image.GetSpacing())
        return centroid_ok and abs(volume - side ** 3) <= 6 * side ** 2 * s

    assert holds(out)
    array = sitk.GetArrayFromImage(out)
    for axis in range(3):
        shifted = sitk.GetImageFromArray(np.roll(array, 1, axis=axis))
        shifted.CopyInformation(out)
        assert not holds(shifted)


def test_every_output_keeps_the_input_direction_exactly():
    geometry = dict(spacing=(0.9, 1.1, 1.3), origin=(4.0, -3.0, 8.0), direction=ROTATED)
    ct = make_image(np.zeros((8, 8, 8)), **geometry)
    dose = make_image(np.ones((8, 8, 8)), **geometry)
    mask = sitk.Cast(dose, sitk.sitkUInt8)
    steps = dataclasses.replace(LEGACY_RNT, new_shape=(10, 10, 10))
    final = pre.process_ct(ct, steps)
    outputs = [final, pre.process_dose(dose, final), pre.process_mask(mask, final, steps)]
    assert all(image.GetDirection() == ROTATED for image in outputs)
    assert all(image.GetSize() == final.GetSize() and image.GetOrigin() == final.GetOrigin() for image in outputs)


def test_names_newest_step_first_and_geometric_prefixes_only_for_dose_and_masks():
    assert pre.output_name('planning_ct.nii.gz', LEGACY_RNT, ct=True) == 'reshape_resample_rescale_initialresample_planning_ct.nii.gz'
    assert pre.output_name('dose.nii.gz', LEGACY_RNT) == 'reshape_resample_initialresample_dose.nii.gz'
    assert pre.output_name('ptv.nii.gz', LEGACY_RNT) == 'reshape_resample_initialresample_ptv.nii.gz'
    rescale = PreprocessSteps(rescale=True)
    assert pre.output_name('planning_ct.nii.gz', rescale, ct=True) == 'rescale_planning_ct.nii.gz'
    assert pre.output_name('dose.nii.gz', rescale) == 'dose.nii.gz' and pre.output_name('ptv.nii.gz', rescale) == 'ptv.nii.gz'
    assert pre.output_name('dose.nii.gz', PreprocessSteps(harmonize_spacing=True)) == 'resample_dose.nii.gz'
    assert pre.output_name('planning_ct.nii.gz', PreprocessSteps(), ct=True) == 'planning_ct.nii.gz'


def test_third_party_failures_are_one_fixed_reason(tmp_path):
    bad = tmp_path / 'ct.nii.gz'
    bad.write_bytes(b'not an image')
    with pytest.raises(CaseRefused) as caught:
        pre.read_image(bad)
    assert str(caught.value) == pre.REASON_FILE_UNREADABLE and str(bad) not in str(caught.value)
    for call in (lambda: pre.process_ct(None, LEGACY_RNT), lambda: pre.process_dose(None, blank((2, 2, 2))),
                 lambda: pre.process_mask(None, blank((2, 2, 2)), LEGACY_RNT)):
        with pytest.raises(CaseRefused) as caught:
            call()
        assert str(caught.value) == pre.REASON_INTERNAL


def test_reasons_hold_no_digits():
    assert not any(c.isdigit() for c in pre.REASON_FILE_UNREADABLE + pre.REASON_INTERNAL)


def test_no_array_view_of_a_temporary_image():
    tree = ast.parse(inspect.getsource(pre))
    views = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'GetArrayViewFromImage']
    assert all(isinstance(n.args[0], ast.Name) for n in views)


# ---- task-02: the tree walk. A synthetic converted tree lives in tmp_path; patient IDs hold no digits or underscores. ----

SMALL = dataclasses.replace(LEGACY_RNT, new_shape=(10, 10, 10))
CT_FILE = 'reshape_resample_rescale_initialresample_planning_ct.nii.gz'
DOSE_FILE = 'reshape_resample_initialresample_dose.nii.gz'
PTV_FILE, BLADDER_FILE = 'reshape_resample_initialresample_ptv.nii.gz', 'reshape_resample_initialresample_bladder.nii.gz'
APPLICATOR_BYTES = b'1.5 2.5 3.5\n4.5 5.5 6.5\n'


def grid(n=12, spacing=1.0):
    """Geometry of an n^3 grid centred on the origin."""
    return dict(spacing=(spacing,) * 3, origin=(-(n - 1) * spacing / 2,) * 3)


def box(lo, hi, n=12, value=1.0):
    array = np.zeros((n, n, n))
    array[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = value
    return array


DEFAULT_DOSE = box((3, 3, 3), (9, 9, 9), value=2.0)
DEFAULT_MASKS = {'ptv': box((4, 4, 4), (8, 8, 8)), 'bladder': box((5, 5, 5), (8, 8, 8))}


def series_row(file, kind, **extra):
    return dict.fromkeys(conv.SERIES_COLUMNS, '') | dict(file=file, kind=kind, **extra)


def write_plan(root, patient='SYNTHETICPTA', plan='plan_1', n=12, spacing=1.0, dose=DEFAULT_DOSE, masks=None, applicator=True):
    """<root>/<patient>/<plan>/ as F2a writes it. Returns (folder, series.csv rows)."""
    masks = DEFAULT_MASKS if masks is None else masks
    geometry = grid(n, spacing)
    folder = root / patient / plan
    folder.mkdir(parents=True)
    ct = make_image(np.linspace(-1000, 200, n ** 3).reshape(n, n, n), **geometry)
    sitk.WriteImage(ct, str(folder / conv.CT_NAME))
    rows = [series_row(conv.CT_NAME, 'ct', modality='CT', series_instance_uid='SYNTHETICCTUID', rtplan_label='SYNTHETICPLAN')]
    if dose is not None:
        sitk.WriteImage(make_image(dose, **geometry), str(folder / conv.DOSE_NAME))
        rows.append(series_row(conv.DOSE_NAME, 'dose', modality='RTDOSE', dose_summation_type='PLAN'))
    for name, array in masks.items():
        sitk.WriteImage(sitk.Cast(make_image(array, **geometry), sitk.sitkUInt8), str(folder / f'{name}.nii.gz'))
        rows.append(series_row(f'{name}.nii.gz', 'mask', modality='RTSTRUCT', roi_name=name))
    if applicator:
        (folder / 'applicator_a.txt').write_bytes(APPLICATOR_BYTES)
        rows.append(series_row('applicator_a.txt', 'applicator', modality='RTSTRUCT', roi_name='applicator_a'))
    save_rows(folder, rows)
    return folder, rows


def save_rows(folder, rows):
    conv.write_series_csv(rows, folder / conv.SERIES_NAME)


def state(root):
    """Every file under root with its bytes: a before/after comparison."""
    return {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file()}


def report_rows(out):
    with open(out / 'preprocess_report.csv', newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ['patient', 'folder', 'outcome', 'reason', 'files']
        return list(reader)


def plan_row(report, patient='SYNTHETICPTA', folder='plan_1'):
    return next(r for r in report.rows if (r.patient, r.folder) == (patient, folder))


def test_preset_writes_the_prefixed_files_and_leaves_the_input_alone(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    before = state(converted)
    report = pre.preprocess(converted, out, SMALL)
    assert isinstance(report, pre.PreprocessReport)
    expected = [CT_FILE, DOSE_FILE, PTV_FILE, BLADDER_FILE, 'applicator_a.txt', 'series.csv']
    row = plan_row(report)
    assert (row.outcome, row.reason, row.files) == ('preprocessed', '', expected) and len(report.rows) == 1
    assert sorted(p.name for p in (out / 'SYNTHETICPTA' / 'plan_1').iterdir()) == sorted(expected)
    assert sorted(p.name for p in out.iterdir()) == ['SYNTHETICPTA', 'preprocess_report.csv']  # no partial left
    assert report_rows(out) == [dict(patient='SYNTHETICPTA', folder='plan_1', outcome='preprocessed', reason='', files=';'.join(expected))]
    done = out / 'SYNTHETICPTA' / 'plan_1'
    assert (done / 'applicator_a.txt').read_bytes() == APPLICATOR_BYTES
    final = sitk.ReadImage(str(done / CT_FILE))
    assert final.GetSize() == (10, 10, 10) and final.GetSpacing() == (1.25, 1.25, 1.25)
    assert np.array_equal(sitk.GetArrayFromImage(final), sitk.GetArrayFromImage(pre.process_ct(sitk.ReadImage(str(folder / conv.CT_NAME)), SMALL)))
    for name in expected[1:4]:
        image = sitk.ReadImage(str(done / name))
        assert (image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection()) == (
            final.GetSize(), final.GetOrigin(), final.GetSpacing(), final.GetDirection())
    assert sitk.ReadImage(str(done / PTV_FILE)).GetPixelID() == sitk.sitkUInt8
    assert state(converted) == before


def test_rescale_only_renames_the_ct_and_copies_dose_and_masks_byte_equal(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, _ = write_plan(converted)
    steps = PreprocessSteps(rescale=True, legacy_linear_masks=True)  # linear masks have no effect without a geometric step
    report = pre.preprocess(converted, out, steps)
    done = out / 'SYNTHETICPTA' / 'plan_1'
    assert plan_row(report).files == ['rescale_planning_ct.nii.gz', 'dose.nii.gz', 'ptv.nii.gz', 'bladder.nii.gz', 'applicator_a.txt', 'series.csv']
    for name in ('dose.nii.gz', 'ptv.nii.gz', 'bladder.nii.gz', 'applicator_a.txt'):
        assert (done / name).read_bytes() == (folder / name).read_bytes()
    hu = sitk.GetArrayFromImage(sitk.ReadImage(str(folder / conv.CT_NAME)))
    assert np.abs(sitk.GetArrayFromImage(sitk.ReadImage(str(done / 'rescale_planning_ct.nii.gz'))) - (hu + 1024) / 4096).max() < 1e-6


def test_series_csv_keeps_every_column_renames_files_and_drops_series_rows(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    series = series_row('series_MR_1.nii.gz', 'series', modality='MR')  # its file is not in the folder: series rows are not read
    save_rows(folder, [*rows[:2], series, *rows[2:]])
    pre.preprocess(converted, out, SMALL)
    with open(out / 'SYNTHETICPTA' / 'plan_1' / 'series.csv', newline='') as handle:
        written = list(csv.DictReader(handle))
    names = [CT_FILE, DOSE_FILE, PTV_FILE, BLADDER_FILE, 'applicator_a.txt']
    assert [r['file'] for r in written] == names
    assert [{k: v for k, v in r.items() if k != 'file'} for r in written] == [{k: v for k, v in r.items() if k != 'file'} for r in rows]


def test_a_plan_without_a_dose_writes_no_dose(tmp_path):
    write_plan(tmp_path / 'converted', dose=None, applicator=False)
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', SMALL)
    assert plan_row(report).files == [CT_FILE, PTV_FILE, BLADDER_FILE, 'series.csv']


def test_the_walk_skips_images_hidden_and_folders_without_a_series_csv(tmp_path):
    converted = tmp_path / 'converted'
    write_plan(converted, plan='plan_2')
    write_plan(converted, plan='plan_10')
    write_plan(converted, patient='.hidden')
    (converted / 'SYNTHETICPTA' / 'images').mkdir()
    (converted / 'SYNTHETICPTA' / 'images' / 'series.csv').write_text('file,kind\n')
    (converted / 'SYNTHETICPTA' / 'plan_3').mkdir()
    report = pre.preprocess(converted, tmp_path / 'out', SMALL)
    assert [(r.patient, r.folder, r.outcome) for r in report.rows] == [
        ('SYNTHETICPTA', 'plan_2', 'preprocessed'), ('SYNTHETICPTA', 'plan_10', 'preprocessed')]
    assert sorted(p.name for p in (tmp_path / 'out' / 'SYNTHETICPTA').iterdir()) == ['plan_10', 'plan_2']


def test_other_kinds_refuse_the_plan(tmp_path):
    folder, rows = write_plan(tmp_path / 'converted')
    save_rows(folder, [*rows, series_row('extra.nii.gz', 'surprise')])
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', SMALL)
    assert (plan_row(report).outcome, plan_row(report).reason) == ('refused', pre.REASON_KIND)
    assert not (tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1').exists()


@pytest.mark.parametrize('value', ['../outside.nii.gz', 'sub/ptv.nii.gz', '..', 'ABSOLUTE', 'sub', 'LINK'])
def test_a_file_value_outside_the_plan_folder_counts_as_missing(tmp_path, value, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    sitk.WriteImage(sitk.Cast(make_image(box((4, 4, 4), (8, 8, 8)), **grid()), sitk.sitkUInt8), str(converted / 'SYNTHETICPTA' / 'outside.nii.gz'))
    (folder / 'sub').mkdir()
    sitk.WriteImage(sitk.Cast(make_image(box((4, 4, 4), (8, 8, 8)), **grid()), sitk.sitkUInt8), str(folder / 'sub' / 'ptv.nii.gz'))
    (folder / 'link.nii.gz').symlink_to(converted / 'SYNTHETICPTA' / 'outside.nii.gz')  # a regular file only: not a link out of the folder
    value = {'ABSOLUTE': str(converted / 'SYNTHETICPTA' / 'outside.nii.gz'), 'LINK': 'link.nii.gz'}.get(value, value)
    save_rows(folder, [*rows, series_row(value, 'mask')])
    before = state(converted)  # the link reads through to outside.nii.gz, which is in the snapshot
    seen = []
    for name in ('read_image', 'read_info'):
        real = getattr(pre, name)
        monkeypatch.setattr(pre, name, lambda path, real=real: (seen.append(path), real(path))[1])
    report = pre.preprocess(converted, out, SMALL)
    assert (plan_row(report).outcome, plan_row(report).reason) == ('refused', pre.REASON_FILE_MISSING)
    assert all(path.parent == folder for path in seen)  # nothing outside the plan folder was opened
    assert state(converted) == before
    assert not (out / 'SYNTHETICPTA' / 'plan_1').exists() and not any(p.name.startswith('outside') for p in out.rglob('*'))


def test_out_root_in_or_equal_to_the_input_raises_before_any_write(tmp_path):
    converted = tmp_path / 'converted'
    write_plan(converted)
    before = state(converted)
    for out in (converted, converted / 'sub', converted / 'sub' / 'deeper'):
        with pytest.raises(ValueError) as caught:
            pre.preprocess(converted, out, SMALL)
        assert str(caught.value) == pre.REASON_OUTSIDE_INPUT
    with pytest.raises(ValueError) as caught:  # checked first: this one has no step on either
        pre.preprocess(converted, converted / 'sub', PreprocessSteps())
    assert str(caught.value) == pre.REASON_OUTSIDE_INPUT
    assert state(converted) == before and not (converted / 'sub').exists()


def test_no_step_on_raises_before_any_write_even_with_linear_masks(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    for steps in (PreprocessSteps(), PreprocessSteps(legacy_linear_masks=True)):
        with pytest.raises(ValueError) as caught:
            pre.preprocess(converted, out, steps)
        assert str(caught.value) == pre.REASON_NO_STEPS and not out.exists()


def test_an_existing_patient_is_skipped_and_a_rerun_skips_everything(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    write_plan(converted, patient='SYNTHETICPTB')
    (out / 'SYNTHETICPTA').mkdir(parents=True)
    report = pre.preprocess(converted, out, SMALL)
    assert [(r.patient, r.folder, r.outcome, r.reason) for r in report.rows] == [
        ('SYNTHETICPTA', '', 'skipped', pre.REASON_EXISTS), ('SYNTHETICPTB', 'plan_1', 'preprocessed', '')]
    assert not any((out / 'SYNTHETICPTA').iterdir())
    again = pre.preprocess(converted, out, SMALL)
    assert [(r.folder, r.outcome, r.reason) for r in again.rows] == [('', 'skipped', pre.REASON_EXISTS)] * 2


def test_a_patient_without_a_plan_folder_with_a_series_csv_is_skipped_and_writes_nothing(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    (converted / 'SYNTHETICPTA' / 'images').mkdir(parents=True)
    (converted / 'SYNTHETICPTA' / 'plan_1').mkdir()
    report = pre.preprocess(converted, out, SMALL)
    assert [(r.patient, r.folder, r.outcome, r.reason, r.files) for r in report.rows] == [
        ('SYNTHETICPTA', '', 'skipped', pre.REASON_NO_PLANS, [])]
    assert sorted(p.name for p in out.iterdir()) == ['preprocess_report.csv']


def refuse(tmp_path, reason, change):
    """Build a plan, let `change(folder, rows)` break it, and assert the plan is refused with `reason` and leaves no folder."""
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    write_plan(converted, plan='plan_2')
    change(folder, rows)
    report = pre.preprocess(converted, out, SMALL)
    assert (plan_row(report).outcome, plan_row(report).reason) == ('refused', reason)
    assert not (out / 'SYNTHETICPTA' / 'plan_1').exists()
    assert plan_row(report, folder='plan_2').outcome == 'preprocessed' and (out / 'SYNTHETICPTA' / 'plan_2' / CT_FILE).is_file()
    assert [r for r in report.rows if r.folder == ''] == []


def test_zero_or_several_ct_rows_refuse(tmp_path):
    refuse(tmp_path / 'a', pre.REASON_NO_CT, lambda folder, rows: save_rows(folder, rows[1:]))
    refuse(tmp_path / 'b', pre.REASON_NO_CT, lambda folder, rows: save_rows(folder, [*rows, series_row('ptv.nii.gz', 'ct')]))


def test_several_dose_rows_refuse(tmp_path):
    refuse(tmp_path, pre.REASON_DOSE_COUNT, lambda folder, rows: save_rows(folder, [*rows, series_row('dose.nii.gz', 'dose')]))


def test_a_row_naming_a_missing_file_refuses(tmp_path):
    refuse(tmp_path, pre.REASON_FILE_MISSING, lambda folder, rows: (folder / 'ptv.nii.gz').unlink())


def test_an_unreadable_nifti_refuses_for_every_nifti_kind(tmp_path):
    for i, name in enumerate(('planning_ct.nii.gz', 'dose.nii.gz', 'bladder.nii.gz')):
        refuse(tmp_path / str(i), pre.REASON_FILE_UNREADABLE, lambda folder, rows, name=name: (folder / name).write_bytes(b'not an image'))


GRID_BREAKS = {
    'size': dict(),
    'origin': dict(origin=(-5.49, -5.5, -5.5)),
    'spacing': dict(spacing=(1.01, 1.0, 1.0)),
    'direction': dict(direction=ROTATED),
}


@pytest.mark.parametrize('kind', ['dose.nii.gz', 'ptv.nii.gz'])
@pytest.mark.parametrize('what', list(GRID_BREAKS))
def test_a_dose_or_mask_off_the_native_ct_grid_refuses(tmp_path, kind, what):
    size = (12, 12, 13) if what == 'size' else (12, 12, 12)
    change = lambda folder, rows: sitk.WriteImage(blank(size, **{**grid(), **GRID_BREAKS[what]}), str(folder / kind))  # noqa: E731
    refuse(tmp_path, pre.REASON_GRID, change)


def test_a_grid_difference_inside_the_tolerance_is_accepted(tmp_path):
    folder, _ = write_plan(tmp_path / 'converted')
    sitk.WriteImage(make_image(DEFAULT_DOSE, **{**grid(), 'origin': (-5.5 + 5e-4, -5.5, -5.5)}), str(folder / 'dose.nii.gz'))
    assert plan_row(pre.preprocess(tmp_path / 'converted', tmp_path / 'out', SMALL)).outcome == 'preprocessed'


def test_a_refusal_on_the_last_mask_happens_before_the_first_write(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, _ = write_plan(converted)
    (folder / 'bladder.nii.gz').write_bytes(b'not an image')  # the last mask row
    writes = []
    real = conv.write_atomic
    monkeypatch.setattr(conv, 'write_atomic', lambda image, path: (writes.append(path), real(image, path))[1])
    report = pre.preprocess(converted, out, SMALL)
    assert plan_row(report).reason == pre.REASON_FILE_UNREADABLE and writes == []
    assert not (out / 'SYNTHETICPTA' / 'plan_1').exists()


def test_a_mask_is_written_before_the_next_one_is_read(tmp_path, monkeypatch):
    converted = tmp_path / 'converted'
    write_plan(converted)
    events = []
    real_read, real_write = pre.read_image, conv.write_atomic
    monkeypatch.setattr(pre, 'read_image', lambda path: (events.append('r'), real_read(path))[1])
    monkeypatch.setattr(conv, 'write_atomic', lambda image, path: (events.append('w'), real_write(image, path))[1])
    pre.preprocess(converted, tmp_path / 'out', SMALL)
    assert events == ['r', 'w'] * 4  # CT, dose, two masks: each written before the next is read


def crop_case(tmp_path, **plan):
    """A plan whose objects sit in the corner voxels, cut off by new_shape (4, 4, 4); the final grid covers +-2.5 mm."""
    small = dataclasses.replace(LEGACY_RNT, new_shape=(4, 4, 4))
    write_plan(tmp_path / 'converted', **plan)
    write_plan(tmp_path / 'converted', plan='plan_2', dose=box((5, 5, 5), (7, 7, 7), value=2.0), masks={'ptv': box((5, 5, 5), (7, 7, 7))})
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', small)
    assert (plan_row(report).outcome, plan_row(report).reason) == ('refused', pre.REASON_CROP)
    assert not (tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1').exists()
    assert plan_row(report, folder='plan_2').outcome == 'preprocessed'  # the sibling inside the crop is kept


def test_a_dose_cut_off_by_the_shape_refuses(tmp_path):
    crop_case(tmp_path, dose=box((0, 0, 0), (3, 3, 3), value=2.0), masks={'ptv': box((5, 5, 5), (7, 7, 7))})


def test_a_mask_cut_off_by_the_shape_refuses(tmp_path):
    # the last cut-off voxel centre sits at 3.5 mm: continuous index 4.3 on the 4-voxel axis, beyond the buffer edge 3.5
    crop_case(tmp_path, dose=box((5, 5, 5), (7, 7, 7), value=2.0), masks={'ptv': box((5, 5, 5), (7, 7, 7)), 'bladder': box((5, 5, 5), (10, 7, 7))})


def test_a_one_voxel_mask_that_nearest_neighbour_empties_refuses(tmp_path):
    # 24^3 at 0.5 mm: the final centres sit on native indices 1.5, 4, 6.5, 9, 11.5, ... so index 10 is never the nearest
    one = np.zeros((24, 24, 24))
    one[10, 10, 10] = 1
    write_plan(tmp_path / 'converted', n=24, spacing=0.5, dose=None, masks={'ptv': one}, applicator=False)
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', SMALL)
    assert (plan_row(report).outcome, plan_row(report).reason) == ('refused', pre.REASON_CROP)
    assert not (tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1').exists()


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize(('index', 'refused'), [(-0.4, False), (-0.6, True), (9.4, False), (9.6, True)])
def test_check_crop_refuses_exactly_outside_half_a_voxel_on_both_sides(axis, index, refused):
    final = blank((10, 10, 10))
    origin = [4.0, 4.0, 4.0]
    origin[axis] = index  # the one nonzero native voxel's centre lands at this continuous index of the final grid
    array = np.zeros((1, 1, 1))
    array[0, 0, 0] = 1
    native = make_image(array, origin=tuple(origin))
    if refused:
        with pytest.raises(CaseRefused, match=pre.REASON_CROP):
            pre.check_crop(native, final)
    else:
        pre.check_crop(native, final)


def test_an_oserror_while_writing_refuses_the_patient_and_removes_the_partial(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    write_plan(converted, plan='plan_2')
    write_plan(converted, patient='SYNTHETICPTB')
    real = conv.write_atomic

    def write(image, path):
        if 'SYNTHETICPTA' in path.parts[-3] and path.parts[-2] == 'plan_2':
            raise OSError(conv.REASON_WRITE)
        real(image, path)
    monkeypatch.setattr(conv, 'write_atomic', write)
    report = pre.preprocess(converted, out, SMALL)
    assert [(r.patient, r.folder, r.outcome, r.reason) for r in report.rows] == [
        ('SYNTHETICPTA', '', 'refused', pre.REASON_IO_ERROR), ('SYNTHETICPTB', 'plan_1', 'preprocessed', '')]
    assert sorted(p.name for p in out.iterdir()) == ['SYNTHETICPTB', 'preprocess_report.csv']


def test_a_compute_failure_aborts_with_a_flushed_report_and_nothing_chained(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    write_plan(converted, patient='SYNTHETICPTB')

    def boom(image, spacing):
        raise RuntimeError('SYNTHETICPTA')
    monkeypatch.setattr(pre, 'resample_to', boom)
    with pytest.raises(pre.PreprocessAborted) as caught:
        pre.preprocess(converted, out, SMALL)
    assert str(caught.value) == pre.REASON_INTERNAL
    chain = ''.join(traceback.format_exception(caught.value))
    assert 'direct cause' not in chain and 'During handling' not in chain  # nothing chained
    assert not issubclass(pre.PreprocessAborted, (ValueError, OSError))
    assert report_rows(out) == [dict(patient='SYNTHETICPTA', folder='', outcome='refused', reason=pre.REASON_INTERNAL, files='')]
    assert sorted(p.name for p in out.iterdir()) == ['preprocess_report.csv']  # no partial, and the next patient was not started


def test_running_out_of_memory_reading_a_nifti_aborts_instead_of_refusing_the_plan(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    write_plan(converted, patient='SYNTHETICPTB')

    def no_memory(path):
        raise RuntimeError('SYNTHETICPTA:179:\nFailed to allocate memory for image.')  # what ITK raises
    monkeypatch.setattr(sitk, 'ReadImage', no_memory)
    with pytest.raises(pre.PreprocessAborted):
        pre.preprocess(converted, out, SMALL)
    assert report_rows(out) == [dict(patient='SYNTHETICPTA', folder='', outcome='refused', reason=pre.REASON_INTERNAL, files='')]
    assert sorted(p.name for p in out.iterdir()) == ['preprocess_report.csv']


def test_each_patient_is_built_under_a_partial_name_and_renamed(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    stale = out / '.SYNTHETICPTA.partial'
    stale.mkdir(parents=True)
    (stale / 'junk').write_text('left by a dead run')
    seen = []
    real = conv.write_atomic
    monkeypatch.setattr(conv, 'write_atomic', lambda image, path: (seen.append(path.parts[-3]), real(image, path))[1])
    pre.preprocess(converted, out, SMALL)
    assert set(seen) == {'.SYNTHETICPTA.partial'}
    assert not stale.exists() and not (out / 'SYNTHETICPTA' / 'junk').exists() and (out / 'SYNTHETICPTA' / 'plan_1').is_dir()


def test_a_patient_whose_plans_are_all_refused_is_renamed_and_a_rerun_skips_it(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, _ = write_plan(converted)
    (folder / 'ptv.nii.gz').unlink()
    report = pre.preprocess(converted, out, SMALL)
    assert plan_row(report).reason == pre.REASON_FILE_MISSING
    assert (out / 'SYNTHETICPTA').is_dir() and not any((out / 'SYNTHETICPTA').iterdir())
    assert [(r.folder, r.outcome, r.reason) for r in pre.preprocess(converted, out, SMALL).rows] == [('', 'skipped', pre.REASON_EXISTS)]


def test_no_log_line_carries_a_path_a_folder_name_or_a_series_value(tmp_path, monkeypatch, caplog):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    folder, _ = write_plan(converted, plan='plan_2')
    (folder / 'ptv.nii.gz').write_bytes(b'not an image')
    write_plan(converted, patient='SYNTHETICPTB')
    with caplog.at_level('DEBUG'):
        pre.preprocess(converted, out, SMALL)
        monkeypatch.setattr(pre, 'resample_to', lambda image, spacing: 1 / 0)
        with pytest.raises(pre.PreprocessAborted):
            pre.preprocess(converted, tmp_path / 'out2', SMALL)
    assert caplog.records and not any(s in caplog.text for s in ('SYNTHETICPT', str(tmp_path), 'ptv', 'SYNTHETICCTUID', 'SYNTHETICPLAN'))


def test_reasons_hold_no_separator_digit_run_or_uid():
    reasons = {name: value for name, value in vars(pre).items() if name.startswith('REASON_')}
    assert {'REASON_OUTSIDE_INPUT', 'REASON_NO_STEPS', 'REASON_EXISTS', 'REASON_NO_PLANS', 'REASON_NO_CT', 'REASON_DOSE_COUNT',
            'REASON_KIND', 'REASON_FILE_MISSING', 'REASON_FILE_UNREADABLE', 'REASON_GRID', 'REASON_CROP', 'REASON_IO_ERROR',
            'REASON_INTERNAL'} <= set(reasons)
    assert len(set(reasons.values())) == len(reasons)
    for value in reasons.values():
        assert not re.search(r'[/\\]|\d{3,}|\d+(\.\d+){2,}', value)


def test_convert_exposes_crash_site():
    assert callable(conv.crash_site) and not hasattr(conv, '_crash_site')


def test_matches_torchio(tmp_path):
    """A2.6: the SimpleITK chain equals torchio 1.2.1 Resample -> rescale -> Resample -> CropOrPad on a synthetic CT (rescale on)."""
    tio = pytest.importorskip('torchio')
    # native 11x9x7, anisotropic: 1.5 mm gives 8x6x9, 1.25 mm gives 9x7x10; new_shape (14, 4, 13) pads x by 3+2, crops y by 2+1, pads z by 2+1
    rng = np.random.default_rng(7)
    ct = make_image(rng.uniform(-1000, 1000, (7, 9, 11)), spacing=(1.2, 1.0, 2.0), origin=(-5.0, 3.0, 40.0))
    mask_array = np.zeros((7, 9, 11), np.uint8)
    mask_array[2:5, 3:7, 4:9] = 1
    mask_array[3, 4, 5] = 2
    mask = sitk.Cast(make_image(mask_array, spacing=ct.GetSpacing(), origin=ct.GetOrigin()), sitk.sitkUInt8)
    sitk.WriteImage(ct, str(tmp_path / 'ct.nii.gz'))
    sitk.WriteImage(mask, str(tmp_path / 'mask.nii.gz'))
    steps = dataclasses.replace(LEGACY_RNT, new_spacing=(1.5, 1.5, 1.5), new_shape=(14, 4, 13))

    final = pre.process_ct(pre.read_image(tmp_path / 'ct.nii.gz'), steps)
    mine_mask = pre.process_mask(pre.read_image(tmp_path / 'mask.nii.gz'), final, steps)
    sitk.WriteImage(final, str(tmp_path / 'mine_ct.nii.gz'))
    sitk.WriteImage(mine_mask, str(tmp_path / 'mine_mask.nii.gz'))

    subject = tio.Subject(ct=tio.ScalarImage(tmp_path / 'ct.nii.gz'))
    subject = tio.Resample((1.5, 1.5, 1.5), antialias=False)(subject)
    subject['ct'].set_data((subject['ct'].data + 1024) / 4096)  # the rescale step, between the two resamples as in legacy
    subject = tio.Resample((1.25, 1.25, 1.25), antialias=False)(subject)
    pad = float(subject['ct'].data.min())  # the minimum of the CT entering reshape
    subject = tio.CropOrPad((14, 4, 13), padding_mode=pad)(subject)
    subject['ct'].save(tmp_path / 'tio_ct.nii.gz')
    tio_final = tio.ScalarImage(tmp_path / 'tio_ct.nii.gz')  # the final CT image, the mask's reference grid
    mask_subject = tio.Subject(mask=tio.LabelMap(tmp_path / 'mask.nii.gz'))
    mask_subject = tio.Resample(target=tio_final, antialias=False)(mask_subject)  # no CropOrPad on the mask
    mask_subject['mask'].save(tmp_path / 'tio_mask.nii.gz')

    mine, theirs = (sitk.ReadImage(str(tmp_path / name)) for name in ('mine_ct.nii.gz', 'tio_ct.nii.gz'))
    assert mine.GetSize() == theirs.GetSize() == (14, 4, 13)
    assert mine.GetSpacing() == pytest.approx(theirs.GetSpacing(), abs=1e-6)
    assert mine.GetOrigin() == pytest.approx(theirs.GetOrigin(), abs=1e-4)
    assert np.abs(sitk.GetArrayFromImage(mine) - sitk.GetArrayFromImage(theirs)).max() < 1e-4
    mine_labels, tio_labels = (sitk.ReadImage(str(tmp_path / name)) for name in ('mine_mask.nii.gz', 'tio_mask.nii.gz'))
    assert mine_labels.GetSize() == tio_labels.GetSize() == mine.GetSize()
    assert np.array_equal(sitk.GetArrayFromImage(mine_labels), sitk.GetArrayFromImage(tio_labels).astype(np.uint8))
    assert sitk.GetArrayFromImage(mine_labels).any()
