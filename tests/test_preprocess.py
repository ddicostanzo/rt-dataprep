"""preprocess: grid, crop/pad, rescale, names, CT pipeline, dose and mask resampling (task-01); the tree walk (task-02)."""
import ast
import csv
import dataclasses
import inspect
import re
import traceback
import weakref

import numpy as np
import pytest
import SimpleITK as sitk

from rt_dataprep import convert as conv
from rt_dataprep import preprocess as pre
from rt_dataprep.dicom_io import CaseRefused
from rt_dataprep.preprocess import LEGACY_RNT, PreprocessSteps

ROTATED = (0.8660254037844387, -0.5, 0.0, 0.5, 0.8660254037844387, 0.0, 0.0, 0.0, 1.0)  # 30 degrees about z
QUARTER = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # 90 degrees about z: D @ (a, b, c) = (-b, a, c)

def listing(root):
    """Sorted names under root, without the dated per-run copies of the report."""
    return sorted(p.name for p in root.iterdir() if not re.fullmatch(r'\w+_report_\d{8}T\d{6}Z\.csv', p.name))



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
        assert reader.fieldnames == ['patient', 'folder', 'outcome', 'reason', 'files', 'skipped_series']
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
    assert listing(out) == ['SYNTHETICPTA', 'preprocess_report.csv']  # no partial left
    assert report_rows(out) == [dict(patient='SYNTHETICPTA', folder='plan_1', outcome='preprocessed', reason='', files=';'.join(expected), skipped_series='0')]
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


def test_series_csv_keeps_every_column_and_renames_files_and_lists_the_aligned_series_last(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    sitk.WriteImage(make_image(np.ones((4, 4, 4))), str(folder / 'series_MR_1.nii.gz'))  # present, as check_rows now requires (task-02)
    series = series_row('series_MR_1.nii.gz', 'series', modality='MR', reg_matrix='identity')  # an empty matrix would leave it out
    save_rows(folder, [*rows[:2], series, *rows[2:]])
    pre.preprocess(converted, out, SMALL)
    with open(out / 'SYNTHETICPTA' / 'plan_1' / 'series.csv', newline='') as handle:
        written = list(csv.DictReader(handle))
    names = [CT_FILE, DOSE_FILE, PTV_FILE, BLADDER_FILE, 'applicator_a.txt', pre.output_name('series_MR_1.nii.gz', SMALL)]
    assert [r['file'] for r in written] == names
    assert [{k: v for k, v in r.items() if k != 'file'} for r in written] == [{k: v for k, v in r.items() if k != 'file'} for r in [*rows, series]]


def add_series(folder, rows, file='series_MR_1.nii.gz', modality='MR', write=True, **extra):
    """Append a kind=series row (and a small image file, unless write is False) to a write_plan plan; returns the row."""
    if write:
        sitk.WriteImage(make_image(np.ones((4, 4, 4))), str(folder / file))
    row = series_row(file, 'series', modality=modality, reg_matrix='identity', **extra)
    save_rows(folder, [*rows, row])
    return row


def test_check_rows_returns_kept_ct_and_the_series_rows_in_input_order(tmp_path):
    folder, rows = write_plan(tmp_path)
    for k in (2, 1):  # not sorted: input order must survive
        rows = [*rows, add_series(folder, rows, f'series_MR_{k}.nii.gz')]
    kept, ct, series = pre.check_rows(folder)
    assert [r['file'] for r in kept] == [conv.CT_NAME, conv.DOSE_NAME, 'ptv.nii.gz', 'bladder.nii.gz', 'applicator_a.txt']
    assert [r['file'] for r in series] == ['series_MR_2.nii.gz', 'series_MR_1.nii.gz'] and all(r['kind'] == 'series' for r in series)
    assert ct[0] == (12, 12, 12)


def test_a_series_row_naming_a_missing_file_refuses_with_file_missing(tmp_path):
    refuse(tmp_path, pre.REASON_FILE_MISSING, lambda folder, rows: add_series(folder, rows, write=False))


@pytest.mark.parametrize('file, modality, reason', [
    ('series_MR_1.nii.gz', 'RTDOSE', pre.REASON_STUDY_MODALITY),
    ('series_MR_1.nii.gz', 'mr', pre.REASON_STUDY_MODALITY),
    ('series_MR_1.nii.gz', '', pre.REASON_STUDY_MODALITY),
    ('series_CT_1.nii.gz', 'MR', pre.REASON_STUDY_MODALITY),  # the name carries a different MOD than the modality column
    ('mr_image.nii.gz', 'MR', pre.REASON_STUDY_NAME),
    ('series_MR_1.nii', 'MR', pre.REASON_STUDY_NAME),
    ('../series_MR_1.nii.gz', 'MR', pre.REASON_STUDY_NAME),
])
def test_a_bad_series_row_refuses_the_plan(tmp_path, file, modality, reason):
    refuse(tmp_path, reason, lambda folder, rows: add_series(folder, rows, file, modality, write=False))


def test_a_series_file_with_step_prefixes_passes_check_rows(tmp_path):
    folder, rows = write_plan(tmp_path)
    add_series(folder, rows, 'reshape_resample_initialresample_series_MR_1.nii.gz')
    assert [r['file'] for r in pre.check_rows(folder)[2]] == ['reshape_resample_initialresample_series_MR_1.nii.gz']


def test_check_rows_reads_no_series_header(tmp_path):
    folder, rows = write_plan(tmp_path)
    add_series(folder, rows, write=False)
    (folder / 'series_MR_1.nii.gz').write_bytes(b'SYNTHETIC not a nifti')
    assert len(pre.check_rows(folder)[2]) == 1


def test_the_study_reasons_say_a_series_csv_series_row():
    for reason in (pre.REASON_STUDY_MODALITY, pre.REASON_STUDY_NAME):
        assert 'series.csv series row' in reason and 'study' not in reason


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
        ('SYNTHETICPTA', '', 'skipped', pre.REASON_NO_CASES, [])]
    assert listing(out) == ['preprocess_report.csv']


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



def rewrite_csv(folder, change):
    path = folder / 'series.csv'
    path.write_text(change(path.read_text()))


SERIES_CSV_BREAKS = {
    'missing column': lambda folder, rows: rewrite_csv(folder, lambda text: text.replace('file,kind,', 'file,', 1)),
    'extra column': lambda folder, rows: rewrite_csv(folder, lambda text: text.replace('\n', ',extra\n', 1)),
    'long row': lambda folder, rows: rewrite_csv(folder, lambda text: text.rstrip('\n') + ',extra\n'),
    'short row': lambda folder, rows: rewrite_csv(folder, lambda text: text + 'ptv.nii.gz,mask\n'),
    'not utf-8': lambda folder, rows: (folder / 'series.csv').write_bytes((folder / 'series.csv').read_bytes() + b'\xff\xfe,mask\n'),
    'one file twice': lambda folder, rows: save_rows(folder, [*rows, rows[-1]]),
}


@pytest.mark.parametrize('what', list(SERIES_CSV_BREAKS))
def test_a_malformed_series_csv_refuses_the_plan_only(tmp_path, what):
    refuse(tmp_path, pre.REASON_SERIES_CSV, SERIES_CSV_BREAKS[what])


def test_a_ct_shorter_than_one_output_voxel_refuses_the_plan_only(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted, n=12, spacing=0.05)  # 12 x 0.05 mm = 0.6 mm < 1 mm on every axis
    write_plan(converted, plan='plan_2')
    report = pre.preprocess(converted, out, SMALL)
    assert (plan_row(report).outcome, plan_row(report).reason) == ('refused', pre.REASON_TOO_SMALL)
    assert plan_row(report, folder='plan_2').outcome == 'preprocessed'


@pytest.mark.parametrize('resample', [False, True])  # True: 2 voxels after a 0.5 mm resample, none at 1.25 mm (#55)
def test_a_ct_that_only_harmonize_spacing_would_empty_refuses(tmp_path, resample):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted, n=2, spacing=0.6, dose=None, masks={})  # 1.2 mm per axis: none at 1.25 mm
    # an axis at 1 voxel stays 1 (torchio), so only a finer initial spacing can survive it and then empty
    report = pre.preprocess(converted, out, dataclasses.replace(SMALL, resample=resample, new_spacing=(0.5, 0.5, 0.5)))
    assert (plan_row(report).outcome, plan_row(report).reason) == ('refused', pre.REASON_TOO_SMALL)


def test_the_ct_is_processed_first_whatever_its_series_csv_row(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    save_rows(folder, [*rows[1:], rows[0]])
    assert plan_row(pre.preprocess(converted, out, SMALL)).outcome == 'preprocessed'


def test_an_integer_ct_is_interpolated_in_float32(tmp_path):
    ct = sitk.Cast(make_image(np.indices((6, 6, 6)).sum(axis=0) % 2), sitk.sitkInt16)  # a 0/1 checkerboard
    image = pre.resample_to(ct, (0.5, 0.5, 0.5))
    values = sitk.GetArrayViewFromImage(image)
    assert image.GetPixelID() == sitk.sitkFloat32
    assert ((values > 0) & (values < 1)).any()  # truncated to int16 there would be none


def test_a_refused_plan_folder_that_cannot_be_removed_drops_the_patient(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    real = pre.shutil.rmtree

    def stuck(path, *args, **kwargs):
        if path.name == 'plan_1':  # fails; ignore_errors would hide it and leave the folder
            if kwargs.get('ignore_errors'):
                return None
            raise OSError('SYNTHETICPTA')
        return real(path, *args, **kwargs)

    def crop(native, final):
        raise CaseRefused(pre.REASON_CROP)
    monkeypatch.setattr(pre.shutil, 'rmtree', stuck)
    monkeypatch.setattr(pre, 'check_crop', crop)
    report = pre.preprocess(converted, out, SMALL)
    assert [(r.folder, r.reason) for r in report.rows] == [('', pre.REASON_IO_ERROR)]
    assert listing(out) == ['preprocess_report.csv']


def test_a_missing_converted_root_writes_no_output_root(tmp_path):
    with pytest.raises(OSError):
        pre.preprocess(tmp_path / 'missing', tmp_path / 'out', SMALL)
    assert not (tmp_path / 'out').exists()

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


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize(('index', 'refused'), [(-0.5, False), (9.5, True)])  # the continuous index range is [-0.5, n-0.5)
def test_check_crop_on_the_exact_half_voxel_boundary_keeps_the_low_edge_and_refuses_the_high_edge(axis, index, refused):
    final = blank((10, 10, 10))
    origin = [4.0, 4.0, 4.0]
    origin[axis] = index
    native = make_image(np.ones((1, 1, 1)), origin=tuple(origin))
    if refused:
        with pytest.raises(CaseRefused, match=pre.REASON_CROP):
            pre.check_crop(native, final)
    else:
        pre.check_crop(native, final)


def test_each_dose_or_mask_image_is_released_before_the_next_file_is_read(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)  # CT, dose, two masks, applicator
    tracked, alive_at_read = [], []
    read_image, process_dose, process_mask = pre.read_image, pre.process_dose, pre.process_mask

    def track(image):
        tracked.append(weakref.ref(image))
        return image

    def read(path):
        alive_at_read.append(sum(ref() is not None for ref in tracked))  # what earlier steps still hold when this read starts
        image = read_image(path)
        return image if len(alive_at_read) == 1 else track(image)  # the CT is the reference grid: it stays

    monkeypatch.setattr(pre, 'read_image', read)
    monkeypatch.setattr(pre, 'process_dose', lambda *args: track(process_dose(*args)))
    monkeypatch.setattr(pre, 'process_mask', lambda *args: track(process_mask(*args)))
    report = pre.preprocess(converted, out, SMALL)
    assert plan_row(report).outcome == 'preprocessed'
    assert alive_at_read == [0, 0, 0, 0]  # CT, dose, ptv, bladder: the previous native and processed images are gone
    assert sum(ref() is not None for ref in tracked) == 0  # the last mask's images are released too


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
    assert listing(out) == ['SYNTHETICPTB', 'preprocess_report.csv']


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
    assert report_rows(out) == [dict(patient='SYNTHETICPTA', folder='', outcome='refused', reason=pre.REASON_INTERNAL, files='', skipped_series='0')]
    assert listing(out) == ['preprocess_report.csv']  # no partial, and the next patient was not started


def test_running_out_of_memory_reading_a_nifti_aborts_instead_of_refusing_the_plan(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    write_plan(converted, patient='SYNTHETICPTB')

    def no_memory(path):
        raise RuntimeError('SYNTHETICPTA:179:\nFailed to allocate memory for image.')  # what ITK raises
    monkeypatch.setattr(sitk, 'ReadImage', no_memory)
    with pytest.raises(pre.PreprocessAborted):
        pre.preprocess(converted, out, SMALL)
    assert report_rows(out) == [dict(patient='SYNTHETICPTA', folder='', outcome='refused', reason=pre.REASON_INTERNAL, files='', skipped_series='0')]
    assert listing(out) == ['preprocess_report.csv']


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
    third, rows = write_plan(converted, plan='plan_3')  # a registered series, kept, and one left out
    sitk.WriteImage(make_image(np.ones((4, 4, 4))), str(third / 'series_MR_1.nii.gz'))
    sitk.WriteImage(make_image(np.ones((4, 4, 4))), str(third / 'series_MR_2.nii.gz'))
    save_rows(third, [*rows, series_row('series_MR_1.nii.gz', 'series', modality='MR', reg_matrix='identity'),
                      series_row('series_MR_2.nii.gz', 'series', modality='MR', reg_matrix='')])
    with caplog.at_level('DEBUG'):
        pre.preprocess(converted, out, SMALL)
        monkeypatch.setattr(pre, 'resample_to', lambda image, spacing: 1 / 0)
        with pytest.raises(pre.PreprocessAborted):
            pre.preprocess(converted, tmp_path / 'out2', SMALL)
    assert caplog.records and not any(s in caplog.text for s in ('SYNTHETICPT', str(tmp_path), 'ptv', 'SYNTHETICCTUID', 'SYNTHETICPLAN', 'series_MR'))


def test_reasons_hold_no_separator_digit_run_or_uid():
    reasons = {name: value for name, value in vars(pre).items() if name.startswith('REASON_')}
    assert {'REASON_OUTSIDE_INPUT', 'REASON_NO_STEPS', 'REASON_EXISTS', 'REASON_NO_CASES', 'REASON_NO_CT', 'REASON_DOSE_COUNT',
            'REASON_KIND', 'REASON_FILE_MISSING', 'REASON_FILE_UNREADABLE', 'REASON_GRID', 'REASON_CROP', 'REASON_IO_ERROR',
            'REASON_INTERNAL'} <= set(reasons)
    assert len(set(reasons.values())) == len(reasons)
    assert 'REASON_NO_PLANS' not in reasons
    for value in reasons.values():
        assert not re.search(r'[/\\]|\d{3,}|\d+(\.\d+){2,}', value)


def test_convert_exposes_crash_site():
    assert callable(conv.crash_site) and not hasattr(conv, '_crash_site')


def rotation(z_degrees, x_degrees):
    """A direction (row-major, columns the axes) rotated about z, then about x; orthonormal."""
    z, x = np.radians(z_degrees), np.radians(x_degrees)
    about_z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    about_x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    return tuple((about_x @ about_z).ravel())


@pytest.mark.parametrize('direction, y_spacing', [
    ((1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0), 1.0), ((-1.0, 0, 0, 0, -1.0, 0, 0, 0, 1.0), 1.0), ((1.0, 0, 0, 0, 1.0, 0, 0, 0, -1.0), 1.0),
    (rotation(20, 10), 1.1)], ids=['axial', 'flipped-xy', 'flipped-z', 'oblique'])
def test_matches_torchio(tmp_path, direction, y_spacing):
    """A2.6: the SimpleITK chain equals torchio 1.2.1 Resample -> rescale -> Resample -> CropOrPad on a synthetic CT (rescale on)."""
    tio = pytest.importorskip('torchio')
    # native 11x9x7, anisotropic: 1.5 mm gives 8x6x9, 1.25 mm gives 9x7x10; new_shape (14, 4, 13) pads x by 3+2, crops y by 2+1, pads z by 2+1
    # y spacing 1.0 keeps the exact whole-number ratio (9 * 1.0 / 1.5 = 6) under test. Only oblique uses 1.1 (#100): torchio takes
    # the spacing from the affine (0.9999999999999999 on an oblique grid), so there that ratio floors one voxel lower than on
    # sitk's 1.0. The sizes above are the same for 1.1.
    rng = np.random.default_rng(7)
    ct = make_image(rng.uniform(-1000, 1000, (7, 9, 11)), spacing=(1.2, y_spacing, 2.0), origin=(-5.0, 3.0, 40.0), direction=direction)
    mask_array = np.zeros((7, 9, 11), np.uint8)
    mask_array[2:5, 3:7, 4:9] = 1
    mask_array[3, 4, 5] = 2
    mask = sitk.Cast(make_image(mask_array, spacing=ct.GetSpacing(), origin=ct.GetOrigin(), direction=direction), sitk.sitkUInt8)
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
    assert mine.GetDirection() == pytest.approx(theirs.GetDirection(), abs=1e-5)
    assert np.abs(sitk.GetArrayFromImage(mine) - sitk.GetArrayFromImage(theirs)).max() < 1e-4
    mine_labels, tio_labels = (sitk.ReadImage(str(tmp_path / name)) for name in ('mine_mask.nii.gz', 'tio_mask.nii.gz'))
    assert mine_labels.GetSize() == tio_labels.GetSize() == mine.GetSize()
    assert np.array_equal(sitk.GetArrayFromImage(mine_labels), sitk.GetArrayFromImage(tio_labels).astype(np.uint8))
    assert sitk.GetArrayFromImage(mine_labels).any()


def test_legacy_fx_folders_are_preprocessed(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted, plan='fx_1')
    report = pre.preprocess(converted, out, SMALL)
    assert plan_row(report, folder='fx_1').outcome == 'preprocessed'


def test_ctrl_c_still_writes_the_report_of_what_was_done(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    write_plan(converted, patient='SYNTHETICPTB')
    real, calls = pre.preprocess_patient, []

    def interrupt(*args):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args)

    monkeypatch.setattr(pre, 'preprocess_patient', interrupt)
    with pytest.raises(KeyboardInterrupt):
        pre.preprocess(converted, out, SMALL)
    assert [(r['patient'], r['outcome']) for r in report_rows(out)] == [('SYNTHETICPTA', 'preprocessed')]


# ---- task-03: the study series.csv validator (headers are not read; files may be empty placeholders). ----

def study_row(file, modality='MR', kind='series'):
    return {**dict.fromkeys(conv.SERIES_COLUMNS, ''), 'file': file, 'kind': kind, 'modality': modality}


def write_study(folder, rows, *, files=None):
    """A study folder holding series.csv for `rows` and an empty placeholder per file (default: every row's)."""
    folder.mkdir(parents=True, exist_ok=True)
    conv.write_series_csv(rows, folder / conv.SERIES_NAME)
    for name in (files if files is not None else [r['file'] for r in rows]):
        (folder / name).write_bytes(b'')
    return folder


def refused_with(folder):
    with pytest.raises(CaseRefused) as caught:
        pre.check_study_rows(folder)
    return str(caught.value)


def test_check_study_rows_returns_the_valid_rows(tmp_path):
    rows = [study_row('series_CT_1.nii.gz', 'CT'), study_row('reshape_resample_rescale_initialresample_series_MR_2.nii.gz'),
            study_row('rescale_series_PT_3.nii.gz', 'PT')]
    got = pre.check_study_rows(write_study(tmp_path / 'study_1', rows))
    assert got == rows and list(got[0]) == conv.SERIES_COLUMNS


def test_check_study_rows_reads_no_header(tmp_path, monkeypatch):
    folder = write_study(tmp_path / 'study_1', [study_row('series_MR_1.nii.gz')])  # an empty file is no NIfTI
    for name in ('read_info', 'read_image'):
        monkeypatch.setattr(pre, name, lambda *a: pytest.fail('a header was read'))
    monkeypatch.setattr(sitk, 'ReadImage', lambda *a, **k: pytest.fail('a header was read'))
    assert len(pre.check_study_rows(folder)) == 1


def test_series_id_is_group_five_of_a_full_match():
    assert pre.series_id('series_MR_12.nii.gz') == 'series_MR_12'
    assert pre.series_id('reshape_resample_rescale_initialresample_series_CT_3.nii.gz') == 'series_CT_3'
    assert pre.series_id('resample_initialresample_series_PT_1.nii.gz') == 'series_PT_1'
    for bad in ('series_MR_1.nii', 'xseries_MR_1.nii.gz', 'series_MR_1.nii.gz.bak', 'series_RTDOSE_1.nii.gz', 'series_MR_.nii.gz',
                'resample_reshape_series_MR_1.nii.gz', 'rescale_rescale_series_MR_1.nii.gz', 'planning_ct.nii.gz', 'series_MR_1\n.nii.gz'):
        assert pre.series_id(bad) is None


@pytest.mark.parametrize('reason, make', [
    (pre.REASON_SERIES_CSV, lambda f: (f / conv.SERIES_NAME).write_bytes(b'\xff\xfe\x00bad')),
    (pre.REASON_SERIES_CSV, lambda f: (f / conv.SERIES_NAME).write_text('file,kind\nseries_MR_1.nii.gz,series\n', encoding='utf-8')),
    (pre.REASON_SERIES_CSV, lambda f: (f / conv.SERIES_NAME).write_text(','.join(conv.SERIES_COLUMNS) + '\nseries_MR_1.nii.gz,series,MR' + ',' * 20 + '\n',
                                                                       encoding='utf-8')),
    (pre.REASON_SERIES_CSV, lambda f: (f / conv.SERIES_NAME).write_text(','.join(conv.SERIES_COLUMNS) + '\n', encoding='utf-8')),
    (pre.REASON_SERIES_CSV, lambda f: (f / conv.SERIES_NAME).write_text('', encoding='utf-8')),
    (pre.REASON_SERIES_CSV, lambda f: conv.write_series_csv([study_row('series_MR_1.nii.gz')] * 2, f / conv.SERIES_NAME)),
    (pre.REASON_SERIES_CSV, lambda f: conv.write_series_csv([study_row('series_MR_1.nii.gz'), study_row('resample_series_MR_1.nii.gz')],
                                                           f / conv.SERIES_NAME)),
    (pre.REASON_KIND, lambda f: conv.write_series_csv([study_row('series_MR_1.nii.gz'), study_row('series_MR_2.nii.gz', kind='ct')],
                                                      f / conv.SERIES_NAME)),
    (pre.REASON_KIND, lambda f: conv.write_series_csv([study_row('series_MR_1.nii.gz', kind='')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_MODALITY, lambda f: conv.write_series_csv([study_row('series_MR_1.nii.gz', 'RTDOSE')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_MODALITY, lambda f: conv.write_series_csv([study_row('series_MR_1.nii.gz', 'mr')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_MODALITY, lambda f: conv.write_series_csv([study_row('series_CT_1.nii.gz', 'MR')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_NAME, lambda f: conv.write_series_csv([study_row('series_MR_\u0661.nii.gz')], f / conv.SERIES_NAME)),  # an Arabic-Indic digit
    (pre.REASON_STUDY_NAME, lambda f: conv.write_series_csv([study_row('planning_ct.nii.gz')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_NAME, lambda f: conv.write_series_csv([study_row('series_MR_1.nii')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_NAME, lambda f: conv.write_series_csv([study_row('../series_MR_1.nii.gz')], f / conv.SERIES_NAME)),
    (pre.REASON_FILE_MISSING, lambda f: (f / 'series_MR_1.nii.gz').unlink()),
    (pre.REASON_FILE_MISSING, lambda f: ((f / 'series_MR_1.nii.gz').unlink(), (f / 'series_MR_1.nii.gz').symlink_to(f / conv.SERIES_NAME))),
    (pre.REASON_FILE_MISSING, lambda f: ((f / 'series_MR_1.nii.gz').unlink(), (f / 'series_MR_1.nii.gz').mkdir())),
], ids=['encoding', 'columns', 'long-row', 'zero-rows', 'empty-file', 'duplicate-file', 'duplicate-id', 'kind-ct', 'kind-blank',
        'modality-rtdose', 'modality-case', 'modality-vs-name', 'name-digit', 'name-other', 'name-suffix', 'name-separator', 'missing', 'symlink', 'directory'])
def test_check_study_rows_refuses(tmp_path, reason, make):
    folder = write_study(tmp_path / 'study_1', [study_row('series_MR_1.nii.gz')])
    make(folder)
    assert refused_with(folder) == reason


def test_study_reasons_are_distinct_fixed_strings():
    assert pre.REASON_STUDY_MODALITY != pre.REASON_STUDY_NAME
    assert 'case folder' in pre.REASON_FILE_MISSING and 'plan folder' not in pre.REASON_FILE_MISSING
    assert 'the image' in pre.REASON_TOO_SMALL and 'the CT' not in pre.REASON_TOO_SMALL


@pytest.mark.parametrize('text, expected', [
    (','.join(conv.SERIES_COLUMNS) + '\n' + 'a,series\n' * 3, 3),  # short rows still count: the validator refuses them whole
    (','.join(conv.SERIES_COLUMNS) + '\n', 1),
    ('', 1),
    ('file,kind\nseries_MR_1.nii.gz,series\nseries_MR_2.nii.gz,series\n', 2),  # wrong columns: rows are still the series
])
def test_study_refusal_count(tmp_path, text, expected):
    (tmp_path / conv.SERIES_NAME).write_text(text, encoding='utf-8')
    assert pre.study_refusal_count(tmp_path) == expected


def test_study_refusal_count_of_an_unparseable_file_is_one(tmp_path):
    (tmp_path / conv.SERIES_NAME).write_bytes(b'file,kind\n\xff\xfe,series\nseries_MR_2.nii.gz,series\n')
    assert pre.study_refusal_count(tmp_path) == 1


def test_study_refusal_count_of_a_valid_study(tmp_path):
    rows = [study_row(f'series_MR_{k}.nii.gz') for k in (1, 2, 3, 4)]
    assert pre.study_refusal_count(write_study(tmp_path / 'study_1', rows)) == 4


def test_check_study_rows_logs_nothing_with_identifiers(tmp_path, caplog):
    folder = write_study(tmp_path / 'SYNTHETICSTUDYFOLDER', [study_row('series_MR_1.nii.gz', 'RTDOSE')])
    with caplog.at_level('DEBUG'):
        reason = refused_with(folder)
    assert reason == pre.REASON_STUDY_MODALITY and not caplog.records
    assert 'SYNTHETICSTUDYFOLDER' not in reason and str(tmp_path) not in reason


# ---- task-05: study folders, one series at a time. Series are written straight to NIfTI; one test runs convert on synthetic DICOM. ----

VALUE_OF = {'CT': -500.0, 'MR': 300.0, 'PT': 5.0}
STUDY_CT = 'reshape_resample_rescale_initialresample_series_CT_1.nii.gz'
STUDY_MR = 'reshape_resample_initialresample_series_MR_1.nii.gz'  # no rescale prefix: the HU rescale is for CT only


def write_study_tree(root, patient='SYNTHETICPTA', study='study_1', series=(('CT', 1), ('MR', 1)), n=8, spacing=1.0):
    """<root>/<patient>/<study>/ as convert writes it: series_<MOD>_<k>.nii.gz (a ramp on a constant) and series.csv. Returns the folder."""
    folder = root / patient / study
    folder.mkdir(parents=True)
    rows = []
    for modality, k in series:
        ramp = np.linspace(0, 10, n ** 3).reshape(n, n, n) + VALUE_OF[modality]
        sitk.WriteImage(make_image(ramp, **grid(n, spacing)), str(folder / f'series_{modality}_{k}.nii.gz'))
        rows.append(series_row(f'series_{modality}_{k}.nii.gz', 'series', modality=modality, series_instance_uid=f'SYNTHETICUID{modality}{k}'))
    save_rows(folder, rows)
    return folder


def imaging_rows(report, patient='SYNTHETICPTA'):
    return [(r.folder, r.outcome, r.reason, r.files) for r in report.rows if r.patient == patient and r.folder.startswith('study_')]


def test_a_study_ct_gets_the_ct_prefixes_and_mr_and_pt_the_geometric_ones(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted, series=(('CT', 1), ('MR', 1), ('MR', 2), ('PT', 1)))
    before = state(converted)
    report = pre.preprocess(converted, out, SMALL)
    names = [STUDY_CT, STUDY_MR, STUDY_MR.replace('MR_1', 'MR_2'), STUDY_MR.replace('MR_1', 'PT_1')]
    assert imaging_rows(report) == [
        ('study_1/series_CT_1', 'preprocessed', '', [names[0], 'series.csv']),
        ('study_1/series_MR_1', 'preprocessed', '', [names[1]]),
        ('study_1/series_MR_2', 'preprocessed', '', [names[2]]),
        ('study_1/series_PT_1', 'preprocessed', '', [names[3]])]
    study = out / 'SYNTHETICPTA' / 'study_1'
    assert sorted(p.name for p in study.iterdir()) == sorted([*names, 'series.csv'])
    assert [r['file'] for r in conv_rows(study)] == names
    assert {r['kind'] for r in conv_rows(study)} == {'series'} and conv_rows(study)[0]['series_instance_uid'] == 'SYNTHETICUIDCT1'
    assert all(sitk.ReadImage(str(study / name)).GetSize() == (10, 10, 10) for name in names)
    assert state(converted) == before


def conv_rows(folder):
    with open(folder / 'series.csv', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def test_mr_and_pt_intensities_are_not_rescaled_and_ct_is(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted, series=(('CT', 1), ('MR', 1), ('PT', 1)))
    steps = PreprocessSteps(resample=True, rescale=True)  # a resample to the grid it already has keeps the voxels
    pre.preprocess(converted, out, steps)
    study = out / 'SYNTHETICPTA' / 'study_1'
    assert sorted(p.name for p in study.iterdir()) == ['initialresample_series_MR_1.nii.gz', 'initialresample_series_PT_1.nii.gz',
                                                       'rescale_initialresample_series_CT_1.nii.gz', 'series.csv']
    for modality in ('MR', 'PT'):
        got = sitk.GetArrayFromImage(sitk.ReadImage(str(study / f'initialresample_series_{modality}_1.nii.gz')))
        native = sitk.GetArrayFromImage(sitk.ReadImage(str(converted / 'SYNTHETICPTA' / 'study_1' / f'series_{modality}_1.nii.gz')))
        np.testing.assert_allclose(got, native, atol=1e-4)
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(study / 'rescale_initialresample_series_CT_1.nii.gz')))
    native = sitk.GetArrayFromImage(sitk.ReadImage(str(converted / 'SYNTHETICPTA' / 'study_1' / 'series_CT_1.nii.gz')))
    np.testing.assert_allclose(ct, (native + 1024) / 4096, atol=1e-4)


def test_rescale_alone_refuses_an_mr_and_still_preprocesses_a_ct(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted)
    report = pre.preprocess(converted, out, PreprocessSteps(rescale=True))
    assert imaging_rows(report) == [('study_1/series_CT_1', 'preprocessed', '', ['rescale_series_CT_1.nii.gz', 'series.csv']),
                                    ('study_1/series_MR_1', 'refused', pre.REASON_NO_STEPS, [])]
    study = out / 'SYNTHETICPTA' / 'study_1'
    assert sorted(p.name for p in study.iterdir()) == ['rescale_series_CT_1.nii.gz', 'series.csv']  # no MR file, whatever its name
    assert [r['file'] for r in conv_rows(study)] == ['rescale_series_CT_1.nii.gz']


def two_bad_series(converted):
    """study_1: CT_1 fine, MR_1 too small, MR_2 unreadable bytes, PT_1 fine."""
    folder = write_study_tree(converted, series=(('CT', 1), ('MR', 1), ('MR', 2), ('PT', 1)))
    sitk.WriteImage(make_image(np.ones((3, 3, 3)), spacing=(0.3,) * 3), str(folder / 'series_MR_1.nii.gz'))  # 0.9 mm per axis
    (folder / 'series_MR_2.nii.gz').write_bytes(b'not an image')
    return folder


def test_a_too_small_and_an_unreadable_series_are_refused_on_their_own_rows(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    two_bad_series(converted)
    report = pre.preprocess(converted, out, SMALL)
    pt = STUDY_MR.replace('MR_1', 'PT_1')
    assert imaging_rows(report) == [
        ('study_1/series_CT_1', 'preprocessed', '', [STUDY_CT, 'series.csv']),
        ('study_1/series_MR_1', 'refused', pre.REASON_TOO_SMALL, []),
        ('study_1/series_MR_2', 'refused', pre.REASON_FILE_UNREADABLE, []),
        ('study_1/series_PT_1', 'preprocessed', '', [pt])]
    study = out / 'SYNTHETICPTA' / 'study_1'
    assert sorted(p.name for p in study.iterdir()) == sorted([STUDY_CT, pt, 'series.csv'])  # no MR file, no leftover temp file
    assert [r['file'] for r in conv_rows(study)] == [STUDY_CT, pt]


def test_a_pixel_read_failure_refuses_that_series_only(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted, series=(('CT', 1), ('MR', 1), ('PT', 1)))
    real = pre.read_image

    def read(path):
        if path.name == 'series_MR_1.nii.gz':
            raise CaseRefused(pre.REASON_FILE_UNREADABLE)  # the header read fine, the pixels did not
        return real(path)
    monkeypatch.setattr(pre, 'read_image', read)
    report = pre.preprocess(converted, out, SMALL)
    assert [(f, o, r) for f, o, r, _ in imaging_rows(report)] == [
        ('study_1/series_CT_1', 'preprocessed', ''), ('study_1/series_MR_1', 'refused', pre.REASON_FILE_UNREADABLE), ('study_1/series_PT_1', 'preprocessed', '')]
    assert [r['modality'] for r in conv_rows(out / 'SYNTHETICPTA' / 'study_1')] == ['CT', 'PT']
    assert not list((out / 'SYNTHETICPTA' / 'study_1').glob('*MR*'))


def test_a_study_with_every_series_refused_leaves_no_study_folder_and_the_next_study_is_written(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder = write_study_tree(converted, series=(('MR', 1), ('PT', 1)))
    (folder / 'series_MR_1.nii.gz').write_bytes(b'not an image')
    sitk.WriteImage(make_image(np.ones((3, 3, 3)), spacing=(0.3,) * 3), str(folder / 'series_PT_1.nii.gz'))
    write_study_tree(converted, study='study_2', series=(('MR', 1),))
    report = pre.preprocess(converted, out, SMALL)
    assert [(f, o, r) for f, o, r, _ in imaging_rows(report)] == [
        ('study_1/series_MR_1', 'refused', pre.REASON_FILE_UNREADABLE), ('study_1/series_PT_1', 'refused', pre.REASON_TOO_SMALL),
        ('study_2/series_MR_1', 'preprocessed', '')]
    assert sorted(p.name for p in (out / 'SYNTHETICPTA').iterdir()) == ['study_2']
    assert [r for r in report.rows if r.folder == ''] == []


def test_a_study_refused_by_check_study_rows_is_one_row_and_writes_nothing(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder = write_study_tree(converted)
    (folder / 'series_MR_1.nii.gz').unlink()
    write_study_tree(converted, study='study_2', series=(('PT', 1),))
    report = pre.preprocess(converted, out, SMALL)
    assert imaging_rows(report) == [
        ('study_1', 'refused', pre.REASON_FILE_MISSING, []), ('study_2/series_PT_1', 'preprocessed', '', [STUDY_MR.replace('MR_1', 'PT_1'), 'series.csv'])]
    assert sorted(p.name for p in (out / 'SYNTHETICPTA').iterdir()) == ['study_2']


def test_plans_come_first_and_studies_follow_in_numeric_order(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted, study='study_10', series=(('MR', 1),))
    write_study_tree(converted, study='study_2', series=(('MR', 1),))
    write_plan(converted)
    (converted / 'SYNTHETICPTA' / 'study_3').mkdir()  # no series.csv
    (converted / 'SYNTHETICPTA' / 'study_x').mkdir()
    (converted / 'SYNTHETICPTA' / '.study_0').mkdir()
    (converted / 'SYNTHETICPTA' / 'images').mkdir()
    report = pre.preprocess(converted, out, SMALL)
    assert [(r.folder, r.outcome) for r in report.rows] == [
        ('plan_1', 'preprocessed'), ('study_2/series_MR_1', 'preprocessed'), ('study_10/series_MR_1', 'preprocessed')]
    assert sorted(p.name for p in (out / 'SYNTHETICPTA').iterdir()) == ['plan_1', 'study_10', 'study_2']
    assert (out / 'SYNTHETICPTA' / 'plan_1' / CT_FILE).is_file()


def test_a_patient_with_neither_plans_nor_studies_is_skipped_with_no_cases(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    (converted / 'SYNTHETICPTA' / 'study_1').mkdir(parents=True)  # no series.csv
    report = pre.preprocess(converted, out, SMALL)
    assert [(r.folder, r.outcome, r.reason) for r in report.rows] == [('', 'skipped', pre.REASON_NO_CASES)]
    assert listing(out) == ['preprocess_report.csv']


def test_an_oserror_in_a_study_series_refuses_the_patient_and_removes_the_partial(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted)
    write_study_tree(converted, patient='SYNTHETICPTB')
    real = conv.write_atomic

    def write(image, path):
        if path.name == STUDY_MR and 'SYNTHETICPTA' in str(path):
            raise OSError(conv.REASON_WRITE)
        real(image, path)
    monkeypatch.setattr(conv, 'write_atomic', write)
    report = pre.preprocess(converted, out, SMALL)
    assert [(r.patient, r.folder, r.outcome, r.reason) for r in report.rows] == [
        ('SYNTHETICPTA', '', 'refused', pre.REASON_IO_ERROR), ('SYNTHETICPTB', 'study_1/series_CT_1', 'preprocessed', ''),
        ('SYNTHETICPTB', 'study_1/series_MR_1', 'preprocessed', '')]
    assert listing(out) == ['SYNTHETICPTB', 'preprocess_report.csv']


def test_an_internal_error_in_a_study_series_aborts_the_run_and_removes_the_partial(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted)
    write_study_tree(converted, patient='SYNTHETICPTB')
    real = pre.resample_to
    calls = []

    def boom(image, spacing):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError('SYNTHETICPTA')
        return real(image, spacing)  # the CT passes, the MR fails
    monkeypatch.setattr(pre, 'resample_to', boom)
    with pytest.raises(pre.PreprocessAborted) as caught:
        pre.preprocess(converted, out, SMALL)
    assert str(caught.value) == pre.REASON_INTERNAL
    chain = ''.join(traceback.format_exception(caught.value))
    assert 'direct cause' not in chain and 'During handling' not in chain
    assert report_rows(out) == [dict(patient='SYNTHETICPTA', folder='', outcome='refused', reason=pre.REASON_INTERNAL, files='', skipped_series='0')]
    assert listing(out) == ['preprocess_report.csv']


def test_the_run_log_counts_plan_rows_and_imaging_rows_separately(tmp_path, caplog):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted, patient='SYNTHETICPTB')
    two_bad_series(converted)
    (converted / 'SYNTHETICPTC' / 'study_1').mkdir(parents=True)
    with caplog.at_level('INFO'):
        pre.preprocess(converted, out, SMALL)
    messages = [r.getMessage() for r in caplog.records if r.name == 'rt_dataprep.preprocess' and r.getMessage().startswith('preprocess: ')]
    summary = [m for m in messages if 'plan rows' in m]
    assert summary == ['preprocess: 1 plan rows (1 preprocessed, 0 refused), 4 imaging rows (2 preprocessed, 2 refused), 1 patient rows (0 refused, 1 skipped); 0 registered series skipped']


def test_study_refusals_log_only_reasons(tmp_path, caplog):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    two_bad_series(converted)
    folder = write_study_tree(converted, study='study_2')
    (folder / 'series_CT_1.nii.gz').unlink()
    with caplog.at_level('DEBUG'):
        pre.preprocess(converted, out, SMALL)
    assert caplog.records and not any(s in caplog.text for s in ('SYNTHETICPT', str(tmp_path), 'study_', 'series_', 'SYNTHETICUID'))


def test_manifest_lists_the_same_cases_on_the_converted_and_the_preprocessed_tree(tmp_path):
    from test_convert import BOX, PID, dose_spec, make_plan
    from test_series import STUDY_D1, STUDY_D2, STUDY_T1, STUDY_T2, put

    from rt_dataprep import manifest as man
    org, converted, out = tmp_path / 'org', tmp_path / 'converted', tmp_path / 'out'
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10)], rois=[BOX])
    misc = org / PID / 'misc_files'
    put(misc, 'ct', 'CT', study_date=STUDY_D1)
    mr_a = put(misc, 'mra', 'MR', study_date=STUDY_D2, series_time=STUDY_T1)
    put(misc, 'mrb', 'MR', study_uid=mr_a['study_uid'], study_date=STUDY_D2, series_time=STUDY_T2)
    conv.convert(org, converted)
    steps = dataclasses.replace(SMALL, new_shape=(8, 8, 8))
    report = pre.preprocess(converted, out, steps)
    assert [(r.folder, r.outcome) for r in report.rows] == [
        ('plan_1', 'preprocessed'), ('study_1/series_CT_1', 'preprocessed'), ('study_2/series_MR_1', 'preprocessed'), ('study_2/series_MR_2', 'preprocessed')]
    assert man.manifest(converted) == (4, 0) and man.manifest(out) == (4, 0)

    def read(root):
        with open(root / 'manifest.csv', newline='') as handle:
            return list(csv.DictReader(handle))
    before, after = read(converted), read(out)
    assert [r['case_dir'] for r in after] == [r['case_dir'] for r in before]
    assert [r['case_dir'].split('/', 1)[1] for r in after] == ['plan_1', 'study_1/series_CT_1', 'study_2/series_MR_1', 'study_2/series_MR_2']
    for kept, done in zip(before[1:], after[1:], strict=True):
        assert kept['image'] == f"{kept['case_dir'].split('/')[-1]}.nii.gz"
        assert done['image'] == pre.output_name(kept['image'], steps, ct=kept['modality'] == 'CT')
        assert (out / PID / done['case_dir'].split('/')[1] / done['image']).is_file()
    assert [r['image'] for r in after[2:]] == [pre.output_name(f'series_MR_{k}.nii.gz', steps) for k in (1, 2)]


# ---- task-03: registered series aligned onto the final CT grid (reg-resampling D1-D9, D13). SYNTHETIC volumes with one bright landmark voxel. ----

CT_FRAME, MR_FRAME = 'SYNTHETICFORCT', 'SYNTHETICFORMR'
FINAL = pre.process_ct(make_image(np.zeros((12, 12, 12)), **grid(12)), SMALL)  # the geometry of write_plan's final CT under SMALL: 10^3 at 1.25 mm, origin -6.25
FINAL_ORIGIN = np.array(FINAL.GetOrigin())
MR_GRID = dict(spacing=(1.25,) * 3, origin=tuple(FINAL_ORIGIN + (1.9 * 1.25, 1.7 * 1.25, 2.9 * 1.25)))  # a different grid: shifted by fractions of a voxel
MR_SIZE = (8, 9, 7)
NON_RIGID = '2 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'
BASE_FILES = [CT_FILE, DOSE_FILE, PTV_FILE, BLADDER_FILE, 'applicator_a.txt']
MR_OUT = pre.output_name('series_MR_1.nii.gz', SMALL)


def matrix_text(matrix):
    return ' '.join('%.10g' % v for v in np.asarray(matrix).ravel())


def rigid(rotation_degrees=0.0, translation=(0, 0, 0)):
    """A 4x4 rotation about z plus translation (mm)."""
    c, s = np.cos(np.radians(rotation_degrees)), np.sin(np.radians(rotation_degrees))
    matrix = np.eye(4)
    matrix[:2, :2] = [[c, -s], [s, c]]
    matrix[:3, 3] = translation
    return matrix


def landmark(size, index, value=100.0):
    array = np.zeros(size[::-1])
    array[index[2], index[1], index[0]] = value
    return array


def run_plan(tmp_path, specs, steps=SMALL):
    """write_plan plus the series of `specs` ((file, modality, array or raw bytes or None, reg_matrix text, geometry), in series.csv order); preprocessed.
    Returns (report, output plan folder, converted plan folder)."""
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    rows[0]['frame_of_reference_uid'] = CT_FRAME
    for file, modality, array, text, geometry in specs:
        if isinstance(array, bytes):
            (folder / file).write_bytes(array)
        elif array is not None:
            sitk.WriteImage(make_image(array, **geometry), str(folder / file))
        rows.append(series_row(file, 'series', modality=modality, reg_matrix=text, frame_of_reference_uid=MR_FRAME))
    save_rows(folder, rows)
    report = pre.preprocess(converted, out, steps)
    return report, out / 'SYNTHETICPTA' / 'plan_1', folder


def mr_spec(array, matrix, geometry=MR_GRID, file='series_MR_1.nii.gz', modality='MR'):
    return (file, modality, array, matrix if isinstance(matrix, str) else matrix_text(matrix), geometry)


def peak_index(image):
    return np.array(np.unravel_index(np.argmax(sitk.GetArrayViewFromImage(image)), image.GetSize()[::-1])[::-1], float)


def landmark_indices(out_image, series_geometry, size, index, matrix):
    """(expected, inverse-mapped) continuous indices on the output of the landmark's physical point mapped through `matrix` and through its inverse."""
    p = make_image(np.zeros(size[::-1]), **series_geometry).TransformIndexToPhysicalPoint(index)
    def at(m):
        return np.array(out_image.TransformPhysicalPointToContinuousIndex((m @ [*p, 1])[:3].tolist()))
    return at(matrix), at(np.linalg.inv(matrix))


def check_landmark(out_image, expected):
    assert ((expected >= 0) & (expected <= np.array(out_image.GetSize()) - 1)).all()  # the landmark lies inside the final grid by construction
    assert (np.abs(peak_index(out_image) - expected) <= 0.5 + 1e-6).all()


def check_direction(out_image, expected, inverse_mapped):
    difference = np.abs(inverse_mapped - expected)
    axis = int(difference.argmax())
    assert difference[axis] > 1 and abs(peak_index(out_image)[axis] - inverse_mapped[axis]) > 0.5


def read_out(folder, name=MR_OUT):
    return sitk.ReadImage(str(folder / name))


def test_ac1_a_registered_mr_is_written_on_exactly_the_final_ct_grid_and_listed_after_the_existing_rows(tmp_path):
    report, done, _ = run_plan(tmp_path, [mr_spec(landmark(MR_SIZE, (3, 4, 2)), rigid(0, (1.25, 0, 0)))])
    assert plan_row(report).outcome == 'preprocessed' and plan_row(report).files == [*BASE_FILES, MR_OUT, 'series.csv']
    image, ct = read_out(done), read_out(done, CT_FILE)
    assert (image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection()) == (ct.GetSize(), ct.GetOrigin(), ct.GetSpacing(), ct.GetDirection())
    assert image.GetPixelID() == sitk.sitkFloat32
    rows = conv_rows(done)
    assert [r['file'] for r in rows] == [*BASE_FILES, MR_OUT]
    assert (rows[-1]['kind'], rows[-1]['modality'], rows[-1]['reg_matrix'], rows[-1]['frame_of_reference_uid']) == ('series', 'MR', 'identity', CT_FRAME)
    assert rows[0]['frame_of_reference_uid'] == CT_FRAME


def test_ac2a_identity_on_a_different_grid(tmp_path):
    _, done, _ = run_plan(tmp_path, [mr_spec(landmark(MR_SIZE, (3, 4, 2)), 'identity')])
    image = read_out(done)
    expected, _ = landmark_indices(image, MR_GRID, MR_SIZE, (3, 4, 2), np.eye(4))
    assert np.abs(expected - (4.9, 5.7, 4.9)).max() < 1e-6  # the grids differ by fractions of a voxel
    check_landmark(image, expected)


def test_ac2b_a_whole_voxel_translation_lands_the_landmark_at_the_mapped_index_and_not_at_the_inverse_one(tmp_path):
    matrix = rigid(0, (3 * 1.25, -2 * 1.25, 1.25))
    _, done, _ = run_plan(tmp_path, [mr_spec(landmark(MR_SIZE, (2, 3, 3)), matrix)])
    image = read_out(done)
    expected, inverse_mapped = landmark_indices(image, MR_GRID, MR_SIZE, (2, 3, 3), matrix)
    check_landmark(image, expected)
    check_direction(image, expected, inverse_mapped)


def test_ac2c_a_rotation_about_z_plus_translation_lands_the_landmark_at_the_mapped_index_and_not_at_the_inverse_one(tmp_path):
    geometry, size, index = dict(spacing=(1.0, 1.5, 2.0), origin=(-7.0, -9.0, -6.0)), (16, 12, 10), (6, 5, 3)
    p = np.array(make_image(np.zeros(size[::-1]), **geometry).TransformIndexToPhysicalPoint(index))
    q = np.array(FINAL.TransformIndexToPhysicalPoint((5, 4, 5)))  # the landmark is built to land on this final voxel centre
    matrix = rigid(30)
    matrix[:3, 3] = q - matrix[:3, :3] @ p
    _, done, _ = run_plan(tmp_path, [mr_spec(landmark(size, index), matrix, geometry)])
    image = read_out(done)
    expected, inverse_mapped = landmark_indices(image, geometry, size, index, matrix)
    assert np.abs(expected - (5, 4, 5)).max() < 1e-6
    check_landmark(image, expected)
    check_direction(image, expected, inverse_mapped)


def coverage(out_image, series_geometry, size, matrix):
    """(interior, outside) boolean (z, y, x) arrays: final voxels whose source continuous index is at least one voxel from the series boundary, and outside its buffer."""
    series = make_image(np.zeros(size[::-1]), **series_geometry)
    inverse = np.linalg.inv(matrix)
    interior, outside = np.zeros(out_image.GetSize()[::-1], bool), np.zeros(out_image.GetSize()[::-1], bool)
    for x, y, z in np.ndindex(*out_image.GetSize()):
        p = (inverse @ [*out_image.TransformIndexToPhysicalPoint((x, y, z)), 1])[:3]
        c = np.array(series.TransformPhysicalPointToContinuousIndex(p.tolist()))
        interior[z, y, x] = ((c >= 1) & (c <= np.array(size) - 2)).all()
        outside[z, y, x] = ((c < -0.5 - 1e-3) | (c > np.array(size) - 0.5 + 1e-3)).any()
    assert interior.any() and outside.any()
    return interior, outside


def test_ac3_outside_the_series_fov_is_zero_and_a_constant_mr_keeps_its_value_at_interior_voxels(tmp_path):
    size, matrix = (8, 8, 8), rigid(0, (0.6, 0.3, -0.4))
    _, done, _ = run_plan(tmp_path, [mr_spec(np.full((8, 8, 8), 7.0), matrix, dict(spacing=(1.25,) * 3, origin=(-3.875, -4.125, -2.625)))])
    values = sitk.GetArrayFromImage(read_out(done))
    interior, outside = coverage(read_out(done), dict(spacing=(1.25,) * 3, origin=(-3.875, -4.125, -2.625)), size, matrix)
    assert np.abs(values[interior] - 7.0).max() < 1e-4 and (values[outside] == 0).all()


def test_ac4_a_registered_ct_is_rescaled_before_the_resample_and_gets_the_rescale_prefix(tmp_path):
    geometry, matrix = dict(spacing=(1.25,) * 3, origin=(-3.875, -4.125, -2.625)), rigid(0, (0.6, 0.3, -0.4))
    name = pre.output_name('series_CT_1.nii.gz', SMALL, ct=True)
    report, done, _ = run_plan(tmp_path, [mr_spec(np.full((8, 8, 8), 100.0), matrix, geometry, 'series_CT_1.nii.gz', 'CT')])
    assert name.startswith('reshape_resample_rescale_initialresample_') and plan_row(report).files[-2] == name
    values = sitk.GetArrayFromImage(read_out(done, name))
    interior, outside = coverage(read_out(done, name), geometry, (8, 8, 8), matrix)
    assert np.abs(values[interior] - (100.0 + 1024) / 4096).max() < 1e-4 and (values[outside] == 0).all()
    assert conv_rows(done)[-1]['file'] == name and conv_rows(done)[-1]['modality'] == 'CT'


@pytest.mark.parametrize('text', ['', NON_RIGID])
def test_ac5_an_unusable_matrix_leaves_the_series_out_and_counts_it(tmp_path, text):
    report, done, _ = run_plan(tmp_path, [mr_spec(np.ones((4, 4, 4)), text)])
    assert (plan_row(report).outcome, plan_row(report).skipped_series) == ('preprocessed', 1)
    assert plan_row(report).files == [*BASE_FILES, 'series.csv'] and sorted(p.name for p in done.iterdir()) == sorted(plan_row(report).files)
    assert [r['file'] for r in conv_rows(done)] == BASE_FILES


def test_ac6_a_series_entirely_outside_the_ct_is_left_out_and_one_partly_outside_is_kept(tmp_path):
    ones = np.ones((8, 8, 8))
    report, done, _ = run_plan(tmp_path, [mr_spec(ones, rigid(0, (500, 0, 0))), mr_spec(ones, rigid(0, (8, 0, 0)), file='series_MR_2.nii.gz')])
    kept = pre.output_name('series_MR_2.nii.gz', SMALL)
    assert plan_row(report).skipped_series == 1 and plan_row(report).files == [*BASE_FILES, kept, 'series.csv']
    values = sitk.GetArrayFromImage(read_out(done, kept))
    assert (values > 0).any() and (values == 0).any() and [r['file'] for r in conv_rows(done)][-1] == kept


def test_ac7_an_unreadable_series_file_is_left_out_and_counted(tmp_path):
    report, done, _ = run_plan(tmp_path, [mr_spec(b'SYNTHETIC not a nifti', 'identity')])
    assert (plan_row(report).outcome, plan_row(report).skipped_series) == ('preprocessed', 1)
    assert plan_row(report).files == [*BASE_FILES, 'series.csv'] and not (done / MR_OUT).exists()


def test_ac7_an_internal_error_while_resampling_a_series_still_stops_the_run(tmp_path, monkeypatch):
    real = sitk.Resample

    def resample(*args):
        if isinstance(args[2], sitk.AffineTransform):  # only the series resample passes an affine
            raise RuntimeError('SYNTHETICPTA')
        return real(*args)
    monkeypatch.setattr(sitk, 'Resample', resample)
    with pytest.raises(pre.PreprocessAborted):
        run_plan(tmp_path, [mr_spec(np.ones((4, 4, 4)), rigid(0, (1, 0, 0)))])
    assert report_rows(tmp_path / 'out') == [dict(patient='SYNTHETICPTA', folder='', outcome='refused', reason=pre.REASON_INTERNAL, files='', skipped_series='0')]


def test_ac9_rescale_only_resamples_an_mr_onto_the_native_ct_grid_and_keeps_its_name(tmp_path):
    matrix = rigid(0, (1.0, -2.0, 1.0))
    report, done, folder = run_plan(tmp_path, [mr_spec(landmark(MR_SIZE, (2, 3, 3)), matrix)], PreprocessSteps(rescale=True))
    assert plan_row(report).files == ['rescale_planning_ct.nii.gz', 'dose.nii.gz', 'ptv.nii.gz', 'bladder.nii.gz', 'applicator_a.txt', 'series_MR_1.nii.gz', 'series.csv']
    image, native = read_out(done, 'series_MR_1.nii.gz'), sitk.ReadImage(str(folder / conv.CT_NAME))
    assert (image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection()) == (native.GetSize(), native.GetOrigin(), native.GetSpacing(), native.GetDirection())
    assert (done / 'series_MR_1.nii.gz').read_bytes() != (folder / 'series_MR_1.nii.gz').read_bytes()  # resampled, not copied
    expected, _ = landmark_indices(image, MR_GRID, MR_SIZE, (2, 3, 3), matrix)
    check_landmark(image, expected)


def test_ac10_the_report_counts_skipped_series_on_plan_rows_only_and_the_closing_line_has_the_total(tmp_path, caplog):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    rows += [add_series(folder, rows)]  # identity: kept
    sitk.WriteImage(make_image(np.ones((4, 4, 4))), str(folder / 'series_MR_2.nii.gz'))
    save_rows(folder, [*rows, series_row('series_MR_2.nii.gz', 'series', modality='MR', reg_matrix='')])  # skipped
    refused, _ = write_plan(converted, plan='plan_2')
    (refused / 'ptv.nii.gz').unlink()
    write_study_tree(converted)
    write_plan(converted, patient='SYNTHETICPTB')
    with caplog.at_level('INFO'):
        report = pre.preprocess(converted, out, SMALL)
    csv_rows = [(r['folder'], r['skipped_series']) for r in report_rows(out)]
    assert csv_rows[0] == ('plan_1', '1') and all(count == '0' for _, count in csv_rows[1:]) and len(csv_rows) == 5
    assert plan_row(report).skipped_series == 1 and sum(r.skipped_series for r in report.rows) == 1
    assert any('1 registered series skipped' in record.getMessage() for record in caplog.records)


def test_two_registered_series_are_written_in_input_order_after_the_existing_rows(tmp_path):
    ones = np.ones((4, 4, 4))
    report, done, _ = run_plan(tmp_path, [mr_spec(ones, 'identity', file='series_MR_2.nii.gz'), mr_spec(ones, 'identity', file='series_PT_1.nii.gz', modality='PT')])
    names = [pre.output_name('series_MR_2.nii.gz', SMALL), pre.output_name('series_PT_1.nii.gz', SMALL)]
    assert plan_row(report).files == [*BASE_FILES, *names, 'series.csv']
    assert [r['file'] for r in conv_rows(done)] == [*BASE_FILES, *names]
    assert [r['modality'] for r in conv_rows(done)[-2:]] == ['MR', 'PT']


def test_each_series_is_released_before_the_next_file_is_read(tmp_path, monkeypatch):
    tracked, alive_at_read = [], []
    read_image, process_series = pre.read_image, pre.process_series

    def track(image):
        tracked.append(weakref.ref(image))
        return image

    def read(path):
        alive_at_read.append(sum(ref() is not None for ref in tracked))
        image = read_image(path)
        return image if len(alive_at_read) == 1 else track(image)  # the CT is the reference grid: it stays

    monkeypatch.setattr(pre, 'read_image', read)
    monkeypatch.setattr(pre, 'process_series', lambda *args: track(process_series(*args)))
    ones = np.ones((4, 4, 4))
    report, _, _ = run_plan(tmp_path, [mr_spec(ones, 'identity'), mr_spec(ones, 'identity', file='series_MR_2.nii.gz')])
    assert plan_row(report).outcome == 'preprocessed'
    assert alive_at_read == [0] * 6  # CT, dose, two masks, two series
    assert sum(ref() is not None for ref in tracked) == 0
