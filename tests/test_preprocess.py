"""preprocess core (task-01): grid, crop/pad, rescale, names, CT pipeline, dose and mask resampling. Images are in memory."""
import ast
import dataclasses
import inspect

import numpy as np
import pytest
import SimpleITK as sitk

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
