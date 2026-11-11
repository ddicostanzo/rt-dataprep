"""preprocess: grid, crop/pad, rescale, names, CT pipeline, dose and mask resampling (task-01); the tree walk (task-02)."""
import ast
import csv
import dataclasses
import errno
import inspect
import json
import logging
import os
import re
import time
import traceback
import weakref

import numpy as np
import pytest
import series_files as sf
import SimpleITK as sitk
from test_dicom_io import fail_python_open

from rt_dataprep import convert as conv
from rt_dataprep import dicom_io
from rt_dataprep import preprocess as pre
from rt_dataprep.dicom_io import CaseRefused
from rt_dataprep.preprocess import PreprocessSteps
from skimage.exposure import equalize_adapthist

FOUR_STEPS = PreprocessSteps(resample=True, rescale=True, harmonize_spacing=True, reshape=True)

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


def test_steps_defaults():
    steps = PreprocessSteps()
    assert (steps.resample, steps.rescale, steps.harmonize_spacing, steps.reshape, steps.legacy_linear_masks) == (False,) * 5
    assert steps.new_spacing == (1, 1, 1) and steps.final_spacing == (1.25, 1.25, 1.25) and steps.new_shape == (701, 701, 433)
    with pytest.raises(dataclasses.FrozenInstanceError):
        steps.resample = True


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


STEPS_SMALL = dataclasses.replace(FOUR_STEPS, new_spacing=(1.5, 1.5, 1.5), new_shape=(9, 9, 9))


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
                           dataclasses.replace(FOUR_STEPS, new_shape=(28, 28, 28)))
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
    steps = dataclasses.replace(FOUR_STEPS, new_shape=(20, 20, 20))
    final = pre.process_ct(ct, steps)
    out = pre.process_mask(mask, final, steps)
    assert out.GetPixelID() == sitk.sitkUInt8 and set(np.unique(sitk.GetArrayFromImage(out))) == {0, 1}
    linear = dataclasses.replace(steps, legacy_linear_masks=True)
    soft = pre.process_mask(mask, final, linear)
    values = sitk.GetArrayFromImage(soft)
    assert soft.GetPixelID() == sitk.sitkFloat32 and ((values > 0) & (values < 1)).any()


def test_default_masks_take_the_nearest_native_voxel():
    ct, mask = cube_fixture(side=8)
    steps = dataclasses.replace(FOUR_STEPS, new_shape=(20, 20, 20))
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
    steps = dataclasses.replace(FOUR_STEPS, new_shape=(20, 20, 20))
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
    steps = dataclasses.replace(FOUR_STEPS, new_shape=(10, 10, 10))
    final = pre.process_ct(ct, steps)
    outputs = [final, pre.process_dose(dose, final), pre.process_mask(mask, final, steps)]
    assert all(image.GetDirection() == ROTATED for image in outputs)
    assert all(image.GetSize() == final.GetSize() and image.GetOrigin() == final.GetOrigin() for image in outputs)


def test_names_newest_step_first_and_geometric_prefixes_only_for_dose_and_masks():
    assert pre.output_name('planning_ct.nii.gz', FOUR_STEPS, ct=True) == 'reshape_resample_rescale_initialresample_planning_ct.nii.gz'
    assert pre.output_name('dose.nii.gz', FOUR_STEPS) == 'reshape_resample_initialresample_dose.nii.gz'
    assert pre.output_name('ptv.nii.gz', FOUR_STEPS) == 'reshape_resample_initialresample_ptv.nii.gz'
    rescale = PreprocessSteps(rescale=True)
    assert pre.output_name('planning_ct.nii.gz', rescale, ct=True) == 'rescale_planning_ct.nii.gz'
    assert pre.output_name('dose.nii.gz', rescale) == 'dose.nii.gz' and pre.output_name('ptv.nii.gz', rescale) == 'ptv.nii.gz'
    assert pre.output_name('dose.nii.gz', PreprocessSteps(harmonize_spacing=True)) == 'resample_dose.nii.gz'
    assert pre.output_name('planning_ct.nii.gz', PreprocessSteps(), ct=True) == 'planning_ct.nii.gz'


def garbage_file(tmp_path):
    bad = tmp_path / 'ct.nii.gz'
    bad.write_bytes(b'not an image')
    return bad


READERS = (pre.read_image, pre.read_info)


@pytest.mark.parametrize('read', READERS)
def test_a_garbage_file_python_reads_cleanly_is_still_unreadable(tmp_path, monkeypatch, read):
    bad = garbage_file(tmp_path)
    calls = []
    real = dicom_io.read_failure
    monkeypatch.setattr(dicom_io, 'read_failure', lambda path: (calls.append(path), real(path))[1])
    with pytest.raises(CaseRefused) as caught:
        read(bad)
    assert str(caught.value) == pre.REASON_FILE_UNREADABLE and calls == [bad]
    assert caught.value.__context__ is None and caught.value.__cause__ is None


@pytest.mark.parametrize('read', READERS)
def test_a_file_that_vanishes_before_the_python_reread_is_unreadable(tmp_path, monkeypatch, read):
    bad = garbage_file(tmp_path)
    modes = fail_python_open(monkeypatch, bad, FileNotFoundError(errno.ENOENT, 'gone', str(bad)))
    with pytest.raises(CaseRefused) as caught:
        read(bad)
    assert str(caught.value) == pre.REASON_FILE_UNREADABLE and modes == ['rb']


@pytest.mark.parametrize('read', READERS)
@pytest.mark.parametrize('code', [errno.EACCES, errno.EIO])
def test_a_file_system_error_from_the_python_reread_propagates_as_oserror(tmp_path, monkeypatch, read, code):
    bad = garbage_file(tmp_path)
    modes = fail_python_open(monkeypatch, bad, OSError(code, 'forced', str(bad)))
    with pytest.raises(OSError) as caught:
        read(bad)
    assert not isinstance(caught.value, CaseRefused) and caught.value.errno == code and modes == ['rb']
    assert caught.value.__context__ is None and caught.value.__cause__ is None


@pytest.mark.parametrize('read', READERS)
def test_enomem_from_the_python_reread_is_a_bare_memory_error(tmp_path, monkeypatch, read):
    bad = garbage_file(tmp_path)
    fail_python_open(monkeypatch, bad, OSError(errno.ENOMEM, 'forced', str(bad)))
    with pytest.raises(MemoryError) as caught:
        read(bad)
    assert type(caught.value) is MemoryError and caught.value.args == ()
    assert caught.value.__context__ is None and caught.value.__cause__ is None


@pytest.mark.parametrize('read', READERS)
def test_a_successful_read_does_not_run_the_python_check(tmp_path, monkeypatch, read):
    good = tmp_path / 'ct.nii.gz'
    sitk.WriteImage(make_image(box((2, 2, 2), (4, 4, 4)), **grid()), str(good))
    monkeypatch.setattr(dicom_io, 'read_failure', lambda path: pytest.fail('the Python check ran'))
    read(good)


@pytest.mark.parametrize('read', READERS)
def test_a_memory_error_from_quiet_call_does_not_run_the_python_check(tmp_path, monkeypatch, read):
    bad = garbage_file(tmp_path)

    def boom(*args, **kwargs):
        raise MemoryError()
    monkeypatch.setattr(pre.sitk, 'ReadImage', boom)
    monkeypatch.setattr(pre.sitk, 'ImageFileReader', boom)
    monkeypatch.setattr(dicom_io, 'read_failure', lambda path: pytest.fail('the Python check ran'))
    with pytest.raises(MemoryError):
        read(bad)


def test_third_party_failures_are_one_fixed_reason(tmp_path):
    bad = tmp_path / 'ct.nii.gz'
    bad.write_bytes(b'not an image')
    with pytest.raises(CaseRefused) as caught:
        pre.read_image(bad)
    assert str(caught.value) == pre.REASON_FILE_UNREADABLE and str(bad) not in str(caught.value)
    for call in (lambda: pre.process_ct(None, FOUR_STEPS), lambda: pre.process_dose(None, blank((2, 2, 2))),
                 lambda: pre.process_mask(None, blank((2, 2, 2)), FOUR_STEPS)):
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

SMALL = dataclasses.replace(FOUR_STEPS, new_shape=(10, 10, 10))
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


def write_plan(root, patient='SYNTHETICPTA', plan='plan_1', n=12, spacing=1.0, dose=DEFAULT_DOSE, masks=None, applicator=True, direction=(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)):
    """<root>/<patient>/<plan>/ as F2a writes it. Returns (folder, series.json rows)."""
    masks = DEFAULT_MASKS if masks is None else masks
    geometry = dict(grid(n, spacing), direction=direction)
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
    sf.write(rows, folder / conv.SERIES_NAME)


def state(root):
    """Every file under root with its bytes: a before/after comparison."""
    return {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file()}


def report_rows(out):
    with open(out / 'preprocess_report.csv', newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ['patient', 'folder', 'outcome', 'reason', 'files', 'skipped_series', 'cropped']
        return list(reader)


def plan_row(report, patient='SYNTHETICPTA', folder='plan_1'):
    return next(r for r in report.rows if (r.patient, r.folder) == (patient, folder))


def test_a_line_file_is_copied_unchanged_under_its_own_name(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    (folder / 'refline.nii.gz').write_bytes(APPLICATOR_BYTES)
    save_rows(folder, rows + [series_row('refline.nii.gz', 'line', modality='RTSTRUCT', roi_name='RefLine')])
    report = pre.preprocess(converted, out, SMALL)
    assert plan_row(report).outcome == 'preprocessed'
    done = out / 'SYNTHETICPTA' / 'plan_1'
    assert (done / 'refline.nii.gz').read_bytes() == APPLICATOR_BYTES
    assert [(r['kind'], r['retained']) for r in sf.read(done) if r['file'] == 'refline.nii.gz'] == [('line', '')]  # a geometric run leaves a line's retained blank


def test_four_steps_write_the_prefixed_files_and_leave_the_input_alone(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    before = state(converted)
    report = pre.preprocess(converted, out, SMALL)
    assert isinstance(report, pre.PreprocessReport)
    expected = [CT_FILE, DOSE_FILE, PTV_FILE, BLADDER_FILE, 'applicator_a.txt', 'series.json']
    row = plan_row(report)
    assert (row.outcome, row.reason, row.files) == ('preprocessed', '', expected) and len(report.rows) == 1
    assert sorted(p.name for p in (out / 'SYNTHETICPTA' / 'plan_1').iterdir()) == sorted(expected)
    assert listing(out) == ['SYNTHETICPTA', 'preprocess_report.csv']  # no partial left
    assert report_rows(out) == [dict(patient='SYNTHETICPTA', folder='plan_1', outcome='preprocessed', reason='', files=';'.join(expected), skipped_series='0', cropped='0')]
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
    assert plan_row(report).files == ['rescale_planning_ct.nii.gz', 'dose.nii.gz', 'ptv.nii.gz', 'bladder.nii.gz', 'applicator_a.txt', 'series.json']
    for name in ('dose.nii.gz', 'ptv.nii.gz', 'bladder.nii.gz', 'applicator_a.txt'):
        assert (done / name).read_bytes() == (folder / name).read_bytes()
    hu = sitk.GetArrayFromImage(sitk.ReadImage(str(folder / conv.CT_NAME)))
    assert np.abs(sitk.GetArrayFromImage(sitk.ReadImage(str(done / 'rescale_planning_ct.nii.gz'))) - (hu + 1024) / 4096).max() < 1e-6


def test_series_json_keeps_every_column_and_renames_files_and_lists_the_aligned_series_last(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    sitk.WriteImage(make_image(np.ones((4, 4, 4))), str(folder / 'series_MR_1.nii.gz'))  # present, as check_rows now requires (task-02)
    series = series_row('series_MR_1.nii.gz', 'series', modality='MR', reg_matrix='identity')  # an empty matrix would leave it out
    save_rows(folder, [*rows[:2], series, *rows[2:]])
    pre.preprocess(converted, out, SMALL)
    written = sf.read(out / 'SYNTHETICPTA' / 'plan_1')
    names = [CT_FILE, DOSE_FILE, PTV_FILE, BLADDER_FILE, 'applicator_a.txt', pre.output_name('series_MR_1.nii.gz', SMALL)]
    assert [r['file'] for r in written] == names
    assert [{k: v for k, v in r.items() if k != 'file'} for r in written] == [{k: v for k, v in r.items() if k != 'file'} | {'retained': '1.0' if r['kind'] in ('dose', 'mask') else ''} for r in [*rows, series]]  # SMALL cuts nothing


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


def test_two_series_rows_with_one_series_id_refuse_the_plan(tmp_path):
    def change(folder, rows):
        add_series(folder, [*rows, add_series(folder, rows)], 'resample_series_MR_1.nii.gz')
    refuse(tmp_path, pre.REASON_SERIES_CSV, change)


def test_a_series_file_with_step_prefixes_passes_check_rows(tmp_path):
    folder, rows = write_plan(tmp_path)
    add_series(folder, rows, 'reshape_resample_initialresample_series_MR_1.nii.gz')
    assert [r['file'] for r in pre.check_rows(folder)[2]] == ['reshape_resample_initialresample_series_MR_1.nii.gz']


def test_check_rows_reads_no_series_header(tmp_path):
    folder, rows = write_plan(tmp_path)
    add_series(folder, rows, write=False)
    (folder / 'series_MR_1.nii.gz').write_bytes(b'SYNTHETIC not a nifti')
    assert len(pre.check_rows(folder)[2]) == 1


def test_the_study_reasons_say_a_series_json_series_row():
    for reason in (pre.REASON_STUDY_MODALITY, pre.REASON_STUDY_NAME):
        assert 'series.json series row' in reason and 'study' not in reason


def test_a_plan_without_a_dose_writes_no_dose(tmp_path):
    write_plan(tmp_path / 'converted', dose=None, applicator=False)
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', SMALL)
    assert plan_row(report).files == [CT_FILE, PTV_FILE, BLADDER_FILE, 'series.json']


def test_the_walk_skips_images_hidden_and_folders_without_a_series_json(tmp_path):
    converted = tmp_path / 'converted'
    write_plan(converted, plan='plan_2')
    write_plan(converted, plan='plan_10')
    write_plan(converted, patient='.hidden')
    (converted / 'SYNTHETICPTA' / 'images').mkdir()
    (converted / 'SYNTHETICPTA' / 'images' / 'series.json').write_text('[]')
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
    for steps in (PreprocessSteps(), PreprocessSteps(legacy_linear_masks=True), PreprocessSteps(series_fov_mask=True)):
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


def test_a_patient_without_a_plan_folder_with_a_series_json_is_skipped_and_writes_nothing(tmp_path):
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



def test_a_series_json_naming_one_file_twice_refuses_the_plan_only(tmp_path):
    refuse(tmp_path, pre.REASON_SERIES_CSV, lambda folder, rows: save_rows(folder, [*rows, rows[-1]]))


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


def test_the_ct_is_processed_first_whatever_its_series_json_row(tmp_path):
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
    small = dataclasses.replace(FOUR_STEPS, new_shape=(4, 4, 4))
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


NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason='root ignores file permissions')
UNREADABLE = ['forced', pytest.param('chmod', marks=NOT_ROOT)]


def make_unreadable(monkeypatch, path, how):
    """'chmod': read permission removed. 'forced': garbage bytes, and preprocess's own open raises PermissionError for this path only (runs as root)."""
    if how == 'chmod':
        path.chmod(0)
        return
    path.write_bytes(b'not an image')
    fail_python_open(monkeypatch, path, PermissionError(errno.EACCES, 'forced', str(path)))


def run_two_patients(tmp_path, monkeypatch, how, unreadable, add=None, workers=1):
    """Patient A's `unreadable` file (relative to its patient folder) is made unreadable; patient B is untouched. Returns (report, out, the file).
    'forced' needs workers=1: a worker process would not see the patch."""
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted)
    if add:
        add(folder, rows)
    write_plan(converted, patient='SYNTHETICPTB')
    bad = folder.parent / unreadable
    make_unreadable(monkeypatch, bad, how)
    return pre.preprocess(converted, out, SMALL, workers=workers), out, bad


def assert_patient_io_error(report, out):
    assert [(r.patient, r.folder, r.outcome, r.reason) for r in report.rows] == [
        ('SYNTHETICPTA', '', 'refused', pre.REASON_IO_ERROR), ('SYNTHETICPTB', 'plan_1', 'preprocessed', '')]
    assert [(r['patient'], r['folder'], r['reason']) for r in report_rows(out)] == [('SYNTHETICPTA', '', pre.REASON_IO_ERROR), ('SYNTHETICPTB', 'plan_1', '')]
    assert listing(out) == ['SYNTHETICPTB', 'preprocess_report.csv'] and (out / 'SYNTHETICPTB' / 'plan_1' / CT_FILE).is_file()


@pytest.mark.parametrize('how', UNREADABLE)
def test_an_unreadable_plan_file_fails_the_patient_with_the_io_error_and_the_next_patient_runs(tmp_path, monkeypatch, how):
    report, out, _ = run_two_patients(tmp_path, monkeypatch, how, 'plan_1/ptv.nii.gz')
    assert_patient_io_error(report, out)


@pytest.mark.parametrize('how', UNREADABLE)
def test_an_unreadable_registered_series_fails_the_patient_with_the_io_error(tmp_path, monkeypatch, how):
    report, out, _ = run_two_patients(tmp_path, monkeypatch, how, 'plan_1/series_MR_1.nii.gz', add=lambda folder, rows: add_series(folder, rows))
    assert_patient_io_error(report, out)


@pytest.mark.parametrize('how', UNREADABLE)
def test_an_unreadable_file_of_a_second_plan_drops_the_good_first_plan_too(tmp_path, monkeypatch, how):
    """plan_1 of the failing patient preprocesses cleanly first; the patient's I/O error still leaves no plan row or folder (#212)."""
    add = lambda folder, rows: write_plan(folder.parent.parent, plan='plan_2')  # noqa: E731
    report, out, _ = run_two_patients(tmp_path, monkeypatch, how, 'plan_2/ptv.nii.gz', add=add)
    assert_patient_io_error(report, out)


@NOT_ROOT
def test_an_unreadable_plan_file_fails_the_patient_with_the_io_error_under_two_workers(tmp_path, monkeypatch):
    """The real chmod end to end with workers=2: the patient's row crosses the process boundary (#212)."""
    report, out, _ = run_two_patients(tmp_path, monkeypatch, 'chmod', 'plan_1/ptv.nii.gz', workers=2)
    assert_patient_io_error(report, out)


def test_a_forced_read_error_puts_no_path_and_no_patient_name_in_a_log_or_the_report_text(tmp_path, monkeypatch, caplog):
    with caplog.at_level('DEBUG'):
        report, out, bad = run_two_patients(tmp_path, monkeypatch, 'forced', 'plan_1/ptv.nii.gz')
    assert_patient_io_error(report, out)
    text = (out / 'preprocess_report.csv').read_text()
    assert caplog.records and not any(s in caplog.text or s in text for s in (str(bad), str(tmp_path), 'forced'))
    assert 'SYNTHETICPTA' not in caplog.text
    assert [k for r in report_rows(out) for k, v in r.items() if 'SYNTHETICPTA' in v] == ['patient'] and text.count('SYNTHETICPTA') == 1


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
    assert report_rows(out) == [dict(patient='SYNTHETICPTA', folder='', outcome='refused', reason=pre.REASON_INTERNAL, files='', skipped_series='0', cropped='')]
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
    assert report_rows(out) == [dict(patient='SYNTHETICPTA', folder='', outcome='refused', reason=pre.REASON_INTERNAL, files='', skipped_series='0', cropped='')]
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
            'REASON_INTERNAL', 'REASON_NO_SERIES_STEPS'} <= set(reasons)
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
    steps = dataclasses.replace(FOUR_STEPS, new_spacing=(1.5, 1.5, 1.5), new_shape=(14, 4, 13))

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


# ---- task-03: the study series.json validator (headers are not read; files may be empty placeholders). ----

def study_row(file, modality='MR', kind='series'):
    return {**dict.fromkeys(conv.SERIES_COLUMNS, ''), 'file': file, 'kind': kind, 'modality': modality}


def write_study(folder, rows, *, files=None):
    """A study folder holding series.json for `rows` and an empty placeholder per file (default: every row's)."""
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(rows, folder / conv.SERIES_NAME)
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
    (pre.REASON_SERIES_CSV, lambda f: (f / conv.SERIES_NAME).write_text('[{"file": "series_MR_1.nii.gz", "kind": "series"}]', encoding='utf-8')),
    (pre.REASON_SERIES_CSV, lambda f: (f / conv.SERIES_NAME).write_text(json.dumps([study_row('series_MR_1.nii.gz') | {'extra': ''}]), encoding='utf-8')),
    (pre.REASON_SERIES_CSV, lambda f: (f / conv.SERIES_NAME).write_text('[]', encoding='utf-8')),
    (pre.REASON_SERIES_CSV, lambda f: (f / conv.SERIES_NAME).write_text('', encoding='utf-8')),
    (pre.REASON_SERIES_CSV, lambda f: sf.write([study_row('series_MR_1.nii.gz')] * 2, f / conv.SERIES_NAME)),
    (pre.REASON_SERIES_CSV, lambda f: sf.write([study_row('series_MR_1.nii.gz'), study_row('resample_series_MR_1.nii.gz')],
                                                           f / conv.SERIES_NAME)),
    (pre.REASON_KIND, lambda f: sf.write([study_row('series_MR_1.nii.gz'), study_row('series_MR_2.nii.gz', kind='ct')],
                                                      f / conv.SERIES_NAME)),
    (pre.REASON_KIND, lambda f: sf.write([study_row('series_MR_1.nii.gz', kind='')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_MODALITY, lambda f: sf.write([study_row('series_MR_1.nii.gz', 'RTDOSE')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_MODALITY, lambda f: sf.write([study_row('series_MR_1.nii.gz', 'mr')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_MODALITY, lambda f: sf.write([study_row('series_CT_1.nii.gz', 'MR')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_NAME, lambda f: sf.write([study_row('series_MR_\u0661.nii.gz')], f / conv.SERIES_NAME)),  # an Arabic-Indic digit
    (pre.REASON_STUDY_NAME, lambda f: sf.write([study_row('planning_ct.nii.gz')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_NAME, lambda f: sf.write([study_row('series_MR_1.nii')], f / conv.SERIES_NAME)),
    (pre.REASON_STUDY_NAME, lambda f: sf.write([study_row('../series_MR_1.nii.gz')], f / conv.SERIES_NAME)),
    (pre.REASON_FILE_MISSING, lambda f: (f / 'series_MR_1.nii.gz').unlink()),
    (pre.REASON_FILE_MISSING, lambda f: ((f / 'series_MR_1.nii.gz').unlink(), (f / 'series_MR_1.nii.gz').symlink_to(f / conv.SERIES_NAME))),
    (pre.REASON_FILE_MISSING, lambda f: ((f / 'series_MR_1.nii.gz').unlink(), (f / 'series_MR_1.nii.gz').mkdir())),
], ids=['encoding', 'keys', 'extra-key', 'zero-rows', 'empty-file', 'duplicate-file', 'duplicate-id', 'kind-ct', 'kind-blank',
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
    ('[{"file": "a"}, {"file": "a"}, {"file": "a"}]', 3),  # short items still count: the validator refuses them whole
    ('[]', 1),
    ('', 1),
    ('[{"file": "series_MR_1.nii.gz"}, {"file": "series_MR_2.nii.gz"}]', 2),  # wrong keys: items are still the series
])
def test_study_refusal_count(tmp_path, text, expected):
    (tmp_path / conv.SERIES_NAME).write_text(text, encoding='utf-8')
    assert pre.study_refusal_count(pre.read_series_json(tmp_path)[0]) == expected


def test_study_refusal_count_of_an_unparseable_file_is_one(tmp_path):
    (tmp_path / conv.SERIES_NAME).write_bytes(b'[{"file": "a"}, {"file": "\xff\xfe"}]')
    assert pre.study_refusal_count(pre.read_series_json(tmp_path)[0]) == 1


def test_study_refusal_count_of_a_valid_study(tmp_path):
    rows = [study_row(f'series_MR_{k}.nii.gz') for k in (1, 2, 3, 4)]
    assert pre.study_refusal_count(pre.read_series_json(write_study(tmp_path / 'study_1', rows))[0]) == 4


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
    """<root>/<patient>/<study>/ as convert writes it: series_<MOD>_<k>.nii.gz (a ramp on a constant) and series.json. Returns the folder."""
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
        ('study_1/series_CT_1', 'preprocessed', '', [names[0], 'series.json']),
        ('study_1/series_MR_1', 'preprocessed', '', [names[1]]),
        ('study_1/series_MR_2', 'preprocessed', '', [names[2]]),
        ('study_1/series_PT_1', 'preprocessed', '', [names[3]])]
    study = out / 'SYNTHETICPTA' / 'study_1'
    assert sorted(p.name for p in study.iterdir()) == sorted([*names, 'series.json'])
    assert [r['file'] for r in conv_rows(study)] == names
    assert {r['kind'] for r in conv_rows(study)} == {'series'} and conv_rows(study)[0]['series_instance_uid'] == 'SYNTHETICUIDCT1'
    assert all(sitk.ReadImage(str(study / name)).GetSize() == (10, 10, 10) for name in names)
    assert state(converted) == before


def conv_rows(folder):
    return sf.read(folder)


def test_mr_and_pt_intensities_are_not_rescaled_and_ct_is(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted, series=(('CT', 1), ('MR', 1), ('PT', 1)))
    steps = PreprocessSteps(resample=True, rescale=True)  # a resample to the grid it already has keeps the voxels
    pre.preprocess(converted, out, steps)
    study = out / 'SYNTHETICPTA' / 'study_1'
    assert sorted(p.name for p in study.iterdir()) == ['initialresample_series_MR_1.nii.gz', 'initialresample_series_PT_1.nii.gz',
                                                       'rescale_initialresample_series_CT_1.nii.gz', 'series.json']
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
    assert imaging_rows(report) == [('study_1/series_CT_1', 'preprocessed', '', ['rescale_series_CT_1.nii.gz', 'series.json']),
                                    ('study_1/series_MR_1', 'refused', pre.REASON_NO_SERIES_STEPS, [])]
    study = out / 'SYNTHETICPTA' / 'study_1'
    assert sorted(p.name for p in study.iterdir()) == ['rescale_series_CT_1.nii.gz', 'series.json']  # no MR file, whatever its name
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
        ('study_1/series_CT_1', 'preprocessed', '', [STUDY_CT, 'series.json']),
        ('study_1/series_MR_1', 'refused', pre.REASON_TOO_SMALL, []),
        ('study_1/series_MR_2', 'refused', pre.REASON_FILE_UNREADABLE, []),
        ('study_1/series_PT_1', 'preprocessed', '', [pt])]
    study = out / 'SYNTHETICPTA' / 'study_1'
    assert sorted(p.name for p in study.iterdir()) == sorted([STUDY_CT, pt, 'series.json'])  # no MR file, no leftover temp file
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


def test_a_too_small_series_is_refused_from_its_header_without_a_pixel_read(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    two_bad_series(converted)
    real, read = pre.read_image, []

    def spy(path):
        read.append(path.name)
        return real(path)
    monkeypatch.setattr(pre, 'read_image', spy)
    report = pre.preprocess(converted, out, SMALL)
    assert ('study_1/series_MR_1', 'refused', pre.REASON_TOO_SMALL) in [(f, o, r) for f, o, r, _ in imaging_rows(report)]
    assert 'series_MR_1.nii.gz' not in read  # #141


NOT_A_VOLUME = [  # convert writes 3D volumes of one real value per voxel only (#127, #138, #139)
    lambda: sitk.Image([4, 4], sitk.sitkFloat32),
    lambda: sitk.Image([4, 4, 4, 2], sitk.sitkFloat32),
    lambda: sitk.Image([4, 4, 4], sitk.sitkVectorFloat32, 3),
    lambda: sitk.Image([4, 4, 4], sitk.sitkComplexFloat32),
]
NOT_A_VOLUME_IDS = ['2d', '4d', 'multi-component', 'complex']


@pytest.mark.parametrize('image', NOT_A_VOLUME, ids=NOT_A_VOLUME_IDS)
@pytest.mark.parametrize('series, bad, steps', [
    ((('CT', 1), ('MR', 1), ('PT', 1)), 'MR_1', SMALL),
    ((('CT', 1), ('CT', 2)), 'CT_1', dataclasses.replace(SMALL, resample=False, harmonize_spacing=False, reshape=False)),
], ids=['mr-every-step', 'ct-rescale-only'])
def test_a_study_series_that_is_not_a_3d_real_volume_is_refused_on_its_own_row(tmp_path, image, series, bad, steps):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder = write_study_tree(converted, series=series)
    sitk.WriteImage(image(), str(folder / f'series_{bad}.nii.gz'))
    report = pre.preprocess(converted, out, steps)
    assert [(f, o, r) for f, o, r, _ in imaging_rows(report)] == [
        (f'study_1/series_{m}_{k}', *(('refused', pre.REASON_NOT_VOLUME) if f'{m}_{k}' == bad else ('preprocessed', ''))) for m, k in series]
    assert not list((out / 'SYNTHETICPTA' / 'study_1').glob(f'*{bad}*'))

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
        ('study_1', 'refused', pre.REASON_FILE_MISSING, []), ('study_2/series_PT_1', 'preprocessed', '', [STUDY_MR.replace('MR_1', 'PT_1'), 'series.json'])]
    assert sorted(p.name for p in (out / 'SYNTHETICPTA').iterdir()) == ['study_2']


def test_plans_come_first_and_studies_follow_in_numeric_order(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted, study='study_10', series=(('MR', 1),))
    write_study_tree(converted, study='study_2', series=(('MR', 1),))
    write_plan(converted)
    (converted / 'SYNTHETICPTA' / 'study_3').mkdir()  # no series.json
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
    (converted / 'SYNTHETICPTA' / 'study_1').mkdir(parents=True)  # no series.json
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
    assert report_rows(out) == [dict(patient='SYNTHETICPTA', folder='', outcome='refused', reason=pre.REASON_INTERNAL, files='', skipped_series='0', cropped='')]
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
    assert summary == ['preprocess: 1 plan rows (1 preprocessed, 0 refused), 4 imaging rows (2 preprocessed, 2 refused), 1 patient rows (0 refused, 1 skipped); 0 registered series skipped; 0 plans cropped, 0 with retained < 0.9; 0 replaced, 0 kept']


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
        return [json.loads(line) for line in (root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    before, after = read(converted), read(out)
    assert [r['case_dir'] for r in after] == [r['case_dir'] for r in before]
    assert [r['case_dir'].split('/', 1)[1] for r in after] == ['plan_1', 'study_1/series_CT_1', 'study_2/series_MR_1', 'study_2/series_MR_2']
    for kept, done in zip(before[1:], after[1:], strict=True):
        assert kept['image'] == f"{kept['case_dir'].split('/')[-1]}.nii.gz"
        assert done['image'] == pre.output_name(kept['image'], steps, ct=kept['modality'] == 'CT')
        assert (out / PID / done['case_dir'].split('/')[1] / done['image']).is_file()
    assert [r['image'] for r in after[2:]] == [pre.output_name(f'series_MR_{k}.nii.gz', steps) for k in (1, 2)]


# ---- rtplan-parquet task-03: plan.parquet is carried through on a non-geometric run, its isocenter index rewritten on a geometric one ----

def photon_converted(tmp_path):
    """A converted tree whose one plan folder is a photon plan (plan.parquet + its series.json row); returns (converted root, plan folder)."""
    import synthetic_dicom as sd
    from test_convert import BOX, PID, dose_spec, make_plan
    org, converted = tmp_path / 'org', tmp_path / 'converted'
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10)], rois=[BOX], plan_writer=sd.write_photon_rtplan,
              ct_meta=dict(tags=dict(PatientPosition='HFS')))  # beam_mu models HFS/HFP/FFS/FFP only
    conv.convert(org, converted)
    folder = converted / PID / 'plan_1'
    assert (folder / conv.PLAN_NAME).is_file()
    return converted, folder


@pytest.mark.parametrize('steps', [PreprocessSteps(rescale=True), PreprocessSteps(ahe=True)], ids=['rescale', 'ahe'])
def test_plan_parquet_is_copied_byte_identical_on_a_non_geometric_run(tmp_path, steps):
    converted, folder = photon_converted(tmp_path)
    out = tmp_path / 'out'
    report = pre.preprocess(converted, out, steps)
    assert [(r.folder, r.outcome, r.reason) for r in report.rows] == [('plan_1', 'preprocessed', '')]
    done = out / folder.parent.name / 'plan_1'
    assert (done / conv.PLAN_NAME).read_bytes() == (folder / conv.PLAN_NAME).read_bytes()
    assert conv.PLAN_NAME in plan_row(report).files
    assert [(r['file'], r['kind'], r['retained']) for r in sf.read(done) if r['kind'] == 'plan'] == [(conv.PLAN_NAME, 'plan', '')]


GEOMETRIC = [SMALL, PreprocessSteps(resample=True), PreprocessSteps(reshape=True, new_shape=(8, 8, 8))]


@pytest.mark.parametrize('steps', GEOMETRIC, ids=['four_steps', 'resample', 'reshape'])
def test_a_geometric_run_rewrites_the_plan_parquet_isocenter_index_for_the_final_ct(tmp_path, steps):
    import pyarrow.parquet as pq

    from rt_dataprep import plan_table
    converted, folder = photon_converted(tmp_path)
    out = tmp_path / 'out'
    report = pre.preprocess(converted, out, steps)
    assert [(r.folder, r.outcome, r.reason) for r in report.rows] == [('plan_1', 'preprocessed', '')]
    done = out / folder.parent.name / 'plan_1'
    final = sitk.ReadImage(str(done / pre.output_name(conv.CT_NAME, steps, ct=True, image=True)))
    before, after = pq.read_table(folder / conv.PLAN_NAME), pq.read_table(done / conv.PLAN_NAME)
    assert after.schema == plan_table.SCHEMA and after.num_rows == before.num_rows > 0
    index_columns = ('isocenter_i', 'isocenter_j', 'isocenter_k')
    for name in before.column_names:
        if name not in index_columns:
            assert after[name].equals(before[name]), name
    for name in index_columns:
        assert after[name].null_count == 0  # a non-nullable schema field does not stop nulls: check the data
    points = zip(*(after[name].to_pylist() for name in ('isocenter_x', 'isocenter_y', 'isocenter_z')), strict=True)
    expected = np.array([final.TransformPhysicalPointToContinuousIndex(p) for p in points])
    assert np.allclose(np.column_stack([after[name].to_numpy() for name in index_columns]), expected, rtol=0, atol=1e-6)
    if steps.reshape:  # the final grid is not the native one: the index really moved
        assert not np.allclose(np.column_stack([after[n].to_numpy() for n in index_columns]), np.column_stack([before[n].to_numpy() for n in index_columns]))
    assert conv.PLAN_NAME in plan_row(report).files
    assert sorted(p.name for p in done.iterdir() if p.name.startswith('.')) == []


def test_a_geometric_run_leaves_out_an_unreadable_or_non_schema_plan_parquet(tmp_path, caplog):
    import pyarrow as pa
    import pyarrow.parquet as pq
    converted, folder = photon_converted(tmp_path)
    for k, writer in enumerate([lambda path: path.write_bytes(b'not parquet'),
                                lambda path: pq.write_table(pa.table({'isocenter_x': [1.0]}), path)]):
        writer(folder / conv.PLAN_NAME)
        out = tmp_path / f'out_{k}'
        with caplog.at_level(logging.WARNING, logger='rt_dataprep.preprocess'):
            caplog.clear()
            report = pre.preprocess(converted, out, SMALL)
        assert [(r.folder, r.outcome, r.reason) for r in report.rows] == [('plan_1', 'preprocessed', '')]
        done = out / folder.parent.name / 'plan_1'
        assert not (done / conv.PLAN_NAME).exists() and conv.PLAN_NAME not in plan_row(report).files
        assert [r for r in sf.read(done) if r['kind'] == 'plan'] == []
        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == [f'preprocess: {pre.REASON_PLAN_INDEX}']


BEAM_FILE = pre.output_name(conv.BEAM_NAME, SMALL)


def add_beam(folder, array, n=12, **geometry):
    """A beam_mu volume (float32) and its series.json row beside the plan's other rows."""
    sitk.WriteImage(make_image(array, **(grid(n) | geometry)), str(folder / conv.BEAM_NAME))
    save_rows(folder, [*sf.read(folder), series_row(conv.BEAM_NAME, 'beam', modality='RTPLAN')])


@pytest.mark.parametrize(('shape', 'array'), [((4, 4, 4), np.ones((12, 12, 12))), ((4, 4, 4), box((9, 9, 9), (12, 12, 12))),
                                              ((16, 16, 16), np.ones((12, 12, 12)))], ids=['cut', 'cut_out_entirely', 'padded'])
@pytest.mark.parametrize('allow_crop', [False, True])
def test_a_beam_volume_cut_by_the_shape_is_resampled_and_neither_refuses_nor_counts_as_cropped(tmp_path, shape, array, allow_crop):
    steps = dataclasses.replace(PreprocessSteps(reshape=True, new_shape=shape), allow_crop=allow_crop, min_mask_retention=0.5)
    folder, _ = write_plan(tmp_path / 'converted', dose=box((5, 5, 5), (7, 7, 7), value=2.0), masks={'ptv': box((5, 5, 5), (7, 7, 7))})  # inside every final grid
    add_beam(folder, array)
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', steps)
    row = plan_row(report)
    assert (row.outcome, row.reason, row.cropped) == ('preprocessed', '', 0)
    assert row.retained == 1.0
    done = tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1'
    name = pre.output_name(conv.BEAM_NAME, steps)
    assert name == f'reshape_{conv.BEAM_NAME}' and name in row.files
    assert [(r['file'], r['retained']) for r in sf.read(done) if r['kind'] == 'beam'] == [(name, '')]
    image = sitk.ReadImage(str(done / name))
    final = sitk.ReadImage(str(done / pre.output_name(conv.CT_NAME, steps, ct=True, image=True)))
    assert image.GetPixelID() == sitk.sitkFloat32 and image.GetSize() == shape
    assert (image.GetOrigin(), image.GetSpacing(), image.GetDirection()) == (final.GetOrigin(), final.GetSpacing(), final.GetDirection())
    values = sitk.GetArrayFromImage(image)
    if shape[0] > 12:  # padded: the two voxels at each end lie outside the native CT
        assert (values[:2] == 0).all() and (values[-2:] == 0).all() and (values[:, :2] == 0).all() and (values[:, :, -2:] == 0).all()
        assert (values[2:-2, 2:-2, 2:-2] == 1).all()
    if array.max() and shape[0] < 12:
        assert (values == 1).all() or not values.any()  # a nearest/linear copy of a constant field or an empty one
    assert report_rows(tmp_path / 'out')[0]['cropped'] == '0'


def test_a_beam_volume_off_the_ct_grid_refuses_like_a_dose(tmp_path):
    folder, _ = write_plan(tmp_path / 'converted')
    add_beam(folder, np.ones((12, 12, 12)), origin=(-5.0, -5.5, -5.5))
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', SMALL)
    assert (plan_row(report).outcome, plan_row(report).reason) == ('refused', pre.REASON_GRID)
    assert not (tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1').exists()


def test_a_beam_volume_is_copied_under_the_dose_name_rule_on_a_non_geometric_run(tmp_path):
    folder, _ = write_plan(tmp_path / 'converted')
    add_beam(folder, np.full((12, 12, 12), 3.0))
    steps = PreprocessSteps(rescale=True)
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', steps)
    name = pre.output_name(conv.BEAM_NAME, steps)
    assert (plan_row(report).outcome, plan_row(report).cropped) == ('preprocessed', '')
    assert (tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1' / name).read_bytes() == (folder / conv.BEAM_NAME).read_bytes()


FLUENCE_NAME = 'fluence_mu.nii.gz'  # convert's name for it (a later task); preprocess only copies the file, so any bytes will do
FLUENCE_BYTES = b'SYNTHETIC fluence bytes'


def add_fluence(folder):
    """A fluence_mu file and its series.json row beside the plan's other rows."""
    (folder / FLUENCE_NAME).write_bytes(FLUENCE_BYTES)
    save_rows(folder, [*sf.read(folder), series_row(FLUENCE_NAME, 'fluence', modality='RTPLAN')])


@pytest.mark.parametrize('steps', [SMALL, PreprocessSteps(rescale=True)], ids=['geometric', 'non_geometric'])
def test_a_fluence_file_is_copied_unchanged_under_its_own_name_and_keeps_its_row(tmp_path, steps):
    folder, _ = write_plan(tmp_path / 'converted')
    add_fluence(folder)
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', steps)
    done = tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1'
    assert plan_row(report).outcome == 'preprocessed'
    assert (done / FLUENCE_NAME).read_bytes() == FLUENCE_BYTES
    assert [(r['file'], r['retained']) for r in sf.read(done) if r['kind'] == 'fluence'] == [(FLUENCE_NAME, '')]


def test_a_tree_without_a_fluence_row_is_unaffected(tmp_path):
    write_plan(tmp_path / 'converted')
    pre.preprocess(tmp_path / 'converted', tmp_path / 'out', SMALL)
    done = tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1'
    assert not (done / FLUENCE_NAME).exists() and all(r['kind'] != 'fluence' for r in sf.read(done))


def test_manifest_rows_are_unchanged_by_plan_parquet(tmp_path):
    import shutil

    from rt_dataprep import manifest as man
    converted, folder = photon_converted(tmp_path)
    bare = tmp_path / 'bare'
    shutil.copytree(converted, bare)
    bare_folder = bare / folder.parent.name / 'plan_1'
    (bare_folder / conv.PLAN_NAME).unlink()
    sf.write([r for r in sf.read(bare_folder) if r['kind'] != 'plan'], bare_folder / conv.SERIES_NAME)
    steps = dataclasses.replace(SMALL, new_shape=(8, 8, 8))
    outs = {}
    for name, root in (('with', converted), ('without', bare)):
        outs[name] = tmp_path / f'out_{name}'
        assert plan_row(pre.preprocess(root, outs[name], steps)).outcome == 'preprocessed'
    assert (outs['with'] / folder.parent.name / 'plan_1' / conv.PLAN_NAME).is_file()
    assert not (outs['without'] / folder.parent.name / 'plan_1' / conv.PLAN_NAME).exists()
    assert man.manifest(outs['with']) == man.manifest(outs['without']) == (1, 0)
    lines = {name: json.loads((root / 'manifest.jsonl').read_text(encoding='utf-8')) for name, root in outs.items()}
    assert lines['with'].pop('plan') == conv.PLAN_NAME  # the beam row stays in both trees, so only `plan` differs
    assert 'plan' not in lines['without']
    assert lines['with'] == lines['without']


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


def axis_rotation(axis, degrees):
    """The 3x3 rotation by `degrees` about `axis` (Rodrigues)."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(np.radians(degrees)) * cross + (1 - np.cos(np.radians(degrees))) * cross @ cross


def landmark(size, index, value=100.0):
    array = np.zeros(size[::-1])
    array[index[2], index[1], index[0]] = value
    return array


def run_plan(tmp_path, specs, steps=SMALL, **plan):
    """write_plan (with `plan`) plus the series of `specs` ((file, modality, array or raw bytes or None, reg_matrix text, geometry), in series.json order);
    preprocessed. Returns (report, output plan folder, converted plan folder)."""
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, rows = write_plan(converted, **plan)
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
    assert plan_row(report).outcome == 'preprocessed' and plan_row(report).files == [*BASE_FILES, MR_OUT, 'series.json']
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


def test_ac2d_an_oblique_rotation_with_oblique_ct_and_mr_directions_lands_the_landmark_at_the_mapped_index_and_not_at_the_inverse_one(tmp_path):
    ct_direction = tuple(axis_rotation((0, 1, 1), 20).ravel())
    geometry = dict(spacing=(1.0, 1.5, 2.0), origin=(-7.0, -9.0, -6.0), direction=tuple(axis_rotation((1, 0, 1), 15).ravel()))
    size, index = (16, 12, 10), (6, 5, 3)
    final = pre.process_ct(make_image(np.zeros((12, 12, 12)), **grid(12), direction=ct_direction), SMALL)
    p = np.array(make_image(np.zeros(size[::-1]), **geometry).TransformIndexToPhysicalPoint(index))
    q = np.array(final.TransformIndexToPhysicalPoint((5, 4, 5)))  # the landmark is built to land on this final voxel centre
    matrix = np.eye(4)
    matrix[:3, :3] = axis_rotation((1, 2, -1), 34)
    matrix[:3, 3] = q - matrix[:3, :3] @ p
    _, done, _ = run_plan(tmp_path, [mr_spec(landmark(size, index), matrix, geometry)], direction=ct_direction)
    image = read_out(done)
    assert np.allclose(image.GetDirection(), ct_direction)
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
    assert plan_row(report).files == [*BASE_FILES, 'series.json'] and sorted(p.name for p in done.iterdir()) == sorted(plan_row(report).files)
    assert [r['file'] for r in conv_rows(done)] == BASE_FILES


def test_ac6_a_series_entirely_outside_the_ct_is_left_out_and_one_partly_outside_is_kept(tmp_path):
    ones = np.ones((8, 8, 8))
    report, done, _ = run_plan(tmp_path, [mr_spec(ones, rigid(0, (500, 0, 0))), mr_spec(ones, rigid(0, (4, 0, 0)), file='series_MR_2.nii.gz')])
    kept = pre.output_name('series_MR_2.nii.gz', SMALL)
    assert plan_row(report).skipped_series == 1 and plan_row(report).files == [*BASE_FILES, kept, 'series.json']
    values = sitk.GetArrayFromImage(read_out(done, kept))
    interior, outside = coverage(read_out(done, kept), MR_GRID, (8, 8, 8), rigid(0, (4, 0, 0)))  # asserts both are nonempty: partial coverage
    assert np.abs(values[interior] - 1.0).max() < 1e-4 and (values[outside] == 0).all() and [r['file'] for r in conv_rows(done)][-1] == kept


def test_a_series_whose_nonzero_voxels_all_miss_the_ct_is_kept_when_its_fov_overlaps_it(tmp_path):
    ends = []
    for x in (0, 7):  # the overlap of one of the two is zero-valued background only
        end = np.zeros((8, 8, 8))
        end[:, :, x] = 1.0
        ends.append(end)
    report, done, _ = run_plan(tmp_path, [mr_spec(ends[0], rigid(0, (4, 0, 0))), mr_spec(ends[1], rigid(0, (4, 0, 0)), file='series_MR_2.nii.gz')])
    assert plan_row(report).skipped_series == 0
    assert plan_row(report).files == [*BASE_FILES, MR_OUT, pre.output_name('series_MR_2.nii.gz', SMALL), 'series.json']
    assert sorted(not sitk.GetArrayFromImage(read_out(done, name)).any() for name in (MR_OUT, pre.output_name('series_MR_2.nii.gz', SMALL))) == [False, True]  # a copy: a view of a temporary image reads freed memory


def test_an_all_zero_series_is_left_out_by_where_it_sits_not_its_values(tmp_path):
    zeros = np.zeros((8, 8, 8))
    report, done, _ = run_plan(tmp_path, [mr_spec(zeros, rigid(0, (500, 0, 0))), mr_spec(zeros, rigid(0, (4, 0, 0)), file='series_MR_2.nii.gz')])
    kept = pre.output_name('series_MR_2.nii.gz', SMALL)
    assert plan_row(report).skipped_series == 1 and plan_row(report).files == [*BASE_FILES, kept, 'series.json']
    assert not sitk.GetArrayFromImage(read_out(done, kept)).any()


def test_a_series_outside_the_ct_is_never_resampled(tmp_path, monkeypatch):
    monkeypatch.setattr(pre, 'process_series', lambda *a: pytest.fail('a series outside the ct was resampled'))
    report, _, _ = run_plan(tmp_path, [mr_spec(np.ones((8, 8, 8)), rigid(0, (500, 0, 0)))])
    assert plan_row(report).skipped_series == 1


def test_a_registered_ct_without_rescale_is_filled_with_its_own_minimum_outside_its_fov(tmp_path):
    geometry, matrix = dict(spacing=(1.25,) * 3, origin=(-3.875, -4.125, -2.625)), rigid(0, (4, 0, 0))
    values = np.full((8, 8, 8), 100.0)
    values[0, 0, 0] = -3024.0  # SYNTHETIC scanner padding value
    steps = dataclasses.replace(SMALL, rescale=False)
    name = pre.output_name('series_CT_1.nii.gz', steps, ct=True)
    _, done, _ = run_plan(tmp_path, [mr_spec(values, matrix, geometry, 'series_CT_1.nii.gz', 'CT')], steps=steps)
    _, outside = coverage(read_out(done, name), geometry, (8, 8, 8), matrix)
    assert (sitk.GetArrayFromImage(read_out(done, name))[outside] == -3024.0).all()


@pytest.mark.parametrize('image', NOT_A_VOLUME, ids=NOT_A_VOLUME_IDS)
def test_a_series_that_is_not_a_3d_scalar_volume_is_left_out_and_counted(tmp_path, image):
    path = tmp_path / 'SYNTHETIC_series.nii.gz'
    sitk.WriteImage(image(), str(path))
    report, done, _ = run_plan(tmp_path, [mr_spec(path.read_bytes(), 'identity')])
    assert (plan_row(report).outcome, plan_row(report).skipped_series) == ('preprocessed', 1)
    assert plan_row(report).files == [*BASE_FILES, 'series.json'] and not (done / MR_OUT).exists()


def test_ac7_an_unreadable_series_file_is_left_out_and_counted(tmp_path):
    report, done, _ = run_plan(tmp_path, [mr_spec(b'SYNTHETIC not a nifti', 'identity')])
    assert (plan_row(report).outcome, plan_row(report).skipped_series) == ('preprocessed', 1)
    assert plan_row(report).files == [*BASE_FILES, 'series.json'] and not (done / MR_OUT).exists()


def test_ac7_an_internal_error_while_resampling_a_series_still_stops_the_run(tmp_path, monkeypatch):
    real = sitk.Resample

    def resample(*args):
        if isinstance(args[2], sitk.AffineTransform):  # only the series resample passes an affine
            raise RuntimeError('SYNTHETICPTA')
        return real(*args)
    monkeypatch.setattr(sitk, 'Resample', resample)
    with pytest.raises(pre.PreprocessAborted):
        run_plan(tmp_path, [mr_spec(np.ones((4, 4, 4)), rigid(0, (1, 0, 0)))])
    assert report_rows(tmp_path / 'out') == [dict(patient='SYNTHETICPTA', folder='', outcome='refused', reason=pre.REASON_INTERNAL, files='', skipped_series='0', cropped='')]


def test_ac9_rescale_only_resamples_an_mr_onto_the_native_ct_grid_and_keeps_its_name(tmp_path):
    matrix = rigid(0, (1.0, -2.0, 1.0))
    report, done, folder = run_plan(tmp_path, [mr_spec(landmark(MR_SIZE, (2, 3, 3)), matrix)], PreprocessSteps(rescale=True))
    assert plan_row(report).files == ['rescale_planning_ct.nii.gz', 'dose.nii.gz', 'ptv.nii.gz', 'bladder.nii.gz', 'applicator_a.txt', 'series_MR_1.nii.gz', 'series.json']
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
    assert plan_row(report).files == [*BASE_FILES, *names, 'series.json']
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


FOV = dataclasses.replace(SMALL, series_fov_mask=True)
FOV_OUT = f'fovmask_{MR_OUT}'


@pytest.mark.parametrize('matrix, geometry', [(rigid(0, (4, 0, 0)), MR_GRID),
                                              (rigid(17, (2.3, -1.1, 0.6)), dict(MR_GRID, spacing=(1.1, 0.9, 1.7)))],
                         ids=['translated', 'rotated_sub_voxel_odd_spacing'])
def test_fov_mask_is_the_final_grid_uint8_where_the_series_lands_and_listed_right_after_its_series(tmp_path, matrix, geometry):
    report, done, _ = run_plan(tmp_path, [mr_spec(np.full((8, 8, 8), 3.0), matrix, geometry)], steps=FOV)
    assert plan_row(report).files == [BASE_FILES[0], f'fovmask_{CT_FILE}', *BASE_FILES[1:], MR_OUT, FOV_OUT, 'series.json']
    assert [r['file'] for r in conv_rows(done)] == [*BASE_FILES, MR_OUT]  # no mask row in series.json
    mask, ct, image = read_out(done, FOV_OUT), read_out(done, CT_FILE), read_out(done)
    assert mask.GetPixelID() == sitk.sitkUInt8
    assert (mask.GetSize(), mask.GetOrigin(), mask.GetSpacing(), mask.GetDirection()) == (ct.GetSize(), ct.GetOrigin(), ct.GetSpacing(), ct.GetDirection())
    flags, values = sitk.GetArrayFromImage(mask), sitk.GetArrayFromImage(image)
    assert (flags == 1).any() and (flags == 0).any()
    assert ((flags == 1) == (values != 0)).all() and (values[flags == 1] == 3.0).all()  # exact: the mask is geometry, never loosened


def test_fov_mask_of_a_registered_ct_filled_with_its_minimum_equals_the_mr_mask(tmp_path):
    steps, matrix = dataclasses.replace(FOV, rescale=False), rigid(0, (4, 0, 0))
    ct_out = pre.output_name('series_CT_1.nii.gz', steps, ct=True)
    _, done, _ = run_plan(tmp_path, [mr_spec(np.full((8, 8, 8), 3.0), matrix), mr_spec(np.full((8, 8, 8), -1000.0), matrix, file='series_CT_1.nii.gz', modality='CT')], steps=steps)
    assert (sitk.GetArrayFromImage(read_out(done, f'fovmask_{ct_out}')) == sitk.GetArrayFromImage(read_out(done, f'fovmask_{MR_OUT}'))).all()


@pytest.mark.parametrize('spec', [mr_spec(np.ones((4, 4, 4)), ''), mr_spec(np.ones((8, 8, 8)), rigid(0, (500, 0, 0))), mr_spec(b'SYNTHETIC not a nifti', 'identity')],
                         ids=['no_matrix', 'no_overlap', 'unreadable'])
def test_a_series_left_out_gets_no_fov_mask(tmp_path, spec):
    report, done, _ = run_plan(tmp_path, [spec], steps=FOV)
    assert plan_row(report).skipped_series == 1
    assert plan_row(report).files == [BASE_FILES[0], f'fovmask_{CT_FILE}', *BASE_FILES[1:], 'series.json']
    assert [p.name for p in done.iterdir() if p.name.startswith('fovmask_')] == [f'fovmask_{CT_FILE}']


def test_no_fov_mask_is_written_by_default(tmp_path):
    report, done, _ = run_plan(tmp_path, [mr_spec(np.full((8, 8, 8), 3.0), rigid(0, (4, 0, 0)))])
    assert plan_row(report).files == [*BASE_FILES, MR_OUT, 'series.json'] and not any(p.name.startswith('fovmask_') for p in done.iterdir())


def test_the_fov_mask_leaves_every_other_output_unchanged(tmp_path):
    outs = []
    for steps in (SMALL, FOV):
        root = tmp_path / str(steps.series_fov_mask)
        folder, rows = write_plan(root / 'converted')
        rows[0]['frame_of_reference_uid'] = CT_FRAME
        sitk.WriteImage(make_image(np.full((8, 8, 8), 3.0), **MR_GRID), str(folder / 'series_MR_1.nii.gz'))
        rows.append(series_row('series_MR_1.nii.gz', 'series', modality='MR', reg_matrix=matrix_text(rigid(0, (4, 0, 0))), frame_of_reference_uid=MR_FRAME))
        save_rows(folder, rows)
        write_study_tree(root / 'converted')
        pre.preprocess(root / 'converted', root / 'out', steps)
        outs.append(root / 'out' / 'SYNTHETICPTA')
    off, on = ({p.relative_to(out) for p in out.rglob('*') if p.is_file() and not p.name.startswith('fovmask_')} for out in outs)
    assert off == on and len(off) > 7  # plan files and series, study series and csvs
    assert sum(p.name.startswith('fovmask_') for p in outs[1].rglob('*')) == 4  # CT, aligned MR, two study images
    for name in off:
        if name.suffix == '.gz':
            a, b = sitk.ReadImage(str(outs[0] / name)), sitk.ReadImage(str(outs[1] / name))
            assert (sitk.GetArrayFromImage(a) == sitk.GetArrayFromImage(b)).all()
            assert (a.GetOrigin(), a.GetSpacing(), a.GetDirection(), a.GetPixelID()) == (b.GetOrigin(), b.GetSpacing(), b.GetDirection(), b.GetPixelID())
        else:
            assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes()


# ---- AHE task-02: the ahe step for the planning CT and study series. Volumes are linear gradients, never constant or two-level. ----

AHE = PreprocessSteps(ahe=True)
SMALL_AHE = dataclasses.replace(SMALL, ahe=True)
AHE_CT_FILE = 'reshape_resample_ahe_rescale_initialresample_planning_ct.nii.gz'
AHE_STUDY_CT = 'reshape_resample_ahe_rescale_initialresample_series_CT_1.nii.gz'


def reference_equalize(image):
    """The array of `image` min-max scaled in float32, then skimage's equalize_adapthist with the spec constants: never the code under test."""
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    normalized = (array - array.min()) / (array.max() - array.min())
    equalized = equalize_adapthist(normalized, nbins=128, clip_limit=0.05)
    return normalized if equalized.max() == equalized.min() else equalized


def reference_pipeline(image, steps):
    """initial resample -> rescale -> reference equalize -> final spacing -> reshape, each step on only when `steps` says so."""
    if steps.resample:
        image = pre.resample_to(image, steps.new_spacing)
    if steps.rescale:
        image = pre.rescale(image)
    equalized = sitk.GetImageFromArray(reference_equalize(image))
    equalized.CopyInformation(image)
    image = equalized
    if steps.harmonize_spacing:
        image = pre.resample_to(image, steps.final_spacing)
    if steps.reshape:
        image = pre.crop_or_pad(image, steps.new_shape, pre.minimum(image))
    return image


def header(image):
    return image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection()


def ahe_plan(root, n=16, **plan):
    """write_plan with a gradient CT on an n^3 grid and no dose or masks (their 12^3 grid would not match)."""
    return write_plan(root, n=n, dose=None, masks={}, applicator=False, **plan)


def test_ahe_field_defaults_off():
    assert PreprocessSteps().ahe is False


def test_ac1a_ahe_only_equalizes_the_ct_on_its_own_non_default_grid(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, _ = ahe_plan(converted, spacing=1.5)
    pre.preprocess(converted, out, AHE)
    source, done = sitk.ReadImage(str(folder / conv.CT_NAME)), sitk.ReadImage(str(out / 'SYNTHETICPTA' / 'plan_1' / f'ahe_{conv.CT_NAME}'))
    assert source.GetOrigin() == (-11.25,) * 3 and source.GetSpacing() == (1.5,) * 3
    expected = reference_equalize(source)
    values = sitk.GetArrayFromImage(done)
    np.testing.assert_array_equal(values, expected)
    assert values.dtype == np.float32 and done.GetPixelID() == sitk.sitkFloat32 and values.min() == 0 and values.max() == 1
    assert not np.array_equal(values, (sitk.GetArrayFromImage(source) - np.float32(-1000)) / np.float32(1200))  # the step did something
    assert header(done) == header(source)


def test_ac1b_small_plus_ahe_equals_the_reference_pipeline(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, _ = ahe_plan(converted, spacing=1.5)
    pre.preprocess(converted, out, SMALL_AHE)
    done = sitk.ReadImage(str(out / 'SYNTHETICPTA' / 'plan_1' / AHE_CT_FILE))
    expected = reference_pipeline(sitk.ReadImage(str(folder / conv.CT_NAME)), SMALL_AHE)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(done), sitk.GetArrayFromImage(expected))
    assert header(done) == header(expected) and done.GetSize() == (10, 10, 10)
    assert done.GetPixelID() == sitk.sitkFloat32
    assert sitk.GetArrayFromImage(done).min() >= 0 and sitk.GetArrayFromImage(done).max() <= 1


def test_a_ct_is_cast_to_float32_before_the_min_max(tmp_path):
    ct = sitk.Cast(make_image(np.linspace(-1000, 200, 16 ** 3).reshape(16, 16, 16), spacing=(1.5,) * 3, origin=(-11.25,) * 3), sitk.sitkInt16)
    done = pre.process_ct(ct, AHE)
    assert done.GetPixelID() == sitk.sitkFloat32
    np.testing.assert_array_equal(sitk.GetArrayFromImage(done), reference_equalize(ct))
    assert header(done) == header(ct)


def test_a_constant_ct_becomes_zeros_without_calling_skimage_and_the_plan_is_preprocessed(tmp_path, monkeypatch):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, _ = ahe_plan(converted, n=8)
    sitk.WriteImage(make_image(np.full((8, 8, 8), -300.0), **grid(8)), str(folder / conv.CT_NAME))
    monkeypatch.setattr(pre, 'equalize_adapthist', lambda *a, **k: pytest.fail('skimage was called on a constant volume'))
    report = pre.preprocess(converted, out, AHE)
    assert plan_row(report).outcome == 'preprocessed'
    done = sitk.ReadImage(str(out / 'SYNTHETICPTA' / 'plan_1' / f'ahe_{conv.CT_NAME}'))
    assert done.GetPixelID() == sitk.sitkFloat32 and not sitk.GetArrayFromImage(done).any()


def test_ac4_study_mr_pt_and_ct_series_are_each_equalized(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder = write_study_tree(converted, series=(('CT', 1), ('MR', 1), ('PT', 1)), n=16)
    pre.preprocess(converted, out, SMALL_AHE)
    study = out / 'SYNTHETICPTA' / 'study_1'
    names = {'CT': AHE_STUDY_CT, 'MR': 'reshape_resample_ahe_initialresample_series_MR_1.nii.gz', 'PT': 'reshape_resample_ahe_initialresample_series_PT_1.nii.gz'}
    assert sorted(p.name for p in study.iterdir()) == sorted([*names.values(), 'series.json'])
    for modality, name in names.items():
        expected = reference_pipeline(sitk.ReadImage(str(folder / f'series_{modality}_1.nii.gz')), SMALL_AHE if modality == 'CT' else dataclasses.replace(SMALL_AHE, rescale=False))
        done = sitk.ReadImage(str(study / name))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(done), sitk.GetArrayFromImage(expected))
        assert header(done) == header(expected) and done.GetPixelID() == sitk.sitkFloat32


def test_ac4_an_ahe_only_run_processes_an_mr_study_series(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder = write_study_tree(converted, series=(('MR', 1),), n=16)
    report = pre.preprocess(converted, out, AHE)
    assert imaging_rows(report) == [('study_1/series_MR_1', 'preprocessed', '', ['ahe_series_MR_1.nii.gz', 'series.json'])]
    source, done = sitk.ReadImage(str(folder / 'series_MR_1.nii.gz')), sitk.ReadImage(str(out / 'SYNTHETICPTA' / 'study_1' / 'ahe_series_MR_1.nii.gz'))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(done), reference_equalize(source))
    assert header(done) == header(source)


def test_an_ahe_only_run_is_not_refused_as_no_step(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    ahe_plan(converted)
    assert plan_row(pre.preprocess(converted, out, AHE)).outcome == 'preprocessed'


def test_output_name_puts_ahe_between_resample_and_rescale_for_image_volumes_only():
    assert pre.output_name('planning_ct.nii.gz', SMALL_AHE, ct=True, image=True) == AHE_CT_FILE
    assert pre.output_name('series_MR_1.nii.gz', SMALL_AHE, image=True) == 'reshape_resample_ahe_initialresample_series_MR_1.nii.gz'
    assert pre.output_name('planning_ct.nii.gz', AHE, ct=True, image=True) == 'ahe_planning_ct.nii.gz'
    assert pre.output_name('dose.nii.gz', SMALL_AHE) == 'reshape_resample_initialresample_dose.nii.gz'
    assert pre.output_name('ptv.nii.gz', AHE) == 'ptv.nii.gz'
    assert pre.output_name('planning_ct.nii.gz', PreprocessSteps(rescale=True), image=True) == 'planning_ct.nii.gz'  # rescale_ still needs ct
    assert pre.output_name('planning_ct.nii.gz', PreprocessSteps(rescale=True), ct=True, image=True) == 'rescale_planning_ct.nii.gz'
    assert pre.output_name('series_MR_1.nii.gz', PreprocessSteps(), image=True) == 'series_MR_1.nii.gz'


def test_study_file_reads_the_id_through_the_named_group():
    assert pre.series_id('reshape_resample_ahe_initialresample_series_MR_1.nii.gz') == 'series_MR_1'
    assert pre.series_id('reshape_resample_ahe_rescale_initialresample_series_CT_2.nii.gz') == 'series_CT_2'
    assert pre.series_id('series_PT_3.nii.gz') == 'series_PT_3'
    assert pre.series_id('ahe_ahe_series_MR_1.nii.gz') is None
    assert pre.STUDY_FILE.pattern == r'(reshape_)?(resample_)?(ahe_)?(rescale_)?(initialresample_)?(?P<id>series_(CT|MR|PT)_[0-9]+)\.nii\.gz'


def test_a_failing_equalize_is_one_fixed_internal_reason(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError('SYNTHETICPTA')
    monkeypatch.setattr(pre, 'equalize_adapthist', boom)
    with pytest.raises(CaseRefused) as caught:
        pre.process_ct(make_image(np.linspace(0, 1, 8 ** 3).reshape(8, 8, 8)), AHE)
    assert str(caught.value) == pre.REASON_INTERNAL


def test_the_module_docstring_describes_ahe_and_not_geometric_only_series():
    doc = getattr(pre, '__doc__')  # noqa: B009
    assert 'ahe' in doc and 'geometric ones only' not in doc


# ---- AHE task-03: the aligned plan series is equalized on its native grid, then resampled once; the zero fill. Gradients, never constants. ----

AHE_SIDE = 24  # CLAHE's kernel is 1/8 of each axis: an 8^3 volume has 1-voxel tiles and equalizes to all 1
MR_RAMP = np.linspace(1.0, 500.0, AHE_SIDE ** 3).reshape((AHE_SIDE,) * 3)
CT_RAMP = np.linspace(-1000.0, 200.0, AHE_SIDE ** 3).reshape((AHE_SIDE,) * 3)
AHE_MR_OUT = 'reshape_resample_ahe_initialresample_series_MR_1.nii.gz'
AHE_PT_OUT = 'reshape_resample_ahe_initialresample_series_PT_1.nii.gz'


def reference_series(native, final_ct, matrix, rescale_hu=False):
    """Resample(reference equalize(native)) onto final_ct with the series transform, float32, fill 0: never the code under test."""
    equalized = sitk.GetImageFromArray(reference_equalize(pre.rescale(native) if rescale_hu else native))
    equalized.CopyInformation(native)
    return sitk.Resample(equalized, final_ct, pre.series_transform(matrix), sitk.sitkLinear, 0, sitk.sitkFloat32)


@pytest.mark.parametrize('matrix, geometry', [(rigid(0, (4, 0, 0)), MR_GRID),
                                              (rigid(17, (2.3, -1.1, 0.6)), dict(MR_GRID, spacing=(1.1, 0.9, 1.7), direction=ROTATED))],
                         ids=['translated', 'rotated_with_a_non_identity_native_direction'])
def test_ac3_an_aligned_mr_is_equalized_on_its_native_grid_then_resampled_and_is_zero_outside_its_fov(tmp_path, matrix, geometry):
    _, done, folder = run_plan(tmp_path, [mr_spec(MR_RAMP, matrix, geometry)], steps=SMALL_AHE)
    image, ct, native = read_out(done, AHE_MR_OUT), read_out(done, AHE_CT_FILE), sitk.ReadImage(str(folder / 'series_MR_1.nii.gz'))
    values, expected = sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(reference_series(native, ct, matrix))
    np.testing.assert_array_equal(values, expected)
    interior, outside = coverage(image, geometry, (AHE_SIDE,) * 3, matrix)
    assert (values[outside] == 0).all() and values[interior].min() >= 0 and values[interior].max() > 0 and len(np.unique(values[interior])) > 10
    assert image.GetPixelID() == sitk.sitkFloat32 and header(image) == header(ct)  # read back on both sides


@pytest.mark.parametrize('rescale', [False, True])
def test_a_non_rescaled_aligned_ct_series_is_filled_with_zero_not_its_own_minimum(tmp_path, rescale):
    steps, matrix = dataclasses.replace(SMALL_AHE, rescale=rescale), rigid(0, (4, 0, 0))
    out_name = pre.output_name('series_CT_1.nii.gz', steps, ct=True, image=True)
    _, done, folder = run_plan(tmp_path, [mr_spec(CT_RAMP, matrix, file='series_CT_1.nii.gz', modality='CT')], steps=steps)
    image, ct = read_out(done, out_name), read_out(done, pre.output_name(conv.CT_NAME, steps, ct=True, image=True))
    native = sitk.ReadImage(str(folder / 'series_CT_1.nii.gz'))
    assert pre.minimum(native) == -1000  # a fill by the native minimum would show here
    values = sitk.GetArrayFromImage(image)
    np.testing.assert_array_equal(values, sitk.GetArrayFromImage(reference_series(native, ct, matrix, rescale_hu=rescale)))
    _, outside = coverage(image, MR_GRID, (AHE_SIDE,) * 3, matrix)
    assert (values[outside] == 0).all() and values.min() == 0 and len(np.unique(values)) > 10


def test_aligned_series_names_carry_ahe_for_mr_pt_and_ct_and_dose_and_masks_do_not(tmp_path):
    matrix = rigid(0, (4, 0, 0))
    specs = [mr_spec(MR_RAMP, matrix), mr_spec(MR_RAMP, matrix, file='series_PT_1.nii.gz', modality='PT'), mr_spec(CT_RAMP, matrix, file='series_CT_1.nii.gz', modality='CT')]
    report, done, _ = run_plan(tmp_path, specs, steps=SMALL_AHE)
    ct_out = 'reshape_resample_ahe_rescale_initialresample_series_CT_1.nii.gz'
    assert plan_row(report).files == [AHE_CT_FILE, DOSE_FILE, PTV_FILE, BLADDER_FILE, 'applicator_a.txt', AHE_MR_OUT, AHE_PT_OUT, ct_out, 'series.json']
    assert [r['file'] for r in conv_rows(done)] == plan_row(report).files[:-1]


def test_ac2_dose_and_mask_outputs_are_identical_with_and_without_ahe(tmp_path):
    outs = {}
    for steps in (SMALL, SMALL_AHE):
        _, done, _ = run_plan(tmp_path / str(steps.ahe), [], steps=steps)
        outs[steps.ahe] = done
    for name in (DOSE_FILE, PTV_FILE, BLADDER_FILE):
        a, b = read_out(outs[False], name), read_out(outs[True], name)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(a), sitk.GetArrayFromImage(b))
        assert (a.GetOrigin(), a.GetSpacing(), a.GetDirection(), a.GetPixelID()) == (b.GetOrigin(), b.GetSpacing(), b.GetDirection(), b.GetPixelID())
    assert (outs[False] / 'applicator_a.txt').read_bytes() == (outs[True] / 'applicator_a.txt').read_bytes()


def test_the_fov_mask_is_written_for_the_ahe_series_and_ahe_does_not_change_it(tmp_path):
    matrix, masks = rigid(0, (4, 0, 0)), {}
    for steps in (FOV, dataclasses.replace(SMALL_AHE, series_fov_mask=True)):
        report, done, _ = run_plan(tmp_path / str(steps.ahe), [mr_spec(MR_RAMP, matrix)], steps=steps)
        name = pre.output_name('series_MR_1.nii.gz', steps, image=True)
        assert plan_row(report).files[-3:] == [name, f'fovmask_{name}', 'series.json']
        masks[steps.ahe] = read_out(done, f'fovmask_{name}')
    assert masks[True].GetPixelID() == sitk.sitkUInt8
    np.testing.assert_array_equal(sitk.GetArrayFromImage(masks[True]), sitk.GetArrayFromImage(masks[False]))
    assert header(masks[True]) == header(masks[False]) and sitk.GetArrayFromImage(masks[True]).any()


def test_a_failing_equalize_in_process_series_is_one_fixed_internal_reason(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError('SYNTHETICPTA')
    monkeypatch.setattr(pre, 'equalize_adapthist', boom)
    with pytest.raises(CaseRefused) as caught:
        pre.process_series(make_image(MR_RAMP, **MR_GRID), FINAL, np.eye(4), False, False, True)
    assert str(caught.value) == pre.REASON_INTERNAL


# ---------------------------------------------------------------- --overwrite

PAST = 1_000_000_000  # an mtime long before any run of these tests


def tree_state(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file()}


def no_hidden(out):
    return [n for n in listing(out) if n.startswith('.')]


def stale_output(tmp_path):
    """One preprocessed patient, then a marker added and every file aged: the state a rerun after a code fix finds."""
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    pre.preprocess(converted, out, SMALL)
    (out / 'SYNTHETICPTA' / 'marker.txt').write_text('stale')
    for p in (out / 'SYNTHETICPTA').rglob('*'):
        os.utime(p, (PAST, PAST))
    return converted, out


def test_overwrite_without_the_flag_skips_an_existing_folder_and_leaves_it_alone(tmp_path):
    converted, out = stale_output(tmp_path)
    before = tree_state(out / 'SYNTHETICPTA')
    rows = pre.preprocess(converted, out, SMALL).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'skipped', pre.REASON_EXISTS)]
    assert tree_state(out / 'SYNTHETICPTA') == before


def test_without_overwrite_a_dangling_link_at_the_output_name_is_an_existing_output(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    out.mkdir()
    (out / 'SYNTHETICPTA').symlink_to(out / 'SYNTHETIC_missing', target_is_directory=True)
    rows = pre.preprocess(converted, out, SMALL).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'skipped', pre.REASON_EXISTS)]
    assert (out / 'SYNTHETICPTA').is_symlink() and no_hidden(out) == []


def test_overwrite_replaces_a_stale_folder_with_a_fresh_run(tmp_path, caplog):
    converted, out = stale_output(tmp_path)
    started = time.time()
    caplog.set_level(logging.INFO)
    report = pre.preprocess(converted, out, SMALL, overwrite=True)
    assert plan_row(report).outcome == 'preprocessed'
    fresh = tmp_path / 'fresh'
    pre.preprocess(converted, fresh, SMALL)
    assert sorted(tree_state(out / 'SYNTHETICPTA')) == sorted(tree_state(fresh / 'SYNTHETICPTA'))
    assert not (out / 'SYNTHETICPTA' / 'marker.txt').exists()
    written = [p for p in (out / 'SYNTHETICPTA').rglob('*') if p.is_file()]
    assert written and all(p.stat().st_mtime >= started - 1 for p in written)  # nothing of the old folder was merged in
    assert no_hidden(out) == []
    assert any(m.endswith('; 1 replaced, 0 kept') for m in caplog.messages)


def test_overwrite_with_a_failed_build_keeps_the_old_folder(tmp_path, monkeypatch, caplog):
    converted, out = stale_output(tmp_path)
    before = tree_state(out / 'SYNTHETICPTA')

    def fail(image, path):
        raise OSError(f'disk full writing {path} under {out}')
    monkeypatch.setattr(conv, 'write_atomic', fail)
    caplog.set_level(logging.DEBUG)
    rows = pre.preprocess(converted, out, SMALL, overwrite=True).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'refused', pre.REASON_IO_ERROR)]
    assert tree_state(out / 'SYNTHETICPTA') == before
    assert no_hidden(out) == []
    assert [m for m in caplog.messages if 'SYNTHETICPTA' in m or str(tmp_path) in m] == []
    assert any(m.endswith('; 0 replaced, 0 kept') for m in caplog.messages)


def test_overwrite_with_a_failed_partial_rename_restores_the_old_folder(tmp_path, monkeypatch, caplog):
    converted, out = stale_output(tmp_path)
    before = tree_state(out / 'SYNTHETICPTA')
    real = dicom_io.rename_dir

    def rename(src, dst):
        if src.name == '.SYNTHETICPTA.partial':
            raise OSError(f'cannot move {src} to {dst}')
        real(src, dst)
    monkeypatch.setattr(dicom_io, 'rename_dir', rename)
    caplog.set_level(logging.DEBUG)
    rows = pre.preprocess(converted, out, SMALL, overwrite=True).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'refused', pre.REASON_IO_ERROR)]
    assert tree_state(out / 'SYNTHETICPTA') == before
    assert no_hidden(out) == []
    assert [m for m in caplog.messages if 'SYNTHETICPTA' in m or str(tmp_path) in m] == []


def test_recovery_restores_or_deletes_the_old_folder_with_or_without_overwrite(tmp_path):
    converted, out = stale_output(tmp_path)
    before = tree_state(out / 'SYNTHETICPTA')
    (out / 'SYNTHETICPTA').rename(out / '.SYNTHETICPTA.old')
    (out / '.SYNTHETICPTA.partial').mkdir()
    rows = pre.preprocess(converted, out, SMALL).rows
    assert [(r.outcome, r.reason) for r in rows] == [('skipped', pre.REASON_EXISTS)]
    assert tree_state(out / 'SYNTHETICPTA') == before and no_hidden(out) == []
    (out / '.SYNTHETICPTA.old').mkdir()
    (out / '.SYNTHETICPTA.old' / 'x').write_text('x')
    pre.preprocess(converted, out, SMALL)
    assert tree_state(out / 'SYNTHETICPTA') == before and no_hidden(out) == []


def test_overwrite_keeps_the_old_folder_when_every_plan_is_refused(tmp_path, caplog):
    converted, out = stale_output(tmp_path)
    before = tree_state(out / 'SYNTHETICPTA')
    (converted / 'SYNTHETICPTA' / 'plan_1' / 'ptv.nii.gz').unlink()
    caplog.set_level(logging.INFO)
    rows = pre.preprocess(converted, out, SMALL, overwrite=True).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('plan_1', 'refused', pre.REASON_FILE_MISSING)]
    assert tree_state(out / 'SYNTHETICPTA') == before and no_hidden(out) == []
    assert any(m.endswith('; 0 replaced, 1 kept') for m in caplog.messages)


def test_overwrite_keeps_the_old_folder_of_a_patient_without_cases(tmp_path, caplog):
    converted, out = stale_output(tmp_path)
    before = tree_state(out / 'SYNTHETICPTA')
    (converted / 'SYNTHETICPTA' / 'plan_1' / conv.SERIES_NAME).unlink()
    caplog.set_level(logging.INFO)
    rows = pre.preprocess(converted, out, SMALL, overwrite=True).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', 'skipped', pre.REASON_NO_CASES)]
    assert tree_state(out / 'SYNTHETICPTA') == before and no_hidden(out) == []
    assert any(m.endswith('; 0 replaced, 1 kept') for m in caplog.messages)


def test_overwrite_refuses_a_second_folder_whose_name_differs_only_in_case(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    write_plan(converted, patient='syntheticpta')
    pre.preprocess(converted, tmp_path / 'ref', SMALL, overwrite=True)  # the same source into an empty root: what the first folder's output must equal
    canary = out / '.syntheticpta.old'  # named for the refused (second) folder: a recover run for it would consume it
    canary.mkdir(parents=True)
    (canary / 'keep.txt').write_text('keep')
    rows = pre.preprocess(converted, out, SMALL, overwrite=True).rows
    first, second = sorted(rows, key=lambda r: r.patient)
    assert (first.patient, first.outcome) == ('SYNTHETICPTA', 'preprocessed')
    assert (second.patient, second.folder, second.outcome, second.reason) == ('syntheticpta', '', 'refused', pre.REASON_NAME_CLAIMED)
    assert (out / 'SYNTHETICPTA' / 'plan_1').is_dir() and tree_state(out / 'SYNTHETICPTA') == tree_state(tmp_path / 'ref' / 'SYNTHETICPTA')
    assert no_hidden(out) == [canary.name] and (canary / 'keep.txt').read_text() == 'keep' and not (out / 'syntheticpta').exists()


def test_without_overwrite_names_differing_in_case_are_both_built_as_today(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    write_plan(converted, patient='syntheticpta')
    rows = pre.preprocess(converted, out, SMALL).rows
    assert sorted((r.patient, r.outcome) for r in rows) == [('SYNTHETICPTA', 'preprocessed'), ('syntheticpta', 'preprocessed')]


def test_overwrite_leaves_an_unrelated_output_folder_alone(tmp_path):
    converted, out = stale_output(tmp_path)
    (out / 'SYNTHETICOTHER').mkdir()
    (out / 'SYNTHETICOTHER' / 'keep.txt').write_text('keep')
    (out / '.SYNTHETICOTHER.old').mkdir()  # a hidden sibling: no recover or publish for SYNTHETICPTA may touch it
    (out / '.SYNTHETICOTHER.old' / 'keep.txt').write_text('keep')
    pre.preprocess(converted, out, SMALL, overwrite=True)
    assert (out / '.SYNTHETICOTHER.old' / 'keep.txt').read_text() == 'keep'
    assert (out / 'SYNTHETICOTHER' / 'keep.txt').read_text() == 'keep'


# ---- patient processes (#171): N=2 equals N=1, parent-side claims, abort ----

def three_patients(root, names=('SYNTHETICPTA', 'SYNTHETICPTB', 'SYNTHETICPTC')):
    for name in names:
        write_plan(root, patient=name)


def written_bytes(root):
    """{relative path: bytes} of every file under root except the report copies (their names carry a timestamp)."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file() and 'report' not in p.name}


def rows_of(report):
    return [(r.patient, r.folder, r.outcome, r.reason, r.files, r.skipped_series) for r in report.rows]


def call_guarded(fn):
    """Run fn in a daemon thread and fail instead of hanging; return the exception it raised, if any."""
    import threading
    outcome = {}

    def target():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            outcome['error'] = exc
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(120)
    assert not thread.is_alive(), 'the call did not return'
    return outcome.get('error')


def test_two_patient_processes_write_the_same_files_and_rows_as_one(tmp_path):
    converted = tmp_path / 'converted'
    three_patients(converted)
    one = pre.preprocess(converted, tmp_path / 'out1', SMALL)
    two = pre.preprocess(converted, tmp_path / 'out2', SMALL, workers=2)
    assert rows_of(two) == rows_of(one) and len(one.rows) >= 3
    assert all(r.outcome == 'preprocessed' for r in one.rows)  # not a vacuous equality
    assert written_bytes(tmp_path / 'out2') == written_bytes(tmp_path / 'out1') and written_bytes(tmp_path / 'out1')


def test_auto_workers_probe_then_a_real_pool_write_the_same_files_and_rows_as_one(tmp_path, monkeypatch):
    from rt_dataprep import pool
    converted = tmp_path / 'converted'
    three_patients(converted)
    one = pre.preprocess(converted, tmp_path / 'out1', SMALL)
    sized = []
    monkeypatch.setattr(pool, 'size_workers', lambda *args: sized.append(args) or 2)  # a small fixture would size to 1: force a spawned pool
    auto = pre.preprocess(converted, tmp_path / 'out2', SMALL, workers='auto')
    assert len(sized) == 1 and sized[0][3] == 2  # sized once, for the two patients after the probe
    assert rows_of(auto) == rows_of(one) and len(one.rows) >= 3
    assert all(r.outcome == 'preprocessed' for r in one.rows)
    assert written_bytes(tmp_path / 'out2') == written_bytes(tmp_path / 'out1') and written_bytes(tmp_path / 'out1')


def test_auto_workers_never_size_when_every_patient_is_skipped(tmp_path, monkeypatch):
    from rt_dataprep import pool
    converted = tmp_path / 'converted'
    three_patients(converted)
    out = tmp_path / 'out'
    pre.preprocess(converted, out, SMALL)  # all three done; the rerun skips them
    sizes = []
    monkeypatch.setattr(pool, 'size_workers', lambda *args: sizes.append(args) or 2)
    report = pre.preprocess(converted, out, SMALL, workers='auto')
    assert report.rows and not any(r.outcome == 'preprocessed' for r in report.rows) and not sizes


@pytest.mark.parametrize('workers', [1, 2])
def test_under_overwrite_a_case_repeat_is_refused_by_the_parent_and_the_first_in_sorted_order_wins(tmp_path, workers):
    converted = tmp_path / 'converted'
    three_patients(converted, ('SYNTHETICPTA', 'syntheticpta', 'SYNTHETICPTC'))  # sorted: ...PTA, ...PTC, syntheticpta
    rows = rows_of(pre.preprocess(converted, tmp_path / 'out', SMALL, workers=workers, overwrite=True))
    claimed = [r for r in rows if r[3] == pre.REASON_NAME_CLAIMED]
    assert claimed == [('syntheticpta', '', conv.OUTCOME_REFUSED, pre.REASON_NAME_CLAIMED, [], 0)]
    assert [r[0] for r in rows if r[1] == 'plan_1'] == ['SYNTHETICPTA', 'SYNTHETICPTC']
    assert not (tmp_path / 'out' / 'syntheticpta').exists()


def test_without_overwrite_a_case_repeat_is_held_and_the_rows_equal_those_of_one_process(tmp_path):
    converted = tmp_path / 'converted'
    three_patients(converted, ('SYNTHETICPTA', 'syntheticpta', 'SYNTHETICPTC'))
    one = rows_of(pre.preprocess(converted, tmp_path / 'out1', SMALL))
    two = rows_of(pre.preprocess(converted, tmp_path / 'out2', SMALL, workers=2))
    assert two == one and not any(r[3] == pre.REASON_NAME_CLAIMED for r in one)
    assert [r[0] for r in one if r[1] == 'plan_1'] == ['SYNTHETICPTA', 'SYNTHETICPTC', 'syntheticpta']  # the repeat ran, as today


def test_a_case_repeat_starts_only_after_its_twin_ends(tmp_path):
    import pool_workers
    converted = tmp_path / 'converted'
    three_patients(converted, ('SYNTHETICPTA', 'syntheticpta', 'SYNTHETICPTC'))
    pre.preprocess(converted, tmp_path / 'out3', SMALL, workers=3, _patient_fn=pool_workers.preprocess_record)  # three slots: only the hold key keeps the twins apart
    times = {n: [float((converted / n / f).read_text()) for f in ('start', 'end')] for n in ('SYNTHETICPTA', 'syntheticpta')}
    assert times['syntheticpta'][0] >= times['SYNTHETICPTA'][1]


def test_an_internal_error_in_one_patient_aborts_a_two_process_run_with_the_report_and_no_message(tmp_path, capfd):
    import pool_workers
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    for name in ('a', 'boom', 'c', 'd', 'e', 'f'):
        (converted / name).mkdir(parents=True)
    error = call_guarded(lambda: pre.preprocess(converted, out, SMALL, workers=2, _patient_fn=pool_workers.preprocess_patient))
    assert isinstance(error, pre.PreprocessAborted) and str(error) == pre.REASON_INTERNAL
    assert [(r['patient'], r['outcome'], r['reason']) for r in report_rows(out)] == [('a', conv.OUTCOME_SKIPPED, ''), ('boom', conv.OUTCOME_REFUSED, pre.REASON_INTERNAL)]  # two slots: 'a' and 'boom' only
    err = capfd.readouterr().err
    assert pool_workers.SECRET not in err and 'RuntimeError' in err  # the worker logged the type and site, never the message


# ---------------------------------------------------------------- allow-crop task-01: the trailing `retained` column

def key_orders(folder):
    return {tuple(item) for item in sf.read(folder)}


def test_every_series_json_preprocess_writes_has_the_retained_column_and_convert_keeps_its_header(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    write_study_tree(converted)
    pre.preprocess(converted, out, SMALL)
    assert key_orders(converted / 'SYNTHETICPTA' / 'plan_1') == {tuple(conv.SERIES_COLUMNS)}
    assert pre.PREPROCESSED_SERIES_COLUMNS == [*conv.SERIES_COLUMNS, 'retained']
    for folder in ('plan_1', 'study_1'):
        assert key_orders(out / 'SYNTHETICPTA' / folder) == {tuple(pre.PREPROCESSED_SERIES_COLUMNS)}
    assert {r['retained'] for r in pre.read_series_json(out / 'SYNTHETICPTA' / 'study_1')[0]} == {''}
    assert {r['kind']: r['retained'] for r in pre.read_series_json(out / 'SYNTHETICPTA' / 'plan_1')[0]} == {'ct': '', 'dose': '1.0', 'mask': '1.0', 'applicator': ''}


@pytest.mark.parametrize('columns, malformed', [
    (conv.SERIES_COLUMNS, False),
    ([*conv.SERIES_COLUMNS, 'retained'], False),
    ([*conv.SERIES_COLUMNS, 'other'], True),
    (['retained', *conv.SERIES_COLUMNS], False),
    (conv.SERIES_COLUMNS[:-1], True),
])
def test_read_series_json_accepts_exactly_the_two_key_sets(tmp_path, columns, malformed):
    (tmp_path / 'series.json').write_text(json.dumps([dict.fromkeys(columns, 'x')]), encoding='utf-8')  # key order is not compared
    assert pre.read_series_json(tmp_path)[1] is malformed


DICOM = {'manufacturer': 'SYNTHETIC_MAKER', 'min_max': {'min': '1', 'max': '2'}, 'distinct': {'distinct': [['A', 'B'], ['C']]}, 'slices': 3}


@pytest.mark.parametrize('columns', [conv.SERIES_COLUMNS, pre.PREPROCESSED_SERIES_COLUMNS])
def test_read_series_json_accepts_a_dicom_object_on_every_item_in_both_key_sets_and_returns_it_as_written(tmp_path, columns):
    rows = [dict.fromkeys(columns, 'x') | {'dicom': DICOM}, dict.fromkeys(columns, 'y') | {'dicom': {}}]  # the values are not strings: exempt
    sf.write(rows, tmp_path / conv.SERIES_NAME)
    assert pre.read_series_json(tmp_path) == (rows, False)


def test_read_series_json_reads_a_file_without_dicom_as_before(tmp_path):
    rows = [dict.fromkeys(conv.SERIES_COLUMNS, 'x'), dict.fromkeys(conv.SERIES_COLUMNS, 'y')]
    sf.write(rows, tmp_path / conv.SERIES_NAME)
    assert pre.read_series_json(tmp_path) == (rows, False)


@pytest.mark.parametrize('dicom', ['x', 1, [], None, [{}]])
def test_read_series_json_calls_a_dicom_that_is_not_an_object_malformed(tmp_path, dicom):
    (tmp_path / conv.SERIES_NAME).write_text(json.dumps([dict.fromkeys(conv.SERIES_COLUMNS, 'x') | {'dicom': dicom}]), encoding='utf-8')
    assert pre.read_series_json(tmp_path)[1] is True


def test_read_series_json_calls_dicom_on_some_items_only_malformed(tmp_path):
    rows = [dict.fromkeys(conv.SERIES_COLUMNS, 'x') | {'dicom': {}}, dict.fromkeys(conv.SERIES_COLUMNS, 'y')]
    for ordered in (rows, rows[::-1]):
        (tmp_path / conv.SERIES_NAME).write_text(json.dumps(ordered), encoding='utf-8')
        rows_read, malformed = pre.read_series_json(tmp_path)
        assert malformed is True and len(rows_read) == 2


def test_preprocess_keeps_the_dicom_object_of_every_row_unchanged_between_the_series_columns_and_retained(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    plan, _ = write_plan(converted)
    study = write_study_tree(converted)
    who = lambda r: {'who': [r['kind'], r['modality'], r['roi_name']], 'nested': DICOM}  # noqa: E731 (names change; these fields do not)
    for folder in (plan, study):
        sf.write([r | {'dicom': who(r)} for r in sf.read(folder)], folder / conv.SERIES_NAME)
    pre.preprocess(converted, out, SMALL)
    for folder in ('plan_1', 'study_1'):
        got = sf.read(out / 'SYNTHETICPTA' / folder)
        assert got and {tuple(item) for item in got} == {(*conv.SERIES_COLUMNS, 'dicom', 'retained')}
        assert all(item['dicom'] == who(item) for item in got)


@pytest.mark.parametrize('columns', [conv.SERIES_COLUMNS, pre.PREPROCESSED_SERIES_COLUMNS])
def test_read_series_json_still_refuses_a_missing_or_extra_key(tmp_path, columns):
    for keys in (columns[1:], [*columns, 'extra']):
        (tmp_path / 'series.json').write_text(json.dumps([dict.fromkeys(keys, 'x')]), encoding='utf-8')
        assert pre.read_series_json(tmp_path)[1] is True


def test_preprocessing_a_preprocessed_tree_succeeds_and_sets_retained_from_this_run(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    write_plan(tmp_path / 'converted')
    write_study_tree(tmp_path / 'converted')
    pre.preprocess(tmp_path / 'converted', first, SMALL)
    for path in first.rglob('series.json'):  # as if the first run had set it: the second must not carry it over
        rows = pre.read_series_json(path.parent)[0]
        sf.write([r | {'retained': 'STALE'} for r in rows], path)
    report = pre.preprocess(first, second, SMALL)
    assert {(r.folder, r.outcome) for r in report.rows} == {('plan_1', 'preprocessed'), ('study_1/series_CT_1', 'preprocessed'),
                                                            ('study_1/series_MR_1', 'preprocessed')}
    for folder in ('plan_1', 'study_1'):
        rows, malformed = pre.read_series_json(second / 'SYNTHETICPTA' / folder)
        assert not malformed and {r['retained'] for r in rows} == ({'', '1.0'} if folder == 'plan_1' else {''})  # never 'STALE'


# ---------------------------------------------------------------- allow-crop task-02: allow_crop, native-grid `retained`, `cropped`

CUT = PreprocessSteps(reshape=True, new_shape=(6, 6, 6), allow_crop=True)  # a centred crop of the 12^3 plan: native indices 3..8 stay
CUT_OFF = dataclasses.replace(CUT, allow_crop=False)
HALF_OUT = box((3, 3, 0), (9, 9, 6))  # z indices 0..5: half of them (0..2) are outside the final grid
# 1 on z 0..2 (outside), 3 on z 3..5 (inside): the in-grid share of the summed dose is 9 / 12
DOSE_SPLIT = np.where(np.arange(12)[:, None, None] < 3, 1.0, 3.0) * HALF_OUT


def series_cells(out, folder='plan_1'):
    """{file: retained cell} of an output plan's or study's series.json."""
    return {r['file']: r['retained'] for r in pre.read_series_json(out / 'SYNTHETICPTA' / folder)[0]}


def test_allow_crop_defaults_off_and_adds_no_filename_prefix(tmp_path):
    assert PreprocessSteps().allow_crop is False
    write_plan(tmp_path / 'converted')
    pre.preprocess(tmp_path / 'converted', tmp_path / 'off', dataclasses.replace(SMALL, allow_crop=False))
    pre.preprocess(tmp_path / 'converted', tmp_path / 'on', dataclasses.replace(SMALL, allow_crop=True))
    assert listing(tmp_path / 'off' / 'SYNTHETICPTA' / 'plan_1') == listing(tmp_path / 'on' / 'SYNTHETICPTA' / 'plan_1')
    assert pre.output_name('dose.nii.gz', CUT) == pre.output_name('dose.nii.gz', CUT_OFF)


def test_allow_crop_writes_a_plan_whose_dose_and_mask_are_cut_on_the_final_ct_grid(tmp_path, caplog):
    write_plan(tmp_path / 'converted', dose=DOSE_SPLIT, masks={'ptv': HALF_OUT, 'bladder': box((5, 5, 5), (8, 8, 8))})
    with caplog.at_level('INFO'):
        report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', CUT)
    assert (plan_row(report).outcome, plan_row(report).reason) == ('preprocessed', '')
    folder = tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1'
    final = sitk.ReadImage(str(folder / pre.output_name(conv.CT_NAME, CUT, ct=True, image=True)))
    assert final.GetSize() == (6, 6, 6)
    for name in ('dose', 'ptv', 'bladder'):
        image = sitk.ReadImage(str(folder / pre.output_name(f'{name}.nii.gz', CUT)))
        assert (image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()) == \
            (final.GetSize(), final.GetSpacing(), final.GetOrigin(), final.GetDirection())
    assert sitk.GetArrayFromImage(sitk.ReadImage(str(folder / pre.output_name('ptv.nii.gz', CUT)))).sum() > 0
    assert not any('SYNTHETIC' in record.getMessage() for record in caplog.records)


def test_retained_is_the_native_in_grid_fraction_of_the_summed_dose_and_of_the_mask(tmp_path):
    write_plan(tmp_path / 'converted', dose=DOSE_SPLIT, masks={'ptv': HALF_OUT, 'bladder': box((5, 5, 5), (8, 8, 8))})
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', CUT)
    cells = series_cells(tmp_path / 'out')
    assert cells == {pre.output_name(conv.CT_NAME, CUT, ct=True, image=True): '', pre.output_name('dose.nii.gz', CUT): '0.75',
                     pre.output_name('ptv.nii.gz', CUT): '0.5', pre.output_name('bladder.nii.gz', CUT): '1.0', 'applicator_a.txt': ''}
    assert all('np.' not in cell and (not cell or float(cell) == float(cell)) for cell in cells.values())
    assert plan_row(report).cropped == 2  # the dose and the ptv


@pytest.mark.parametrize(('masks', 'tail'), [({'ptv': HALF_OUT}, '1 plans cropped, 1 with retained < 0.9'),  # retained 0.5
                                            ({'bladder': box((5, 5, 5), (8, 8, 8))}, '0 plans cropped, 0 with retained < 0.9')])  # 1.0
def test_the_run_log_counts_cropped_plans_and_those_with_a_low_retained(tmp_path, caplog, masks, tail):
    write_plan(tmp_path / 'converted', dose=None, masks=masks)
    with caplog.at_level('INFO'):
        pre.preprocess(tmp_path / 'converted', tmp_path / 'out', CUT)
    assert f'; {tail}; ' in caplog.records[-1].getMessage()


def test_a_cut_plan_without_allow_crop_still_refuses_and_records_no_cropped_count(tmp_path):
    write_plan(tmp_path / 'converted', dose=DOSE_SPLIT)
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', CUT_OFF)
    assert (plan_row(report).outcome, plan_row(report).reason, plan_row(report).cropped) == ('refused', pre.REASON_CROP, '')


def test_check_crop_is_called_as_before_without_allow_crop_and_never_with_it(tmp_path, monkeypatch):
    write_plan(tmp_path / 'converted')
    calls = []
    real = pre.check_crop
    monkeypatch.setattr(pre, 'check_crop', lambda native, final: (calls.append(1), real(native, final))[1])
    pre.preprocess(tmp_path / 'converted', tmp_path / 'off', dataclasses.replace(SMALL, allow_crop=False))
    assert len(calls) == 3  # dose and two masks
    pre.preprocess(tmp_path / 'converted', tmp_path / 'on', dataclasses.replace(SMALL, allow_crop=True))
    assert len(calls) == 3


@pytest.mark.parametrize('what', ['dose', 'mask'])
def test_allow_crop_refuses_a_dose_or_mask_entirely_outside_the_final_grid(tmp_path, what):
    outside = box((0, 0, 0), (2, 2, 2), value=2.0)  # indices 0..1: outside the kept 3..8
    plan = dict(dose=outside) if what == 'dose' else dict(masks={'ptv': outside})
    write_plan(tmp_path / 'converted', **plan)
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', CUT)
    assert (plan_row(report).outcome, plan_row(report).reason) == ('refused', pre.REASON_CROP)
    assert not (tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1').exists()


def test_a_retained_of_zero_alone_does_not_refuse_when_the_output_still_holds_voxels(tmp_path):
    # 2 mm native, 1 mm final, 4 voxels: the one dose voxel (centre x = 3) lies outside the grid (edge at 2), but linear
    # interpolation puts a quarter of it into the final centre at 1.5
    single = np.zeros((12, 12, 12))
    single[6, 6, 7] = 1.0
    steps = PreprocessSteps(harmonize_spacing=True, final_spacing=(1, 1, 1), reshape=True, new_shape=(4, 4, 4), allow_crop=True)
    write_plan(tmp_path / 'converted', spacing=2.0, dose=single, masks={})
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', steps)
    assert (plan_row(report).outcome, plan_row(report).cropped) == ('preprocessed', 1)
    assert series_cells(tmp_path / 'out')[pre.output_name('dose.nii.gz', steps)] == '0.0'
    dose = sitk.ReadImage(str(tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1' / pre.output_name('dose.nii.gz', steps)))
    assert sitk.GetArrayFromImage(dose).sum() > 0


@pytest.mark.parametrize('allow_crop', [False, True])
def test_a_dose_that_resampling_empties_refuses_with_and_without_allow_crop(tmp_path, allow_crop):
    # 12 voxels at 1 mm into 5 mm voxels: the final centres (+-2.5 mm) are 2 mm from the one voxel at 0.5 mm, beyond linear reach
    single = np.zeros((12, 12, 12))
    single[6, 6, 6] = 1.0
    steps = PreprocessSteps(harmonize_spacing=True, final_spacing=(5, 5, 5), allow_crop=allow_crop)
    write_plan(tmp_path / 'converted', dose=single, masks={})
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', steps)
    assert (plan_row(report).outcome, plan_row(report).reason) == ('refused', pre.REASON_CROP)
    assert not (tmp_path / 'out' / 'SYNTHETICPTA' / 'plan_1').exists()


def test_allow_crop_with_a_min_mask_retention_above_the_cropped_masks_retention_refuses(tmp_path):
    write_plan(tmp_path / 'converted', masks={'ptv': HALF_OUT})
    strict = dataclasses.replace(CUT, min_mask_retention=0.9)
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', strict)
    assert (plan_row(report).outcome, plan_row(report).reason) == ('refused', pre.REASON_MASK_RETENTION)
    lax = dataclasses.replace(CUT, min_mask_retention=0.4)
    assert plan_row(pre.preprocess(tmp_path / 'converted', tmp_path / 'out2', lax)).outcome == 'preprocessed'


@pytest.mark.parametrize('steps', [PreprocessSteps(allow_crop=True), PreprocessSteps(allow_crop=True, rescale=True, ahe=True)])
def test_allow_crop_without_a_geometric_step_raises_before_any_mkdir(tmp_path, steps):
    write_plan(tmp_path / 'converted')
    with pytest.raises(ValueError) as caught:
        pre.preprocess(tmp_path / 'converted', tmp_path / 'out', steps)
    assert str(caught.value) == (pre.REASON_NO_STEPS if not (steps.rescale or steps.ahe) else pre.REASON_ALLOW_CROP_STEPS)
    assert not (tmp_path / 'out').exists()


def test_the_allow_crop_reason_is_a_fixed_constant_distinct_from_no_steps():
    assert pre.REASON_ALLOW_CROP_STEPS != pre.REASON_NO_STEPS and 'allow_crop' in pre.REASON_ALLOW_CROP_STEPS


@pytest.mark.parametrize('steps', [dataclasses.replace(SMALL, allow_crop=False), dataclasses.replace(SMALL, allow_crop=True),
                                   dataclasses.replace(CUT, new_shape=(12, 12, 12)), PreprocessSteps(reshape=True, new_shape=(6, 6, 6), allow_crop=True)])
def test_a_dose_and_mask_away_from_the_edges_have_retained_exactly_one(tmp_path, steps):
    inner = box((4, 4, 4), (8, 8, 8), value=0.1)  # float32 values whose sum is not exact: the 1.0 is by the bounding-box test
    write_plan(tmp_path / 'converted', dose=inner, masks={'ptv': box((4, 4, 4), (8, 8, 8))})
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', steps)
    cells = series_cells(tmp_path / 'out')
    assert (plan_row(report).outcome, plan_row(report).cropped) == ('preprocessed', 0)
    assert [cells[pre.output_name(f'{n}.nii.gz', steps)] for n in ('dose', 'ptv')] == ['1.0', '1.0']


def test_a_native_dose_grid_larger_than_the_final_grid_with_values_only_inside_has_retained_one_and_cropped_zero(tmp_path):
    write_plan(tmp_path / 'converted', dose=box((3, 3, 3), (9, 9, 9), value=0.1), masks={})
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', CUT)
    assert series_cells(tmp_path / 'out')[pre.output_name('dose.nii.gz', CUT)] == '1.0'
    assert plan_row(report).cropped == 0


def test_a_negligible_tail_past_the_edge_counts_as_cropped_even_when_retained_rounds_to_one(tmp_path):
    tail = box((3, 3, 3), (9, 9, 9), value=1.0)
    tail[3, 6, 2] = 1e-9  # one native voxel centre just outside the kept indices 3..8
    write_plan(tmp_path / 'converted', dose=tail, masks={})
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', CUT)
    assert plan_row(report).cropped == 1
    assert 0.99999 < float(series_cells(tmp_path / 'out')[pre.output_name('dose.nii.gz', CUT)]) <= 1.0


def test_native_retained_of_an_empty_image_is_one_and_of_a_cut_box_the_in_grid_share():
    final = blank((6, 6, 6), origin=(-2.5, -2.5, -2.5))  # centres -2.5..2.5: native centres 0..11 mm shifted below
    native = make_image(np.zeros((12, 12, 12)), origin=(-5.5, -5.5, -5.5))
    assert pre.native_retained(native, final) == 1.0
    cut = make_image(HALF_OUT, origin=(-5.5, -5.5, -5.5))
    assert pre.native_retained(cut, final) == 0.5


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize(('first', 'share'), [(-1.5, 0.5), (-0.5, 1.0), (8.5, 0.5), (7.5, 1.0)])  # two voxels of 1: the one centred at 9.5 is out, at -0.5 in
def test_native_retained_keeps_the_low_half_voxel_edge_and_drops_the_high_one(axis, first, share):
    final = blank((10, 10, 10))
    origin = [4.0, 4.0, 4.0]
    origin[axis] = first
    size = [1, 1, 1]
    size[axis] = 2
    native = make_image(np.ones(size[::-1]), origin=tuple(origin))
    assert pre.native_retained(native, final) == share


def test_a_non_geometric_run_leaves_retained_and_cropped_blank(tmp_path):
    write_plan(tmp_path / 'converted')
    write_study_tree(tmp_path / 'converted')
    report = pre.preprocess(tmp_path / 'converted', tmp_path / 'out', PreprocessSteps(rescale=True))
    assert {r.cropped for r in report.rows} == {''}
    assert {r['cropped'] for r in report_rows(tmp_path / 'out')} == {''}
    for folder in ('plan_1', 'study_1'):
        assert set(series_cells(tmp_path / 'out', folder).values()) == {''}


def test_the_cropped_column_is_an_int_on_preprocessed_plan_rows_only(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted, dose=DOSE_SPLIT)  # a cut dose
    write_plan(converted, plan='plan_2', masks={'ptv': HALF_OUT, 'bladder': HALF_OUT})  # two cut masks
    write_plan(converted, plan='plan_3')  # nothing cut
    write_plan(converted, plan='plan_4', masks={'ptv': box((0, 0, 0), (2, 2, 2))})  # refused
    write_study_tree(converted)
    report = pre.preprocess(converted, out, CUT)
    assert [(r.folder, r.outcome, r.cropped) for r in report.rows] == [
        ('plan_1', 'preprocessed', 1), ('plan_2', 'preprocessed', 2), ('plan_3', 'preprocessed', 0), ('plan_4', 'refused', ''),
        ('study_1/series_CT_1', 'preprocessed', ''), ('study_1/series_MR_1', 'preprocessed', '')]
    assert [r['cropped'] for r in report_rows(out)] == ['1', '2', '0', '', '', '']
    again = pre.preprocess(converted, out, CUT)  # the patient's output exists: a patient-level row
    assert [(r.folder, r.outcome, r.cropped) for r in again.rows] == [('', 'skipped', '')]
    assert [r['cropped'] for r in report_rows(out)] == ['']


# ---------------------------------------------------------------- json-outputs task-01: series.json, and the legacy series.csv refusal

def good_series_bytes(folder):
    return (folder / conv.SERIES_NAME).read_bytes()


def full_row_json(**extra):
    return json.dumps(series_row('a.nii.gz', 'ct') | extra)


# each: the bytes of a series.json that read_series_json calls malformed, from the plan's good bytes
SERIES_JSON_BREAKS = {
    'invalid json': lambda good: b'[{"file": ',
    'deep nesting': lambda good: b'[' * 100_000,
    'duplicate key': lambda good: ('[' + full_row_json()[:-1] + ', "file": "b.nii.gz"}]').encode(),
    'bom': lambda good: b'\xef\xbb\xbf' + good,
    'not utf-8': lambda good: good + b'\xff\xfe',
    'empty file': lambda good: b'',
    'not an array': lambda good: b'{}',
    'item not an object': lambda good: b'[[]]',
    'null item': lambda good: b'[null]',
    'missing key': lambda good: ('[' + json.dumps({k: '' for k in conv.SERIES_COLUMNS[1:]}) + ']').encode(),
    'extra key': lambda good: ('[' + full_row_json(extra='') + ']').encode(),
    'retained only': lambda good: b'[{"retained": ""}]',
    'key sets differ': lambda good: ('[' + full_row_json() + ',' + full_row_json(retained='') + ']').encode(),
    'number value': lambda good: ('[' + full_row_json(fractions_planned=5) + ']').encode(),
    'null value': lambda good: ('[' + full_row_json(roi_name=None) + ']').encode(),
    'list value': lambda good: ('[' + full_row_json(roi_name=['x']) + ']').encode(),
    'dicom a string': lambda good: ('[' + full_row_json(dicom='x') + ']').encode(),
    'dicom a list': lambda good: ('[' + full_row_json(dicom=[]) + ']').encode(),
    'dicom null': lambda good: ('[' + full_row_json(dicom=None) + ']').encode(),
    'dicom on one item only': lambda good: ('[' + full_row_json(dicom={}) + ',' + full_row_json() + ']').encode(),
    'dicom on the second item only': lambda good: ('[' + full_row_json() + ',' + full_row_json(dicom={}) + ']').encode(),
    'dicom and another extra key': lambda good: ('[' + full_row_json(dicom={}, extra='') + ']').encode(),
}


@pytest.mark.parametrize('what', list(SERIES_JSON_BREAKS))
def test_a_malformed_series_json_refuses_the_plan_only(tmp_path, what):
    def change(folder, rows):
        (folder / conv.SERIES_NAME).write_bytes(SERIES_JSON_BREAKS[what](good_series_bytes(folder)))
    refuse(tmp_path, pre.REASON_SERIES_CSV, change)


@pytest.mark.parametrize('what', list(SERIES_JSON_BREAKS))
def test_read_series_json_reports_malformed_and_never_raises_for_content(tmp_path, what):
    folder, _ = write_plan(tmp_path)
    (folder / conv.SERIES_NAME).write_bytes(SERIES_JSON_BREAKS[what](good_series_bytes(folder)))
    assert pre.read_series_json(folder)[1] is True


@pytest.mark.parametrize('what', list(SERIES_JSON_BREAKS))
def test_a_malformed_series_json_chains_no_exception_and_leaks_no_content(tmp_path, what, caplog):
    folder, _ = write_plan(tmp_path)
    secret = b'SYNTHETICSECRETCONTENT'
    (folder / conv.SERIES_NAME).write_bytes(SERIES_JSON_BREAKS[what](good_series_bytes(folder)) + secret)
    with caplog.at_level('DEBUG'), pytest.raises(CaseRefused) as caught:
        pre.check_rows(folder)
    assert str(caught.value) == pre.REASON_SERIES_CSV and caught.value.__cause__ is None and caught.value.__context__ is None
    assert secret.decode() not in caplog.text and caplog.records == []


def test_a_malformed_study_series_json_chains_no_exception(tmp_path):
    folder = write_study(tmp_path / 'study_1', [study_row('series_MR_1.nii.gz')])
    (folder / conv.SERIES_NAME).write_bytes(b'[' * 100_000)
    with pytest.raises(CaseRefused) as caught:
        pre.check_study_rows(folder)
    assert str(caught.value) == pre.REASON_SERIES_CSV and caught.value.__cause__ is None and caught.value.__context__ is None


def test_read_series_json_returns_the_objects_of_a_good_file_in_both_key_sets(tmp_path):
    for columns in (conv.SERIES_COLUMNS, pre.PREPROCESSED_SERIES_COLUMNS):
        rows = [dict.fromkeys(columns, 'é'), dict.fromkeys(columns, '')]
        sf.write(rows, tmp_path / conv.SERIES_NAME)
        assert pre.read_series_json(tmp_path) == (rows, False)
    (tmp_path / conv.SERIES_NAME).write_text('[]\n', encoding='utf-8')
    assert pre.read_series_json(tmp_path) == ([], False)


def test_read_series_json_keeps_the_items_of_a_valid_array_that_fails_the_key_check(tmp_path):
    (tmp_path / conv.SERIES_NAME).write_text('[{"file": "a"}, {"file": "b"}]', encoding='utf-8')
    rows, malformed = pre.read_series_json(tmp_path)
    assert malformed and len(rows) == 2  # study_refusal_count counts them
    (tmp_path / conv.SERIES_NAME).write_text('{"file": "a"}', encoding='utf-8')
    assert pre.read_series_json(tmp_path) == ([], True)


def test_read_series_json_lets_an_os_error_through(tmp_path):
    with pytest.raises(OSError):
        pre.read_series_json(tmp_path)


def test_write_series_json_orders_the_keys_and_round_trips(tmp_path):
    scrambled = dict(reversed(series_row('a.nii.gz', 'ct', roi_name='é "q"').items()))
    rows = [scrambled | {'zzz': '1', 'retained': '1.0'}, series_row('b.nii.gz', 'mask') | {'zzz': '', 'retained': ''}]
    conv.write_series_json(rows, tmp_path / conv.SERIES_NAME)
    text = (tmp_path / conv.SERIES_NAME).read_bytes().decode('utf-8')
    assert text.startswith('[\n {\n  "file": "a.nii.gz",\n  "kind": "ct",\n') and text.endswith('\n }\n]\n') and 'é' in text
    got = sf.read(tmp_path)
    assert [list(item) for item in got] == [[*conv.SERIES_COLUMNS, 'zzz', 'retained']] * 2
    assert got == rows  # dict equality ignores order
    conv.write_series_json([], tmp_path / conv.SERIES_NAME)
    assert (tmp_path / conv.SERIES_NAME).read_bytes() == b'[]\n'


def test_write_series_json_failure_is_the_fixed_write_reason_and_leaves_no_temp(tmp_path):
    with pytest.raises(OSError) as caught:
        conv.write_series_json([series_row('a', 'ct')], tmp_path / 'missing' / conv.SERIES_NAME)
    assert str(caught.value) == conv.REASON_WRITE and list(tmp_path.iterdir()) == []


def test_series_name_is_series_json_and_the_legacy_name_is_defined_once():
    assert conv.SERIES_NAME == 'series.json' and pre.LEGACY_SERIES_NAME == 'series.csv'
    assert pre.REASON_LEGACY_SERIES == 'converted by an older version (series.csv); reconvert with --overwrite'
    for reason in (pre.REASON_NO_CASES, pre.REASON_NO_CT, pre.REASON_DOSE_COUNT, pre.REASON_KIND, pre.REASON_FILE_MISSING,
                   pre.REASON_STUDY_MODALITY, pre.REASON_STUDY_NAME, pre.REASON_SERIES_CSV):
        assert 'series.json' in reason and 'series.csv' not in reason
    assert 'column' not in pre.REASON_SERIES_CSV and 'row length' not in pre.REASON_SERIES_CSV


def make_legacy(folder):
    """What a v0.1.x convert left: a series.csv and no series.json."""
    (folder / conv.SERIES_NAME).unlink(missing_ok=True)
    (folder / pre.LEGACY_SERIES_NAME).write_text('file,kind\nplanning_ct.nii.gz,ct\n', encoding='utf-8')


def test_a_legacy_plan_next_to_a_good_plan_is_refused_with_its_reason(tmp_path):
    refuse(tmp_path, pre.REASON_LEGACY_SERIES, lambda folder, rows: make_legacy(folder))  # refuse() also asserts no patient-level row (no REASON_IO_ERROR)


def test_a_legacy_study_is_refused_as_one_row_and_a_good_study_is_preprocessed(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted)
    write_study_tree(converted, study='study_2')
    make_legacy(converted / 'SYNTHETICPTA' / 'study_1')
    report = pre.preprocess(converted, out, SMALL)
    assert [(r.folder, r.outcome, r.reason) for r in report.rows] == [
        ('study_1', 'refused', pre.REASON_LEGACY_SERIES),
        ('study_2/series_CT_1', 'preprocessed', ''), ('study_2/series_MR_1', 'preprocessed', '')]
    assert not (out / 'SYNTHETICPTA' / 'study_1').exists()


def test_a_patient_with_only_legacy_folders_gets_refused_rows_not_no_cases(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, _ = write_plan(converted)
    study = write_study_tree(converted)
    make_legacy(folder)
    make_legacy(study)
    report = pre.preprocess(converted, out, SMALL)
    assert [(r.patient, r.folder, r.outcome, r.reason) for r in report.rows] == [
        ('SYNTHETICPTA', 'plan_1', 'refused', pre.REASON_LEGACY_SERIES), ('SYNTHETICPTA', 'study_1', 'refused', pre.REASON_LEGACY_SERIES)]


def test_series_json_wins_over_a_stray_series_csv(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    folder, _ = write_plan(converted)
    (folder / pre.LEGACY_SERIES_NAME).write_text('not read', encoding='utf-8')
    assert plan_row(pre.preprocess(converted, out, SMALL)).outcome == 'preprocessed'


def test_images_and_misc_files_stay_ignored_even_with_a_series_csv_or_json(tmp_path):
    converted = tmp_path / 'converted'
    for name in ('images', 'misc_files'):
        (converted / 'SYNTHETICPTA' / name).mkdir(parents=True)
        (converted / 'SYNTHETICPTA' / name / pre.LEGACY_SERIES_NAME).write_text('file,kind\n', encoding='utf-8')
        (converted / 'SYNTHETICPTA' / name / conv.SERIES_NAME).write_text('[]', encoding='utf-8')
    report = pre.preprocess(converted, tmp_path / 'out', SMALL)
    assert [(r.folder, r.outcome, r.reason) for r in report.rows] == [('', 'skipped', pre.REASON_NO_CASES)]


def test_the_legacy_check_comes_before_any_read(tmp_path, monkeypatch):
    folder, _ = write_plan(tmp_path / 'a')
    study = write_study_tree(tmp_path / 'b')
    make_legacy(folder)
    make_legacy(study)
    monkeypatch.setattr(pre, 'read_series_json', lambda *a: pytest.fail('series.json was read'))
    for check, path in ((pre.check_rows, folder), (pre.check_study_rows, study)):
        with pytest.raises(CaseRefused) as caught:
            check(path)
        assert str(caught.value) == pre.REASON_LEGACY_SERIES
