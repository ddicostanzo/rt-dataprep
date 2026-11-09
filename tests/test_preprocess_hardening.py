"""Synthetic AHE correctness and image-support regressions (#154, #147)."""
import dataclasses
import json
import logging
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk
from test_preprocess import (AHE, FOV, MR_GRID, MR_SIZE, ROTATED, SMALL, ahe_plan, conv_rows, grid, header,
                             make_image, mr_spec, plan_row, run_plan, write_plan, write_study_tree)

from rt_dataprep import convert as conv
from rt_dataprep import manifest as man
from rt_dataprep import preprocess as pre
from rt_dataprep.dicom_io import CaseRefused


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('context', ['plan', 'study', 'registered'])
def test_ahe_refuses_nonfinite_images_and_preserves_valid_siblings(tmp_path, monkeypatch, caplog, capfd, recwarn, bad, context):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    if context == 'plan':
        ahe_plan(converted)
        ahe_plan(converted, plan='plan_2')
    elif context == 'study':
        write_study_tree(converted, series=(('CT', 1), ('MR', 1), ('PT', 1)), n=16)
    real_read = pre.read_image

    def read(path):
        image = real_read(path)
        affected = path.parent.name == 'plan_1' and path.name == conv.CT_NAME if context == 'plan' else path.name == 'series_MR_1.nii.gz'
        if affected:
            values = sitk.GetArrayFromImage(image)
            values.flat[0] = bad
            replacement = sitk.GetImageFromArray(values)
            replacement.CopyInformation(image)
            return replacement  # inject after reading: ITK may sanitize non-finite NIfTI pixels on disk
        return image

    monkeypatch.setattr(pre, 'read_image', read)
    caplog.set_level(logging.DEBUG)
    capfd.readouterr()
    if context == 'registered':
        report, done, _ = run_plan(tmp_path, [mr_spec(np.ones(MR_SIZE[::-1]), 'identity', MR_GRID)], steps=dataclasses.replace(SMALL, ahe=True))
        assert plan_row(report).outcome == 'preprocessed' and plan_row(report).skipped_series == 1
        assert not any(p.name.endswith('series_MR_1.nii.gz') for p in done.iterdir())
    else:
        report = pre.preprocess(converted, out, AHE)
        if context == 'plan':
            assert plan_row(report).reason == pre.REASON_NONFINITE
            assert plan_row(report, folder='plan_2').outcome == 'preprocessed'
            assert not (out / 'SYNTHETICPTA' / 'plan_1').exists()
        else:
            assert [(r.folder, r.outcome, r.reason) for r in report.rows] == [
                ('study_1/series_CT_1', 'preprocessed', ''), ('study_1/series_MR_1', 'refused', pre.REASON_NONFINITE),
                ('study_1/series_PT_1', 'preprocessed', '')]
            assert not (out / 'SYNTHETICPTA' / 'study_1' / 'ahe_series_MR_1.nii.gz').exists()
    captured = capfd.readouterr()
    text = captured.out + captured.err + caplog.text + ''.join(str(w.message) for w in recwarn.list)
    assert str(tmp_path) not in text and 'SYNTHETICPTA' not in text


@pytest.mark.parametrize('shape', [(8, 8, 8), (24, 24, 24)])
@pytest.mark.parametrize('values', ['ramp', 'two-level', 'constant', 'extremes'])
def test_ahe_keeps_finite_contrast_for_degenerate_volumes(shape, values):
    array = np.linspace(-1000, 200, np.prod(shape), dtype=np.float32).reshape(shape)
    if values == 'two-level':
        array = (array > 0).astype(np.float32)
    elif values == 'constant':
        array.fill(-300)
    elif values == 'extremes':
        array = np.linspace(-3e38, 3e38, np.prod(shape), dtype=np.float64).reshape(shape).astype(np.float32)
    image = make_image(array, **grid(shape[0], spacing=1.5))
    done = pre.process_ct(image, AHE)
    got = sitk.GetArrayFromImage(done)
    assert header(done) == header(image) and got.dtype == np.float32
    assert np.isfinite(got).all() and 0 <= got.min() <= got.max() <= 1
    if values == 'constant':
        assert not got.any()
    else:
        assert got.max() > got.min()  # no successful conversion may erase all input contrast


def test_nonfinite_ahe_backend_output_aborts_with_fixed_reason(monkeypatch):
    monkeypatch.setattr(pre, 'equalize_adapthist', lambda values, **kwargs: np.full_like(values, np.nan))
    with pytest.raises(CaseRefused) as caught:
        pre.process_ct(make_image(np.arange(4096).reshape(16, 16, 16)), AHE)
    assert str(caught.value) == pre.REASON_INTERNAL
    assert caught.value.__context__ is None and caught.value.__cause__ is None


@pytest.mark.parametrize('aligned', [False, True])
def test_physical_ahe_kernel_is_converted_from_xyz_mm_to_zyx_voxels(monkeypatch, aligned):
    seen = []

    def backend(values, **options):
        seen.append(options)
        return values

    monkeypatch.setattr(pre, 'equalize_adapthist', backend)
    image = make_image(np.arange(4096).reshape(16, 16, 16), spacing=(1.1, 1.8, 2.2))
    steps = dataclasses.replace(AHE, ahe_kernel_mm=(4.4, 5.4, 6.6), ahe_bins=64, ahe_clip_limit=0.02)
    if aligned:
        pre.process_series(image, image, np.eye(4), False, False, True, pre.ahe_options(steps))
    else:
        pre.process_ct(image, steps)
    assert seen == [{'kernel_size': (3, 3, 4), 'nbins': 64, 'clip_limit': 0.02}]


@pytest.mark.parametrize('options', [dict(ahe_bins=1), dict(ahe_clip_limit=np.nan), dict(ahe_clip_limit=1.1),
                                     dict(ahe_kernel_mm=(1, 0, 2)), dict(ahe_kernel_mm=(1, 2))])
def test_invalid_ahe_configuration_is_rejected_before_any_output(tmp_path, options):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    ahe_plan(converted)
    with pytest.raises(ValueError) as caught:
        pre.preprocess(converted, out, dataclasses.replace(AHE, **options))
    assert str(caught.value) == pre.REASON_AHE_OPTIONS and not out.exists()


def test_ahe_cli_parameters_and_support_masks_work_together(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted, series=(('MR', 1),), n=16)
    command = [sys.executable, '-m', 'rt_dataprep', 'preprocess', str(converted), str(out), '--ahe',
               '--ahe-kernel-mm', '4', '4', '4', '--ahe-bins', '64', '--ahe-clip-limit', '0.02', '--series-fov-mask']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    folder = out / 'SYNTHETICPTA' / 'study_1'
    assert (folder / 'ahe_series_MR_1.nii.gz').is_file() and (folder / 'fovmask_ahe_series_MR_1.nii.gz').is_file()
    assert 'SYNTHETICPTA' not in result.stdout + result.stderr and str(tmp_path) not in result.stdout + result.stderr


@pytest.mark.parametrize('modality', ['CT', 'MR', 'PT'])
def test_study_mask_propagates_both_resamples_and_marks_only_real_support(tmp_path, modality):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted, series=((modality, 1),), n=5)
    steps = dataclasses.replace(FOV, new_spacing=(1.5,) * 3, new_shape=(9,) * 3)
    report = pre.preprocess(converted, out, steps)
    folder = out / 'SYNTHETICPTA' / 'study_1'
    name = pre.output_name(f'series_{modality}_1.nii.gz', steps, ct=modality == 'CT', image=True)
    assert report.rows[0].files == [name, f'fovmask_{name}', 'series.json']
    mask, image = [sitk.ReadImage(str(folder / file)) for file in (f'fovmask_{name}', name)]
    assert header(mask) == header(image) and mask.GetPixelID() == sitk.sitkUInt8
    # 5^3 -> 3^3 at 1.5 mm -> 3^3 at 1.25 mm -> 9^3, with 3 padded voxels at each end.
    expected = np.zeros((9, 9, 9), np.uint8)
    expected[3:6, 3:6, 3:6] = 1
    np.testing.assert_array_equal(sitk.GetArrayFromImage(mask), expected)
    assert [row['file'] for row in conv_rows(folder)] == [name]  # auxiliary masks never change series.json kinds


@pytest.mark.parametrize('context', ['plan', 'study'])
def test_support_mask_handles_rotated_anisotropic_odd_pad_and_crop_with_real_zero_voxels(tmp_path, context):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    source = make_image(np.zeros((7, 6, 5)), spacing=(0.8, 1.1, 2.3), origin=(2.1, -4.2, 7.3), direction=ROTATED)
    if context == 'plan':
        folder, _ = write_plan(converted, n=5, dose=None, masks={}, applicator=False)
        name = conv.CT_NAME
    else:
        folder = write_study_tree(converted, series=(('MR', 1),), n=5)
        name = 'series_MR_1.nii.gz'
    sitk.WriteImage(source, str(folder / name))
    steps = pre.PreprocessSteps(reshape=True, new_shape=(9, 3, 10), series_fov_mask=True)
    report = pre.preprocess(converted, out, steps)
    assert all(row.outcome == 'preprocessed' for row in report.rows)
    done = out / folder.relative_to(converted)
    output = pre.output_name(name, steps)
    mask, image = [sitk.ReadImage(str(done / file)) for file in (f'fovmask_{output}', output)]
    assert header(mask) == header(image) and mask.GetPixelID() == sitk.sitkUInt8
    expected = np.zeros((10, 3, 9), np.uint8)
    expected[2:9, :, 2:7] = 1  # x pad 2+2; y crop 2+1; z pad 2+1
    np.testing.assert_array_equal(sitk.GetArrayFromImage(mask), expected)
    assert not sitk.GetArrayFromImage(image).any()  # genuine zeros are still valid, padding is not


def test_primary_masks_are_optional_manifest_columns_on_their_own_image_grid(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    ahe_plan(converted)
    write_study_tree(converted, series=(('MR', 1), ('PT', 1)), n=16)
    report = pre.preprocess(converted, out, dataclasses.replace(AHE, series_fov_mask=True))
    assert len(report.rows) == 3
    assert man.manifest(out) == (3, 0)
    rows = [json.loads(line) for line in (out / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(list(row) == man.FOV_COLUMNS for row in rows)
    for row in rows:
        folder = out / row['patient'] / row['case_dir'].split('/')[1]
        image = row['image'] or f'ahe_{conv.CT_NAME}'
        assert row['fov_mask'] == f'fovmask_{image}'
        assert man.image_fov_mask(folder, image, pre.read_info(folder / image)) == row['fov_mask']
        assert sitk.GetArrayFromImage(sitk.ReadImage(str(folder / row['fov_mask']))).all()  # AHE-only preserves support


@pytest.mark.parametrize('bad_mask', ['absent', 'off-grid', 'unreadable', 'symlink'])
def test_invalid_primary_support_masks_do_not_remove_images_or_add_a_manifest_column(tmp_path, bad_mask):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted, series=(('MR', 1),), n=8)
    pre.preprocess(converted, out, dataclasses.replace(SMALL, series_fov_mask=True))
    folder = out / 'SYNTHETICPTA' / 'study_1'
    mask = next(folder.glob('fovmask_*'))
    if bad_mask == 'absent':
        mask.unlink()
    elif bad_mask == 'off-grid':
        sitk.WriteImage(sitk.Image((2, 2, 2), sitk.sitkUInt8), str(mask))
    elif bad_mask == 'unreadable':
        mask.write_bytes(b'SYNTHETIC not an image')
    else:
        target = tmp_path / 'SYNTHETIC_external.nii.gz'
        mask.rename(target)
        mask.symlink_to(target)
    assert man.manifest(out) == (1, 0)
    rows = [json.loads(line) for line in (out / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [list(row) for row in rows] == [man.COLUMNS]
