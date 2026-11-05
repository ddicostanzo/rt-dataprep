"""#51: detect major partial mask loss without choosing a clinical acceptance threshold for the user."""
import dataclasses
import logging
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk
from test_preprocess import SMALL, make_image, plan_row, write_plan

from rt_dataprep import preprocess as pre


@pytest.mark.parametrize('linear', [False, True])
def test_partial_mask_loss_refuses_the_plan_at_a_configured_limit_and_keeps_its_sibling(tmp_path, caplog, capfd, recwarn, linear):
    converted = tmp_path / 'converted'
    thin = np.zeros((24, 24, 24), np.uint8)
    thin[10, 8:16, 8:16] = 1  # this plane is missed by the final nearest-neighbor lattice
    thin[12, 12, 12] = 1  # one foreground voxel survives: the existing empty-mask check cannot catch the loss
    kept = np.zeros_like(thin)
    kept[8:16, 8:16, 8:16] = 1
    write_plan(converted, n=24, spacing=0.5, dose=None, masks={'ptv': thin}, applicator=False)
    write_plan(converted, plan='plan_2', n=24, spacing=0.5, dose=None, masks={'ptv': kept}, applicator=False)
    compatible = dataclasses.replace(SMALL, legacy_linear_masks=linear)
    baseline = pre.preprocess(converted, tmp_path / 'baseline', compatible)
    assert plan_row(baseline).outcome == 'preprocessed'  # default output remains compatible
    native = pre.read_image(converted / 'SYNTHETICPTA' / 'plan_1' / 'ptv.nii.gz')
    image = pre.read_image(tmp_path / 'baseline' / 'SYNTHETICPTA' / 'plan_1' / pre.output_name('ptv.nii.gz', compatible))
    assert np.count_nonzero(sitk.GetArrayFromImage(image)) == 1
    assert 0 < pre.mask_retention(native, image, linear) < 0.5
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    recwarn.clear()
    capfd.readouterr()
    out = tmp_path / 'guarded'
    report = pre.preprocess(converted, out, dataclasses.replace(compatible, min_mask_retention=0.5))
    assert plan_row(report).reason == pre.REASON_MASK_RETENTION
    assert not (out / 'SYNTHETICPTA' / 'plan_1').exists()
    assert plan_row(report, folder='plan_2').outcome == 'preprocessed'
    captured = capfd.readouterr()
    text = captured.out + captured.err + caplog.text + ''.join(str(w.message) for w in recwarn.list)
    assert 'SYNTHETICPTA' not in text and str(tmp_path) not in text and 'ptv' not in text


def test_retention_compares_physical_volume_instead_of_voxel_counts():
    native = make_image(np.ones((2, 2, 2)), spacing=(0.5,) * 3)
    image = make_image(np.ones((1, 1, 1)), spacing=(1.0,) * 3)
    assert pre.mask_retention(native, image) == 1.0  # 8 x 0.125 mm3 == 1 x 1 mm3


def test_retention_accepts_equality_and_roundoff_but_refuses_a_real_shortfall(tmp_path):
    converted = tmp_path / 'converted'
    thin = np.zeros((24, 24, 24), np.uint8)
    thin[10, 8:16, 8:16] = 1
    thin[12, 12, 12] = 1
    folder, _ = write_plan(converted, n=24, spacing=0.5, dose=None, masks={'ptv': thin}, applicator=False)
    native = pre.read_image(folder / 'ptv.nii.gz')
    final = pre.process_ct(pre.read_image(folder / 'planning_ct.nii.gz'), SMALL)
    retained = pre.mask_retention(native, pre.process_mask(native, final, SMALL))
    for index, (scale, expected) in enumerate([(1.0, 'preprocessed'), (1.0000005, 'preprocessed'), (1.000002, 'refused')]):
        report = pre.preprocess(converted, tmp_path / f'out{index}', dataclasses.replace(SMALL, min_mask_retention=retained * scale))
        assert plan_row(report).outcome == expected
        assert plan_row(report).reason == (pre.REASON_MASK_RETENTION if expected == 'refused' else '')


def test_linear_retention_integrates_fractional_occupancy():
    native = make_image(np.ones((2, 2, 2)))
    image = make_image(np.full((2, 2, 2), 0.25))
    assert pre.mask_retention(native, image, linear=True) == 0.25
    assert pre.mask_retention(native, image, linear=False) == 1.0
    assert pre.mask_retention(image, image, linear=True) == 1.0  # already fractional inputs retain their physical occupancy
    assert pre.mask_retention(make_image(np.zeros((2, 2, 2))), image) == 1.0


@pytest.mark.parametrize('limit', [0, -0.1, 1.01, np.nan, np.inf])
def test_invalid_retention_limits_fail_before_creating_output(tmp_path, limit):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    with pytest.raises(ValueError) as caught:
        pre.preprocess(converted, out, dataclasses.replace(SMALL, min_mask_retention=limit))
    assert str(caught.value) == pre.REASON_MASK_RETENTION_LIMIT and not out.exists()


def test_retention_cli_validation_is_fixed_text_and_writes_nothing(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    result = subprocess.run([sys.executable, '-m', 'rt_dataprep', 'preprocess', str(converted), str(out),
                             '--resample', '--min-mask-retention', 'nan'], capture_output=True, text=True)
    assert result.returncode == 1 and not out.exists()
    assert result.stderr == f'rt_dataprep: {pre.REASON_MASK_RETENTION_LIMIT}\n' and not result.stdout


def test_retention_cli_option_reaches_the_steps(monkeypatch):
    from rt_dataprep import __main__ as cli
    seen = []
    monkeypatch.setattr(cli, 'preprocess', lambda *args: seen.append(args))
    cli.main(['preprocess', 'converted', 'out', '--resample', '--min-mask-retention', '0.75'])
    assert len(seen) == 1 and seen[0][2].min_mask_retention == 0.75
    assert seen[0][2].harmonize_spacing and not seen[0][2].resample
