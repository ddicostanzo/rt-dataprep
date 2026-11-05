"""scripts/parity.py on a synthetic plan against a fake pooled folder built from the same output."""
import importlib.util
import json
import shutil
from pathlib import Path

import SimpleITK as sitk

import test_convert as tc
from rt_dataprep import convert as conv
from rt_dataprep import preprocess as pre

spec = importlib.util.spec_from_file_location('parity', Path(__file__).resolve().parents[1] / 'scripts' / 'parity.py')
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)

ROIS = [dict(tc.BOX, number=1, name='HR-CTV'), dict(tc.BOX, number=2, name='Bladder', box=((1, 3), (1, 3), (1, 3)))]


def build(tmp_path):
    """(preprocessed root, pooled root) where the pooled folder is a copy of the new output under legacy names."""
    org, converted, out, pooled = (tmp_path / n for n in ('org', 'conv', 'pre', 'pooled'))
    tc.make_plan(org / tc.PID / 'plan_1', doses=[tc.dose_spec(10, 'PLAN')], rois=ROIS)
    conv.convert(org, converted)
    pre.preprocess(converted, out, pre.PreprocessSteps(rescale=True))
    new = parity.new_files(out / tc.PID / 'plan_1')
    old = pooled / f'{tc.PID}_x_1'
    old.mkdir(parents=True)
    shutil.copy(new['ct'], old / 'ct.nii.gz')
    shutil.copy(new['dose'], old / 'dose_gy.nii.gz')
    for s in ('hrctv', 'bladder'):
        shutil.copy(new['mask'][s], old / f'{s}.nii.gz')
    return out, pooled, old


def test_identical_output_passes_every_check(tmp_path):
    out, pooled, _ = build(tmp_path)
    s = parity.run(out, pooled)
    assert s['counts'] == {'pooled_folders': 1, 'ct_geometry_equal': 1, 'ct_pass': 1, 'dose_compared': 1, 'compared': 1}
    assert s['refused'] == {}
    assert s['masks'] == {'hrctv': {'legacy': 1, 'pass': 1}, 'bladder': {'legacy': 1, 'pass': 1}}
    assert s['distributions']['ct_pearson_r']['min'] > 0.999 and s['distributions']['dose_max_abs_diff_gy']['max'] == 0


def test_differences_are_counted(tmp_path):
    out, pooled, old = build(tmp_path)
    mask = sitk.ReadImage(str(old / 'bladder.nii.gz'))
    sitk.WriteImage(mask * 0, str(old / 'bladder.nii.gz'))
    (pooled / f'{tc.PID}_x_2').mkdir()
    (pooled / f'{tc.PID}_x_2' / 'ct.nii.gz').write_bytes(b'')
    s = parity.run(out, pooled)
    assert s['masks']['bladder'] == {'legacy': 1, 'fail': 1}
    assert s['refused'] == {parity.REASON_NEW_MISSING: 1}
    ct = sitk.ReadImage(str(old / 'ct.nii.gz'))
    sitk.WriteImage(ct[:-1, :, :], str(old / 'ct.nii.gz'))  # legacy one voxel shorter in x: the #100 pattern
    s = parity.run(out, pooled)
    assert s['counts']['ct_geometry_differs'] == 1 and s['ct_size_delta_xyz'] == {'(1, 0, 0)': 1}
    assert 'ct_pass' not in s['counts'] and 'bladder' not in s['masks']


def test_cli_prints_no_identifiers(tmp_path, capfd):
    out, pooled, old = build(tmp_path)
    (old / 'hrctv.nii.gz').write_bytes(b'not nifti')
    assert parity.main([str(out), str(pooled), '--json', str(tmp_path / 's.json')]) == 0
    printed = capfd.readouterr()  # capfd: ITK writes to fd 2 directly
    assert json.loads(printed.out) == json.loads((tmp_path / 's.json').read_text())
    s = json.loads(printed.out)
    assert s['refused'] == {parity.REASON_UNREADABLE: 1}
    assert s['counts'] == {'pooled_folders': 1} and s['masks'] == {}  # a failed pair adds nothing but its refusal
    assert all(secret not in printed.out + printed.err for secret in (tc.PID, str(tmp_path)))
