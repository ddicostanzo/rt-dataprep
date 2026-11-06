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

# The plan's step set on the synthetic CT's own grid: every name gets the step prefixes, resolved through series.csv
STEPS = pre.PreprocessSteps(resample=True, rescale=True, harmonize_spacing=True, reshape=True, legacy_linear_masks=True,
                            new_spacing=tc.SPACING, final_spacing=tc.SPACING, new_shape=tc.SHAPE[::-1])
ROIS = [dict(tc.BOX, number=1, name='HR-CTV'), dict(tc.BOX, number=2, name='Bladder', box=((1, 3), (1, 3), (1, 3)))]


def build(tmp_path):
    """(preprocessed root, pooled root) where the pooled folder is a copy of the new output under legacy names."""
    org, converted, out, pooled = (tmp_path / n for n in ('org', 'conv', 'pre', 'pooled'))
    tc.make_plan(org / tc.PID / 'plan_1', doses=[tc.dose_spec(10, 'PLAN')], rois=ROIS)
    conv.convert(org, converted)
    pre.preprocess(converted, out, STEPS)
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
    new = parity.new_files(out / tc.PID / 'plan_1')
    prefix = 'reshape_resample_'  # the geometric steps every file gets; the CT also rescale_
    assert new['ct'].name.startswith(prefix + 'rescale_') and new['dose'].name.startswith(prefix)
    assert all(path.name.startswith(prefix) for path in new['mask'].values())
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
    assert 'ct_pass' not in s['counts'] and s['counts']['ct_resampled'] == 1
    assert s['distributions']['resampled_ct_max_abs_diff']['max'] == 0  # the legacy grid is a sub-grid: same voxels
    assert s['masks']['bladder'] == {'legacy': 1, 'fail': 1}  # the mask's own grid still matches


def test_a_shifted_legacy_grid_is_compared_after_resampling(tmp_path):
    out, pooled, old = build(tmp_path)
    for name in ('ct', 'hrctv', 'bladder', 'dose_gy'):
        image = sitk.ReadImage(str(old / f'{name}.nii.gz'))
        ref = sitk.Image(image)
        ref.SetOrigin([o + 0.5 * sp for o, sp in zip(image.GetOrigin(), image.GetSpacing(), strict=True)])  # half a voxel
        sitk.WriteImage(sitk.Resample(image, ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32), str(old / f'{name}.nii.gz'))
    s = parity.run(out, pooled)
    assert s['counts'] == {'pooled_folders': 1, 'ct_geometry_differs': 1, 'ct_resampled': 1, 'resampled_dose_compared': 1, 'compared': 1}
    assert s['ct_size_delta_xyz'] == {'(0, 0, 0)': 1}
    assert s['distributions']['resampled_ct_max_abs_diff']['max'] < 1e-5
    assert s['distributions']['resampled_ct_pearson_r']['min'] > 0.9999
    assert s['masks'] == {'hrctv': {'legacy': 1, 'resampled': 1}, 'bladder': {'legacy': 1, 'resampled': 1}}
    assert {'resampled_dice_hrctv', 'resampled_dice_bladder'} <= set(s['distributions'])


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


def test_a_three_part_root_or_grouping_folder_is_searched(tmp_path):
    out, pooled, old = build(tmp_path)
    nested = tmp_path / 'legacy_pooled_data' / 'site_a_b'
    nested.mkdir(parents=True)
    shutil.move(old, nested / old.name)
    assert parity.run(out, tmp_path / 'legacy_pooled_data')['counts']['compared'] == 1


def test_a_root_of_symlinked_split_folders_is_searched(tmp_path):
    out, pooled, old = build(tmp_path)
    train_val = tmp_path / 'train_val'
    train_val.mkdir()
    shutil.move(old, train_val / old.name)
    linked = tmp_path / 'linked'
    linked.mkdir()
    (linked / 'train_val').symlink_to(train_val, target_is_directory=True)
    assert parity.run(out, linked)['counts']['compared'] == 1


def test_a_pair_with_no_overlapping_voxels_is_not_counted_as_compared(tmp_path):
    out, pooled, old = build(tmp_path)
    ct = sitk.ReadImage(str(old / 'ct.nii.gz'))
    ct.SetOrigin([o + 1000.0 for o in ct.GetOrigin()])  # 1 m away: no new voxel lands on the legacy grid
    sitk.WriteImage(ct, str(old / 'ct.nii.gz'))
    s = parity.run(out, pooled)
    assert s['counts'] == {'pooled_folders': 1, 'ct_geometry_differs': 1, 'ct_no_overlap': 1}
