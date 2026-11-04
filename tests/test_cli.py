"""CLI (task-06, task-03 of F2b): `python -m rt_dataprep convert` and `preprocess`, always run as a subprocess. Every tree is synthetic and built in tmp_path."""
import csv
import dataclasses
import json
import re
import subprocess
import sys

import pytest
import SimpleITK as sitk
from test_convert import PID, make_plan
from test_preprocess import SMALL, report_rows, write_plan

from rt_dataprep import convert as conv
from rt_dataprep import preprocess as pre
from rt_dataprep import __main__ as cli
from rt_dataprep import split as sp
from rt_dataprep.manifest import MANIFEST_NAME


@pytest.fixture
def org(tmp_path):
    root = tmp_path / 'org'
    make_plan(root / PID / 'plan_1')
    return root


def run_convert(org, out):
    return subprocess.run([sys.executable, '-m', 'rt_dataprep', 'convert', str(org), str(out)], capture_output=True, text=True)


def test_a2_12_convert_runs_end_to_end_and_writes_the_report(tmp_path, org):
    out = tmp_path / 'out'
    result = run_convert(org, out)
    assert result.returncode == 0
    with open(out / conv.REPORT_NAME, newline='') as handle:
        assert [(r['patient'], r['folder'], r['outcome']) for r in csv.DictReader(handle)] == [(PID, 'plan_1', 'converted')]
    assert sorted(p.name for p in (out / PID / 'plan_1').iterdir()) == ['planning_ct.nii.gz', 'series.csv']
    assert PID not in result.stdout + result.stderr and str(tmp_path) not in result.stdout + result.stderr


@pytest.mark.parametrize('where', ['same', 'inside'])
def test_out_inside_org_exits_1_with_the_reason_and_no_path(tmp_path, org, where):
    out = org if where == 'same' else org / 'deeper' / 'out'
    result = run_convert(org, out)
    assert result.returncode == 1
    assert result.stderr == f'rt_dataprep: {conv.REASON_OUTSIDE_ORG}\n' and result.stdout == ''
    assert not (org / 'deeper').exists() and not (org / conv.REPORT_NAME).exists()


def test_oserror_exits_1_with_a_path_free_message(tmp_path, org):
    blocker = tmp_path / 'blocker'
    blocker.write_text('a file, not a folder')
    result = run_convert(org, blocker / 'out')  # the output root cannot be made
    assert result.returncode == 1
    assert result.stderr == 'rt_dataprep: file system error on the source or output root\n' and result.stdout == ''


# Runs the entry module as __main__ with build_masks patched to raise; argv comes after -c, so no path is embedded.
CRASH = """
import runpy, sys
from rt_dataprep import masks
def boom(*args, **kwargs):
    raise TypeError('1.2.3.4.5')
masks.build_masks = boom
sys.argv = ['rt_dataprep', 'convert', sys.argv[1], sys.argv[2]]
runpy.run_module('rt_dataprep', run_name='__main__')
"""


def test_an_unexpected_error_exits_1_and_the_last_stderr_line_is_the_fixed_text(tmp_path, org):
    out = tmp_path / 'out'
    result = subprocess.run([sys.executable, '-c', CRASH, str(org), str(out)], capture_output=True, text=True)
    assert result.returncode == 1
    lines = result.stderr.splitlines()
    assert lines[-1] == f'rt_dataprep: {conv.REASON_INTERNAL}' and result.stdout == ''
    assert [line for line in lines if '1.2.3.4.5' in line or str(tmp_path) in line] == []  # log lines stay, minus the secrets
    with open(out / conv.REPORT_NAME, newline='') as handle:
        assert [(r['outcome'], r['reason']) for r in csv.DictReader(handle)] == [(conv.OUTCOME_REFUSED, conv.REASON_INTERNAL)]


# F2b: the preprocess subcommand. The preset pads to 701 cubed, so these runs shrink new_shape inside the child process
# (as CRASH patches build_masks); argv comes after -c, so no path is embedded. %s is extra patch code, run before main.
PRE_RUN = """
import dataclasses, runpy, sys
from rt_dataprep import preprocess
preprocess.LEGACY_RNT = dataclasses.replace(preprocess.LEGACY_RNT, new_shape=%r)
%s
sys.argv = ['rt_dataprep', 'preprocess', *sys.argv[1:]]
runpy.run_module('rt_dataprep', run_name='__main__')
"""
PRE_CRASH = "def boom(*args, **kwargs):\n    raise TypeError('1.2.3.4.5')\npreprocess.process_ct = boom"


def run_pre(converted, out, *flags, patch=''):
    code = PRE_RUN % (SMALL.new_shape, patch)
    return subprocess.run([sys.executable, '-c', code, str(converted), str(out), *flags], capture_output=True, text=True)


def mask_pixel_type(out, steps):
    return sitk.ReadImage(str(out / 'SYNTHETICPTA' / 'plan_1' / pre.output_name('ptv.nii.gz', steps))).GetPixelID()


@pytest.fixture
def converted(tmp_path):
    root = tmp_path / 'converted'
    write_plan(root)
    return root


def test_preprocess_legacy_rnt_runs_end_to_end(tmp_path, converted):
    out = tmp_path / 'out'
    result = run_pre(converted, out, '--legacy-rnt')
    assert result.returncode == 0 and result.stdout == ''
    assert [(r['patient'], r['folder'], r['outcome']) for r in report_rows(out)] == [('SYNTHETICPTA', 'plan_1', pre.OUTCOME_PREPROCESSED)]
    names = {p.name for p in (out / 'SYNTHETICPTA' / 'plan_1').iterdir()}
    assert {'reshape_resample_rescale_initialresample_planning_ct.nii.gz', 'reshape_resample_initialresample_dose.nii.gz'} <= names
    assert mask_pixel_type(out, SMALL) == sitk.sitkUInt8
    assert str(tmp_path) not in result.stderr


def test_legacy_linear_masks_flag_sets_the_field(tmp_path, converted):
    out = tmp_path / 'out'
    result = run_pre(converted, out, '--legacy-rnt', '--legacy-linear-masks')
    assert result.returncode == 0
    assert mask_pixel_type(out, dataclasses.replace(SMALL, legacy_linear_masks=True)) == sitk.sitkFloat32


def test_preprocess_without_the_preset_exits_1_with_no_steps(tmp_path, converted):
    out = tmp_path / 'out'
    for flags in ([], ['--legacy-linear-masks']):
        result = subprocess.run([sys.executable, '-m', 'rt_dataprep', 'preprocess', str(converted), str(out), *flags], capture_output=True, text=True)
        assert result.returncode == 1
        assert result.stderr == f'rt_dataprep: {pre.REASON_NO_STEPS}\n' and result.stdout == ''
        assert not out.exists()


@pytest.mark.parametrize('where', ['same', 'inside'])
def test_preprocess_out_inside_the_input_exits_1_with_the_reason(tmp_path, converted, where):
    out = converted if where == 'same' else converted / 'deeper' / 'out'
    result = run_pre(converted, out, '--legacy-rnt')
    assert result.returncode == 1
    assert result.stderr == f'rt_dataprep: {pre.REASON_OUTSIDE_INPUT}\n' and result.stdout == ''
    assert not (converted / 'deeper').exists() and not (converted / pre.REPORT_NAME).exists()


def test_preprocess_oserror_exits_1_with_a_path_free_message(tmp_path, converted):
    blocker = tmp_path / 'blocker'
    blocker.write_text('a file, not a folder')
    result = run_pre(converted, blocker / 'out', '--legacy-rnt')
    assert result.returncode == 1
    assert result.stderr == 'rt_dataprep: file system error on the source or output root\n' and result.stdout == ''


def test_preprocess_unexpected_error_exits_1_with_its_own_fixed_text(tmp_path, converted):
    out = tmp_path / 'out'
    result = run_pre(converted, out, '--legacy-rnt', patch=PRE_CRASH)
    assert result.returncode == 1 and result.stdout == ''
    lines = result.stderr.splitlines()
    assert lines[-1] == f'rt_dataprep: {pre.REASON_INTERNAL}' and pre.REASON_INTERNAL != conv.REASON_INTERNAL
    assert [line for line in lines if '1.2.3.4.5' in line or str(tmp_path) in line] == []
    assert [(r['outcome'], r['reason']) for r in report_rows(out)] == [(conv.OUTCOME_REFUSED, pre.REASON_INTERNAL)]


def test_preprocess_help_says_the_preset_is_required_for_any_output():
    result = subprocess.run([sys.executable, '-m', 'rt_dataprep', 'preprocess', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'required for any output' in ' '.join(result.stdout.split())


# convert itself raises outside the patient loop, so the CLI's own handler reports it; argv comes after -c.
BUG = """
import runpy, sys
from rt_dataprep import convert
def bug(*args, **kwargs):
    raise %s('SYNTHETIC_ID in ' + sys.argv[1])
convert.convert = bug
sys.argv = ['rt_dataprep', 'convert', sys.argv[1], sys.argv[2]]
runpy.run_module('rt_dataprep', run_name='__main__')
"""


@pytest.mark.parametrize('kind', ['ValueError', 'TypeError'])
def test_an_unexpected_exception_prints_its_type_and_site_never_its_message(tmp_path, org, kind):
    result = subprocess.run([sys.executable, '-c', BUG % kind, str(org), str(tmp_path / 'out')], capture_output=True, text=True)
    assert result.returncode == 1 and result.stdout == ''
    assert re.fullmatch(r'rt_dataprep: unexpected error \(' + kind + r' at [\w.]+:\d+\)\n', result.stderr)
    assert 'SYNTHETIC_ID' not in result.stderr and str(tmp_path) not in result.stderr


# F3: the manifest and split subcommands, called as cli.main([...]); capfd reads the file descriptors.
NAMES = [f'SYNTHETIC_P{i}' for i in range(5)]


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / 'converted'
    for name in NAMES:
        write_plan(root, patient=name)
    return root


def split_args(root, *extra, test_size='0.2', val_size='0.2', seed='0'):
    return ['split', str(root), '--test-size', test_size, '--val-size', val_size, '--seed', seed, *extra]


def refusal(capsys_result, reason):
    assert capsys_result.out == '' and capsys_result.err.splitlines()[-1] == f'rt_dataprep: {reason}'


def test_manifest_then_split_run_end_to_end_and_print_counts_only(tree, capfd):
    cli.main(['manifest', str(tree)])
    first = capfd.readouterr()
    assert first.out == 'manifest: 5 written, 0 refused\n'
    cli.main(split_args(tree))
    second = capfd.readouterr()
    assert second.out == 'train: 3 patients, 3 cases\nval: 1 patients, 1 cases\ntest: 1 patients, 1 cases\n'
    with open(tree / MANIFEST_NAME, newline='') as handle:
        assert sorted(r['patient'] for r in csv.DictReader(handle)) == NAMES
    splits = json.loads((tree / sp.SPLITS_NAME).read_text())
    assert sorted(p for name in sp.SPLIT_NAMES for p in splits[name]) == NAMES
    for out in (first, second):
        assert 'SYNTHETIC' not in out.out + out.err and str(tree) not in out.out + out.err and '/plan_' not in out.out + out.err


def test_seed_test_size_and_val_size_are_required(tree, capfd):
    for missing in ('--seed', '--test-size', '--val-size'):
        argv = split_args(tree)
        del argv[argv.index(missing):argv.index(missing) + 2]
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2
        capfd.readouterr()
    assert not (tree / sp.SPLITS_NAME).exists()


def test_overlapping_external_exits_1_with_the_fixed_overlap_text(tree, tmp_path, capfd):
    cli.main(['manifest', str(tree)])
    other = tmp_path / 'other'
    for name in NAMES:
        write_plan(other, patient=name)
    cli.main(['manifest', str(other)])
    capfd.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(split_args(tree, '--external', str(other / MANIFEST_NAME)))
    assert exc.value.code == 1
    out = capfd.readouterr()
    refusal(out, sp.REASON_OVERLAP)
    assert 'SYNTHETIC' not in out.out + out.err and not (tree / sp.SPLITS_NAME).exists()


def test_a_disjoint_external_manifest_splits(tree, tmp_path, capfd):
    other = tmp_path / 'other'
    write_plan(other, patient='SYNTHETIC_OTHER')
    cli.main(['manifest', str(tree)])
    cli.main(['manifest', str(other)])
    capfd.readouterr()
    cli.main(split_args(tree, '--external', str(other / MANIFEST_NAME)))
    assert (tree / sp.SPLITS_NAME).exists()


@pytest.mark.parametrize(('kwargs', 'reason'), [
    ({'test_size': '1'}, sp.REASON_SIZE), ({'val_size': '-0.1'}, sp.REASON_SIZE),
    ({'test_size': '0.5', 'val_size': '0.5'}, sp.REASON_NO_TRAIN)])
def test_split_refusals_exit_1_with_their_fixed_text(tree, capfd, kwargs, reason):
    cli.main(['manifest', str(tree)])
    capfd.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(split_args(tree, **kwargs))
    assert exc.value.code == 1
    out = capfd.readouterr()
    refusal(out, reason)
    assert 'SYNTHETIC' not in out.out + out.err and not (tree / sp.SPLITS_NAME).exists()


def test_a_different_existing_splits_file_is_refused(tree, capfd):
    cli.main(['manifest', str(tree)])
    cli.main(split_args(tree))
    capfd.readouterr()
    cli.main(split_args(tree))  # the same arguments rewrite the same bytes: fine
    capfd.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(split_args(tree, seed='1'))
    assert exc.value.code == 1
    refusal(capfd.readouterr(), sp.REASON_EXISTS)


def test_split_without_a_manifest_exits_1_with_the_path_free_message(tree, capfd):
    with pytest.raises(SystemExit) as exc:
        cli.main(split_args(tree))
    assert exc.value.code == 1
    out = capfd.readouterr()
    assert out.out == '' and out.err.splitlines()[-1] == 'rt_dataprep: file system error on the source or output root'
    assert str(tree) not in out.err
