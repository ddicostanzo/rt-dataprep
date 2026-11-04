"""CLI (task-06, task-03 of F2b): `python -m rt_dataprep convert` and `preprocess`, always run as a subprocess. Every tree is synthetic and built in tmp_path."""
import csv
import dataclasses
import subprocess
import sys

import pytest
import SimpleITK as sitk
from test_convert import PID, make_plan
from test_preprocess import SMALL, report_rows, write_plan

from rt_dataprep import convert as conv
from rt_dataprep import preprocess as pre


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
