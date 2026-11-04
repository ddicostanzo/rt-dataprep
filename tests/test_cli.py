"""CLI (task-06): `python -m rt_dataprep convert`, always run as a subprocess. Every tree is synthetic and built in tmp_path."""
import csv
import subprocess
import sys

import pytest
from test_convert import PID, make_plan

from rt_dataprep import convert as conv


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
