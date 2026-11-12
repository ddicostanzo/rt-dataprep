"""Example exit status must stop downstream automation when a converted cohort has no usable cases."""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('empty_patient', [False, True], ids=['empty-root', 'skipped-patient'])
def test_custom_preprocess_fails_when_no_cases_are_processed(tmp_path, empty_patient):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    converted.mkdir()
    if empty_patient:
        (converted / 'SYNTHETIC_EMPTY').mkdir()
    script = Path(__file__).resolve().parents[1] / 'examples' / 'custom_preprocess.py'
    result = subprocess.run([sys.executable, str(script), str(converted), str(out), '--shape', '16', '16', '8'],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert 'no cases were processed, or cases/registered series were refused or skipped' in result.stderr
    assert 'Custom preprocessing complete' not in result.stdout
    assert all(secret not in result.stdout + result.stderr for secret in ('SYNTHETIC_EMPTY', str(tmp_path)))


def test_synthetic_workflow_runs_end_to_end(tmp_path):  # #250: about 4 s
    script = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic_workflow.py'
    result = subprocess.run([sys.executable, str(script), str(tmp_path / 'workspace')], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert 'Synthetic workflow complete: 6 patients, 18 imaging cases; train=2, val=2, test=2; no refusals.' in result.stdout
