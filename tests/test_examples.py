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
