import logging
import operator
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pydicom
import pytest
import SimpleITK as sitk

import synthetic_dicom as sd
from rt_dataprep import dicom_io
from rt_dataprep.dicom_io import CaseRefused, quiet_call, read_header, sanitize_path_component


def test_package_init_has_no_imports():
    import rt_dataprep

    assert 'import' not in Path(rt_dataprep.__file__).read_text()


def test_help_lists_organize():
    out = subprocess.run([sys.executable, '-m', 'rt_dataprep', '--help'], capture_output=True, text=True)
    assert out.returncode == 0
    assert 'organize' in out.stdout


def test_read_header_has_no_pixel_data(tmp_path):
    ct = sd.write_ct_series(tmp_path, [[[1, 2], [3, 4]]])
    ds = read_header(ct['paths'][0])
    assert isinstance(ds, pydicom.Dataset)
    assert 'PixelData' not in ds
    assert ds.Modality == 'CT'


def test_read_header_is_silent_for_invalid_values(tmp_path, caplog):
    path = tmp_path / 'dose.dcm'
    sd.write_rtdose(path, [[[1]]], invalid_values=True)
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.DEBUG):
        warnings.simplefilter('always')
        ds = read_header(path)
        # first access after the quiet block: must hit cached values, not convert (and warn) now
        _ = ds.PatientName, ds.StudyDate, ds.SOPInstanceUID, ds.StudyInstanceUID
        for element in ds.iterall():
            _ = element.value
    assert caught == []
    assert caplog.records == []


def test_read_header_raises_oserror_for_non_dicom(tmp_path):
    path = tmp_path / 'notes.dcm'
    path.write_text('SYNTHETIC_ not a dicom file')
    with pytest.raises(OSError) as err:
        read_header(path)
    assert 'notes' not in str(err.value) and str(tmp_path) not in str(err.value)


def test_read_header_raises_oserror_for_missing_file(tmp_path):
    with pytest.raises(OSError):
        read_header(tmp_path / 'absent.dcm')


@pytest.mark.parametrize('value', ['../evil', 'a/b', '..', '/etc/passwd', 'C:\\Windows\\x', '', None])
def test_sanitize_is_one_safe_component(value):
    out = sanitize_path_component(value)
    assert out and '/' not in out and '\\' not in out and out not in ('.', '..')


def test_sanitize_values():
    assert sanitize_path_component('../evil') == 'evil'
    assert sanitize_path_component('a/b') == 'b'
    assert sanitize_path_component('ok-1.2_x') == 'ok-1.2_x'
    assert sanitize_path_component('a b') == 'a_b'
    assert sanitize_path_component('') == 'UNKNOWN'
    assert sanitize_path_component('CON') == '_CON'
    assert len(sanitize_path_component('x' * 200)) == 64


# the extended synthetic helpers

def _hdr(path):
    return pydicom.dcmread(path, stop_before_pixels=True)


def test_ct_writer_metadata(tmp_path):
    study = pydicom.uid.generate_uid()
    ct = sd.write_ct_series(tmp_path, [[[1]], [[2]]], patient_id='SYNTHETIC_A', study_uid=study, series_date='20240102',
                            series_time='101500', manufacturer_model_name='SYNTHETIC_Acuity',
                            image_type=['ORIGINAL', 'PRIMARY', 'LOCALIZER'])
    assert ct['study_uid'] == study
    for p in ct['paths']:
        ds = _hdr(p)
        assert (ds.PatientID, ds.StudyInstanceUID) == ('SYNTHETIC_A', study)
        assert (ds.SeriesDate, ds.SeriesTime) == ('20240102', '101500')
        assert ds.ManufacturerModelName == 'SYNTHETIC_Acuity'
        assert list(ds.ImageType) == ['ORIGINAL', 'PRIMARY', 'LOCALIZER']


def test_rt_writers_metadata(tmp_path):
    study = pydicom.uid.generate_uid()
    ct = sd.write_ct_series(tmp_path, [[[1]]])
    kw = dict(patient_id='SYNTHETIC_B', study_uid=study)
    struct = sd.write_rtstruct(tmp_path / 's.dcm', ct, **kw)
    plan = sd.write_rtplan(tmp_path / 'p.dcm', struct, [[0, 0, 0]], plan_date='20240301', plan_time='083000', **kw)
    sd.write_rtdose(tmp_path / 'd.dcm', [[[1]]], plan_sop_uid=plan, **kw)
    sd.write_rtstruct_contours(tmp_path / 'c.dcm', [(1, 'R', [[0, 0, 0], [1, 1, 1]])], **kw)
    for name in 'spdc':
        ds = _hdr(tmp_path / f'{name}.dcm')
        assert (ds.PatientID, ds.StudyInstanceUID) == ('SYNTHETIC_B', study)
    plan_ds = _hdr(tmp_path / 'p.dcm')
    assert (plan_ds.RTPlanDate, plan_ds.RTPlanTime) == ('20240301', '083000')


def test_write_reg_references_other_study(tmp_path):
    study, other = pydicom.uid.generate_uid(), pydicom.uid.generate_uid()
    sop = sd.write_reg(tmp_path / 'r.dcm', study, other, patient_id='SYNTHETIC_C')
    ds = _hdr(tmp_path / 'r.dcm')
    assert ds.SOPInstanceUID == sop and ds.Modality == 'REG'
    assert (ds.PatientID, ds.StudyInstanceUID) == ('SYNTHETIC_C', study)
    assert [i.StudyInstanceUID for i in ds.StudiesContainingOtherReferencedInstancesSequence] == [other]


def test_write_ct_series_mr_modality(tmp_path):
    import pydicom
    import synthetic_dicom as sd

    out = sd.write_ct_series(tmp_path, [[[1]]], modality='MR', prefix='mr')
    ds = pydicom.dcmread(out['paths'][0])
    assert ds.Modality == 'MR' and ds.SOPClassUID == pydicom.uid.MRImageStorage


# the PHI-safe third-party call wrapper

FAKE_PATH = '/data/SYNTHETIC_PATIENT_X/plan_1/ct_slice.dcm'
FAKE_ID = 'SYNTHETIC_ID_ABC'
REASON = 'SYNTHETIC reason constant'


_chain = operator.attrgetter('__cause__', '__context__')  # (cause, context) of an exception


def _leaky():
    """Everything a third-party loader does with a path: warn, log, print, write to the fds, then raise."""
    for text in (FAKE_PATH, FAKE_ID):
        warnings.warn(f'bad value {text}', stacklevel=2)
        logging.getLogger('some.loader').error('Failed on %s', text)
        logging.getLogger('unhandled.worker').warning('Failed on %s', text)  # no handler: lastResort writes to stderr
        print(f'stdout {text}')
        print(f'stderr {text}', file=sys.stderr)
        os.write(1, f'fd1 {text}\n'.encode())
        os.write(2, f'fd2 {text}\n'.encode())
    sitk.ProcessObject.SetGlobalWarningDisplay(False)  # irrelevant to the leak; the wrapper must restore the original
    raise ValueError(f'cannot read {FAKE_PATH} for {FAKE_ID}')


def test_quiet_call_leaks_nothing(capfd, caplog):
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.DEBUG):
        warnings.simplefilter('always')
        with pytest.raises(CaseRefused) as err:
            quiet_call(_leaky, REASON)
    out = capfd.readouterr()
    for text in (FAKE_PATH, FAKE_ID, 'SYNTHETIC_PATIENT_X'):
        assert text not in out.out + out.err
        assert text not in caplog.text and not caplog.records
        assert not any(text in str(w.message) for w in caught)
        assert text not in str(err.value)
    assert str(err.value) == REASON
    assert _chain(err.value) == (None, None)
    assert caught == []


def test_case_refused_str_is_exactly_the_reason():
    assert str(CaseRefused(REASON)) == REASON


def test_quiet_call_returns_the_value():
    assert quiet_call(lambda: 42, REASON) == 42


def test_quiet_call_restores_everything(capfd):
    shown = sitk.ProcessObject.GetGlobalWarningDisplay()
    sitk.ProcessObject.SetGlobalWarningDisplay(True)
    prior_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    mode = pydicom.config.settings.reading_validation_mode
    stdout, stderr = sys.stdout, sys.stderr
    try:
        with pytest.raises(CaseRefused):
            quiet_call(_leaky, REASON)
        assert sitk.ProcessObject.GetGlobalWarningDisplay() is True
        assert logging.root.manager.disable == logging.WARNING  # the prior level, not 0
        assert pydicom.config.settings.reading_validation_mode == mode
        assert (sys.stdout, sys.stderr) == (stdout, stderr)
        capfd.readouterr()
        os.write(1, b'after-out')
        os.write(2, b'after-err')
        out = capfd.readouterr()
        assert (out.out, out.err) == ('after-out', 'after-err')  # fds 1 and 2 are back
    finally:
        logging.disable(prior_disable)
        sitk.ProcessObject.SetGlobalWarningDisplay(shown)


def test_quiet_call_refuses_an_exception_whose_str_fails():
    class Odd(Exception):
        def __str__(self):
            raise ValueError(FAKE_PATH)

    def fail():
        raise Odd()
    with pytest.raises(CaseRefused) as err:
        quiet_call(fail, REASON)
    assert str(err.value) == REASON
    assert _chain(err.value) == (None, None)


@pytest.mark.parametrize('error', [MemoryError(FAKE_PATH), RuntimeError(f'{FAKE_PATH}:179:\nFailed to allocate memory for image.')])
def test_quiet_call_raises_a_bare_memory_error_when_out_of_memory(error):
    def fail():
        raise error
    with pytest.raises(MemoryError) as err:
        quiet_call(fail, REASON)
    assert err.value.args == ()
    assert _chain(err.value) == (None, None)


def test_read_header_error_has_no_context(tmp_path):
    path = tmp_path / 'notes.dcm'
    path.write_text('SYNTHETIC_ not a dicom file')
    with pytest.raises(OSError) as err:
        read_header(path)
    assert _chain(err.value) == (None, None)


def test_a_failed_latest_report_copy_leaves_no_hidden_temp(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError('SYNTHETIC_PATH')
    monkeypatch.setattr(dicom_io.os, 'replace', fail)
    with pytest.raises(OSError):
        dicom_io.write_report(tmp_path, 'r.csv', ['a'], [['1']])
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith('.')] == []
