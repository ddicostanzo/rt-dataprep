import logging
import errno
import operator
import os
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pydicom
import pytest
import SimpleITK as sitk

import synthetic_dicom as sd
from rt_dataprep import dicom_io
from rt_dataprep.dicom_io import CaseRefused, quiet_call, read_header, read_header_core, read_pooled, sanitize_path_component


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


@pytest.mark.parametrize('error', [MemoryError(FAKE_PATH), OSError(errno.ENOMEM, FAKE_PATH),
                                  *[RuntimeError(f'{FAKE_PATH}:179:\n{wording}') for wording in (
                                      'Failed to allocate memory for image.', 'Out of memory.', 'std::bad_alloc',
                                      'Cannot allocate memory', 'Memory allocation failed', 'bad allocation')]])
def test_quiet_call_raises_a_bare_memory_error_when_out_of_memory(error):
    def fail():
        raise error
    with pytest.raises(MemoryError) as err:
        quiet_call(fail, REASON)
    assert err.value.args == ()
    assert _chain(err.value) == (None, None)


@pytest.mark.parametrize('error', [OSError(errno.EIO, FAKE_PATH), RuntimeError('memory tag is unreadable'),
                                  ValueError('Out of memory is a tag value, not a runtime allocation failure')])
def test_ordinary_input_failures_are_not_promoted_to_memory_errors(error):
    def fail():
        raise error
    with pytest.raises(CaseRefused) as caught:
        quiet_call(fail, REASON)
    assert str(caught.value) == REASON and _chain(caught.value) == (None, None)


class _NeedsArg(BaseException):
    def __init__(self, value):
        super().__init__(value)


class _NoisyInit(BaseException):
    def __init__(self, *args):
        super().__init__(*args)
        os.write(2, b'SYNTHETIC_x')


def test_quiet_call_never_runs_a_third_party_constructor_outside_the_quiet_block(capfd):
    error = _NoisyInit()
    capfd.readouterr()  # the instance's own construction is not under test
    with pytest.raises(BaseException) as err:  # noqa: PT011
        quiet_call(_raiser(error), REASON)
    assert type(err.value) is KeyboardInterrupt
    assert capfd.readouterr().err == ''


def _raiser(exc):
    def fail():
        raise exc
    return fail


@pytest.mark.parametrize('error, kind, args, code', [
    (KeyboardInterrupt('SYNTHETIC_x'), KeyboardInterrupt, (), None),
    (SystemExit('SYNTHETIC_x'), SystemExit, (1,), 1),
    (SystemExit(3), SystemExit, (3,), 3),
    (SystemExit(), SystemExit, (), None),
    (SystemExit(None), SystemExit, (), None),
    (_NeedsArg('SYNTHETIC_x'), KeyboardInterrupt, (), None),
    (GeneratorExit('SYNTHETIC_x'), GeneratorExit, (), None),
])
def test_quiet_call_re_raises_a_bare_base_exception(capfd, error, kind, args, code):
    shown = sitk.ProcessObject.GetGlobalWarningDisplay()
    prior_disable = logging.root.manager.disable
    mode = pydicom.config.settings.reading_validation_mode
    disabled = logging.getLogger('pydicom').disabled
    stdout, stderr = sys.stdout, sys.stderr
    with pytest.raises(BaseException) as err:  # noqa: PT011
        quiet_call(_raiser(error), REASON)
    assert type(err.value) is kind
    assert err.value.args == args
    assert _chain(err.value) == (None, None)
    if kind is SystemExit:
        assert err.value.code == code
    assert sitk.ProcessObject.GetGlobalWarningDisplay() is shown
    assert logging.root.manager.disable == prior_disable
    assert pydicom.config.settings.reading_validation_mode == mode
    assert logging.getLogger('pydicom').disabled == disabled
    assert (sys.stdout, sys.stderr) == (stdout, stderr)
    capfd.readouterr()
    os.write(1, b'after-out')
    os.write(2, b'after-err')
    out = capfd.readouterr()
    assert (out.out, out.err) == ('after-out', 'after-err')  # fds 1 and 2 are back


@pytest.mark.skipif(not os.path.isdir('/proc/self/fd'), reason='needs /proc/self/fd')
def test_fds_quiet_leaks_no_fd_when_the_second_dup_fails(monkeypatch):
    real_dup, calls = os.dup, []

    def flaky_dup(fd):
        calls.append(fd)
        if len(calls) == 2:
            raise OSError('SYNTHETIC dup failure')
        return real_dup(fd)
    fds_quiet = getattr(dicom_io, '_fds_quiet')  # noqa: B009
    before = set(os.listdir('/proc/self/fd'))
    failed = False
    with monkeypatch.context() as patch:  # os is global: keep the patch to the one call
        patch.setattr(os, 'dup', flaky_dup)
        try:
            with fds_quiet():
                pass
        except OSError:  # info dropped on leaving: a kept traceback would hold the generator frame
            failed = True
    assert failed and len(calls) == 2
    assert set(os.listdir('/proc/self/fd')) == before


@pytest.mark.skipif(not os.path.isdir('/proc/self/fd'), reason='needs /proc/self/fd')
def test_fds_quiet_restores_fd_2_and_leaks_no_fd_when_the_first_restore_fails(monkeypatch, capfd):
    real_dup2, calls = os.dup2, []

    def flaky_dup2(fd, target):
        calls.append(target)
        real_dup2(fd, target)  # done, then fails: the rest of the restore must still run
        if len(calls) == 3:
            raise OSError('SYNTHETIC dup2 failure')
    fds_quiet = getattr(dicom_io, '_fds_quiet')  # noqa: B009
    before = set(os.listdir('/proc/self/fd'))
    failed = False
    with monkeypatch.context() as patch:  # os is global: keep the patch to the one call
        patch.setattr(os, 'dup2', flaky_dup2)
        try:
            with fds_quiet():
                pass
        except OSError:  # info dropped on leaving: a kept traceback would hold the generator frame
            failed = True
    assert failed and calls == [1, 2, 1, 2]
    assert set(os.listdir('/proc/self/fd')) == before
    os.write(2, b'after-err')
    assert capfd.readouterr().err == 'after-err'


def test_quiet_call_swallows_a_warning_from_a_thread_started_inside(capfd):
    def warn_in_thread():
        worker = threading.Thread(target=lambda: warnings.warn('SYNTHETIC_x', stacklevel=2))
        worker.start()
        worker.join()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        quiet_call(warn_in_thread, REASON)
    out = capfd.readouterr()
    assert caught == []
    assert (out.out, out.err) == ('', '')


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


def test_read_header_core_error_has_no_context(tmp_path):
    bad = tmp_path / 'notes.dcm'
    bad.write_text('SYNTHETIC_ not a dicom file')
    for path in (bad, tmp_path / 'absent.dcm'):
        with pytest.raises(OSError) as err:
            read_header_core(path)
        assert str(err.value) == dicom_io.REASON_UNREADABLE
        assert _chain(err.value) == (None, None)


def _quiet_state():
    return (pydicom.config.settings.reading_validation_mode, logging.getLogger('pydicom').disabled, list(warnings.filters))


def _invalid_value_files(folder, count=16):
    for i in range(count):
        sd.write_rtdose(folder / f'dose{i:02d}.dcm', [[[1]]], invalid_values=(i == 7))
    return sorted(folder.glob('*.dcm'))


def test_read_pooled_is_silent_under_threads_and_restores_state(tmp_path, capfd, caplog):
    paths = _invalid_value_files(tmp_path)
    with warnings.catch_warnings(record=True) as control, caplog.at_level(logging.DEBUG):  # control: warns unquieted
        warnings.simplefilter('always')
        for element in pydicom.dcmread(paths[7], stop_before_pixels=True).iterall():
            _ = element.value
    assert control or caplog.records
    caplog.clear()
    capfd.readouterr()
    before = _quiet_state()
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.DEBUG):
        warnings.simplefilter('always')
        inside = _quiet_state()
        headers = read_pooled(read_header_core, paths, 4)
        after = _quiet_state()
    out = capfd.readouterr()
    assert len(headers) == 16 and all(isinstance(h, pydicom.Dataset) for h in headers)
    assert caught == [] and caplog.records == [] and (out.out, out.err) == ('', '')
    assert after == inside  # filters compared inside the recording block, where they were set
    assert _quiet_state() == before


def test_read_pooled_returns_input_order(tmp_path, monkeypatch):
    paths = [tmp_path / f'p{i}.dcm' for i in range(8)]

    def core(path):
        time.sleep(0.05 * (len(paths) - paths.index(path)))  # earlier paths finish last
        return path

    monkeypatch.setattr(dicom_io, 'read_header_core', core)
    assert read_pooled(lambda path: dicom_io.read_header_core(path), paths, 4) == paths


def test_read_pooled_one_worker_is_one_thread(tmp_path, monkeypatch):
    threads = set()

    def core(path):
        threads.add(threading.get_ident())
        return path

    monkeypatch.setattr(dicom_io, 'read_header_core', core)
    paths = [tmp_path / f'p{i}.dcm' for i in range(5)]
    assert read_pooled(lambda path: dicom_io.read_header_core(path), paths, 1) == paths
    assert len(threads) == 1 and threads != {threading.get_ident()}


def test_read_pooled_ctrl_c_cancels_queued_steps_and_restores_state(tmp_path, monkeypatch):
    paths = [tmp_path / f'p{i:02d}.dcm' for i in range(50)]
    calls = []

    def core(path):
        calls.append(path)
        if path == paths[0]:
            raise KeyboardInterrupt
        time.sleep(0.5)  # structural margin: the queue is cancelled long before these finish

    monkeypatch.setattr(dicom_io, 'read_header_core', core)
    before = _quiet_state()
    with pytest.raises(KeyboardInterrupt):
        read_pooled(lambda path: dicom_io.read_header_core(path), paths, 4)
    assert len(calls) <= 2 * 4 and paths[-1] not in calls
    assert _quiet_state() == before


@pytest.mark.parametrize('first', [KeyboardInterrupt, OSError])  # a Ctrl-C mid-join wins over a step's OSError too
def test_read_pooled_waits_for_running_steps_even_when_the_shutdown_join_is_interrupted(monkeypatch, first):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    shutdowns = []
    real_shutdown = ThreadPoolExecutor.shutdown

    def step(path):
        if path == 'SYNTHETIC_1':
            assert started.wait(timeout=5)
            raise first('SYNTHETIC_x')
        started.set()
        release.wait(timeout=5)  # a timeout fails the test instead of hanging it
        finished.set()

    def shutdown(self, *args, **kwargs):
        shutdowns.append(1)
        if len(shutdowns) == 1:
            raise KeyboardInterrupt  # the second Ctrl-C, mid-join
        release.set()
        return real_shutdown(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, 'shutdown', shutdown)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            read_pooled(step, ['SYNTHETIC_1', 'SYNTHETIC_2'], 2)
        assert len(shutdowns) == 2 and getattr(raised.value, '__suppress_context__')  # noqa: B009
        assert finished.is_set()  # the running step ended inside the block
    finally:
        release.set()  # on a regression worker 2 ends at once instead of waiting out its 5 s timeout


def test_read_pooled_raises_the_ctrl_c_of_the_join_even_when_every_step_succeeded(monkeypatch):
    shutdowns = []
    real_shutdown = ThreadPoolExecutor.shutdown

    def shutdown(self, *args, **kwargs):
        shutdowns.append(1)
        if len(shutdowns) == 1:
            raise KeyboardInterrupt  # Ctrl-C mid-join, without delegating
        return real_shutdown(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, 'shutdown', shutdown)
    with pytest.raises(KeyboardInterrupt):  # not a list of results
        read_pooled(lambda path: path, ['SYNTHETIC_1', 'SYNTHETIC_2'], 2)
    assert len(shutdowns) == 2


# recover / publish: the --overwrite swap helper. Folders are SYNTHETIC_ names in tmp_path.
NAME = 'SYNTHETIC_pt'


def _make(path, files):
    for rel, data in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_bytes(data)


def _snap(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob('*')) if p.is_file()}


def _scene(tmp_path, target=None, old=None, partial=None):
    """An output root holding the given contents (dicts of relative path -> bytes) and an untouched sibling with hidden twins."""
    root = tmp_path / 'out'
    for folder, files in {NAME: target, f'.{NAME}.old': old, f'.{NAME}.partial': partial}.items():
        if files is not None:
            (root / folder).mkdir(parents=True)
            _make(root / folder, files)
    for folder in ('SYNTHETIC_other', '.SYNTHETIC_other.old', '.SYNTHETIC_other.partial'):
        _make(root / folder, {'a.bin': b'SYNTHETIC_sibling'})
    return root


def _others(root):
    return {p.name: _snap(p) for p in root.iterdir() if NAME not in p.name}


def _names(root):
    return sorted(p.name for p in root.iterdir() if NAME in p.name)


def _failing_rename(monkeypatch, *sources):
    real = dicom_io.rename_dir

    def fake(src, dst):
        if src.name in sources:
            raise OSError('SYNTHETIC_ rename failed')
        real(src, dst)
    monkeypatch.setattr(dicom_io, 'rename_dir', fake)


OLD = {'a.bin': b'SYNTHETIC_old', 'sub/b.bin': b'SYNTHETIC_old_b'}
NEW = {'a.bin': b'SYNTHETIC_new', 'c.bin': b'SYNTHETIC_new_c'}


@pytest.mark.parametrize('target, old, partial, expected', [
    (None, OLD, None, OLD),                       # interrupted swap: restore
    (None, OLD, NEW, OLD),                        # restore, and the leftover partial goes
    ({'a.bin': b'SYNTHETIC_kept'}, OLD, None, {'a.bin': b'SYNTHETIC_kept'}),  # finished swap: old goes
    ({'a.bin': b'SYNTHETIC_kept'}, OLD, NEW, {'a.bin': b'SYNTHETIC_kept'}),
    ({'a.bin': b'SYNTHETIC_kept'}, None, NEW, {'a.bin': b'SYNTHETIC_kept'}),  # only a partial
    (None, None, NEW, None),
    (None, None, None, None),
])
def test_recover(tmp_path, target, old, partial, expected):
    root = _scene(tmp_path, target, old, partial)
    before = _others(root)
    dicom_io.recover(root, NAME)
    assert _names(root) == ([NAME] if expected is not None else [])
    if expected is not None:
        assert _snap(root / NAME) == expected
    assert _others(root) == before


@pytest.mark.parametrize('overwrite, success', [(False, False), (False, True), (True, False), (True, True)])
def test_publish_without_a_target_is_new(tmp_path, overwrite, success):
    root = _scene(tmp_path, partial=NEW)
    before = _others(root)
    assert dicom_io.publish(root, NAME, overwrite=overwrite, success=success) == 'new'
    assert _names(root) == [NAME] and _snap(root / NAME) == NEW
    assert _others(root) == before


def test_publish_without_a_target_publishes_an_empty_partial(tmp_path):
    root = _scene(tmp_path, partial={})
    assert dicom_io.publish(root, NAME, overwrite=False, success=False) == 'new'
    assert _names(root) == [NAME] and _snap(root / NAME) == {}


def test_publish_replaces_an_existing_target(tmp_path):
    root = _scene(tmp_path, OLD, partial=NEW)
    before = _others(root)
    assert dicom_io.publish(root, NAME, overwrite=True, success=True) == 'replaced'
    assert _names(root) == [NAME] and _snap(root / NAME) == NEW
    assert _others(root) == before


@pytest.mark.parametrize('partial', [NEW, None])
def test_publish_keeps_the_target_when_the_build_had_no_success(tmp_path, partial):
    root = _scene(tmp_path, OLD, partial=partial)
    before = _others(root)
    assert dicom_io.publish(root, NAME, overwrite=True, success=False) == 'kept'
    assert _names(root) == [NAME] and _snap(root / NAME) == OLD
    assert _others(root) == before


@pytest.mark.parametrize('success', [True, False])
def test_publish_refuses_an_existing_target_without_overwrite(tmp_path, success):
    root = _scene(tmp_path, OLD, partial=NEW)
    before = _others(root)
    with pytest.raises(ValueError):
        dicom_io.publish(root, NAME, overwrite=False, success=success)
    assert _names(root) == [f'.{NAME}.partial', NAME]
    assert _snap(root / NAME) == OLD and _snap(root / f'.{NAME}.partial') == NEW
    assert _others(root) == before


def test_publish_rolls_back_when_the_partial_cannot_be_moved_in(tmp_path, monkeypatch, caplog):
    root = _scene(tmp_path, OLD, partial=NEW)
    before = _others(root)
    _failing_rename(monkeypatch, f'.{NAME}.partial')
    with caplog.at_level(logging.DEBUG), pytest.raises(OSError):
        dicom_io.publish(root, NAME, overwrite=True, success=True)
    assert _snap(root / NAME) == OLD
    assert _names(root) == [f'.{NAME}.partial', NAME]  # the stage's own cleanup removes the partial
    assert _others(root) == before
    assert caplog.records == []


def test_publish_leaves_the_old_folder_for_recover_when_the_rollback_also_fails(tmp_path, monkeypatch, caplog):
    root = _scene(tmp_path, OLD, partial=NEW)
    before = _others(root)
    _failing_rename(monkeypatch, f'.{NAME}.partial', f'.{NAME}.old')
    with caplog.at_level(logging.DEBUG), pytest.raises(OSError):
        dicom_io.publish(root, NAME, overwrite=True, success=True)
    assert _names(root) == [f'.{NAME}.old', f'.{NAME}.partial']
    assert _snap(root / f'.{NAME}.old') == OLD
    assert _others(root) == before
    assert caplog.records == []
    monkeypatch.undo()
    dicom_io.recover(root, NAME)
    assert _names(root) == [NAME] and _snap(root / NAME) == OLD


def test_publish_ignores_a_failed_delete_of_the_old_folder(tmp_path, caplog):
    if os.geteuid() == 0:
        pytest.skip('root ignores directory permissions')
    root = _scene(tmp_path, {'locked/a.bin': b'SYNTHETIC_old'}, partial=NEW)
    (root / NAME / 'locked').chmod(0o500)  # r-x: its file cannot be unlinked
    try:
        with caplog.at_level(logging.DEBUG):
            assert dicom_io.publish(root, NAME, overwrite=True, success=True) == 'replaced'
        assert _snap(root / NAME) == NEW
        assert caplog.records == []
    finally:
        (root / f'.{NAME}.old' / 'locked').chmod(0o700)
