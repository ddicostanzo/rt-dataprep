import builtins
import logging
import errno
import operator
import os
import stat
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import pydicom
import pytest
import SimpleITK as sitk

import synthetic_dicom as sd
from rt_dataprep import dicom_io
from rt_dataprep.dicom_io import CaseRefused, quiet_call, read_header, read_header_core, sanitize_path_component


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


# recover / publish: the --overwrite swap helper. Folders are SYNTHETIC_ names in tmp_path.
NAME = 'SYNTHETIC_pt'
_OURS = (NAME, f'.{NAME}.old', f'.{NAME}.partial')  # the only folders the helpers may touch


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
    # '.<name>2.old' is hidden and shares the name prefix, so a '.{name}*' glob in the helper would take it (AC-H7)
    for folder in ('SYNTHETIC_other', '.SYNTHETIC_other.old', '.SYNTHETIC_other.partial', f'.{NAME}2.old'):
        _make(root / folder, {'a.bin': b'SYNTHETIC_sibling'})
    return root


def _others(root):
    return {p.name: _snap(p) for p in root.iterdir() if p.name not in _OURS}


def _names(root):
    return sorted(p.name for p in root.iterdir() if p.name in _OURS)


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
    before = _others(root)
    assert dicom_io.publish(root, NAME, overwrite=False, success=False) == 'new'
    assert _names(root) == [NAME] and _snap(root / NAME) == {}
    assert _others(root) == before


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


def test_recover_raises_and_keeps_the_old_folder_when_it_cannot_restore_it(tmp_path, monkeypatch):
    root = _scene(tmp_path, old=OLD)
    before = _others(root)
    _failing_rename(monkeypatch, f'.{NAME}.old')
    with pytest.raises(OSError):
        dicom_io.recover(root, NAME)
    assert _names(root) == [f'.{NAME}.old'] and _snap(root / f'.{NAME}.old') == OLD
    assert _others(root) == before


def test_publish_ignores_a_failed_delete_of_the_old_folder(tmp_path, caplog):
    if os.geteuid() == 0:
        pytest.skip('root ignores directory permissions')
    root = _scene(tmp_path, {'locked/a.bin': b'SYNTHETIC_old'}, partial=NEW)
    (root / NAME / 'locked').chmod(0o500)  # r-x: its file cannot be unlinked
    before = _others(root)
    try:
        with caplog.at_level(logging.DEBUG):
            assert dicom_io.publish(root, NAME, overwrite=True, success=True) == 'replaced'
        assert _snap(root / NAME) == NEW
        assert caplog.records == []
        assert _others(root) == before
    finally:
        (root / f'.{NAME}.old' / 'locked').chmod(0o700)


def _outside(tmp_path):
    outside = tmp_path / 'outside'
    _make(outside, {'keep.bin': b'SYNTHETIC_outside'})
    return outside


@pytest.mark.parametrize('dangling', [False, True])
def test_recover_unlinks_a_symlinked_old_folder_and_never_follows_it(tmp_path, dangling):
    root = _scene(tmp_path, {'a.bin': b'SYNTHETIC_kept'})
    outside = _outside(tmp_path)
    (root / f'.{NAME}.old').symlink_to(tmp_path / 'missing' if dangling else outside, target_is_directory=True)
    before = _others(root)
    dicom_io.recover(root, NAME)
    assert _others(root) == before
    assert not os.path.lexists(root / f'.{NAME}.old')
    assert _snap(root / NAME) == {'a.bin': b'SYNTHETIC_kept'}
    assert _snap(outside) == {'keep.bin': b'SYNTHETIC_outside'}


def test_publish_replaces_a_symlinked_target_without_touching_the_folder_it_points_at(tmp_path):
    root = _scene(tmp_path, partial=NEW)
    outside = _outside(tmp_path)
    (root / NAME).symlink_to(outside, target_is_directory=True)
    before = _others(root)
    assert dicom_io.publish(root, NAME, overwrite=True, success=True) == 'replaced'
    assert _others(root) == before
    assert not (root / NAME).is_symlink() and _snap(root / NAME) == NEW
    assert not os.path.lexists(root / f'.{NAME}.old')
    assert _snap(outside) == {'keep.bin': b'SYNTHETIC_outside'}


def test_the_rmtree_handler_clears_the_read_only_bit_before_it_retries(tmp_path, monkeypatch):
    file = tmp_path / 'ro.bin'
    file.write_bytes(b'SYNTHETIC_ro')
    file.chmod(0o444)
    seen = []
    unlink = os.unlink

    def func(p):
        seen.append(os.stat(p).st_mode & stat.S_IWRITE)
        assert seen[-1]  # the bit is back before the retry
        unlink(p)
    monkeypatch.setattr(os, 'unlink', func)  # the handler retries only os.unlink / os.rmdir
    getattr(dicom_io, '_clear_readonly')(os.unlink, str(file), OSError())  # noqa: B009
    assert seen and not file.exists()


def test_the_rmtree_handler_never_changes_a_link_target(tmp_path):
    outside = tmp_path / 'outside.bin'
    outside.write_bytes(b'SYNTHETIC_outside')
    outside.chmod(0o444)
    link = tmp_path / 'link'
    link.symlink_to(outside)
    getattr(dicom_io, '_clear_readonly')(os.unlink, str(link), OSError())  # noqa: B009
    assert not os.path.lexists(link)
    assert stat.S_IMODE(outside.stat().st_mode) == 0o444


@pytest.mark.skipif(os.name != 'posix' or os.geteuid() == 0, reason='needs a POSIX permission check root bypasses')
def test_remove_tree_swallows_an_unreadable_folder_inside_the_tree(tmp_path):
    tree = tmp_path / 'tree'
    (tree / 'locked').mkdir(parents=True)
    (tree / 'locked').chmod(0)
    try:
        dicom_io.remove_tree(tree)  # rmtree reports it with func=os.open, which takes more than a path
    finally:
        (tree / 'locked').chmod(0o700)
    assert (tree / 'locked').is_dir()  # left for the next recover, nothing raised


def fail_python_open(monkeypatch, bad, error):
    """Only dicom_io's own open, and only for `bad`, raises `error`; the modes asked for are returned."""
    modes = []

    def raiser(path, *args, **kwargs):
        if str(path) != str(bad):
            return builtins.open(path, *args, **kwargs)
        modes.append(args[0] if args else kwargs.get('mode', 'r'))
        raise error
    monkeypatch.setattr(dicom_io, 'open', raiser, raising=False)
    return modes


def test_read_failure_reads_in_chunks_to_the_end_and_returns(tmp_path, monkeypatch):
    big = tmp_path / 'big.bin'
    big.write_bytes(bytes(10))
    sizes, real = [], builtins.open

    def spy(path, mode='r', *args, **kwargs):
        handle = real(path, mode, *args, **kwargs)
        read = handle.read
        handle.read = lambda size=-1: (sizes.append(size), read(size))[1]
        return handle
    monkeypatch.setattr(dicom_io, 'open', spy, raising=False)
    monkeypatch.setattr(dicom_io, 'READ_CHUNK', 4)
    assert dicom_io.read_failure(big) is None
    assert sizes == [4, 4, 4, 4]  # 4 + 4 + 2 bytes, then the empty read that ends it


def test_read_failure_swallows_a_missing_file(tmp_path):
    assert dicom_io.read_failure(tmp_path / 'absent.nii.gz') is None


@pytest.mark.parametrize('code', [errno.ENOENT, errno.ENOTDIR, errno.ELOOP])  # gone, under a file now, or a symlink loop now
def test_read_failure_swallows_a_file_that_vanishes_before_the_reread(tmp_path, monkeypatch, code):
    bad = tmp_path / 'gone.bin'
    bad.write_bytes(b'x')
    modes = fail_python_open(monkeypatch, bad, OSError(code, 'gone', str(bad)))
    assert dicom_io.read_failure(bad) is None and modes == ['rb']


@pytest.mark.parametrize('error', [PermissionError(errno.EACCES, 'forced'), OSError(errno.EIO, 'forced')])
def test_read_failure_propagates_any_other_oserror_unchained(tmp_path, monkeypatch, error):
    bad = tmp_path / 'bad.bin'
    bad.write_bytes(b'x')
    modes = fail_python_open(monkeypatch, bad, error)
    with pytest.raises(OSError) as caught:
        dicom_io.read_failure(bad)
    assert caught.value is error and modes == ['rb']
    assert caught.value.__context__ is None and caught.value.__cause__ is None


def test_read_failure_turns_enomem_into_a_bare_memory_error(tmp_path, monkeypatch):
    bad = tmp_path / 'bad.bin'
    bad.write_bytes(b'x')
    fail_python_open(monkeypatch, bad, OSError(errno.ENOMEM, 'forced'))
    with pytest.raises(MemoryError) as caught:
        dicom_io.read_failure(bad)
    assert type(caught.value) is MemoryError and caught.value.args == ()
    assert caught.value.__context__ is None and caught.value.__cause__ is None


# ---------------------------------------------------------------- entry_is: Python 3.13's is_file/is_dir on every version

def test_entry_is_answers_false_for_an_entry_that_is_not_there(tmp_path):
    (tmp_path / 'plain').write_text('x')
    (tmp_path / 'dangling').symlink_to(tmp_path / 'nowhere')
    (tmp_path / 'loop').symlink_to(tmp_path / 'loop')
    absent = [tmp_path / 'missing', tmp_path / 'dangling', tmp_path / 'loop', tmp_path / 'plain' / 'below', tmp_path / 'with\0nul']
    assert [dicom_io.entry_is(p, kind) for p in absent for kind in ('file', 'dir')] == [False] * 10


def test_entry_is_tells_a_file_from_a_folder_and_follows_a_link(tmp_path):
    (tmp_path / 'plain').write_text('x')
    (tmp_path / 'folder').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'folder')
    assert [dicom_io.entry_is(tmp_path / n, k) for n in ('plain', 'folder', 'link') for k in ('file', 'dir')] == \
        [True, False, False, True, False, True]


def test_entry_is_ignores_ebadf_and_raises_every_other_error(tmp_path, monkeypatch):
    real = Path.stat

    def stat(self, *, follow_symlinks=True):
        if self.name != 'target':
            return real(self, follow_symlinks=follow_symlinks)
        raise OSError(code, 'forced')
    monkeypatch.setattr(Path, 'stat', stat)
    code = errno.EBADF
    assert dicom_io.entry_is(tmp_path / 'target', 'file') is False
    for code in (errno.EACCES, errno.EIO):
        with pytest.raises(OSError) as raised:
            dicom_io.entry_is(tmp_path / 'target', 'dir')
        assert raised.value.errno == code


@pytest.mark.parametrize('winerror', [21, 123, 1921])
def test_entry_is_answers_false_for_the_windows_errors_python_reads_as_not_there(tmp_path, monkeypatch, winerror):
    """Windows' not-ready, bad-name and unresolvable-link errors carry an errno Python ignores elsewhere (#215)."""
    real = Path.stat

    def stat(self, *, follow_symlinks=True):
        if self.name != 'target':
            return real(self, follow_symlinks=follow_symlinks)
        error = OSError(errno.EINVAL, 'forced')
        error.winerror = winerror  # set by hand: only Windows builds the attribute
        raise error
    monkeypatch.setattr(Path, 'stat', stat)
    assert dicom_io.entry_is(tmp_path / 'target', 'file') is False


def test_entry_is_raises_permission_error_for_a_parent_without_search_permission_twin(tmp_path, monkeypatch):
    real = Path.stat

    def stat(self, *, follow_symlinks=True):
        if self.parent == tmp_path:
            raise PermissionError(errno.EACCES, 'denied')
        return real(self, follow_symlinks=follow_symlinks)
    monkeypatch.setattr(Path, 'stat', stat)
    with pytest.raises(PermissionError):
        dicom_io.entry_is(tmp_path / 'any', 'file')


@pytest.mark.skipif(os.name != 'posix' or os.geteuid() == 0, reason='mode bits do not stop root')
def test_entry_is_raises_permission_error_for_a_parent_with_mode_0o444(tmp_path):
    (tmp_path / 'inside').write_text('x')
    tmp_path.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            dicom_io.entry_is(tmp_path / 'inside', 'file')
    finally:
        tmp_path.chmod(0o755)
