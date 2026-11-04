"""convert error paths (task-01, #33): the narrowed tag catch and the REASON_INTERNAL crash path. Every tree is synthetic."""
import ast
import csv
import inspect
import logging
import random
import re
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import synthetic_dicom as sd
from test_convert import PID, dose_spec, edit, make_plan, run_plan

from rt_dataprep import convert as conv
from rt_dataprep import masks
from rt_dataprep import series as ser
from rt_dataprep.dicom_io import CaseRefused, read_header

PID_B = 'SYNTHETICPTB'
SECRET = '1.2.3.4.5'  # UID-like, in the message of the exception the tests raise


@pytest.fixture
def tree(tmp_path):
    org, out = tmp_path / 'org', tmp_path / 'out'
    org.mkdir()
    return org, out


def report(out):
    with open(out / conv.REPORT_NAME, newline='') as handle:
        return list(csv.DictReader(handle))


def crash_in_masks_for(monkeypatch, patient, exc):
    """build_masks as is, except that it raises `exc` for the plan folders of `patient`."""
    real = masks.build_masks

    def build(kept, ct_paths, struct_path, *args):
        if patient in str(struct_path):
            raise exc
        return real(kept, ct_paths, struct_path, *args)
    monkeypatch.setattr(masks, 'build_masks', build)


def raise_in(monkeypatch, name, exc):
    def boom(*args, **kwargs):
        raise exc
    monkeypatch.setattr(conv, name, boom)


# ---------------------------------------------------------------- the tag catch is per header-only call

@pytest.mark.parametrize('what', ['plan_struct_reference', 'ct_spacing', 'dose_units'])
def test_a_missing_header_tag_still_refuses_the_plan_with_missing_tag(tree, what):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', doses=[dose_spec(10)] if what == 'dose_units' else [])
    if what == 'plan_struct_reference':
        edit(plan['folder'] / 'plan.dcm', lambda ds: delattr(ds, 'ReferencedStructureSetSequence'))
    elif what == 'ct_spacing':
        edit(plan['ct']['paths'][1], lambda ds: delattr(ds, 'PixelSpacing'))
    else:
        edit(plan['folder'] / 'dose_0.dcm', lambda ds: delattr(ds, 'DoseUnits'))
    with pytest.raises(CaseRefused) as caught:
        run_plan(plan['folder'], out / 'case')
    assert str(caught.value) == conv.REASON_MISSING_TAG and not (out / 'case').exists()
    assert getattr(caught.value, '__cause__') is None and getattr(caught.value, '__context__') is None  # noqa: B009


def test_a_tag_error_while_building_the_series_rows_refuses_the_plan_and_removes_the_folder(tree, monkeypatch):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    raise_in(monkeypatch, '_series_rows', KeyError(SECRET))
    with pytest.raises(CaseRefused) as caught:
        run_plan(plan['folder'], out / 'case')
    assert str(caught.value) == conv.REASON_MISSING_TAG and not (out / 'case').exists()


@pytest.mark.parametrize('name', ['_read_ct', 'write_atomic'])
def test_calls_outside_the_tag_catch_do_not_turn_a_type_error_into_a_refusal(tree, monkeypatch, name):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    raise_in(monkeypatch, name, TypeError(SECRET))
    with pytest.raises(TypeError):
        run_plan(plan['folder'], out / 'case')


# ---------------------------------------------------------------- the crash path

def test_convert_aborted_is_a_plain_exception_with_the_fixed_text():
    assert issubclass(conv.ConvertAborted, Exception)
    assert not issubclass(conv.ConvertAborted, (ValueError, OSError, CaseRefused))
    assert str(conv.ConvertAborted(conv.REASON_INTERNAL)) == conv.REASON_INTERNAL


def two_patients(org):
    make_plan(org / PID / 'plan_1', patient=PID)
    make_plan(org / PID_B / 'plan_1', patient=PID_B)
    make_plan(org / PID_B / 'plan_2', patient=PID_B)


def test_an_unexpected_error_on_the_second_patient_keeps_the_first_and_flushes_the_report(tree, tmp_path, monkeypatch, capfd, caplog):
    org, out = tree
    two_patients(org)
    crash_in_masks_for(monkeypatch, PID_B, TypeError(SECRET))
    caplog.set_level(logging.DEBUG)
    capfd.readouterr()
    with pytest.raises(conv.ConvertAborted) as caught:
        conv.convert(org, out)
    err = caught.value
    assert str(err) == conv.REASON_INTERNAL and err.args == (conv.REASON_INTERNAL,)
    assert getattr(err, '__cause__') is None and getattr(err, '__context__') is None  # noqa: B009
    assert [(r['patient'], r['folder'], r['outcome'], r['reason']) for r in report(out)] == [
        (PID, 'plan_1', conv.OUTCOME_CONVERTED, ''), (PID_B, '', conv.OUTCOME_REFUSED, conv.REASON_INTERNAL)]
    assert sorted(p.name for p in out.iterdir()) == sorted([conv.REPORT_NAME, PID])  # no output and no partial folder of the second

    captured = capfd.readouterr()
    texts = [captured.out, captured.err, caplog.text, f'{err!r} {err}']
    assert [s for s in (SECRET, str(tmp_path)) for t in texts if s in t] == []

    records = [r.getMessage() for r in caplog.records if 'TypeError' in r.getMessage()]
    assert len(records) == 1
    module, line = re.fullmatch(r'.*TypeError.* (\S+):(\d+)', records[0]).groups()
    source = inspect.getsource(conv).splitlines()
    call = next(i for i, text in enumerate(source, 1) if 'masks.build_masks(' in text)
    assert module == 'rt_dataprep.convert' and call <= int(line) <= call + 2  # the innermost frame inside rt_dataprep
    assert '/' not in records[0] and '.py' not in records[0] and SECRET not in records[0]


@pytest.mark.parametrize('exc', [KeyboardInterrupt, SystemExit])
def test_ctrl_c_and_exit_add_no_row_and_still_remove_the_partial(tree, monkeypatch, exc):
    org, out = tree
    two_patients(org)
    crash_in_masks_for(monkeypatch, PID_B, exc())
    with pytest.raises(exc):
        conv.convert(org, out)
    assert [p.name for p in out.iterdir()] == [PID]  # the first patient is done; no partial, no report


def test_an_output_side_oserror_is_still_an_io_error_row_and_the_run_goes_on(tree, monkeypatch):
    org, out = tree
    two_patients(org)
    real = conv.write_atomic

    def write(image, path):
        if PID_B in str(path):
            raise OSError(conv.REASON_WRITE)
        return real(image, path)
    monkeypatch.setattr(conv, 'write_atomic', write)
    rows = conv.convert(org, out).rows
    assert [(r.patient, r.outcome, r.reason) for r in rows] == [
        (PID, conv.OUTCOME_CONVERTED, ''), (PID_B, conv.OUTCOME_REFUSED, conv.REASON_IO_ERROR)]


# ---------------------------------------------------------------- task-03: series.py (generic volume check and read)

SAG = (0, 1, 0, 0, 0, -1)  # row cosine +y, column cosine -z, normal -x
VALUES = np.random.default_rng(3).integers(-500, 500, size=(4, 3, 5))  # (slices, rows, columns)
PID_S = 'SYNTHETICPTS'


def make_series(folder, values=VALUES, shuffle=True, **kwargs):
    """(the write_ct_series dict, [(path, header)] in a shuffled order) for a synthetic MR series in `folder`."""
    kwargs.setdefault('modality', 'MR')
    ct = sd.write_ct_series(folder, values, patient_id=PID_S, **kwargs)
    items = [(Path(p), read_header(p)) for p in ct['paths']]
    if shuffle:
        random.Random(1).shuffle(items)
    return ct, items


def test_the_sagittal_mr_is_read_with_the_geometry_of_its_headers(tmp_path):
    _, items = make_series(tmp_path, spacing=(1.5, 2.0, 3.0), origin=(10, 20, 30), orientation=SAG)
    image = ser.read_series(items)
    assert image.GetPixelID() == sitk.sitkFloat32 and image.GetSize() == (5, 3, 4)
    assert np.allclose(image.GetSpacing(), (1.5, 2.0, 3.0), rtol=0, atol=1e-9)
    assert np.allclose(image.GetOrigin(), (10, 20, 30), rtol=0, atol=1e-9)
    assert np.allclose(image.GetDirection(), (0, 0, -1, 1, 0, 0, 0, -1, 0), rtol=0, atol=1e-9)  # columns: row, column, normal
    assert np.array_equal(sitk.GetArrayFromImage(image), VALUES.astype(np.float32))


def test_check_volume_returns_the_slices_sorted_along_the_normal(tmp_path):
    ct, items = make_series(tmp_path)
    assert [str(p) for p, _ in ser.check_volume(items)] == [str(p) for p in ct['paths']]


def test_a_tilted_in_plane_rotation_is_read(tmp_path):
    c, s = np.cos(0.5), np.sin(0.5)
    _, items = make_series(tmp_path, orientation=(c, s, 0, -s, c, 0))
    image = ser.read_series(items)
    assert np.allclose(image.GetDirection(), (c, -s, 0, s, c, 0, 0, 0, 1), rtol=0, atol=1e-9)


def spaced(**kwargs):
    return dict(spacing=(1.0, 1.0, 2.0), **kwargs)


@pytest.mark.parametrize('kwargs', [
    dict(positions=[(0, 0, 0), (0, 0, 0), (0, 0, 2), (0, 0, 2)]),  # two echoes: duplicate positions
    dict(number_of_frames=3),  # multi-frame
    dict(positions=[(0, 0, 0), (0, 0, 2), (0, 0, 6), (0, 0, 8)]),  # missing middle slice
    dict(pixel_spacing=False),
    dict(orientation=(1, 0, 0, 0.001, 1, 0)),  # off-orthogonal by 1e-3
    dict(orientation=(1.001, 0, 0, 0, 1, 0)),  # off unit length by 1e-3
    dict(positions=[(0, 0, 0), (0.02, 0, 2), (0.04, 0, 4), (0.06, 0, 6)]),  # shear
    dict(positions=[(0, 0, 0), (0, 0, 2.02), (0, 0, 4), (0, 0, 6)]),  # 0.02 mm jitter
], ids=['two-echo', 'multi-frame', 'gap', 'no-pixel-spacing', 'off-orthogonal', 'off-unit', 'shear', 'big-jitter'])
def test_a_series_that_is_not_a_uniform_single_frame_volume_is_skipped(tmp_path, kwargs):
    _, items = make_series(tmp_path, **spaced(**kwargs))
    assert ser.check_volume(items) is None and ser.read_series(items) is None


@pytest.mark.parametrize('kwargs', [
    dict(positions=[(0, 0, 0), (0, 0, 2.005), (0, 0, 3.995), (0, 0, 6)]),  # 0.005 mm jitter
    dict(orientation=(1, 0, 0, 1e-6, 1, 0)),  # DS rounding, orthogonality
    dict(orientation=(1.000001, 0, 0, 0, 1, 0)),  # DS rounding, length
], ids=['jitter', 'ds-orthogonality', 'ds-length'])
def test_scanner_rounding_is_read(tmp_path, kwargs):
    _, items = make_series(tmp_path, **spaced(**kwargs))
    assert ser.read_series(items).GetSize() == (5, 3, 4)


def header_edit(tmp_path, change, count=None):
    """items of a good series, `change(index, header)` applied to each header (in memory)."""
    _, items = make_series(tmp_path, shuffle=False, **spaced())
    for index, (_, ds) in enumerate(items):
        change(index, ds)
    return items


def test_one_slice_off_in_orientation_size_spacing_or_frame_is_skipped(tmp_path):
    def last(change):
        return lambda i, ds: change(ds) if i == 3 else None
    changes = [lambda ds: setattr(ds, 'ImageOrientationPatient', [1, 0, 0, 0.001, 1, 0]),
               lambda ds: setattr(ds, 'Rows', 4),
               lambda ds: setattr(ds, 'PixelSpacing', [1.0, 1.02]),
               lambda ds: setattr(ds, 'FrameOfReferenceUID', '1.2.3')]
    for change in changes:
        assert ser.check_volume(header_edit(tmp_path, last(change))) is None
    assert ser.check_volume(header_edit(tmp_path, last(lambda ds: setattr(ds, 'PixelSpacing', [1.0, 1.005])))) is not None


def test_a_single_slice_and_an_empty_list_are_skipped(tmp_path):
    _, items = make_series(tmp_path, shuffle=False, **spaced())
    assert ser.check_volume(items[:1]) is None and ser.check_volume([]) is None


@pytest.mark.parametrize('damage', [
    lambda ds: delattr(ds, 'ImagePositionPatient'),
    lambda ds: delattr(ds, 'FrameOfReferenceUID'),
    lambda ds: delattr(ds, 'Rows'),
    lambda ds: setattr(ds, 'ImageOrientationPatient', [1, 0, 0]),
    lambda ds: setattr(ds, 'ImagePositionPatient', [0, 0]),
    lambda ds: setattr(ds, 'PixelSpacing', [1.0]),
    lambda ds: setattr(ds, 'NumberOfFrames', None),
    lambda ds: setattr(ds, 'ImageOrientationPatient', None),
], ids=['no-position', 'no-frame', 'no-rows', 'short-orientation', 'short-position', 'short-spacing', 'frames-empty', 'none'])
def test_a_missing_or_malformed_tag_skips_the_series_without_an_exception(tmp_path, damage):
    items = header_edit(tmp_path, lambda i, ds: damage(ds) if i == 1 else None)
    assert ser.check_volume(items) is None and ser.read_series(items) is None


@pytest.mark.parametrize('change', [
    lambda i, ds: setattr(ds, 'PixelSpacing', [0.0, 1.0]),
    lambda i, ds: setattr(ds, 'PixelSpacing', [-1.0, 1.0]),
    lambda i, ds: setattr(ds, 'ImagePositionPatient', [0.0, 0.0, 0.0]),
], ids=['zero-spacing', 'negative-spacing', 'all-duplicate-positions'])
def test_a_degenerate_spacing_or_step_skips_the_series_without_an_exception(tmp_path, change):
    items = header_edit(tmp_path, change)  # every slice agrees, so only the degenerate value can refuse
    assert ser.check_volume(items) is None and ser.read_series(items) is None


def test_a_size_mismatch_or_an_unreadable_file_skips_the_series(tmp_path):
    items = header_edit(tmp_path, lambda i, ds: setattr(ds, 'Rows', 4))  # all headers agree, the pixels do not
    assert ser.read_series(items) is None
    items = header_edit(tmp_path, lambda i, ds: None)
    items[2][0].write_bytes(b'not dicom')
    assert ser.read_series(items) is None


def test_only_execute_runs_inside_quiet_call_the_geometry_is_set_after_it(tmp_path, monkeypatch):
    _, items = make_series(tmp_path, spacing=(1.5, 2.0, 3.0), origin=(10, 20, 30), orientation=SAG)
    inside, calls = [False], []
    real = ser.quiet_call

    def spy(fn, reason):
        inside[0] = True
        try:
            return real(fn, reason)
        finally:
            inside[0] = False
    monkeypatch.setattr(ser, 'quiet_call', spy)
    def recording(name, real_method):
        def wrap(self, value):
            calls.append((name, inside[0]))
            return real_method(self, value)
        return wrap
    for name in ('SetOrigin', 'SetSpacing', 'SetDirection'):
        monkeypatch.setattr(sitk.Image, name, recording(name, getattr(sitk.Image, name)))
    assert ser.read_series(items) is not None
    assert sorted(calls) == [('SetDirection', False), ('SetOrigin', False), ('SetSpacing', False)]


def test_pt_reads_stored_times_each_slices_own_slope_plus_intercept(tmp_path):
    rescales = [(1.0, 0.0), (2.0, 5.0), (0.5, -3.0), (3.25, 1.0)]
    _, items = make_series(tmp_path, modality='PT', rescales=rescales, **spaced())
    expected = np.stack([(plane.astype(np.float32) * np.float32(slope) + np.float32(intercept))
                         for plane, (slope, intercept) in zip(VALUES, rescales, strict=True)])
    image = ser.read_series(items)
    assert image.GetPixelID() == sitk.sitkFloat32
    assert np.allclose(sitk.GetArrayFromImage(image), expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(sitk.GetArrayFromImage(image), VALUES)  # the slopes were applied, not just stored values


def test_b8_truncated_pixel_data_skips_the_series_and_nothing_names_the_id_a_path_or_a_uid(tmp_path, capfd, caplog, recwarn):
    ct, items = make_series(tmp_path, **spaced())
    victim = Path(ct['paths'][1])
    victim.write_bytes(victim.read_bytes()[:-40])
    secrets = [PID_S, str(tmp_path), ct['series_uid'], ct['frame_uid'], ct['study_uid'], *ct['sop_uids']]
    caplog.set_level(logging.DEBUG)
    capfd.readouterr()
    assert ser.read_series(items) is None
    captured = capfd.readouterr()
    texts = [captured.out, captured.err, caplog.text, *(r.getMessage() for r in caplog.records),
             *(str(w.message) for w in recwarn.list)]
    assert [s for s in secrets for t in texts if s in t] == []


def test_series_py_imports_nothing_from_convert_and_owns_its_tolerances():
    tree = ast.parse(inspect.getsource(ser))
    modules = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in modules if 'convert' in m]
    assert (ser.SERIES_ATOL_MM, ser.ORTHO_ATOL, ser.DIRECTION_ATOL) == (0.01, 1e-4, 1e-6)
