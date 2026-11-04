"""convert error paths (task-01, #33): the narrowed tag catch and the REASON_INTERNAL crash path. Every tree is synthetic."""
import ast
import csv
import inspect
import logging
import random
import re
import shutil
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
import synthetic_dicom as sd
from pydicom.dataset import Dataset
from test_convert import BOX, PID, arr, dose_spec, edit, every_uid, make_plan, run_plan, tree_hashes

from rt_dataprep import convert as conv
from rt_dataprep import masks
from rt_dataprep import series as ser
from rt_dataprep.dicom_io import CaseRefused, read_header
from rt_dataprep.organize import organize

PID_B = 'SYNTHETICPTB'
SECRET = '1.2.3.4.5'  # UID-like, in the message of the exception the tests raise


@pytest.fixture
def tree(tmp_path):
    org, out = tmp_path / 'org', tmp_path / 'out'
    org.mkdir()
    return org, out

def listing(root):
    """Sorted names under root, without the dated per-run copies of the report."""
    return sorted(p.name for p in root.iterdir() if not re.fullmatch(r'\w+_report_\d{8}T\d{6}Z\.csv', p.name))



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
    assert listing(out) == sorted([conv.REPORT_NAME, PID])  # no output and no partial folder of the second

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
    assert listing(out) == [PID, conv.REPORT_NAME]  # the first patient is done, no partial; the report lists what was done


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


# ---------------------------------------------------------------- task-04: series.py (REG parsing and reg_matrix)

VALUES_SMALL = np.zeros((2, 2, 2), dtype=int)
IDENTITY_TEXT = ' '.join(['1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1'])
RIGID = 'RIGID'


class World:
    """A CT on one frame, an MR on another, and the headers by SOP; `reg` writes a REG and returns its header."""

    def __init__(self, folder):
        self.folder = folder
        self.ct = sd.write_ct_series(folder, VALUES_SMALL, patient_id=PID_S, prefix='ct')
        self.mr = sd.write_ct_series(folder, VALUES_SMALL, patient_id=PID_S, prefix='mr', modality='MR')
        self.lookup = {}
        for series in (self.ct, self.mr):
            for sop, path in zip(series['sop_uids'], series['paths'], strict=True):
                self.lookup[sop] = read_header(path)
        self.count = 0
        self.ct_frame, self.mr_frame = self.ct['frame_uid'], self.mr['frame_uid']

    def reg(self, registrations=None, frame_uid=None, sop_uid=None):
        self.count += 1
        path = self.folder / f'reg_{self.count}.dcm'
        sd.write_reg(path, self.ct['study_uid'], self.mr['study_uid'], sop_uid=sop_uid, frame_uid=frame_uid,
                     registrations=registrations, patient_id=PID_S)
        return read_header(path)

    def items(self, ct_matrix, mr_matrix, **extra):
        """RegistrationSequence for one CT item and one MR item with the given matrix lists."""
        return [dict(frame_uid=self.ct_frame, matrices=ct_matrix, **extra), dict(frame_uid=self.mr_frame, matrices=mr_matrix)]

    def registration(self, regs):
        return ser.registration(self.ct_frame, self.mr_frame, regs, self.lookup)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def rigid(**kwargs):
    return [(RIGID, sd.transformation_matrix(**kwargs))]


def apply(text, point):
    """The parsed reg_matrix applied to a point."""
    values = [float(v) for v in text.split()]
    assert len(values) == 16
    return (np.asarray(values).reshape(4, 4) @ [*point, 1.0])[:3]


MR_POINT = np.array([5.0, 6.0, 7.0])  # a named point in the MR frame; the MR frame sits +10 mm in x of the CT frame
CT_POINT = np.array([15.0, 6.0, 7.0])  # the same place in the CT frame


@pytest.mark.parametrize('in_frame', ['ct', 'mr'])
def test_a2_4_the_reg_matrix_maps_a_named_mr_point_to_its_ct_coordinates(world, in_frame):
    # standard wording: each matrix maps its item's frame into the REG frame; the REG frame is the CT's or the MR's
    if in_frame == 'ct':
        sequence = world.items(rigid(), rigid(translation=(10, 0, 0)))
    else:
        sequence = world.items(rigid(translation=(-10, 0, 0)), rigid())
    registered, text = world.registration([world.reg(sequence)])
    assert registered is True
    assert np.allclose(apply(text, MR_POINT), CT_POINT, rtol=0, atol=1e-6)


def test_a_rotated_registration_maps_the_point_with_the_right_order_of_the_matrices(world):
    quarter = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90 degrees about z
    t_ct, t_mr = np.array([1.0, 2.0, 3.0]), np.array([-4.0, 5.0, 0.5])
    sequence = world.items(rigid(rotation=np.eye(3), translation=t_ct), rigid(rotation=quarter, translation=t_mr))
    _, text = world.registration([world.reg(sequence)])
    in_reg = np.asarray(quarter) @ MR_POINT + t_mr  # MR frame -> REG frame
    assert np.allclose(apply(text, MR_POINT), in_reg - t_ct, rtol=0, atol=1e-6)  # REG frame -> CT frame (identity rotation)
    ct_turned = world.items(rigid(rotation=quarter, translation=t_ct), rigid(translation=t_mr))
    _, text = world.registration([world.reg(ct_turned)])
    assert np.allclose(apply(text, MR_POINT), np.asarray(quarter).T @ (MR_POINT + t_mr - t_ct), rtol=0, atol=1e-6)


def test_the_text_is_16_values_row_major_in_10_significant_digits(world):
    _, text = world.registration([world.reg(world.items(rigid(), rigid(translation=(10, 0, 0))))])
    assert text == '1 0 0 10 0 1 0 0 0 0 1 0 0 0 0 1'
    _, text = world.registration([world.reg(world.items(rigid(), rigid(translation=(1 / 3, 0, 0))))])
    assert text == '1 0 0 0.3333333333 0 1 0 0 0 0 1 0 0 0 0 1'


def test_the_same_frame_is_identity_and_no_reg_is_looked_at(world):
    class Boom:
        def __iter__(self):
            raise AssertionError('REG lookup ran')
    assert ser.registration(world.ct_frame, world.ct_frame, Boom(), world.lookup) == (True, 'identity')
    assert ser.registration(world.ct_frame, world.ct_frame, [world.reg(world.items(rigid(), rigid(), ))], {}) == (True, 'identity')


@pytest.mark.parametrize('series_frame', ['', 'SYNTHETICFRAME'])
def test_a_ct_without_a_frame_registers_nothing(world, series_frame):
    assert ser.registration('', series_frame, [world.reg(world.items(rigid(), rigid()))], world.lookup) == (False, '')


def test_a_computed_zero_prints_as_0(world):
    turn = [[np.cos(np.pi / 2), -np.sin(np.pi / 2), 0], [np.sin(np.pi / 2), np.cos(np.pi / 2), 0], [0, 0, 1]]  # cos is 6e-17
    _, text = world.registration([world.reg(world.items(rigid(), rigid(rotation=turn)))])
    assert text == '0 -1 0 0 1 0 0 0 0 0 1 0 0 0 0 1'


@pytest.mark.parametrize('own, kept', [('ct', 'mr'), ('mr', 'ct')])
def test_a_reg_in_one_of_the_two_frames_has_an_implied_identity_item_for_it(world, own, kept):
    if kept == 'mr':
        sequence, own_frame = [dict(frame_uid=world.mr_frame, matrices=rigid(translation=(10, 0, 0)))], world.ct_frame
    else:
        sequence, own_frame = [dict(frame_uid=world.ct_frame, matrices=rigid(translation=(-10, 0, 0)))], world.mr_frame
    registered, text = world.registration([world.reg(sequence, frame_uid=own_frame)])
    assert registered is True and np.allclose(apply(text, MR_POINT), CT_POINT, rtol=0, atol=1e-6)


def test_a_reg_that_omits_an_item_and_is_in_neither_frame_does_not_qualify(world):
    sequence = [dict(frame_uid=world.mr_frame, matrices=rigid())]
    assert world.registration([world.reg(sequence), world.reg(sequence, frame_uid='1.9.9')]) == (False, '')
    assert world.registration([]) == (False, '')


def test_a_study_link_without_a_frame_match_and_a_deformable_reg_do_not_register(world):
    deformable = world.reg(world.items(rigid(), rigid()))
    deformable.DeformableRegistrationSequence = deformable.RegistrationSequence
    del deformable.RegistrationSequence
    assert world.registration([world.reg(None), deformable]) == (False, '')


@pytest.mark.parametrize('bad', [
    [('DEFORMABLE', sd.transformation_matrix(translation=(10, 0, 0)))],
    [('RIGID_SCALE', sd.transformation_matrix(translation=(10, 0, 0)))],
    [(RIGID, sd.transformation_matrix(rotation=np.eye(3) * 1.1, translation=(10, 0, 0)))],
    [(RIGID, sd.transformation_matrix(rotation=np.diag([1, 1, -1]), translation=(10, 0, 0)))],  # a reflection
    [(RIGID, sd.transformation_matrix()[:15])],
    [(RIGID, [*sd.transformation_matrix()[:12], 0, 0, 0.001, 1])],  # last row off
    [(RIGID, [*sd.transformation_matrix()[:3], 'nan', *sd.transformation_matrix()[4:]])],  # non-finite translation
    [(RIGID, [*sd.transformation_matrix()[:3], 'inf', *sd.transformation_matrix()[4:]])],
    [(RIGID, sd.transformation_matrix(translation=(10, 0, 0))), (RIGID, sd.transformation_matrix())],
    [],
])
def test_b3_an_unusable_matrix_gives_an_empty_reg_matrix_and_the_series_still_counts_as_registered(world, bad):
    assert world.registration([world.reg(world.items(rigid(), bad))]) == (True, '')
    assert world.registration([world.reg(world.items(bad, rigid()))]) == (True, '')


def test_b3_two_items_for_one_frame_give_an_empty_reg_matrix(world):
    sequence = [*world.items(rigid(), rigid(translation=(10, 0, 0))), dict(frame_uid=world.ct_frame, matrices=rigid())]
    assert world.registration([world.reg(sequence)]) == (True, '')
    sequence = [*world.items(rigid(), rigid(translation=(10, 0, 0))), dict(frame_uid=world.mr_frame, matrices=rigid())]
    assert world.registration([world.reg(sequence)]) == (True, '')
    implied = [dict(frame_uid=world.mr_frame, matrices=rigid()), dict(frame_uid=world.mr_frame, matrices=rigid())]
    assert world.registration([world.reg(implied, frame_uid=world.ct_frame)]) == (True, '')


def test_b3_a_matrix_within_the_orthonormal_tolerance_is_used(world):
    near = rigid(rotation=np.eye(3) * (1 + 2e-5), translation=(10, 0, 0))  # |R^T R - I| = 4e-5
    registered, text = world.registration([world.reg(world.items(rigid(), near))])
    assert registered and np.allclose(apply(text, MR_POINT), CT_POINT, rtol=0, atol=1e-3)


def test_the_first_qualifying_reg_by_sop_uid_is_used_with_no_fallback(world):
    good = world.items(rigid(), rigid(translation=(10, 0, 0)))
    bad = world.items(rigid(), [('DEFORMABLE', sd.transformation_matrix())])
    unrelated = [dict(frame_uid=world.mr_frame, matrices=rigid())]
    first = world.reg(bad, sop_uid='1.2.10')  # '1.2.10' < '1.2.9' as strings
    second = world.reg(good, sop_uid='1.2.9')
    none = world.reg(unrelated, sop_uid='1.2.0')  # sorts first but does not qualify
    for order in ([first, second, none], [none, second, first]):
        assert world.registration(order) == (True, '')
    assert world.registration([second, none])[1] == '1 0 0 10 0 1 0 0 0 0 1 0 0 0 0 1'


def test_b9_an_item_without_a_frame_takes_the_one_frame_of_the_present_sops_it_references(world):
    ct_item = dict(frame_uid=world.ct_frame, matrices=rigid())
    mr_item = lambda images: dict(matrices=rigid(translation=(10, 0, 0)), images=images)  # noqa: E731
    present = world.mr['sop_uids']
    assert world.registration([world.reg([ct_item, mr_item(present)])]) == (True, '1 0 0 10 0 1 0 0 0 0 1 0 0 0 0 1')
    assert world.registration([world.reg([ct_item, mr_item([*present, '1.7.7'])])])[0] is True  # an absent SOP is ignored
    assert world.registration([world.reg([ct_item, mr_item(['1.7.7'])])]) == (False, '')  # none present
    assert world.registration([world.reg([ct_item, mr_item([present[0], world.ct['sop_uids'][0]])])]) == (False, '')  # two frames
    assert world.registration([world.reg([ct_item, mr_item(None)])]) == (False, '')  # no frame and no references
    ct_by_image = dict(matrices=rigid(), images=world.ct['sop_uids'])  # the CT item resolved the same way
    assert world.registration([world.reg([ct_by_image, mr_item(present)])])[0] is True


def test_a_malformed_reg_tag_makes_that_reg_not_qualify_and_no_exception_escapes(world):
    good = world.reg(world.items(rigid(), rigid(translation=(10, 0, 0))), sop_uid='1.2.9')
    damaged = []
    for change in ('no_ref_uid', 'empty_reference_item', 'frame_none', 'no_sop'):
        reg = world.reg([*world.items(rigid(), rigid()), dict(matrices=rigid(), images=world.mr['sop_uids'])], sop_uid='1.2.0')
        if change == 'no_ref_uid':
            del reg.RegistrationSequence[2].ReferencedImageSequence[0].ReferencedSOPInstanceUID
        elif change == 'empty_reference_item':
            reg.RegistrationSequence[2].ReferencedImageSequence = [Dataset()]
        elif change == 'frame_none':
            reg.RegistrationSequence[0].FrameOfReferenceUID = None
        else:
            del reg.SOPInstanceUID
        damaged.append(reg)
    for reg in damaged:
        assert world.registration([reg]) == (False, '')
        assert world.registration([reg, good]) == (True, '1 0 0 10 0 1 0 0 0 0 1 0 0 0 0 1')
    assert world.registration([world.reg(None), 7, None, 'text', good])[0] is True  # not even a Dataset


# ---------------------------------------------------------------- task-06: placement, images/, plan-less patients, report rows

PLAN_COLUMNS = ['rtplan_uid', 'rtplan_label', 'rtplan_date', 'fractions_planned']
EMPTY_PLAN = dict.fromkeys(PLAN_COLUMNS, '')
MODALITY_ORDER = {'CT': 0, 'MR': 1, 'PT': 2}


def put(folder, prefix='mr', modality='MR', values=VALUES, patient=PID, **kwargs):
    """A synthetic MR (or PT, CT) series in `folder`; the write_ct_series dict."""
    folder.mkdir(parents=True, exist_ok=True)
    return sd.write_ct_series(folder, values, patient_id=patient, prefix=prefix, modality=modality, **kwargs)


def nii(path):
    return sitk.ReadImage(str(path))


def csv_at(folder):
    with open(folder / conv.SERIES_NAME, newline='') as handle:
        return list(csv.DictReader(handle))


def describe_series(series, text, date):
    for path in series['paths']:
        edit(path, lambda ds: (setattr(ds, 'SeriesDescription', text), setattr(ds, 'SeriesDate', date)))


def reg_for(path, ct, mr, shift=(10, 0, 0), others=None, **kwargs):
    """A REG in the CT study: a CT item (identity) and an MR item (p_ct = p_mr + shift), each in its own frame."""
    return sd.write_reg(path, ct['study_uid'], others or mr['study_uid'], patient_id=PID, registrations=[
        dict(frame_uid=ct['frame_uid'], matrices=rigid()), dict(frame_uid=mr['frame_uid'], matrices=rigid(translation=shift))], **kwargs)


def files_of(rows, folder):
    return next(r for r in rows if r.folder == folder).files


def test_convert_uses_the_one_direction_tolerance_of_series_py():
    tree = ast.parse(inspect.getsource(conv))
    assigned = [t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)]
    assert 'DIRECTION_ATOL' not in assigned and conv.DIRECTION_ATOL is ser.DIRECTION_ATOL and conv.DIRECTION_ATOL == 1e-6


def test_a2_4_a_registered_mr_goes_into_the_plan_folder_on_its_own_grid(tree):
    org, out = tree
    folder = org / PID / 'plan_1'
    plan = make_plan(folder, doses=[dose_spec(10)], rois=[BOX], plan_meta=dict(label='Plan A', plan_date='20240301', fractions=5))
    baseline = org.parent / 'baseline'
    conv.convert(org, baseline)
    mr = put(folder, spacing=(1.5, 2.0, 3.0), origin=(7, -4, 12))
    stray = put(folder, prefix='stray')  # its study is named by the REG, its frame is not
    describe_series(mr, 'T1 axial', '20240201')
    reg_for(folder / 'reg.dcm', plan['ct'], mr, others=[mr['study_uid'], stray['study_uid']])
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason, r.skipped_series) for r in rows] == [
        ('plan_1', 'converted', '', 0), ('images', 'converted', '', 0)]
    assert files_of(rows, 'plan_1') == ['planning_ct.nii.gz', 'dose.nii.gz', 'liver.nii.gz', 'series_MR_1.nii.gz', 'series.csv']
    assert files_of(rows, 'images') == ['series_MR_1.nii.gz', 'series.csv']
    for name in ('planning_ct.nii.gz', 'dose.nii.gz', 'liver.nii.gz'):  # the planning outputs are what they were without the MR
        before, after = nii(baseline / PID / 'plan_1' / name), nii(out / PID / 'plan_1' / name)
        assert np.array_equal(sitk.GetArrayFromImage(before), sitk.GetArrayFromImage(after))
        assert (before.GetSize(), before.GetSpacing(), before.GetOrigin(), before.GetDirection()) == (
            after.GetSize(), after.GetSpacing(), after.GetOrigin(), after.GetDirection())
    image = nii(out / PID / 'plan_1' / 'series_MR_1.nii.gz')
    assert image.GetSize() == (5, 3, 4) and np.allclose(image.GetSpacing(), (1.5, 2.0, 3.0), rtol=0, atol=1e-9)
    assert np.allclose(image.GetOrigin(), (7, -4, 12), rtol=0, atol=1e-9) and np.allclose(image.GetDirection(), np.eye(3).ravel())
    assert np.array_equal(sitk.GetArrayFromImage(image), VALUES.astype(np.float32))

    ct_row, mr_row = csv_at(out / PID / 'plan_1')[0], csv_at(out / PID / 'plan_1')[-1]
    assert len(csv_at(out / PID / 'plan_1')) == 4  # ct, dose, mask, mr
    assert {k: mr_row[k] for k in ('file', 'kind', 'modality', 'series_description', 'series_instance_uid', 'study_instance_uid',
                                   'frame_of_reference_uid', 'series_date', 'roi_name', 'dose_summation_type')} == dict(
        file='series_MR_1.nii.gz', kind='series', modality='MR', series_description='T1 axial', series_instance_uid=mr['series_uid'],
        study_instance_uid=mr['study_uid'], frame_of_reference_uid=mr['frame_uid'], series_date='20240201', roi_name='',
        dose_summation_type='')
    assert {k: mr_row[k] for k in PLAN_COLUMNS} == {k: ct_row[k] for k in PLAN_COLUMNS} and mr_row['rtplan_label'] == 'Plan A'
    assert np.allclose(apply(mr_row['reg_matrix'], MR_POINT), CT_POINT, rtol=0, atol=1e-6)

    [stray_row] = csv_at(out / PID / 'images')
    assert (stray_row['file'], stray_row['kind'], stray_row['series_instance_uid'], stray_row['reg_matrix']) == (
        'series_MR_1.nii.gz', 'series', stray['series_uid'], '')
    assert {k: stray_row[k] for k in PLAN_COLUMNS} == EMPTY_PLAN


def test_a2_4b_the_other_ct_of_the_study_goes_to_images(tree):
    org, out = tree
    folder = org / PID / 'plan_1'
    plan = make_plan(folder)
    other = put(folder, prefix='ctb', modality='CT', study_uid=plan['ct']['study_uid'])
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.files) for r in rows] == [
        ('plan_1', 'converted', ['planning_ct.nii.gz', 'series.csv']), ('images', 'converted', ['series_CT_1.nii.gz', 'series.csv'])]
    assert np.array_equal(sitk.GetArrayFromImage(nii(out / PID / 'plan_1' / 'planning_ct.nii.gz')), plan['values'].astype(np.float32))
    assert np.array_equal(sitk.GetArrayFromImage(nii(out / PID / 'images' / 'series_CT_1.nii.gz')), VALUES.astype(np.float32))
    assert csv_at(out / PID / 'images')[0]['series_instance_uid'] == other['series_uid']


def test_a2_4c_registered_to_two_plans_unregistered_and_refused_ct_and_misc_only(tree):
    org, out = tree
    one = make_plan(org / PID / 'plan_1')
    two = make_plan(org / PID / 'plan_2')
    refused = make_plan(org / PID / 'plan_3', doses=[dose_spec(1, 'BRACHY')])
    misc = org / PID / 'misc_files'
    mr = put(misc)
    free = put(misc, prefix='free')
    sd.write_reg(misc / 'reg.dcm', one['ct']['study_uid'], mr['study_uid'], patient_id=PID, registrations=[
        dict(frame_uid=one['ct']['frame_uid'], matrices=rigid()),
        dict(frame_uid=two['ct']['frame_uid'], matrices=rigid(translation=(0, 5, 0))),
        dict(frame_uid=mr['frame_uid'], matrices=rigid(translation=(10, 0, 0)))])
    misc_only = org / PID_B / 'misc_files'
    lone = put(misc_only, patient=PID_B)
    put(org / '.SYNTHETICPTC.partial' / 'misc_files', patient='SYNTHETICPTC')  # an interrupted organize: never converted
    rows = conv.convert(org, out).rows
    assert [(r.patient, r.folder, r.outcome, r.reason) for r in rows] == [
        (PID, 'plan_1', 'converted', ''), (PID, 'plan_2', 'converted', ''), (PID, 'plan_3', 'refused', conv.REASON_DOSE_RULE),
        (PID, 'images', 'converted', ''), (PID_B, 'images', 'converted', '')]
    assert files_of(rows, 'plan_1') == ['planning_ct.nii.gz', 'series_MR_1.nii.gz', 'series.csv']
    assert sorted(p.name for p in (out / PID).iterdir()) == ['images', 'plan_1', 'plan_2']
    for name, shifted in (('plan_1', (10, 0, 0)), ('plan_2', (10, -5, 0))):
        assert sorted(p.name for p in (out / PID / name).iterdir()) == ['planning_ct.nii.gz', 'series.csv', 'series_MR_1.nii.gz']
        assert np.array_equal(sitk.GetArrayFromImage(nii(out / PID / name / 'series_MR_1.nii.gz')), VALUES.astype(np.float32))
        mr_row = csv_at(out / PID / name)[-1]
        assert mr_row['series_instance_uid'] == mr['series_uid']
        assert np.allclose(apply(mr_row['reg_matrix'], MR_POINT), MR_POINT + shifted, rtol=0, atol=1e-6)
    assert sorted(p.name for p in (out / PID / 'images').iterdir()) == ['series.csv', 'series_CT_1.nii.gz', 'series_MR_1.nii.gz']
    assert [(r['file'], r['series_instance_uid']) for r in csv_at(out / PID / 'images')] == [
        ('series_CT_1.nii.gz', refused['ct']['series_uid']), ('series_MR_1.nii.gz', free['series_uid'])]
    assert sorted(p.name for p in (out / PID_B).iterdir()) == ['images'] and csv_at(out / PID_B / 'images')[0]['series_instance_uid'] == lone['series_uid']
    assert listing(out) == [PID, PID_B, conv.REPORT_NAME]


def test_k_counts_per_folder_and_modality_in_date_time_uid_order_with_no_gap_for_a_skipped_series(tree):
    org, out = tree
    folder = org / PID / 'plan_1'
    plan = make_plan(folder)
    frame = dict(frame_uid=plan['ct']['frame_uid'])  # same frame as the planning CT: registered by identity
    late = put(folder, prefix='a', series_uid='1.9.9', series_date='20240103', **frame)
    tie_b = put(folder, prefix='b', series_uid='1.9.8', series_date='20240102', series_time='090000', **frame)
    tie_a = put(folder, prefix='c', series_uid='1.9.5', series_date='20240102', series_time='090000', **frame)
    undated = put(folder, prefix='d', series_uid='1.9.7', **frame)
    put(folder, prefix='e', series_uid='1.9.1', positions=[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 4)], **frame)  # fails the check
    broken = put(folder, prefix='f', series_uid='1.9.2', **frame)  # passes the check, fails the read
    victim = Path(broken['paths'][1])
    victim.write_bytes(victim.read_bytes()[:-40])
    pet = put(folder, prefix='g', modality='PT', **frame)
    ct2 = put(folder, prefix='h', modality='CT', **frame)
    unregistered = put(folder, prefix='i')
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.skipped_series) for r in rows] == [('plan_1', 0), ('images', 2)]
    assert files_of(rows, 'plan_1') == ['planning_ct.nii.gz', 'series_CT_1.nii.gz', *(f'series_MR_{k}.nii.gz' for k in range(1, 5)),
                                        'series_PT_1.nii.gz', 'series.csv']
    got = [(r['file'], r['series_instance_uid']) for r in csv_at(out / PID / 'plan_1')[1:]]
    assert got == [('series_CT_1.nii.gz', ct2['series_uid']), ('series_MR_1.nii.gz', undated['series_uid']),
                   ('series_MR_2.nii.gz', tie_a['series_uid']), ('series_MR_3.nii.gz', tie_b['series_uid']),
                   ('series_MR_4.nii.gz', late['series_uid']), ('series_PT_1.nii.gz', pet['series_uid'])]
    assert sorted(p.name for p in (out / PID / 'plan_1').iterdir()) == sorted(['planning_ct.nii.gz', 'series.csv', *(n for n, _ in got)])
    [image_row] = csv_at(out / PID / 'images')  # k starts again at 1 in images/
    assert (image_row['file'], image_row['series_instance_uid'], image_row['reg_matrix']) == ('series_MR_1.nii.gz', unregistered['series_uid'], '')
    assert {k: image_row[k] for k in PLAN_COLUMNS} == EMPTY_PLAN


def test_b4_excluded_model_studies_and_localizers_are_written_nowhere_and_not_counted(tree):
    org, out = tree
    folder = org / PID / 'plan_1'
    make_plan(folder)
    misc = org / PID / 'misc_files'
    study = pydicom.uid.generate_uid()
    put(misc, prefix='x', study_uid=study)
    put(misc, prefix='y', study_uid=study, manufacturer_model_name='Acuity Sim')
    put(misc, prefix='loc', image_type=['ORIGINAL', 'PRIMARY', 'LOCALIZER'])
    kept = put(misc, prefix='kept')
    put(misc, prefix='scout', values=VALUES[:1], series_uid=kept['series_uid'], frame_uid=kept['frame_uid'], study_uid=kept['study_uid'],
        image_type=['ORIGINAL', 'PRIMARY', 'LOCALIZER'])  # one localizer file inside a series that is otherwise written
    nostudy = put(misc, prefix='nostudy')
    for path in nostudy['paths']:
        edit(path, lambda ds: delattr(ds, 'StudyInstanceUID'))
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.files, r.skipped_series) for r in rows] == [
        ('plan_1', 'converted', ['planning_ct.nii.gz', 'series.csv'], 0), ('images', 'converted', ['series_MR_1.nii.gz', 'series.csv'], 0)]
    assert [r['series_instance_uid'] for r in csv_at(out / PID / 'images')] == [kept['series_uid']]
    assert sorted(p.name for p in (out / PID / 'images').iterdir()) == ['series.csv', 'series_MR_1.nii.gz']
    assert np.array_equal(sitk.GetArrayFromImage(nii(out / PID / 'images' / 'series_MR_1.nii.gz')), VALUES.astype(np.float32))


def copy_to(paths, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, folder / Path(path).name)


def test_b6_a_series_in_two_plan_folders_and_misc_is_read_once_and_written_once_per_destination(tree, monkeypatch):
    org, out = tree
    one = make_plan(org / PID / 'plan_1')
    two = make_plan(org / PID / 'plan_2')
    mr = put(org / PID / 'plan_1')
    sd.write_reg(org / PID / 'plan_1' / 'reg.dcm', one['ct']['study_uid'], mr['study_uid'], patient_id=PID, registrations=[
        dict(frame_uid=one['ct']['frame_uid'], matrices=rigid()), dict(frame_uid=two['ct']['frame_uid'], matrices=rigid()),
        dict(frame_uid=mr['frame_uid'], matrices=rigid(translation=(10, 0, 0)))])
    copies = [*mr['paths'], org / PID / 'plan_1' / 'reg.dcm']
    copy_to(copies, org / PID / 'plan_2')  # organize copies a REG-linked study into every plan folder, and misc_files/ keeps its own
    copy_to(copies, org / PID / 'misc_files')
    reads, writes = [], []
    real_read, real_write = ser.read_series, conv.write_atomic
    monkeypatch.setattr(ser, 'read_series', lambda slices: reads.append(len(slices)) or real_read(slices))
    monkeypatch.setattr(conv, 'write_atomic', lambda image, path: writes.append(Path(path).parent.name + '/' + Path(path).name) or real_write(image, path))
    rows = conv.convert(org, out).rows
    assert reads == [4]
    assert sorted(w for w in writes if '/series_' in w) == ['plan_1/series_MR_1.nii.gz', 'plan_2/series_MR_1.nii.gz']
    for name in ('plan_1', 'plan_2'):
        assert [r['series_instance_uid'] for r in csv_at(out / PID / name)[1:]] == [mr['series_uid']]
    assert [(r.folder, r.outcome, r.reason, r.files, r.skipped_series) for r in rows if r.folder == 'images'] == [
        ('images', 'skipped', conv.REASON_NO_IMAGES, [], 0)]
    assert not (out / PID / 'images').exists()


@pytest.mark.parametrize('model, destination, other', [(None, 'plan_1', 'images'), ('Acuity', 'images', 'plan_1')])
def test_a_reg_in_an_excluded_study_registers_nothing(tmp_path, model, destination, other):  # the None case is the control
    org, out = tmp_path / 'org', tmp_path / 'out'
    plan = make_plan(org / PID / 'plan_1')
    misc = org / PID / 'misc_files'
    mr = put(misc)
    sd.write_reg(misc / 'reg.dcm', pydicom.uid.generate_uid(), mr['study_uid'], patient_id=PID, manufacturer_model_name=model, registrations=[
        dict(frame_uid=plan['ct']['frame_uid'], matrices=rigid()), dict(frame_uid=mr['frame_uid'], matrices=rigid())])
    conv.convert(org, out)
    assert (out / PID / destination / 'series_MR_1.nii.gz').is_file() and not (out / PID / other / 'series_MR_1.nii.gz').exists()


def test_two_copies_of_one_reg_behave_as_the_first_in_folder_order(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    plan_two = make_plan(org / PID / 'plan_2')
    mr = put(org / PID / 'misc_files')
    sop = '1.2.826.0.1.3680043.9.7156.1'
    sd.write_reg(org / PID / 'plan_1' / 'reg.dcm', plan['ct']['study_uid'], mr['study_uid'], sop_uid=sop, patient_id=PID, registrations=[
        dict(frame_uid=plan['ct']['frame_uid'], matrices=rigid())])  # no MR item: this copy does not qualify
    sd.write_reg(org / PID / 'plan_2' / 'reg.dcm', plan['ct']['study_uid'], mr['study_uid'], sop_uid=sop, patient_id=PID, registrations=[
        dict(frame_uid=plan['ct']['frame_uid'], matrices=rigid()), dict(frame_uid=mr['frame_uid'], matrices=rigid())])
    sd.write_reg(org / PID / 'misc_files' / 'reg.dcm', plan['ct']['study_uid'], mr['study_uid'], sop_uid=sop, patient_id=PID, registrations=[
        dict(frame_uid=plan_two['ct']['frame_uid'], matrices=rigid()), dict(frame_uid=mr['frame_uid'], matrices=rigid())])
    conv.convert(org, out)
    assert not list((out / PID / 'plan_1').glob('series_MR_*')) and not list((out / PID / 'plan_2').glob('series_MR_*'))
    assert (out / PID / 'images' / 'series_MR_1.nii.gz').is_file()


def test_b7_report_rows_in_order_with_skipped_series_counted_once_on_the_images_row(tree):
    org, out = tree
    for n in (10, 2):
        make_plan(org / PID / f'plan_{n}')
    bad = put(org / PID / 'plan_10', prefix='bad', positions=[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 4)])
    copy_to(bad['paths'], org / PID / 'plan_2')
    copy_to(bad['paths'], org / PID / 'misc_files')
    good = put(org / PID / 'misc_files')
    make_plan(org / PID_B / 'plan_1', patient=PID_B)  # no other series at all: no images row
    conv.convert(org, out)
    with open(out / conv.REPORT_NAME, newline='') as handle:
        written = list(csv.DictReader(handle))
    assert [(r['patient'], r['folder'], r['outcome'], r['reason'], r['files'], r['skipped_series']) for r in written] == [
        (PID, 'plan_2', 'converted', '', 'planning_ct.nii.gz;series.csv', '0'),
        (PID, 'plan_10', 'converted', '', 'planning_ct.nii.gz;series.csv', '0'),
        (PID, 'images', 'converted', '', 'series_MR_1.nii.gz;series.csv', '1'),
        (PID_B, 'plan_1', 'converted', '', 'planning_ct.nii.gz;series.csv', '0')]
    assert [r['series_instance_uid'] for r in csv_at(out / PID / 'images')] == [good['series_uid']]


def test_b7_a_patient_with_neither_plans_nor_candidates_keeps_the_no_plans_row_and_gets_no_folder(tree):
    org, out = tree
    put(org / PID / 'misc_files', image_type=['ORIGINAL', 'PRIMARY', 'LOCALIZER'])  # a localizer is not a candidate
    (org / PID_B).mkdir()
    rows = conv.convert(org, out).rows
    assert [(r.patient, r.folder, r.outcome, r.reason) for r in rows] == [
        (PID, '', 'skipped', conv.REASON_NO_PLANS), (PID_B, '', 'skipped', conv.REASON_NO_PLANS)]
    assert listing(out) == [conv.REPORT_NAME]


def test_b7_a_plan_less_patient_whose_candidates_all_fail_gets_an_empty_folder_and_a_rerun_skips_it(tree):
    org, out = tree
    put(org / PID / 'misc_files', positions=[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 4)])
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason, r.files, r.skipped_series) for r in rows] == [
        ('images', 'skipped', conv.REASON_NO_IMAGES, [], 1)]
    assert (out / PID).is_dir() and list((out / PID).iterdir()) == []
    assert [(r.folder, r.outcome, r.reason) for r in conv.convert(org, out).rows] == [('', 'skipped', conv.REASON_EXISTS)]


def test_b8_an_unreadable_series_in_misc_is_skipped_and_nothing_names_the_id_a_path_or_a_uid(tree, tmp_path, capfd, caplog, recwarn):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    misc = org / PID / 'misc_files'
    broken = put(misc)
    victim = Path(broken['paths'][1])
    victim.write_bytes(victim.read_bytes()[:-40])
    secrets = [PID, str(tmp_path), *every_uid(misc), *every_uid(plan['folder'])]
    assert len(secrets) > 10
    caplog.set_level(logging.DEBUG)
    capfd.readouterr()
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason, r.skipped_series) for r in rows] == [
        ('plan_1', 'converted', '', 0), ('images', 'skipped', conv.REASON_NO_IMAGES, 1)]
    captured = capfd.readouterr()
    texts = [captured.out, captured.err, caplog.text, *(r.getMessage() for r in caplog.records), *(str(w.message) for w in recwarn.list)]
    assert [s for s in secrets for t in texts if s in t] == []


def test_the_closing_log_line_counts_images_rows_apart_from_plan_rows(tree, caplog):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    put(org / PID / 'plan_1', frame_uid=plan['ct']['frame_uid'])
    put(org / PID_B / 'misc_files', patient=PID_B)
    (org / 'SYNTHETICPTC').mkdir()
    caplog.set_level(logging.INFO)
    rows = conv.convert(org, out).rows
    assert [(r.patient, r.folder, r.outcome) for r in rows] == [
        (PID, 'plan_1', 'converted'), (PID, 'images', 'skipped'), (PID_B, 'images', 'converted'), ('SYNTHETICPTC', '', 'skipped')]
    assert 'convert: 1 plan rows (1 converted, 0 refused, 0 skipped), 2 images rows, 1 patient rows (0 refused, 1 skipped)' in caplog.messages


def test_a2_11_the_organized_tree_keeps_its_hashes_with_series_and_a_rerun_skips_every_patient(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', doses=[dose_spec(10)])
    mr = put(org / PID / 'misc_files')
    reg_for(org / PID / 'misc_files' / 'reg.dcm', plan['ct'], mr)
    put(org / PID_B / 'misc_files', patient=PID_B)
    make_plan(org / 'SYNTHETICPTC' / 'plan_1', patient='SYNTHETICPTC', doses=[dose_spec(1, 'BRACHY')])
    before = tree_hashes(org)
    first = conv.convert(org, out).rows
    assert tree_hashes(org) == before and sorted({r.patient for r in first}) == [PID, PID_B, 'SYNTHETICPTC']
    rerun = conv.convert(org, out).rows
    assert sorted((r.patient, r.folder, r.outcome, r.reason) for r in rerun) == [
        (p, '', 'skipped', conv.REASON_EXISTS) for p in (PID, PID_B, 'SYNTHETICPTC')]
    assert tree_hashes(org) == before


def test_a_sagittal_mr_on_the_planning_frame_lands_in_the_plan_folder_with_identity(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    put(org / PID / 'plan_1', frame_uid=plan['ct']['frame_uid'], spacing=(1.5, 2.0, 3.0), origin=(10, 20, 30), orientation=SAG)
    conv.convert(org, out)
    image = nii(out / PID / 'plan_1' / 'series_MR_1.nii.gz')
    assert image.GetSize() == (5, 3, 4) and np.allclose(image.GetSpacing(), (1.5, 2.0, 3.0), rtol=0, atol=1e-9)
    assert np.allclose(image.GetOrigin(), (10, 20, 30), rtol=0, atol=1e-9)
    assert np.allclose(image.GetDirection(), (0, 0, -1, 1, 0, 0, 0, -1, 0), rtol=0, atol=1e-9)
    assert csv_at(out / PID / 'plan_1')[-1]['reg_matrix'] == 'identity'


def test_a_type_error_in_placement_is_an_internal_error_row_and_aborts(tree, monkeypatch):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    put(org / PID / 'plan_1', frame_uid=plan['ct']['frame_uid'])

    def boom(slices):
        raise TypeError(SECRET)
    monkeypatch.setattr(ser, 'read_series', boom)
    with pytest.raises(conv.ConvertAborted) as caught:
        conv.convert(org, out)
    assert str(caught.value) == conv.REASON_INTERNAL
    assert [(r['patient'], r['folder'], r['outcome'], r['reason']) for r in report(out)] == [(PID, '', conv.OUTCOME_REFUSED, conv.REASON_INTERNAL)]
    assert listing(out) == [conv.REPORT_NAME]


def test_an_oserror_writing_a_series_is_still_an_io_error_row(tree, monkeypatch):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    put(org / PID / 'plan_1', frame_uid=plan['ct']['frame_uid'])
    real = conv.write_atomic

    def write(image, path):
        if Path(path).name.startswith('series_'):
            raise OSError(conv.REASON_WRITE)
        return real(image, path)
    monkeypatch.setattr(conv, 'write_atomic', write)
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows] == [('', conv.OUTCOME_REFUSED, conv.REASON_IO_ERROR)]
    assert listing(out) == [conv.REPORT_NAME]


# ---------------------------------------------------------------- review: a CT known to be short never reaches images/

@pytest.mark.parametrize('damage', ['end-slice-deleted', 'unreadable-slice'])
def test_a_refused_plans_ct_that_may_be_short_is_skipped_not_written_to_images(tmp_path, damage):
    org, out = tmp_path / 'org', tmp_path / 'out'
    plan = make_plan(org / PID / 'plan_1')
    last = Path(plan['ct']['paths'][-1])
    if damage == 'end-slice-deleted':
        last.unlink()  # refused with REASON_CT_SLICE_MISSING; the rest is still one uniform volume
    else:
        last.write_bytes(b'not a dicom file')  # refused with REASON_FILE_UNREADABLE; the rest is still one volume
    rows = conv.convert(org, out).rows
    plan_row = next(r for r in rows if r.folder == 'plan_1')
    images_row = next(r for r in rows if r.folder == conv.IMAGES_NAME)
    assert plan_row.outcome == conv.OUTCOME_REFUSED
    assert (images_row.outcome, images_row.reason, images_row.skipped_series) == (
        conv.OUTCOME_SKIPPED, conv.REASON_NO_IMAGES, 1)
    assert not (out / PID / conv.IMAGES_NAME).exists()


# ---------------------------------------------------------------- hardening #37, #35

class Unreadable:
    def get(self, *args):
        raise TypeError


def test_a_file_whose_identifying_tags_raise_is_dropped_not_the_patient(tmp_path):
    mr = put(tmp_path / 'misc_files')
    good = [(Path(path), read_header(path)) for path in mr['paths']]
    for headers in ([(tmp_path / 'bad.dcm', Unreadable()), *good], [*good, (tmp_path / 'bad.dcm', Unreadable())]):
        found, regs, by_sop = ser.candidates(headers, ('Acuity',))
        assert [uid for uid, _ in found] == [mr['series_uid']] and regs == [] and len(by_sop) == len(good)
    slice_of_mr = Unreadable()
    slice_of_mr.SeriesInstanceUID = mr['series_uid']  # a dropped slice of the MR: the MR may now lack an end slice
    assert ser.candidates([*good, (tmp_path / 'bad.dcm', slice_of_mr)], ('Acuity',))[0] == []


def copy_renamed(paths, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, folder / f'copy_{Path(path).name}')


def test_a_series_shared_with_a_refused_unreadable_sibling_is_written_from_the_converted_copy(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    two = make_plan(org / PID / 'plan_2')
    (org / PID / 'plan_1' / 'junk.dcm').write_bytes(b'not a dicom file')  # plan_1: REASON_FILE_UNREADABLE
    mr = put(org / PID / 'plan_1', frame_uid=two['ct']['frame_uid'])
    copy_to(mr['paths'], org / PID / 'plan_2')
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.reason) for r in rows if r.folder.startswith('plan_')] == [
        ('plan_1', conv.REASON_FILE_UNREADABLE), ('plan_2', '')]
    assert [r['series_instance_uid'] for r in csv_at(out / PID / 'plan_2')[1:]] == [mr['series_uid']]
    assert next(r for r in rows if r.folder == conv.IMAGES_NAME).skipped_series == 1  # plan_1's own CT: short-only


def test_a_slice_missing_planning_ct_is_skipped_even_where_a_converted_sibling_holds_a_copy(tree):
    org, out = tree
    one = make_plan(org / PID / 'plan_1')
    make_plan(org / PID / 'plan_2')
    Path(one['ct']['paths'][-1]).unlink()  # plan_1: REASON_CT_SLICE_MISSING, the rest still one uniform volume
    copy_renamed(one['ct']['paths'][:-1], org / PID / 'plan_2')
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.reason) for r in rows if r.folder.startswith('plan_')] == [
        ('plan_1', conv.REASON_CT_SLICE_MISSING), ('plan_2', '')]
    images_row = next(r for r in rows if r.folder == conv.IMAGES_NAME)
    assert (images_row.outcome, images_row.skipped_series) == (conv.OUTCOME_SKIPPED, 1)
    assert len(csv_at(out / PID / 'plan_2')) == 1  # the planning CT row only


def test_a_plan_on_a_planning_ct_its_sibling_found_short_is_refused_too(tree):
    org, out = tree
    one = make_plan(org / PID / 'plan_1')
    second = org / PID / 'plan_2'  # a second RTPLAN on the same CT and RTSTRUCT, its CT copy complete
    copy_renamed([*one['ct']['paths'], one['folder'] / 'struct.dcm'], second)
    sd.write_rtplan(second / 'plan.dcm', one['struct'], np.zeros((2, 3)), patient_id=PID, study_uid=one['ct']['study_uid'])
    Path(one['ct']['paths'][-1]).unlink()  # plan_1: REASON_CT_SLICE_MISSING, so this CT series is known short
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows if r.folder.startswith('plan_')] == [
        ('plan_1', 'refused', conv.REASON_CT_SLICE_MISSING), ('plan_2', 'refused', conv.REASON_CT_SLICE_MISSING)]
    assert listing(out / PID) == []


def test_a_sibling_plan_on_a_different_ct_stays_converted_when_one_ct_is_short(tree):
    org, out = tree
    one = make_plan(org / PID / 'plan_1')
    make_plan(org / PID / 'plan_2')  # its own CT series
    Path(one['ct']['paths'][-1]).unlink()  # plan_1: REASON_CT_SLICE_MISSING
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome) for r in rows if r.folder.startswith('plan_')] == [('plan_1', 'refused'), ('plan_2', 'converted')]


def test_legacy_fx_folders_convert_like_plan_folders(tree):
    org, out = tree
    make_plan(org / PID / 'fx_2')
    make_plan(org / PID / 'fx_10')
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome) for r in rows if r.folder.startswith('fx_')] == [('fx_2', 'converted'), ('fx_10', 'converted')]
    assert {'fx_2', 'fx_10'} <= set(listing(out / PID))


# ---------------------------------------------------------------- hardening B6 (#53)

def test_itks_out_of_memory_reading_the_planning_ct_aborts_the_run_and_leaves_no_partial_folder(tree, monkeypatch, caplog):
    org, out = tree
    make_plan(org / PID / 'plan_1')

    def execute(self):
        raise RuntimeError('SYNTHETIC_SOURCE:179:\nFailed to allocate memory for image.')  # ITK's, a RuntimeError
    monkeypatch.setattr(sitk.ImageSeriesReader, 'Execute', execute)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(conv.ConvertAborted) as caught:
        conv.convert(org, out)
    assert str(caught.value) == conv.REASON_INTERNAL
    assert [(r['patient'], r['folder'], r['outcome'], r['reason']) for r in report(out)] == [
        (PID, '', conv.OUTCOME_REFUSED, conv.REASON_INTERNAL)]
    assert listing(out) == [conv.REPORT_NAME]  # no patient folder, no partial
    assert any('MemoryError' in message for message in caplog.messages) and PID not in caplog.text


# ---------------------------------------------------------------- hardening B6 (#21): an RTSTRUCT in a study of its own

@pytest.mark.parametrize('registered', [False, True])
def test_an_rtstruct_study_that_also_holds_another_series_still_converts_and_the_series_is_placed_by_registration(
        tmp_path, registered):
    src, org, out = tmp_path / 'src', tmp_path / 'org', tmp_path / 'out'
    folder = src / PID / 'p'
    folder.mkdir(parents=True)
    ct = sd.write_ct_series(folder, np.zeros((4, 5, 6), int), patient_id=PID)
    struct_study = pydicom.uid.generate_uid()  # the RTSTRUCT was exported in a study of its own ...
    struct = sd.write_rtstruct_rois(folder / 'struct.dcm', ct, [BOX], patient_id=PID, study_uid=struct_study)
    sd.write_rtplan(folder / 'plan.dcm', struct, np.zeros((2, 3)), patient_id=PID, study_uid=ct['study_uid'])
    mr = put(src / PID / 'mr', study_uid=struct_study)  # ... which holds an MR series too, on a frame of its own
    if registered:
        reg_for(folder / 'reg.dcm', ct, mr)
    organize(src, org)
    rows = conv.convert(org, out).rows
    assert (rows[0].folder, rows[0].outcome, rows[0].reason) == ('plan_1', 'converted', '')
    assert arr(out / PID / 'plan_1' / 'liver.nii.gz').sum() == 27
    if registered:
        assert files_of(rows, 'plan_1') == ['planning_ct.nii.gz', 'liver.nii.gz', 'series_MR_1.nii.gz', 'series.csv']
        assert [(r.folder, r.outcome, r.reason) for r in rows[1:]] == [('images', 'skipped', conv.REASON_NO_IMAGES)]
        assert csv_at(out / PID / 'plan_1')[-1]['reg_matrix'] == '1 0 0 10 0 1 0 0 0 0 1 0 0 0 0 1'
    else:
        assert files_of(rows, 'plan_1') == ['planning_ct.nii.gz', 'liver.nii.gz', 'series.csv']
        assert [(r.folder, r.outcome, r.files) for r in rows[1:]] == [('images', 'converted', ['series_MR_1.nii.gz', 'series.csv'])]
        assert csv_at(out / PID / 'images')[0]['reg_matrix'] == ''


# ---------------------------------------------------------------- hardening B6 (#42, subset): tolerance boundaries

@pytest.mark.parametrize('axis', [2, 0], ids=['along-the-normal', 'sideways'])
@pytest.mark.parametrize('deviation, kept', [(0.009, True), (0.011, False)])
def test_a_slice_position_just_under_series_atol_is_kept_and_just_over_is_skipped(tmp_path, axis, deviation, kept):
    positions = [[0, 0, 0], [0, 0, 2], [0, 0, 4], [0, 0, 6]]
    positions[1][axis] += deviation  # one slice off its place on the line
    _, items = make_series(tmp_path, **spaced(positions=positions))
    assert (ser.check_volume(items) is not None) is kept
    assert (ser.read_series(items) is not None) is kept


def last_row(last_row_value, index):
    row = [0, 0, 0, 1]
    row[index] += last_row_value
    return [(RIGID, [*sd.transformation_matrix(translation=(10, 0, 0))[:12], *row])]


@pytest.mark.parametrize('bad, usable', [
    (rigid(rotation=np.eye(3) * (1 + 4.5e-5), translation=(10, 0, 0)), True),  # |R^T R - I| = 9.0e-5
    (rigid(rotation=np.eye(3) * (1 + 5.5e-5), translation=(10, 0, 0)), False),  # 1.1e-4
    (last_row(0.9e-6, 2), True),
    (last_row(1.1e-6, 2), False),
    (last_row(0.9e-6, 3), True),
    (last_row(1.1e-6, 3), False),
], ids=['rigid-inside', 'rigid-outside', 'last-row-inside', 'last-row-outside', 'last-corner-inside', 'last-corner-outside'])
def test_a_matrix_just_inside_rigid_atol_or_last_row_atol_is_used_and_just_outside_gives_an_empty_text(world, bad, usable):
    registered, text = world.registration([world.reg(world.items(rigid(), bad))])
    assert registered is True
    assert (text != '') is usable
    if usable:
        assert np.allclose(apply(text, MR_POINT), CT_POINT, rtol=0, atol=1e-3)


@pytest.mark.parametrize('own', ['ct', 'mr'])
def test_a_reg_whose_own_frame_also_has_an_explicit_item_uses_that_item_not_an_extra_identity(world, own):
    # the implied identity only fills in a frame with no item: with one present there is still exactly one matrix to use
    if own == 'ct':
        sequence, frame = world.items(rigid(translation=(3, 0, 0)), rigid(translation=(13, 0, 0))), world.ct_frame
    else:
        sequence, frame = world.items(rigid(translation=(-13, 0, 0)), rigid(translation=(-3, 0, 0))), world.mr_frame
    registered, text = world.registration([world.reg(sequence, frame_uid=frame)])
    assert registered is True and text != ''
    assert np.allclose(apply(text, MR_POINT), CT_POINT, rtol=0, atol=1e-6)


def test_a_single_slice_is_skipped_by_its_own_guard_before_any_geometry_is_read(tmp_path, monkeypatch):
    _, items = make_series(tmp_path, shuffle=False, **spaced())
    def read_past_the_guard(ds):
        raise AssertionError('a single slice got past the guard')
    monkeypatch.setattr(ser, '_pixel_spacing', read_past_the_guard)
    monkeypatch.setattr(ser, '_position', read_past_the_guard)
    assert ser.check_volume(items[:1]) is None and ser.check_volume([]) is None
    with pytest.raises(AssertionError):  # the spy is live: two slices do reach the geometry
        ser.check_volume(items[:2])


def test_a_blank_patient_id_series_in_misc_files_is_never_placed(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    mr = put(org / PID / 'misc_files', frame_uid=plan['ct']['frame_uid'])  # an MR on the plan's frame
    for path in mr['paths']:  # organize put it in misc_files because its PatientID is blank
        ds = pydicom.dcmread(path)
        ds.PatientID = ''
        ds.save_as(path)
    conv.convert(org, out)
    assert len(csv_at(out / PID / 'plan_1')) == 1  # the planning CT row only
    assert not (out / PID / conv.IMAGES_NAME).exists()


def images_uids(out):
    return {r['series_instance_uid'] for r in csv_at(out / PID / conv.IMAGES_NAME)} if (out / PID / conv.IMAGES_NAME).exists() else set()


def test_an_unreadable_misc_file_whose_study_can_be_salvaged_skips_that_study_only(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    misc = org / PID / 'misc_files'
    at_risk, other = put(misc, prefix='a'), put(misc, prefix='b')
    (misc / 'stray.dcm').write_bytes(Path(at_risk['paths'][0]).read_bytes()[132:])  # no preamble: unreadable, its tags intact
    rows = conv.convert(org, out).rows
    assert images_uids(out) == {other['series_uid']}
    assert next(r for r in rows if r.folder == conv.IMAGES_NAME).skipped_series == 1


def test_an_unreadable_misc_file_with_no_study_skips_every_misc_only_series(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    misc = org / PID / 'misc_files'
    put(misc, prefix='a')
    put(misc, prefix='b')
    kept = put(org / PID / 'plan_1', prefix='c', frame_uid=plan['ct']['frame_uid'])  # in the plan folder, not misc-only
    (misc / 'stray.dcm').write_bytes(b'SYNTHETIC not dicom at all')
    rows = conv.convert(org, out).rows
    assert images_uids(out) == set()
    assert [r['series_instance_uid'] for r in csv_at(out / PID / 'plan_1') if r['kind'] == 'series'] == [kept['series_uid']]
    assert next(r for r in rows if r.folder == conv.IMAGES_NAME).skipped_series == 2


def test_a_file_dropped_for_a_bad_tag_skips_every_series_of_its_study(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    misc = org / PID / 'misc_files'
    first = put(misc, prefix='a')
    sibling = put(misc, prefix='c', study_uid=first['study_uid'])  # another series of the same study
    other = put(misc, prefix='b')
    bad = pydicom.dcmread(first['paths'][0], stop_before_pixels=True).SOPInstanceUID
    real = getattr(ser, '_file_keys')  # noqa: B009 (a private name, never dotted)

    def keys(ds):
        if str(ds.get('SOPInstanceUID', '')) == bad:
            raise ValueError('SYNTHETIC bad tag')
        return real(ds)
    monkeypatch.setattr(ser, '_file_keys', keys)
    rows = conv.convert(org, out).rows
    assert images_uids(out) == {other['series_uid']}
    assert sibling['series_uid'] not in images_uids(out)
    assert next(r for r in rows if r.folder == conv.IMAGES_NAME).skipped_series == 1  # the sibling; the first was dropped whole


def test_a_slice_dropped_by_candidate_series_skips_every_series_of_its_study(tree, monkeypatch):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    misc = org / PID / 'misc_files'
    first = put(misc, prefix='a')
    sibling = put(misc, prefix='c', study_uid=first['study_uid'])
    other = put(misc, prefix='b')
    bad = pydicom.dcmread(first['paths'][-1], stop_before_pixels=True).SOPInstanceUID  # an end slice: the rest would pass
    real = getattr(ser, '_candidate_series')  # noqa: B009 (a private name, never dotted)

    def series_uid(ds):
        if str(ds.get('SOPInstanceUID', '')) == bad:
            raise TypeError('SYNTHETIC bad ImageType')
        return real(ds)
    monkeypatch.setattr(ser, '_candidate_series', series_uid)
    rows = conv.convert(org, out).rows
    assert images_uids(out) == {other['series_uid']}  # not first, one slice short, nor its sibling
    assert sibling['series_uid'] not in images_uids(out)
    assert next(r for r in rows if r.folder == conv.IMAGES_NAME).skipped_series == 2


def test_a_bad_tag_file_with_no_readable_study_skips_every_misc_only_series(tree, monkeypatch):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    misc = org / PID / 'misc_files'
    first = put(misc, prefix='a')
    put(misc, prefix='b')
    kept = put(org / PID / 'plan_1', prefix='c', frame_uid=plan['ct']['frame_uid'])
    ds = pydicom.dcmread(first['paths'][0])
    del ds.StudyInstanceUID
    ds.save_as(first['paths'][0])
    bad = ds.SOPInstanceUID
    real = getattr(ser, '_file_keys')  # noqa: B009 (a private name, never dotted)

    def keys(ds):
        if str(ds.get('SOPInstanceUID', '')) == bad:
            raise ValueError('SYNTHETIC bad tag')
        return real(ds)
    monkeypatch.setattr(ser, '_file_keys', keys)
    conv.convert(org, out)
    assert images_uids(out) == set()
    assert [r['series_instance_uid'] for r in csv_at(out / PID / 'plan_1') if r['kind'] == 'series'] == [kept['series_uid']]


def test_a_plan_folder_series_of_a_salvaged_tainted_study_is_skipped(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    at_risk = put(org / PID / 'plan_1', prefix='c', frame_uid=plan['ct']['frame_uid'])  # registered to plan_1 by its frame
    misc = org / PID / 'misc_files'
    misc.mkdir()
    (misc / 'stray.dcm').write_bytes(Path(at_risk['paths'][0]).read_bytes()[132:])  # unreadable, its study salvaged
    conv.convert(org, out)
    assert [r for r in csv_at(out / PID / 'plan_1') if r['kind'] == 'series'] == []  # the #36 deviation: skipped, not placed


def test_a_series_whose_slices_disagree_on_study_is_skipped(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    mixed = put(org / PID / 'misc_files', prefix='a')
    ds = pydicom.dcmread(mixed['paths'][1])
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.save_as(mixed['paths'][1])
    rows = conv.convert(org, out).rows
    assert images_uids(out) == set()
    assert next(r for r in rows if r.folder == conv.IMAGES_NAME).skipped_series == 1


def test_salvaging_a_study_tag_prints_and_logs_nothing_of_its_value(tree, capfd, caplog, recwarn):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    misc = org / PID / 'misc_files'
    series = put(misc, prefix='a')
    ds = pydicom.dcmread(series['paths'][0])
    ds.StudyInstanceUID = 'FAKE_NAME^SECRET'  # invalid for VR UI: pydicom warns with the value when it is decoded
    ds.save_as(misc / 'stray.dcm')
    (misc / 'stray.dcm').write_bytes((misc / 'stray.dcm').read_bytes()[132:])  # no preamble: unreadable, its tags intact
    caplog.set_level(logging.DEBUG)
    caplog.clear()  # writing the value above warned already; only the run counts
    recwarn.clear()
    capfd.readouterr()
    conv.convert(org, out)
    captured = capfd.readouterr()
    texts = [captured.out, captured.err, caplog.text, *(str(w.message) for w in recwarn.list)]
    assert [t for t in texts if 'SECRET' in t] == []


def test_a_misc_series_with_a_blank_patient_id_slice_is_skipped_not_written_short(tree):
    org, out = tree
    make_plan(org / PID / 'plan_1')
    mr = put(org / PID / 'misc_files', prefix='a')
    ds = pydicom.dcmread(mr['paths'][-1])
    ds.PatientID = ''
    ds.save_as(mr['paths'][-1])
    rows = conv.convert(org, out).rows
    assert images_uids(out) == set()
    assert next(r for r in rows if r.folder == conv.IMAGES_NAME).skipped_series == 1


def test_a_plan_whose_planning_ct_end_slice_sits_in_misc_with_a_blank_id_is_refused(tree):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    last = Path(plan['ct']['paths'][-1])
    struct = pydicom.dcmread(plan['folder'] / 'struct.dcm')  # the slice list leaves the end slice out: its loss passes the CT checks
    sop = str(pydicom.dcmread(last, stop_before_pixels=True).SOPInstanceUID)
    for frame in struct.ReferencedFrameOfReferenceSequence:
        for study in frame.RTReferencedStudySequence:
            for series in study.RTReferencedSeriesSequence:
                series.ContourImageSequence = [i for i in series.ContourImageSequence if str(i.ReferencedSOPInstanceUID) != sop]
    struct.save_as(plan['folder'] / 'struct.dcm')
    ds = pydicom.dcmread(last)
    ds.PatientID = ''
    (org / PID / 'misc_files').mkdir()
    ds.save_as(org / PID / 'misc_files' / last.name)
    last.unlink()
    rows = conv.convert(org, out).rows
    assert [(r.folder, r.outcome, r.reason) for r in rows if r.folder == 'plan_1'] == [
        ('plan_1', 'refused', conv.REASON_CT_SLICE_MISSING)]
    assert not (out / PID / 'plan_1').exists()
