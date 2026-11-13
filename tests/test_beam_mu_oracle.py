"""
Bit-identity oracle for beam_volume.beam_mu (spec 2026-11-13-beam-mu-perf, AC1 and AC3).

The reference is tests/beam_mu_reference.py, a frozen copy of the 324e2b2 algorithm; it runs on the same points
(beam_volume._points) as the code under test. These tests pass against 324e2b2 itself: they guard the rewrite, so
they can never go red before it. Bit-identity means the same dtype and shape and equal bytes.
"""
import copy
import math

import numpy as np
import pytest
import SimpleITK as sitk
from beam_mu_reference import reference_beam_mu
from test_beam_volume import IDENTITY, ROT90, ct_image, set_mlc, static, vmat_arc, written

from rt_dataprep import beam_volume, plan_table
from rt_dataprep.beam_volume import REASON_BEAM_VOLUME
from rt_dataprep.dicom_io import CaseRefused

NO_WARNINGS = pytest.mark.filterwarnings('error::RuntimeWarning')
REFUSED = 'refused'
SAD = 1000.0
BOUNDARIES = list(range(-40, 50, 10))  # 8 leaf pairs
PAIRS = len(BOUNDARIES) - 1


def _oblique():
    a, b = math.radians(30), math.radians(20)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return tuple((rz @ rx).ravel())


OBLIQUE = _oblique()


CALLS = []  # (path, id(point)) per call beam_mu makes for one point in one chunk, in order


@pytest.fixture(autouse=True)
def spy(monkeypatch):
    """AC2: records which path each point took: 'fallback' (`_add`, today's loop) or 'rows' (`_add_rows`)."""
    CALLS.clear()
    for name, path in (('_add', 'fallback'), ('_add_rows', 'rows')):
        real = getattr(beam_volume, name)

        def wrapped(chunk, point, *args, _real=real, _path=path):
            CALLS.append((_path, id(point)))
            return _real(chunk, point, *args)
        monkeypatch.setattr(beam_volume, name, wrapped)


def fallbacks():
    return sum(path == 'fallback' for path, _ in CALLS)


def grid(ct):  # as beam_mu builds it
    return np.array(ct.GetOrigin()), np.array(ct.GetDirection()).reshape(3, 3), np.array(ct.GetSpacing())


def _refusal_or(run):
    try:
        return run()
    except CaseRefused as refused:
        assert str(refused) == REASON_BEAM_VOLUME
        return REFUSED


def new_result(plan, rows, ct, position):
    return _refusal_or(lambda: sitk.GetArrayFromImage(beam_volume.beam_mu(plan, rows, ct, position)))


def ref_result(plan, rows, ct, position):
    return _refusal_or(lambda: reference_beam_mu(beam_volume._points(plan, rows, position, grid(ct)), ct))


def assert_identical(plan, rows, ct, position, *, lit=True, fallback=False):
    CALLS.clear()
    new, ref = new_result(plan, rows, ct, position), ref_result(plan, rows, ct, position)
    if isinstance(ref, str) or isinstance(new, str):
        assert new == ref == REFUSED
        return
    assert (fallbacks() > 0) == fallback, CALLS  # AC2: the interval path is what these cases test
    assert new.dtype == ref.dtype == np.float32 and new.shape == ref.shape
    assert new.tobytes() == ref.tobytes()
    assert not lit or ref.any()  # a case that lights nothing proves nothing


def check(tmp_path, beams, position='HFS', ct=None, **kw):
    ct = ct or ct_image()
    plan = written(tmp_path, beams if isinstance(beams, list) else [beams])
    assert_identical(plan, plan_table.plan_rows(plan, ct), ct, position, **kw)


def base(**kw):
    kw.setdefault('sad', SAD)
    return static(**kw)


def with_isocenter(beam, iso):
    beam['ControlPointSequence'][0]['IsocenterPosition'] = list(iso)
    return beam


def drop_jaw(beam, kind):
    beam['BeamLimitingDeviceSequence'] = [d for d in beam['BeamLimitingDeviceSequence'] if d['RTBeamLimitingDeviceType'] != kind]
    cp = beam['ControlPointSequence'][0]
    cp['BeamLimitingDevicePositionSequence'] = [i for i in cp['BeamLimitingDevicePositionSequence']
                                                if i['RTBeamLimitingDeviceType'] != kind]
    return beam


def mlc_beam(x1=None, x2=None, jaws=(-60.0, 60.0, -60.0, 60.0), **kw):
    """Closed, open, partly open and reversed (x1 > x2) leaf pairs unless given."""
    x1 = x1 or [0.0, -30.0, -10.0, -45.0, 20.0, -20.0, 0.0, -5.0]
    x2 = x2 or [0.0, 30.0, 5.0, 40.0, 10.0, 25.0, 0.0, 50.0]
    beam = base(jaws=jaws, mlc=True, pairs=PAIRS, **kw)
    set_mlc(beam, BOUNDARIES, {0: (x1, x2)})
    return beam


@NO_WARNINGS
@pytest.mark.parametrize('gantry', [0, 45, 90, 180, 270, 315])
def test_ac1_gantry(tmp_path, gantry):
    check(tmp_path, base(gantry=gantry))


@NO_WARNINGS
@pytest.mark.parametrize('coll', [0, 30, 90])
def test_ac1_collimator(tmp_path, coll):
    check(tmp_path, base(coll=coll, gantry=45))


@NO_WARNINGS
@pytest.mark.parametrize('couch', [0, 90, 270])
def test_ac1_couch(tmp_path, couch):
    check(tmp_path, base(couch=couch, gantry=45))


@NO_WARNINGS
@pytest.mark.parametrize('position', ['HFS', 'HFP', 'FFS', 'FFP'])
@pytest.mark.parametrize('gantry', [0, 90])
def test_ac1_patient_position(tmp_path, position, gantry):
    check(tmp_path, base(gantry=gantry, jaws=(-10.0, 60.0, -30.0, 40.0)), position)


@NO_WARNINGS
@pytest.mark.parametrize('direction', [IDENTITY, ROT90, OBLIQUE], ids=['identity', 'rot90', 'oblique'])
def test_ac1_ct_direction(tmp_path, direction):
    check(tmp_path, base(gantry=30, coll=20), ct=ct_image(direction=direction))


@NO_WARNINGS
@pytest.mark.parametrize('spacing', [(3.0, 4.0, 5.0), (5.0, 2.5, 1.5)])
def test_ac1_anisotropic_spacing(tmp_path, spacing):
    check(tmp_path, base(gantry=60, coll=15), ct=ct_image(spacing=spacing))


@NO_WARNINGS
def test_ac1_off_centre_isocenter(tmp_path):
    check(tmp_path, with_isocenter(base(gantry=45, coll=10), (20.0, -35.0, 40.0)))


@NO_WARNINGS
def test_ac1_asymmetric_jaws(tmp_path):
    check(tmp_path, base(gantry=20, jaws=(-7.0, 63.0, -41.0, 12.0)))


@NO_WARNINGS
@pytest.mark.parametrize('gantry', [0, 45, 90, 270])
def test_ac1_mlc_closed_open_and_partly_open_pairs(tmp_path, gantry):
    check(tmp_path, mlc_beam(gantry=gantry))


@NO_WARNINGS
def test_ac1_jaw_only_beam(tmp_path):
    check(tmp_path, base(jaws=(-35.0, 60.0, -20.0, 45.0)))


@NO_WARNINGS
@pytest.mark.parametrize('dropped', ['ASYMX', 'ASYMY'])
@pytest.mark.parametrize('gantry', [0, 90])
def test_ac1_mlc_with_one_jaw_axis_missing(tmp_path, dropped, gantry):
    check(tmp_path, drop_jaw(mlc_beam(gantry=gantry), dropped))


@NO_WARNINGS
@pytest.mark.parametrize('gantry', [0, 90, 270])
@pytest.mark.parametrize('mlc', [False, True])
def test_ac1_jaws_on_voxel_centres(tmp_path, gantry, mlc):  # 5 mm grid: jaws on multiples of 5 sit on voxel centres
    jaws = (-40.0, 45.0, -30.0, 35.0)
    check(tmp_path, mlc_beam(gantry=gantry, jaws=jaws) if mlc else base(gantry=gantry, jaws=jaws))


@NO_WARNINGS
@pytest.mark.parametrize('sad', [400.0, 1000.0])
@pytest.mark.parametrize('gantry', [0, 45])
def test_ac1_sad(tmp_path, sad, gantry):  # SAD 400 at gantry 45: a grid corner is nearer the source than SAD/2
    check(tmp_path, base(sad=sad, gantry=gantry), fallback=sad == 400.0 and gantry == 45)


@NO_WARNINGS
@pytest.mark.parametrize('coll', [0, 90])
@pytest.mark.parametrize('gantry', [90, 270])
@pytest.mark.parametrize('direction', [IDENTITY, ROT90, OBLIQUE], ids=['identity', 'rot90', 'oblique'])
def test_ac1_degenerate_slopes(tmp_path, gantry, coll, direction):  # cos(90) is 6e-17, not 0: near-zero slopes along rows
    check(tmp_path, mlc_beam(gantry=gantry, coll=coll, jaws=(-40.0, 45.0, -30.0, 35.0)), ct=ct_image(direction=direction))


def _random_case(seed):
    rng = np.random.default_rng(seed)
    uniform = rng.uniform
    sad = uniform(800, 1200)
    spacing = tuple(uniform(2.0, 5.0, 3))
    direction = [IDENTITY, ROT90, OBLIQUE][rng.integers(3)]
    iso = tuple(uniform(-50, 50, 3))
    jaws = (-uniform(5, 80), uniform(5, 80), -uniform(5, 80), uniform(5, 80))
    angles = dict(gantry=uniform(0, 360), coll=uniform(0, 360), couch=uniform(0, 360))
    if rng.integers(2):
        x1 = list(uniform(-60, 10, PAIRS))
        x2 = [lo + w for lo, w in zip(x1, uniform(-10, 80, PAIRS), strict=True)]
        x1[0], x2[0] = -20.0, 25.0  # at least one open pair
        beam = mlc_beam(x1, x2, jaws=jaws, sad=sad, **angles)
    else:
        beam = base(jaws=jaws, sad=sad, **angles)
    return with_isocenter(beam, iso), ct_image(spacing=spacing, direction=direction), ['HFS', 'HFP', 'FFS', 'FFP'][rng.integers(4)]


@NO_WARNINGS
@pytest.mark.parametrize('seed', range(60))
def test_ac1_seeded_random_geometries(tmp_path, seed):  # SAD 800..1200, spacing <= 5 mm, |isocenter| <= 50 mm, 61^3
    beam, ct, position = _random_case(seed)
    check(tmp_path, beam, position, ct, lit=False)


@NO_WARNINGS
def test_ac1_random_sample_is_not_vacuous(tmp_path):  # most seeded cases must light voxels, or the sample proves little
    lit_cases = 0
    for seed in range(60):
        beam, ct, position = _random_case(seed)
        plan = written(tmp_path, [beam])
        rows = plan_table.plan_rows(plan, ct)
        lit_cases += bool(ref_result(plan, rows, ct, position).any())
    assert lit_cases >= 45


def _rows_case(tmp_path, mutate, beam=None, ct=None):
    ct = ct or ct_image()
    plan = written(tmp_path, [beam or mlc_beam(gantry=30)])
    rows = copy.deepcopy(plan_table.plan_rows(plan, ct))
    mutate(rows)
    return plan, rows, ct


def _set(key, value):
    def mutate(rows):
        rows[1][key] = value
    return mutate


def _list_item(key, index, value):
    def mutate(rows):
        rows[1][key] = list(rows[1][key])
        rows[1][key][index] = value
    return mutate


NON_FINITE = {
    'jaw_x1_nan': _set('jaw_x1', float('nan')),
    'jaw_y2_inf': _set('jaw_y2', float('inf')),
    'mlc_position_nan': _list_item('mlc_positions', 3, float('nan')),
    'last_boundary_inf': _list_item('leaf_position_boundaries', -1, float('inf')),
    'isocenter_x_inf': _set('isocenter_x', float('inf')),
}


@pytest.mark.parametrize('mutate', NON_FINITE.values(), ids=NON_FINITE)
def test_ac1_non_finite_values_refuse_or_match(tmp_path, mutate):  # runs with warnings permitted: inf warns in both paths
    plan, rows, ct = _rows_case(tmp_path, mutate)
    assert_identical(plan, rows, ct, 'HFS', lit=False, fallback=True)


def test_ac1_ct_origin_beyond_5000_mm_in_beam_coordinates(tmp_path):
    beam = base(jaws=(-10000.0, 10000.0, -10000.0, 10000.0))
    plan, rows, ct = _rows_case(tmp_path, lambda rows: None, beam, ct_image(centre=(6000.0, 0.0, 0.0)))
    assert_identical(plan, rows, ct, 'HFS', lit=False, fallback=True)


def _mixed_beams():
    first = mlc_beam(gantry=0, number=1)
    second = base(gantry=45, sad=400.0, number=2, jaws=(-60.0, 60.0, -60.0, 60.0))  # corner depth < SAD/2: falls back
    third = base(gantry=90, number=3, jaws=(-40.0, 45.0, -30.0, 35.0), mu=60.0)
    return [first, second, third]


@NO_WARNINGS
@pytest.mark.parametrize('reverse', [False, True], ids=['forward', 'reversed'])
def test_ac1_mixed_fallback_plan(tmp_path, reverse):
    beams = _mixed_beams()
    check(tmp_path, beams[::-1] if reverse else beams, fallback=True)


@NO_WARNINGS
@pytest.mark.parametrize('reverse', [False, True], ids=['forward', 'reversed'])
def test_ac2_mixed_plan_runs_both_paths_in_point_order(tmp_path, monkeypatch, reverse):  # float32 sums are order-sensitive
    ct = ct_image()
    plan = written(tmp_path, _mixed_beams()[::-1] if reverse else _mixed_beams())
    rows = plan_table.plan_rows(plan, ct)
    monkeypatch.setattr(beam_volume, 'CHUNK_VOXELS', 61 * 61 * 20)  # four chunks
    CALLS.clear()
    beam_volume.beam_mu(plan, rows, ct, 'HFS')
    paths = {path for path, _ in CALLS}
    assert paths == {'fallback', 'rows'}
    per_chunk = len(CALLS) // 4
    assert len(CALLS) == 4 * per_chunk and all(CALLS[i:i + per_chunk] == CALLS[:per_chunk] for i in range(0, len(CALLS), per_chunk))
    points = beam_volume._points(plan, rows, 'HFS', grid(ct))
    expected = ['fallback' if p['sad'] == 400.0 else 'rows' for p in points]
    assert [path for path, _ in CALLS[:per_chunk]] == expected


@NO_WARNINGS
@pytest.mark.parametrize('fraction', [0.0, 1e9], ids=['all_rows', 'all_fallback'])
def test_ac2_fallback_threshold_does_not_change_the_output(tmp_path, monkeypatch, fraction):
    monkeypatch.setattr(beam_volume, 'FALLBACK_DEPTH_FRACTION', fraction)
    check(tmp_path, mlc_beam(gantry=45), fallback=fraction > 1)


@NO_WARNINGS
@pytest.mark.parametrize('chunk_voxels', [61 * 61 * 20, 1], ids=['four_chunks', 'one_plane_each'])
def test_ac3_chunking_does_not_change_the_output(tmp_path, monkeypatch, chunk_voxels):
    ct = ct_image()
    beams = [vmat_arc(1, 0.0, 2.0), *_mixed_beams()[1:]]
    plan = written(tmp_path, beams)
    rows = plan_table.plan_rows(plan, ct)
    whole = new_result(plan, rows, ct, 'HFS')  # one chunk: the 61 x 61 x 61 grid is far below the default CHUNK_VOXELS
    monkeypatch.setattr(beam_volume, 'CHUNK_VOXELS', chunk_voxels)
    assert -(-61 // max(1, chunk_voxels // (61 * 61))) >= 3
    chunked = new_result(plan, rows, ct, 'HFS')
    ref = ref_result(plan, rows, ct, 'HFS')
    assert whole.any() and whole.tobytes() == chunked.tobytes() == ref.tobytes()
    assert whole.dtype == chunked.dtype == ref.dtype and whole.shape == chunked.shape == ref.shape
