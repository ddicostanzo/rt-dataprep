import ast
import inspect
import time

import numpy as np
import pytest
import SimpleITK as sitk

import synthetic_dicom as sd
from rt_dataprep import beam_volume, fluence, plan_table
from rt_dataprep.dicom_io import CaseRefused
from rt_dataprep.fluence import REASON_FLUENCE
from test_beam_volume import (
    _blocks,
    _flat_boundaries,
    _mlc_not_mlcx,
    _no_y_limit,
    _null_boundaries,
    _reversed_boundaries,
    ct_image,
    set_mlc,
    static,
    vmat_arc,
    written,
)

NOT_MODELLED = [_blocks, _mlc_not_mlcx, _null_boundaries, _flat_boundaries, _reversed_boundaries, _no_y_limit]


def maps(tmp_path, beams):
    """(image, beam_numbers) of fluence_mu for the beams."""
    plan = written(tmp_path, beams if isinstance(beams, list) else [beams])
    return fluence.fluence_mu(plan, plan_table.plan_rows(plan, ct_image()))


def array(tmp_path, beams):
    return sitk.GetArrayFromImage(maps(tmp_path, beams)[0])


def lit_rows_cols(slice_):
    rows, cols = np.nonzero(slice_)
    return set(rows.tolist()), set(cols.tolist())


def mlc_beam(boundaries, x1, x2, jaws=(-100.0, 100.0, -100.0, 100.0), mu=100.0):
    beam = static(jaws=jaws, mu=mu, mlc=True, pairs=len(boundaries) - 1)
    set_mlc(beam, boundaries, {0: (x1, x2)})
    return beam


def test_ac1_static_jaw_field(tmp_path):
    image, numbers = maps(tmp_path, static(jaws=(-50.0, 50.0, -30.0, 30.0), mu=100.0))
    a = sitk.GetArrayFromImage(image)
    assert a.shape == (1, 400, 400) and a.dtype == np.float32 and numbers == [1]
    assert np.allclose(a[0, 171:229, 151:249], 100.0, atol=1e-4, rtol=0)  # Y -29..29, X -49..49
    assert not a[0, :170].any() and not a[0, 230:].any() and not a[0, :, :150].any() and not a[0, :, 250:].any()  # rows 170..229, columns 150..249 are the field
    assert image.GetOrigin() == (-199.5, -199.5, 0.0) and image.GetSpacing() == (1.0, 1.0, 1.0)
    assert image.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_ac2_one_sided_jaws(tmp_path):
    rows, cols = lit_rows_cols(array(tmp_path, static(jaws=(10.0, 60.0, -30.0, 30.0)))[0])
    assert cols == set(range(210, 260)) and rows == set(range(170, 230))
    rows, cols = lit_rows_cols(array(tmp_path, static(jaws=(-50.0, 50.0, 10.0, 40.0)))[0])
    assert rows == set(range(210, 240)) and cols == set(range(150, 250))


def test_ac2_one_sided_mlc(tmp_path):
    boundaries = [-20.0, -10.0, 0.0, 10.0, 20.0]
    closed = [0.0, 0.0, 0.0, -30.0]  # only the highest-Y pair is open
    rows, cols = lit_rows_cols(array(tmp_path, mlc_beam(boundaries, closed, [0.0, 0.0, 0.0, 30.0]))[0])
    assert rows == set(range(210, 220)) and cols == set(range(170, 230))
    rows, cols = lit_rows_cols(array(tmp_path, mlc_beam(boundaries, [10.0] * 4, [40.0] * 4))[0])  # X2 bank opens at +X
    assert cols == set(range(210, 240)) and rows == set(range(180, 220))


@pytest.mark.parametrize('angles', [(0, 0, 0), (90, 0, 0), (30, 270, 45)])
def test_ac3_collimator_gantry_couch_do_not_matter(tmp_path, angles):
    coll, gantry, couch = angles
    got = array(tmp_path, static(gantry=gantry, coll=coll, couch=couch, jaws=(10.0, 60.0, -30.0, 5.0)))
    expected = array(tmp_path, static(jaws=(10.0, 60.0, -30.0, 5.0)))
    assert got.any() and np.array_equal(got, expected)


def test_ac4_exact_area(tmp_path):
    m = 37.5
    a = array(tmp_path, static(jaws=(-10.3, 20.7, -5.25, 7.5), mu=m))[0]
    assert a.sum() == pytest.approx(m * 31.0 * 12.75, rel=1e-3)
    assert a[199, 189] == pytest.approx(m * 0.3, rel=1e-5)  # X -11..-10 covered from -10.3, Y fully
    assert a[194, 200] == pytest.approx(m * 0.25, rel=1e-5)  # Y -6..-5 covered from -5.25
    assert a[194, 189] == pytest.approx(m * 0.3 * 0.25, rel=1e-5)


def test_ac5_mlc_sum_and_degenerate_pairs(tmp_path):
    boundaries = [-30.0, -27.5, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 12.5, 15.0, 20.0, 30.0]
    x1 = [-10.4, -20.15, -3.3, -40.0, 5.0, -8.8, -0.45, 8.0, -12.25, -60.0, -2.0, 0.6]
    x2 = [14.7, 0.35, 9.95, 40.0, 5.0, 30.5, 0.35, 2.0, 6.05, 60.0, 2.6, 2.4]
    x1.append(-1.0), x2.append(1.0)
    assert len(x1) == len(boundaries) - 1  # pair 4 is closed (X1 = X2), pair 7 crossed (X1 > X2)
    jaws = (-25.0, 25.0, -22.0, 22.0)
    mu = 80.0
    expected = 0.0
    for n in range(len(x1)):
        if x1[n] < x2[n]:
            width = max(0.0, min(x2[n], jaws[1]) - max(x1[n], jaws[0]))
            height = max(0.0, min(boundaries[n + 1], jaws[3]) - max(boundaries[n], jaws[2]))
            expected += mu * width * height
    a = array(tmp_path, mlc_beam(boundaries, x1, x2, jaws=jaws, mu=mu))
    assert a.sum() == pytest.approx(expected, rel=1e-4) and a.min() >= 0
    crossed = array(tmp_path, static(jaws=(20.0, -20.0, -30.0, 30.0)))
    assert not crossed.any()


def test_ac6_step_and_shoot_segments(tmp_path):
    beam = sd.photon_beam(1, kind='sas', meterset=200.0, pairs=2)
    beam['SourceAxisDistance'] = 1000.0
    beam['ControlPointSequence'][0].update(GantryAngle=0.0, BeamLimitingDeviceAngle=0.0, PatientSupportAngle=0.0,
                                           IsocenterPosition=[0.0, 0.0, 0.0])
    items = beam['ControlPointSequence'][0]['BeamLimitingDevicePositionSequence']
    items[0]['LeafJawPositions'], items[1]['LeafJawPositions'] = [-60.0, 60.0], [-60.0, 60.0]
    for point, weight in zip(beam['ControlPointSequence'], (0.0, 0.3, 0.3, 1.0), strict=True):
        point['CumulativeMetersetWeight'] = weight
    set_mlc(beam, [-60.0, 0.0, 60.0], {0: ([-40.0] * 2, [10.0] * 2), 2: ([-10.0] * 2, [40.0] * 2)})
    a = array(tmp_path, beam)[0]
    assert a[230, 170] == pytest.approx(60.0) and a[230, 200] == pytest.approx(200.0) and a[230, 230] == pytest.approx(140.0)
    assert a[230, 150] == 0 and a[230, 250] == 0


def test_ac6_arc_sum_is_weight_times_area(tmp_path):
    beam = vmat_arc(1, 181.0, 2.0)
    plan = written(tmp_path, [beam])
    rows = plan_table.plan_rows(plan, ct_image())
    assert len(rows) == 178
    expected = 0.0
    for i in range(178):
        weight = 0.5 * rows[i]['mu'] + (0.5 * rows[i + 1]['mu'] if i + 1 < 178 else 0.0)
        width = 30.0 + 20.0 * np.sin(i / 10.0)
        expected += weight * 2 * width * 200.0
    image, _ = fluence.fluence_mu(plan, rows)
    assert sitk.GetArrayFromImage(image).sum() == pytest.approx(expected, rel=1e-4)
    assert expected > 0


def test_ac7_three_beams_in_plan_order(tmp_path):
    jaws = [(10.0, 60.0, -30.0, 30.0), (-60.0, -10.0, -30.0, 30.0), (-30.0, 30.0, 10.0, 60.0)]
    beams = [static(number=n, jaws=j, mu=10.0 * n) for n, j in zip((7, 2, 5), jaws, strict=True)]
    image, numbers = maps(tmp_path, beams)
    a = sitk.GetArrayFromImage(image)
    assert a.shape == (3, 400, 400) and numbers == [7, 2, 5] and all(type(n) is int for n in numbers)
    assert lit_rows_cols(a[0])[1] == set(range(210, 260)) and a[0].max() == pytest.approx(70.0)
    assert lit_rows_cols(a[1])[1] == set(range(140, 190)) and a[1].max() == pytest.approx(20.0)
    assert lit_rows_cols(a[2])[0] == set(range(210, 260)) and a[2].max() == pytest.approx(50.0)


def refused(tmp_path, beams):
    with pytest.raises(CaseRefused) as raised:
        maps(tmp_path, beams)
    assert str(raised.value) == REASON_FLUENCE


@pytest.mark.parametrize('mutate', NOT_MODELLED)
def test_ac8_ac9_not_modelled_apertures_are_refused_by_both(tmp_path, mutate):
    beam = static(mlc=mutate is not _no_y_limit)
    mutate(beam)
    refused(tmp_path, beam)
    plan = written(tmp_path, [beam])
    with pytest.raises(CaseRefused) as raised:
        beam_volume.beam_mu(plan, plan_table.plan_rows(plan, ct_image()), ct_image(), 'HFS')
    assert str(raised.value) == beam_volume.REASON_BEAM_VOLUME


def test_ac8_edge_at_200_is_inside_and_200_01_is_not(tmp_path):
    assert array(tmp_path, static(jaws=(-100.0, 200.0, -50.0, 50.0))).sum() > 0
    refused(tmp_path, static(jaws=(-100.0, 200.01, -50.0, 50.0)))
    refused(tmp_path, static(jaws=(-200.01, 100.0, -50.0, 50.0)))
    refused(tmp_path, static(jaws=(-100.0, 100.0, -50.0, 200.01)))


@pytest.mark.parametrize('jaws', [(-200.0, 100.0, -50.0, 50.0), (-100.0, 100.0, -50.0, 200.0), (-100.0, 100.0, -200.0, 50.0)],
                         ids=['-X', '+Y', '-Y'])
def test_ac8_edge_at_exactly_200_is_inside_at_minus_x_plus_y_and_minus_y(tmp_path, jaws):
    assert array(tmp_path, static(jaws=jaws)).sum() > 0


@pytest.mark.parametrize('order', ['setup_last', 'setup_first'])
def test_ac8_a_setup_beam_sharing_a_beam_number_with_a_treatment_beam_is_accepted(tmp_path, order):
    beams = [static(number=1), sd.photon_beam(1, kind='setup')]
    image, numbers = maps(tmp_path, beams if order == 'setup_last' else beams[::-1])
    assert numbers == [1] and sitk.GetArrayFromImage(image).max() == pytest.approx(100.0)


def test_ac8_duplicate_treatment_beam_numbers_are_refused_by_fluence_itself(tmp_path):  # #325: not only by plan_rows
    rows = plan_table.plan_rows(written(tmp_path, [static(number=1), static(number=2)]), ct_image())
    duplicated = written(tmp_path, [static(number=1), static(number=1)])
    with pytest.raises(CaseRefused) as raised:
        fluence.fluence_mu(duplicated, rows)
    assert str(raised.value) == REASON_FLUENCE


def test_accumulation_is_float64_so_small_weights_are_not_lost_beside_a_large_one(tmp_path):
    big = 2 ** 25  # float32 spacing is 4 from here: 2^25 + 1 and + 2 round back down
    beam = static(jaws=(-50.0, 50.0, -30.0, 30.0), mu=float(big + 4))
    beam['ControlPointSequence'][0]['CumulativeMetersetWeight'] = 0.0
    points = beam['ControlPointSequence'] = beam['ControlPointSequence'][:1]
    for i, weight in enumerate((big, big, big + 2, big + 2, big + 4), start=1):
        points.append(dict(ControlPointIndex=i, CumulativeMetersetWeight=float(weight)))
    beam.update(NumberOfControlPoints=6, FinalCumulativeMetersetWeight=float(big + 4))
    plan = written(tmp_path, [beam])
    rows = plan_table.plan_rows(plan, ct_image())
    assert [r['mu'] for r in rows] == [0.0, big, 0.0, 2.0, 0.0, 2.0]
    a = sitk.GetArrayFromImage(fluence.fluence_mu(plan, rows)[0])
    assert a[0, 200, 200] == big + 4  # overlap is exactly 1 mm2; trapezoid weights are 2^24, 2^24, 1, 1, 1, 1


@pytest.mark.parametrize('spoil', [dict(number_of_blocks=1), dict(mlc_type='MLCY')], ids=['blocks', 'mlcy'])
def test_ac8_a_not_modelled_aperture_on_a_weight_zero_control_point_is_still_refused(tmp_path, spoil):
    beam = static(jaws=(-50.0, 50.0, -50.0, 50.0), mu=40.0)
    beam['ControlPointSequence'][1]['CumulativeMetersetWeight'] = 0.0
    beam['ControlPointSequence'].append(dict(ControlPointIndex=2, CumulativeMetersetWeight=1.0))
    beam['NumberOfControlPoints'] = 3
    plan = written(tmp_path, [beam])
    rows = plan_table.plan_rows(plan, ct_image())
    assert [r['mu'] for r in rows] == [0.0, 0.0, 40.0] and sitk.GetArrayFromImage(fluence.fluence_mu(plan, rows)[0]).max() == pytest.approx(40.0)  # row 0 has weight 0: accepted as built
    rows[0].update(spoil)
    with pytest.raises(CaseRefused) as raised:
        fluence.fluence_mu(plan, rows)
    assert str(raised.value) == REASON_FLUENCE


def test_ac8_table_top_angle_and_missing_sad_are_accepted(tmp_path):
    beam = static()
    beam['ControlPointSequence'][0]['TableTopPitchAngle'] = 5.0
    del beam['SourceAxisDistance']
    assert array(tmp_path, beam).any()


def test_ac8_duplicate_beam_numbers_are_refused(tmp_path):  # by plan_rows since #273, before fluence sees the rows
    for beams in ([static(number=1), static(number=1)], [static(number=1), static(number=2), static(number=1)]):
        with pytest.raises(CaseRefused) as raised:
            maps(tmp_path, beams)
        assert str(raised.value) == plan_table.REASON_PLAN_TABLE


def off_grid_mlc(x1=-250.0, x2=250.0, jaws=(-100.0, 100.0, -100.0, 100.0)):
    return mlc_beam([-20.0, 0.0, 20.0], [x1, 0.0], [x2, 0.0], jaws=jaws)


def test_ac8_off_grid_is_checked_after_jaws_and_dropping(tmp_path):
    assert array(tmp_path, off_grid_mlc()).any()  # open pair behind X jaws
    assert array(tmp_path, off_grid_mlc(x1=-250.0, x2=-250.0, jaws=(-100.0, 100.0, -100.0, 100.0))).sum() == 0  # parked
    wide = mlc_beam([-250.0, 0.0, 250.0], [-10.0, 0.0], [10.0, 0.0], jaws=(-100.0, 100.0, -100.0, 100.0))
    assert array(tmp_path, wide).any()  # boundaries beyond 200 outside the Y jaws
    dropped = mlc_beam([-250.0, -150.0, 0.0], [-250.0, -10.0], [250.0, 10.0], jaws=(-100.0, 100.0, -100.0, 100.0))
    # no X jaw: the pair wholly outside the Y jaws is dropped before the off-grid check
    del dropped['BeamLimitingDeviceSequence'][0]
    del dropped['ControlPointSequence'][0]['BeamLimitingDevicePositionSequence'][0]
    assert array(tmp_path, dropped).any()
    no_x_jaw = off_grid_mlc()
    del no_x_jaw['BeamLimitingDeviceSequence'][0]
    del no_x_jaw['ControlPointSequence'][0]['BeamLimitingDevicePositionSequence'][0]
    refused(tmp_path, no_x_jaw)


def test_ac8_off_grid_only_counts_for_non_zero_weight(tmp_path):
    beam = static(jaws=(-250.0, 250.0, -50.0, 50.0), mu=40.0)
    points = beam['ControlPointSequence']
    points[1]['CumulativeMetersetWeight'] = 0.0
    points[1]['BeamLimitingDevicePositionSequence'] = [
        dict(RTBeamLimitingDeviceType='ASYMX', LeafJawPositions=[-50.0, 50.0]),
        dict(RTBeamLimitingDeviceType='ASYMY', LeafJawPositions=[-50.0, 50.0])]
    points.append(dict(ControlPointIndex=2, CumulativeMetersetWeight=1.0))
    beam['NumberOfControlPoints'] = 3
    a = array(tmp_path, beam)  # control point 0 has weight 0: accepted
    assert a.max() == pytest.approx(40.0)
    refused(tmp_path, static(jaws=(-250.0, 250.0, -50.0, 50.0)))


def test_ac10_fluence_module_rules():
    assert isinstance(REASON_FLUENCE, str) and REASON_FLUENCE
    tree = ast.parse(inspect.getsource(fluence))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {'quiet_call', '_pydicom_quiet', 'beam_volume'}
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.ExceptHandler, ast.Try))]
    assert [ast.unparse(n.exc) for n in ast.walk(tree) if isinstance(n, ast.Raise)] == ['CaseRefused(REASON_FLUENCE)']
    imported = [(n.module, [a.name for a in n.names]) for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and (n.module or '').startswith('rt_dataprep.beam_volume')]
    assert imported == [('rt_dataprep.beam_volume', ['aperture'])]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Import) and any('beam_volume' in a.name for a in n.names)]


@pytest.mark.slow
def test_ac14_full_size_vmat_plan_under_ten_seconds(tmp_path):
    plan = written(tmp_path, [vmat_arc(1, 181.0, 2.0), vmat_arc(2, 179.0, -2.0)])
    rows = plan_table.plan_rows(plan, ct_image())
    assert len(rows) == 2 * 178
    start = time.perf_counter()
    image, numbers = fluence.fluence_mu(plan, rows)
    wall = time.perf_counter() - start
    print(f'\nAC14 wall {wall:.2f} s')
    assert wall < 10.0 and numbers == [1, 2] and sitk.GetArrayFromImage(image).max() > 0
