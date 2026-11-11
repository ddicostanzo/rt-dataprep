import ast
import copy
import inspect
import resource
import time

import numpy as np
import pytest
import SimpleITK as sitk

import synthetic_dicom as sd
from rt_dataprep import beam_volume, plan_table
from rt_dataprep.beam_volume import REASON_BEAM_VOLUME
from rt_dataprep.dicom_io import CaseRefused, read_header_core

IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
ROT90 = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # image x -> patient +y, image y -> patient -x
SIZE, STEP = 61, 5.0  # +-150 mm about the isocenter; 5 mm voxels put every centre on a multiple of 5
SAD = 400.0
TOL = 1e-2


def ct_image(size=(SIZE, SIZE, SIZE), spacing=(STEP,) * 3, direction=IDENTITY, centre=(0.0, 0.0, 0.0)):
    """A CT whose centre voxel sits exactly at `centre` (patient mm), whatever the direction matrix."""
    image = sitk.Image(list(size), sitk.sitkInt16)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    half = np.array([s * (n - 1) / 2 for s, n in zip(spacing, size, strict=True)])
    image.SetOrigin(tuple(np.array(centre) - np.array(direction).reshape(3, 3) @ half))
    return image


def static(gantry=0.0, coll=0.0, couch=0.0, jaws=(-50.0, 50.0, -50.0, 50.0), mu=100.0, sad=SAD, number=1, mlc=False,
           pairs=4):
    """A 2-control-point 3D beam (all the MU on CP 1), isocenter (0, 0, 0), no MLC unless asked."""
    beam = sd.photon_beam(number, kind='3d', meterset=mu, mlc=mlc, pairs=pairs)
    cp0 = beam['ControlPointSequence'][0]
    cp0.update(GantryAngle=gantry, BeamLimitingDeviceAngle=coll, PatientSupportAngle=couch,
               IsocenterPosition=[0.0, 0.0, 0.0])
    cp0['BeamLimitingDevicePositionSequence'][0]['LeafJawPositions'] = list(jaws[:2])
    cp0['BeamLimitingDevicePositionSequence'][1]['LeafJawPositions'] = list(jaws[2:])
    beam['SourceAxisDistance'] = sad
    return beam


def set_mlc(beam, boundaries, positions):
    """Leaf boundaries, and {control point index: (x1 leaves, x2 leaves)} for the MLCX item of those control points."""
    pairs = len(boundaries) - 1
    device = next(d for d in beam['BeamLimitingDeviceSequence'] if d['RTBeamLimitingDeviceType'] == 'MLCX')
    device.update(NumberOfLeafJawPairs=pairs, LeafPositionBoundaries=list(boundaries))
    for index, (x1, x2) in positions.items():
        cp = beam['ControlPointSequence'][index]
        items = cp.setdefault('BeamLimitingDevicePositionSequence', [])
        item = next((i for i in items if i['RTBeamLimitingDeviceType'] == 'MLCX'), None)
        if item is None:
            item = dict(RTBeamLimitingDeviceType='MLCX')
            items.append(item)
        item['LeafJawPositions'] = list(x1) + list(x2)


def written(tmp_path, beams):
    path = tmp_path / 'plan.dcm'
    sd.write_photon_rtplan(path, sd.generate_uid(), beams=beams)
    return read_header_core(path)


def volume(tmp_path, beams, position='HFS', ct=None):
    ct = ct or ct_image()
    plan = written(tmp_path, beams if isinstance(beams, list) else [beams])
    return beam_volume.beam_mu(plan, plan_table.plan_rows(plan, ct), ct, position)


def lit(image):
    """(patient points mm, values) of the voxels holding more than 0, via the image's full geometry."""
    array = sitk.GetArrayFromImage(image)
    k, j, i = np.nonzero(array > 0)
    index = np.stack([i, j, k], axis=1) * np.array(image.GetSpacing())
    points = np.array(image.GetOrigin()) + index @ np.array(image.GetDirection()).reshape(3, 3).T
    return points, array[k, j, i]


def plane_count(image, axis, value):
    points, _ = lit(image)
    return int(np.sum(np.abs(points[:, axis] - value) < 0.1))


AXIS = dict(x=0, y=1, z=2)


def test_ac1_flat_field_at_the_isocenter_plane(tmp_path):
    image = volume(tmp_path, static(mu=80.0))
    array = sitk.GetArrayFromImage(image)  # (z, y, x); the isocenter plane of a G0 beam is y = 0
    plane = array[:, SIZE // 2, :]
    z, x = np.meshgrid(np.arange(SIZE) * STEP - 150, np.arange(SIZE) * STEP - 150, indexing='ij')
    assert np.allclose(plane[(np.abs(x) < 45) & (np.abs(z) < 45)], 80.0)
    assert not plane[np.abs(x) > 55].any()


@pytest.mark.parametrize('depth', [100, 150])
def test_ac2_field_diverges_with_depth(tmp_path, depth):
    image = volume(tmp_path, static())
    array = sitk.GetArrayFromImage(image)
    row = array[SIZE // 2, SIZE // 2 + depth // 5, :]  # z = 0, y = +depth: deeper for a source anterior (b_z = -y)
    half = max(abs(x * STEP - 150) for x in np.nonzero(row)[0])
    expected = 50 * (SAD + depth) / SAD
    assert abs(half - expected) <= STEP and expected - 50 > STEP


def _field_checks(image, source, jaw_x, jaw_y):
    """The named sides: footprint narrower at the source side, and the one-sided jaws on their named patient sides."""
    (s_axis, s_sign), (x_axis, x_sign), (y_axis, y_sign) = source, jaw_x, jaw_y
    near = plane_count(image, AXIS[s_axis], s_sign * 100.0)
    far = plane_count(image, AXIS[s_axis], -s_sign * 100.0)
    assert 0 < near < far  # the beam diverges away from its source
    points, _ = lit(image)
    for axis, sign in ((x_axis, x_sign), (y_axis, y_sign)):
        coord = points[:, AXIS[axis]] * sign
        assert coord.min() >= -TOL and coord.max() > 20


G90_CASES = [  # position, gantry, (source axis, sign), (axis the X jaw opens toward), (axis the Y jaw opens toward)
    ('HFS', 90, ('x', 1), ('y', 1), ('z', 1)),  # source at patient left
    ('HFS', 180, ('y', 1), ('x', -1), ('z', 1)),  # posterior
    ('HFS', 270, ('x', -1), ('y', -1), ('z', 1)),  # patient right
    ('HFP', 90, ('x', -1), ('y', -1), ('z', 1)),
    ('FFS', 90, ('x', -1), ('y', 1), ('z', -1)),
    ('FFP', 90, ('x', 1), ('y', -1), ('z', -1)),
]


@pytest.mark.parametrize(('position', 'gantry', 'source', 'jaw_x', 'jaw_y'), G90_CASES)
@pytest.mark.parametrize('direction', [IDENTITY, ROT90], ids=['identity', 'rot90'])
def test_ac3_ac4_gantry_and_position_sides(tmp_path, position, gantry, source, jaw_x, jaw_y, direction):
    image = volume(tmp_path, static(gantry=gantry, jaws=(0.0, 50.0, 0.0, 40.0)), position, ct_image(direction=direction))
    _field_checks(image, source, jaw_x, jaw_y)


def test_ac5a_couch_90_puts_the_source_inferior(tmp_path):
    image = volume(tmp_path, static(gantry=90, couch=90))
    assert 0 < plane_count(image, 2, -100.0) < plane_count(image, 2, 100.0)  # deeper (wider) toward superior


def test_ac5b_collimator_90_turns_jaw_x_toward_superior(tmp_path):
    image = volume(tmp_path, static(coll=90, jaws=(0.0, 50.0, -50.0, 50.0)))
    points, _ = lit(image)
    plane = points[np.abs(points[:, 1]) < 0.1]  # the isocenter plane of a G0 beam
    assert len(plane) and plane[:, 2].min() >= -TOL and plane[:, 2].max() > 20
    assert points[:, 2].min() >= -TOL  # and nothing inferior anywhere


def test_ac6_one_open_leaf_pair(tmp_path):
    boundaries = list(range(-40, 50, 10))
    closed = [0.0] * 8
    x1, x2 = list(closed), list(closed)
    x1[5], x2[5] = -20.0, 30.0  # the pair spanning Y 10..20
    beam = static(jaws=(-60.0, 60.0, -60.0, 60.0), mlc=True, pairs=8)
    set_mlc(beam, boundaries, {0: (x1, x2)})
    points, _ = lit(volume(tmp_path, beam))
    plane = points[np.abs(points[:, 1]) < 0.1]  # at G0, C0 the beam's Y is the patient's z and X is x
    assert plane[:, 2].min() >= 10 - STEP and plane[:, 2].max() <= 20 + STEP
    assert plane[:, 0].min() >= -20 - STEP and plane[:, 0].max() <= 30 + STEP
    assert {10.0, 15.0} <= set(plane[:, 2]) and {-20.0, 0.0, 30.0} <= set(plane[:, 0])


def test_ac6_closed_control_points_contribute_nothing(tmp_path):
    boundaries = list(range(-40, 50, 10))
    closed = ([0.0] * 8, [0.0] * 8)
    beam = static(jaws=(-60.0, 60.0, -60.0, 60.0), mlc=True, pairs=8, mu=100.0)
    set_mlc(beam, boundaries, {0: closed, 1: closed})
    assert not sitk.GetArrayFromImage(volume(tmp_path, beam)).any()
    beam = static(jaws=(-60.0, 60.0, -60.0, 60.0), mlc=True, pairs=8, mu=100.0)  # only CP 0 closed: half the MU
    set_mlc(beam, boundaries, {0: closed, 1: ([-20.0] * 8, [20.0] * 8)})
    _, values = lit(volume(tmp_path, beam))
    assert len(values) and np.allclose(values, 50.0)


def test_ac7_static_beam_holds_its_mu(tmp_path):
    _, values = lit(volume(tmp_path, static(mu=37.5)))
    assert np.allclose(values, 37.5)


def test_ac7_step_and_shoot_segments_add_where_they_overlap(tmp_path):
    beam = sd.photon_beam(1, kind='sas', meterset=200.0, pairs=2)
    beam['SourceAxisDistance'] = SAD
    beam['ControlPointSequence'][0].update(GantryAngle=0.0, BeamLimitingDeviceAngle=0.0, PatientSupportAngle=0.0,
                                           IsocenterPosition=[0.0, 0.0, 0.0])
    items = beam['ControlPointSequence'][0]['BeamLimitingDevicePositionSequence']
    items[0]['LeafJawPositions'], items[1]['LeafJawPositions'] = [-60.0, 60.0], [-60.0, 60.0]
    set_mlc(beam, [-60.0, 0.0, 60.0], {0: ([-40.0] * 2, [10.0] * 2), 2: ([-10.0] * 2, [40.0] * 2)})
    points, values = lit(volume(tmp_path, beam))
    plane = np.abs(points[:, 1]) < 0.1
    for x, expected in ((-30.0, 100.0), (0.0, 200.0), (30.0, 100.0)):
        at = plane & (points[:, 0] == x)
        assert at.any() and np.allclose(values[at], expected)
    assert not (plane & (np.abs(points[:, 0]) > 45)).any()


def test_ac7_arc_sum_is_linear_in_mu(tmp_path):
    def arc(mu):
        beam = static(mu=mu)
        beam['ControlPointSequence'][1].update(CumulativeMetersetWeight=0.5, GantryAngle=20.0)
        beam['ControlPointSequence'].append(dict(ControlPointIndex=2, CumulativeMetersetWeight=1.0, GantryAngle=40.0))
        beam['NumberOfControlPoints'] = 3
        return float(sitk.GetArrayFromImage(volume(tmp_path, beam)).sum(dtype=np.float64))

    assert arc(250.0) == pytest.approx(2.5 * arc(100.0), rel=1e-5)


def test_each_row_uses_the_sad_of_its_beam(tmp_path):
    first, second = static(number=1, gantry=0, sad=400.0), static(number=2, gantry=90, sad=800.0)
    both = sitk.GetArrayFromImage(volume(tmp_path, [first, second]))
    alone = sum(sitk.GetArrayFromImage(volume(tmp_path, [b])) for b in (static(number=1, gantry=0, sad=400.0),
                                                                        static(number=2, gantry=90, sad=800.0)))
    assert np.allclose(both, alone)


def test_a_setup_beam_needs_no_sad_and_adds_nothing(tmp_path):
    setup = sd.photon_beam(2, kind='setup')
    del setup['SourceAxisDistance']
    with_setup = sitk.GetArrayFromImage(volume(tmp_path, [static(), setup]))
    assert np.array_equal(with_setup, sitk.GetArrayFromImage(volume(tmp_path, [static()])))


def test_output_is_float32_on_the_ct_geometry(tmp_path):
    ct = ct_image(size=(21, 21, 17), spacing=(5.0, 4.0, 3.0), direction=ROT90, centre=(7.0, -3.0, 11.0))
    beam = static()
    beam['ControlPointSequence'][0]['IsocenterPosition'] = [7.0, -3.0, 11.0]
    image = volume(tmp_path, beam, ct=ct)
    assert image.GetPixelID() == sitk.sitkFloat32 and lit(image)[1].size
    for name in ('Size', 'Spacing', 'Origin', 'Direction'):
        assert getattr(image, f'Get{name}')() == getattr(ct, f'Get{name}')()


def _blocks(beam):
    beam['NumberOfBlocks'] = 1


def _mlc_not_mlcx(beam):
    beam['BeamLimitingDeviceSequence'][2]['RTBeamLimitingDeviceType'] = 'MLCY'
    beam['ControlPointSequence'][0]['BeamLimitingDevicePositionSequence'][2]['RTBeamLimitingDeviceType'] = 'MLCY'


def _null_boundaries(beam):
    del beam['BeamLimitingDeviceSequence'][2]['LeafPositionBoundaries']


def _sad_missing(beam):
    del beam['SourceAxisDistance']


def _sad_zero(beam):
    beam['SourceAxisDistance'] = 0.0


def _sad_negative(beam):
    beam['SourceAxisDistance'] = -1000.0


def _no_y_limit(beam):
    del beam['BeamLimitingDeviceSequence'][1]
    del beam['ControlPointSequence'][0]['BeamLimitingDevicePositionSequence'][1]


@pytest.mark.parametrize('mutate', [_blocks, _mlc_not_mlcx, _null_boundaries, _sad_missing, _sad_zero, _sad_negative,
                                    _no_y_limit])
def test_ac8_not_modelled_beams_are_refused(tmp_path, mutate):
    beam = static(mlc=mutate is not _no_y_limit)  # with an MLC the Y axis would be bounded after all
    mutate(beam)
    with pytest.raises(CaseRefused) as raised:
        volume(tmp_path, beam)
    assert str(raised.value) == REASON_BEAM_VOLUME


@pytest.mark.parametrize('position', ['HFDR', 'hfs', '', None])
def test_ac8_unsupported_patient_position_is_refused(tmp_path, position):
    with pytest.raises(CaseRefused) as raised:
        volume(tmp_path, static(), position)
    assert str(raised.value) == REASON_BEAM_VOLUME


@pytest.mark.parametrize('sad', [float('nan'), float('inf'), '400', None, [400.0]])
def test_ac8_sad_must_be_a_finite_positive_number(tmp_path, sad):
    plan = written(tmp_path, [static()])
    ct = ct_image()
    rows = plan_table.plan_rows(plan, ct)
    fake = {'BeamSequence': [{'BeamNumber': 1, 'TreatmentDeliveryType': 'TREATMENT', 'SourceAxisDistance': sad}]}
    with pytest.raises(CaseRefused) as raised:
        beam_volume.beam_mu(fake, rows, ct, 'HFS')
    assert str(raised.value) == REASON_BEAM_VOLUME


def test_ac8_reason_is_a_fixed_string():
    assert isinstance(REASON_BEAM_VOLUME, str) and REASON_BEAM_VOLUME


def test_beam_volume_catches_and_quiets_nothing():
    tree = ast.parse(inspect.getsource(beam_volume))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {'quiet_call', '_pydicom_quiet'}
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.ExceptHandler, ast.Try))]
    raises = [ast.unparse(n.exc) for n in ast.walk(tree) if isinstance(n, ast.Raise)]
    assert raises == ['CaseRefused(REASON_BEAM_VOLUME)']


@pytest.mark.slow
def test_ac14_full_size_vmat_case(tmp_path):
    """512 x 512 x 150 CT, two 178-point arcs; run with -m slow -s to see the wall time and peak RSS."""
    def arc(number, start, step):
        beam = sd.photon_beam(number, kind='vmat', pairs=40)
        beam['SourceAxisDistance'] = 1000.0
        boundaries = [-100.0 + 5.0 * k for k in range(41)]

        def leaves(i):
            width = 30.0 + 20.0 * np.sin(i / 10.0)
            return [-width] * 40, [width] * 40

        points = copy.deepcopy(beam['ControlPointSequence'][:1])
        points[0].update(GantryAngle=start, BeamLimitingDeviceAngle=0.0, PatientSupportAngle=0.0,
                         IsocenterPosition=[0.0, 0.0, 0.0])
        points[0]['BeamLimitingDevicePositionSequence'][0]['LeafJawPositions'] = [-100.0, 100.0]
        points[0]['BeamLimitingDevicePositionSequence'][1]['LeafJawPositions'] = [-100.0, 100.0]
        for i in range(1, 178):
            points.append(dict(ControlPointIndex=i, CumulativeMetersetWeight=i / 177, GantryAngle=(start + step * i) % 360.0,
                               BeamLimitingDevicePositionSequence=[dict(RTBeamLimitingDeviceType='MLCX')]))
        beam.update(ControlPointSequence=points, NumberOfControlPoints=178, FinalCumulativeMetersetWeight=1.0)
        set_mlc(beam, boundaries, {i: leaves(i) for i in range(178)})
        return beam

    ct = ct_image(size=(512, 512, 150), spacing=(1.0, 1.0, 2.0))
    plan = written(tmp_path, [arc(1, 181.0, 2.0), arc(2, 179.0, -2.0)])
    rows = plan_table.plan_rows(plan, ct)
    assert len(rows) == 2 * 178
    start = time.perf_counter()
    image = beam_volume.beam_mu(plan, rows, ct, 'HFS')
    wall = time.perf_counter() - start
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'\nAC14 wall {wall:.1f} s ({wall / 60:.1f} min), peak RSS {peak_mb:.0f} MB')
    assert image.GetSize() == ct.GetSize() and sitk.GetArrayFromImage(image).max() > 0
