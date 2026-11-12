import ast
import copy
import inspect
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import SimpleITK as sitk

import synthetic_dicom as sd
from rt_dataprep import plan_table
from rt_dataprep.dicom_io import CaseRefused, read_header
from rt_dataprep.plan_table import REASON_PLAN_TABLE, SCHEMA

NULLABLE = {'beam_name', 'treatment_machine', 'fluence_mode', 'mlc_type', 'leaf_position_boundaries', 'wedge_id',
            'wedge_type', 'wedge_orientation', 'wedge_angle', 'dose_rate', 'gantry_rotation_direction', 'jaw_x1',
            'jaw_x2', 'jaw_y1', 'jaw_y2', 'mlc_positions', 'ssd', 'wedge_position'}
LISTS = {'leaf_position_boundaries', 'mlc_positions'}


def ct_image():
    image = sitk.Image([10, 12, 8], sitk.sitkInt16)
    image.SetSpacing((2.0, 2.0, 3.0))
    image.SetOrigin((-4.0, 6.0, -9.0))
    return image


def rows_of(tmp_path, beams=None, ct=None, writer=sd.write_photon_rtplan, **kwargs):
    path = tmp_path / 'plan.dcm'
    writer(path, sd.generate_uid(), **({'beams': beams} if beams is not None else {}), **kwargs)
    rows = plan_table.plan_rows(read_header(path), ct or ct_image())
    if rows is not None:  # AC9: every fixture table has no null in a required column
        required_complete(plan_table.to_table(rows))
    return rows


def refused(tmp_path, beams):
    with pytest.raises(CaseRefused) as raised:
        rows_of(tmp_path, beams)
    assert str(raised.value) == REASON_PLAN_TABLE


def required_complete(table):
    assert all(table[name].null_count == 0 for name in table.schema.names if name not in NULLABLE)


def test_schema_columns_types_nullability():
    assert SCHEMA.names == [
        'beam_number', 'beam_name', 'beam_type', 'treatment_machine', 'beam_meterset', 'final_cumulative_meterset_weight',
        'number_of_control_points', 'fluence_mode', 'mlc_type', 'leaf_position_boundaries', 'number_of_wedges', 'wedge_id',
        'wedge_type', 'wedge_orientation', 'wedge_angle', 'number_of_blocks', 'number_of_boli', 'control_point_index',
        'cumulative_meterset_weight', 'mu', 'mu_cumulative', 'nominal_energy', 'dose_rate', 'gantry_angle',
        'gantry_rotation_direction', 'collimator_angle', 'couch_angle', 'jaw_x1', 'jaw_x2', 'jaw_y1', 'jaw_y2',
        'mlc_positions', 'ssd', 'wedge_position', 'isocenter_x', 'isocenter_y', 'isocenter_z', 'isocenter_i',
        'isocenter_j', 'isocenter_k']
    ints = {'beam_number', 'number_of_control_points', 'number_of_wedges', 'number_of_blocks', 'number_of_boli',
            'control_point_index'}
    strings = {'beam_name', 'beam_type', 'treatment_machine', 'fluence_mode', 'mlc_type', 'wedge_id', 'wedge_type',
               'gantry_rotation_direction', 'wedge_position'}
    for field in SCHEMA:
        expected = (pa.list_(pa.float64()) if field.name in LISTS else pa.int64() if field.name in ints
                    else pa.string() if field.name in strings else pa.float64())
        assert field.type == expected, field.name
        assert field.nullable == (field.name in NULLABLE), field.name


def test_vmat_rows_fill_forward_from_cp0(tmp_path):
    rows = rows_of(tmp_path)
    assert [r['control_point_index'] for r in rows] == list(range(7))
    for key in ('jaw_x1', 'jaw_x2', 'jaw_y1', 'jaw_y2', 'nominal_energy', 'couch_angle', 'collimator_angle', 'dose_rate'):
        assert {r[key] for r in rows} == {rows[0][key]}
    assert (rows[0]['jaw_x1'], rows[0]['jaw_x2'], rows[0]['jaw_y1'], rows[0]['jaw_y2']) == (-40, 50, -30, 35)
    assert math.isclose(sum(r['mu'] for r in rows), 200.0, rel_tol=1e-6) and rows[0]['mu'] == 0
    assert math.isclose(rows[-1]['mu_cumulative'], 200.0, rel_tol=1e-6)
    assert all(len(r['mlc_positions']) == 8 for r in rows) and len(rows[0]['leaf_position_boundaries']) == 5
    assert {r['fluence_mode'] for r in rows} == {'FFF'}
    assert rows[3]['mlc_positions'][0] == -2.0 and rows[3]['gantry_angle'] == 1.0  # (181 + 180) % 360


def test_fill_forward_is_per_device_type(tmp_path):
    beam = sd.photon_beam()
    points = beam['ControlPointSequence']
    points[2]['BeamLimitingDevicePositionSequence'] = [
        dict(RTBeamLimitingDeviceType='ASYMX', LeafJawPositions=[-20.0, 25.0])]  # only a jaw moves
    rows = rows_of(tmp_path, [beam])
    assert (rows[2]['jaw_x1'], rows[2]['jaw_x2']) == (-20, 25) and rows[3]['jaw_x1'] == -20  # carried on
    assert rows[2]['jaw_y2'] == 35
    assert rows[2]['mlc_positions'] == rows[1]['mlc_positions']  # the MLC kept its state
    assert rows[1]['jaw_x1'] == -40 and rows[1]['mlc_positions'] != rows[0]['mlc_positions']


def test_wedge_position_tracked_by_referenced_wedge_number(tmp_path):
    beam = sd.photon_beam(kind='3d', wedge=True)
    beam['ControlPointSequence'][1]['WedgePositionSequence'] = [dict(ReferencedWedgeNumber=2, WedgePosition='OUT')]
    assert [r['wedge_position'] for r in rows_of(tmp_path, [beam])] == ['IN', 'IN']
    beam['ControlPointSequence'][1]['WedgePositionSequence'] = [dict(ReferencedWedgeNumber=1, WedgePosition='OUT')]
    assert [r['wedge_position'] for r in rows_of(tmp_path, [beam])] == ['IN', 'OUT']


def test_empty_values_count_as_missing(tmp_path):
    beam = sd.photon_beam()
    beam['ControlPointSequence'][3].update(DoseRateSet='', GantryAngle='')
    assert read_header(_written(tmp_path, [beam])).BeamSequence[0].ControlPointSequence[3].DoseRateSet in ('', None)
    rows = rows_of(tmp_path, [beam])
    assert rows[3]['dose_rate'] == 600 and rows[3]['gantry_angle'] == rows[2]['gantry_angle']
    first = sd.photon_beam()
    first['ControlPointSequence'][0]['NominalBeamEnergy'] = ''
    refused(tmp_path, [first])


def _written(tmp_path, beams):
    path = tmp_path / 'raw.dcm'
    sd.write_photon_rtplan(path, sd.generate_uid(), beams=beams)
    return path


def test_cumulative_weight_is_never_filled_forward(tmp_path):
    beam = sd.photon_beam()
    del beam['ControlPointSequence'][2]['CumulativeMetersetWeight']
    refused(tmp_path, [beam])


def test_step_and_shoot_equal_weights_build_with_zero_mu(tmp_path):
    rows = rows_of(tmp_path, [sd.photon_beam(kind='sas')])
    assert [r['cumulative_meterset_weight'] for r in rows] == [0, 0.5, 0.5, 1]
    assert rows[2]['mu'] == 0 and rows[1]['mu'] == rows[3]['mu'] == 100


def test_3d_plan_with_wedge_and_setup_beam(tmp_path):
    beams = [sd.photon_beam(1, kind='setup'), sd.photon_beam(2, kind='3d', wedge=True, mlc=False)]
    rows = rows_of(tmp_path, beams)
    assert {r['beam_number'] for r in rows} == {2} and len(rows) == 2
    for row in rows:
        assert (row['wedge_id'], row['wedge_type'], row['wedge_orientation'], row['wedge_angle'], row['wedge_position']) == (
            'SYNTHETIC_W1', 'STANDARD', 90.0, 30.0, 'IN') and row['number_of_wedges'] == 1
        assert row['mlc_type'] is row['leaf_position_boundaries'] is row['mlc_positions'] is None
        assert row['jaw_x1'] == -40


def test_no_declared_device_is_null_not_malformed(tmp_path):
    beam = sd.photon_beam(kind='3d', mlc=False)
    beam['BeamLimitingDeviceSequence'] = beam['BeamLimitingDeviceSequence'][1:]  # no X jaws, as on an Elekta Agility
    del beam['ControlPointSequence'][0]['BeamLimitingDevicePositionSequence'][0]
    rows = rows_of(tmp_path, [beam])
    assert rows[0]['jaw_x1'] is rows[0]['jaw_x2'] is rows[0]['mlc_type'] is None and rows[0]['jaw_y1'] == -30


def test_isocenter_as_voxel_index(tmp_path):
    rows = rows_of(tmp_path)
    assert [rows[0][k] for k in ('isocenter_x', 'isocenter_y', 'isocenter_z')] == [10, 20, 30]
    assert [rows[0][k] for k in ('isocenter_i', 'isocenter_j', 'isocenter_k')] == [7.0, 7.0, 13.0]  # (p - origin) / spacing
    assert {tuple(r[k] for k in ('isocenter_i', 'isocenter_j', 'isocenter_k')) for r in rows} == {(7.0, 7.0, 13.0)}


def test_plans_without_a_table_return_none_silently(tmp_path, caplog):
    electron = sd.photon_beam(2, kind='3d', tags=dict(RadiationType='ELECTRON'))
    plans = [
        dict(writer=sd.write_rtplan, positions_mm=[(0, 0, 0)]),
        dict(writer=sd.write_ion_rtplan),
        dict(beams=[sd.photon_beam(1, kind='3d'), electron]),
        dict(beams=[sd.photon_beam(1, kind='setup')]),
        *(dict(beams=[sd.photon_beam(1, kind='3d'), sd.photon_beam(2, kind='3d', tags=dict(TreatmentDeliveryType=kind))])
          for kind in ('CONTINUATION', 'TRMT_PORTFILM', 'OPEN_PORTFILM')),
    ]
    caplog.set_level(0)
    for spec in plans:
        writer = spec.pop('writer', sd.write_photon_rtplan)
        path = tmp_path / 'p.dcm'
        writer(path, sd.generate_uid(), **spec)
        assert plan_table.plan_rows(read_header(path), ct_image()) is None
    assert caplog.records == []


def vmat():
    return sd.photon_beam()


def _bad_gantry(beam):
    beam['ControlPointSequence'][2]['GantryAngle'] = 'nan'


def _beam_type_gone(beam):
    beam['BeamType'] = None


def _energy_gone(beam):
    beam['ControlPointSequence'][0]['NominalBeamEnergy'] = None


def _no_meterset(beam):
    beam['BeamMeterset'] = None


def _short_mlc(beam):
    beam['ControlPointSequence'][3]['BeamLimitingDevicePositionSequence'][0]['LeafJawPositions'] = [0.0] * 7


def _short_boundaries(beam):
    beam['BeamLimitingDeviceSequence'][2]['LeafPositionBoundaries'] = [0.0] * 4


def _decreasing(beam):
    beam['ControlPointSequence'][3]['CumulativeMetersetWeight'] = 0.1


def _final_zero(beam):
    beam['FinalCumulativeMetersetWeight'] = 0.0


def _first_weight(beam):
    beam['ControlPointSequence'][0]['CumulativeMetersetWeight'] = 0.1


def _last_weight(beam):
    beam['FinalCumulativeMetersetWeight'] = 2.0


def _empty_weight(beam):
    beam['ControlPointSequence'][2]['CumulativeMetersetWeight'] = ''


def _two_mlcs(beam):
    beam['BeamLimitingDeviceSequence'].append(dict(RTBeamLimitingDeviceType='MLCY', NumberOfLeafJawPairs=4))


def _jaw_missing_at_cp0(beam):
    del beam['ControlPointSequence'][0]['BeamLimitingDevicePositionSequence'][1]


def _mlc_missing_at_cp0(beam):
    del beam['ControlPointSequence'][0]['BeamLimitingDevicePositionSequence'][2]


def _truncated(beam):  # #272: NumberOfControlPoints says one more than the sequence holds
    beam['NumberOfControlPoints'] = len(beam['ControlPointSequence']) + 1


def _index_out_of_order(beam):  # #272
    beam['ControlPointSequence'][2]['ControlPointIndex'] = 5


@pytest.mark.filterwarnings('ignore:Invalid value for VR DS')  # the 'nan' gantry case
@pytest.mark.parametrize('mutate', [
    _bad_gantry, _beam_type_gone, _energy_gone, _no_meterset, _short_mlc, _short_boundaries, _decreasing, _final_zero,
    _first_weight, _last_weight, _empty_weight, _two_mlcs, _jaw_missing_at_cp0, _mlc_missing_at_cp0, _truncated, _index_out_of_order])
def test_malformed_photon_plan_is_refused(tmp_path, mutate):
    beam = vmat()
    mutate(beam)
    refused(tmp_path, [beam])


def test_schema_holds_without_wedge_and_parquet_round_trips(tmp_path):
    tables = [plan_table.to_table(rows_of(tmp_path, [sd.photon_beam(kind=kind, wedge=kind == '3d')]))
              for kind in ('vmat', 'sas', '3d')]
    for table in tables:
        assert table.schema.equals(SCHEMA)
        required_complete(table)
    table = tables[0]
    assert table['wedge_id'].null_count == table.num_rows and table.schema.field('wedge_angle').type == pa.float64()
    out = tmp_path / 'plan.parquet'
    plan_table.write_plan_table(table, out)
    assert pq.read_table(out).equals(table) and pq.read_table(out).schema.equals(SCHEMA)


def test_plan_table_catches_and_quiets_nothing():
    tree = ast.parse(inspect.getsource(plan_table))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {'quiet_call', '_pydicom_quiet'}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    raises = [ast.unparse(n.exc) for n in ast.walk(tree) if isinstance(n, ast.Raise)]
    assert raises == ['CaseRefused(REASON_PLAN_TABLE)']


def test_multiple_beams_concatenate(tmp_path):
    rows = rows_of(tmp_path, [sd.photon_beam(1, kind='3d'), copy.deepcopy(sd.photon_beam(2, kind='sas'))])
    assert [r['beam_number'] for r in rows] == [1, 1, 2, 2, 2, 2]


@pytest.mark.filterwarnings('ignore:Invalid value for VR DS')
def test_non_finite_gantry_is_what_is_refused(tmp_path):
    beam = vmat()
    _bad_gantry(beam)
    path = _written(tmp_path, [beam])
    assert math.isnan(float(read_header(path).BeamSequence[0].ControlPointSequence[2].GantryAngle))
