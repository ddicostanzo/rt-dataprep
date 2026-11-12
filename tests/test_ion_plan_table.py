import ast
import copy
import inspect
import math
import time

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import SimpleITK as sitk
from pydicom.datadict import tag_for_keyword
from pydicom.multival import MultiValue

import synthetic_dicom as sd
from rt_dataprep import ion_plan_table
from rt_dataprep.dicom_io import CaseRefused, read_header
from rt_dataprep.ion_plan_table import ION_SCHEMA, REASON_ION_PLAN

NAMES = [
    'beam_number', 'beam_name', 'beam_type', 'radiation_type', 'treatment_machine', 'scan_mode', 'beam_meterset',
    'final_cumulative_meterset_weight', 'number_of_control_points', 'virtual_sad_x', 'virtual_sad_y', 'snout_id',
    'range_shifter_ids', 'control_point_index', 'layer_index', 'cumulative_meterset_weight', 'nominal_energy',
    'gantry_angle', 'couch_angle', 'snout_position', 'scan_spot_tune_id', 'spot_size_x', 'spot_size_y',
    'number_of_paintings', 'range_shifter_settings', 'range_shifter_wet', 'isocenter_x', 'isocenter_y', 'isocenter_z',
    'isocenter_i', 'isocenter_j', 'isocenter_k', 'spot_index', 'spot_x', 'spot_y', 'spot_weight', 'spot_mu']
NULLABLE = {'beam_name', 'treatment_machine', 'virtual_sad_x', 'virtual_sad_y', 'snout_id', 'snout_position',
            'scan_spot_tune_id', 'spot_size_x', 'spot_size_y', 'number_of_paintings', 'range_shifter_wet'}
INTS = {'beam_number', 'number_of_control_points', 'control_point_index', 'layer_index', 'number_of_paintings', 'spot_index'}
STRINGS = {'beam_name', 'beam_type', 'radiation_type', 'treatment_machine', 'scan_mode', 'snout_id', 'scan_spot_tune_id'}
STRING_LISTS = {'range_shifter_ids', 'range_shifter_settings'}


def ct_image():
    image = sitk.Image([10, 12, 8], sitk.sitkInt16)
    image.SetSpacing((2.0, 2.0, 3.0))
    image.SetOrigin((-4.0, 6.0, -9.0))
    return image


def header_of(tmp_path, beams=None, **kwargs):
    path = tmp_path / 'plan.dcm'
    sd.write_pbs_rtplan(path, sd.generate_uid(), beams=beams, **kwargs)
    return read_header(path)


def rows_of(tmp_path, beams=None, ct=None, edit=None, **kwargs):
    """The rows of a written-and-read-back PBS plan; `edit(plan)` changes the read-back header first."""
    plan = header_of(tmp_path, beams, **kwargs)
    if edit:
        edit(plan)
    rows = ion_plan_table.ion_plan_rows(plan, ct or ct_image())
    if rows is not None:
        required_complete(ion_plan_table.to_table(rows))
    return rows


def refused(tmp_path, beams=None, edit=None, **kwargs):
    with pytest.raises(CaseRefused) as raised:
        rows_of(tmp_path, beams, edit=edit, **kwargs)
    assert str(raised.value) == REASON_ION_PLAN


def required_complete(table):
    assert all(table[name].null_count == 0 for name in table.schema.names if name not in NULLABLE)


def beam_rows(rows, number):
    return [r for r in rows if r['beam_number'] == number]


def points_of(beam):
    return beam['IonControlPointSequence']


def test_schema_columns_types_nullability():
    assert ION_SCHEMA.names == NAMES
    for field in ION_SCHEMA:
        expected = (pa.list_(pa.string()) if field.name in STRING_LISTS else pa.list_(pa.float64()) if field.name == 'range_shifter_wet'
                    else pa.int64() if field.name in INTS else pa.string() if field.name in STRINGS else pa.float64())
        assert field.type == expected, field.name
        assert field.nullable == (field.name in NULLABLE), field.name
    assert not [n for n in ION_SCHEMA.names if 'spreading' in n or 'modulator' in n]


def test_table_without_optional_tags_has_the_declared_schema(tmp_path):
    beam = sd.pbs_beam(tags=dict(VirtualSourceAxisDistances=None, SnoutSequence=None, TreatmentMachineName=None, BeamName=None))
    for point in points_of(beam):
        point.update(SnoutPosition=None, ScanSpotTuneID=None, ScanningSpotSize=None)
    table = ion_plan_table.to_table(rows_of(tmp_path, [beam]))
    assert table.schema.equals(ION_SCHEMA)
    assert table['range_shifter_ids'].to_pylist() == [[]] * table.num_rows
    assert table['range_shifter_settings'].to_pylist() == [[]] * table.num_rows
    assert table['range_shifter_wet'].to_pylist() == [[]] * table.num_rows
    for name in ('beam_name', 'treatment_machine', 'virtual_sad_x', 'snout_id', 'snout_position', 'scan_spot_tune_id', 'spot_size_y',
                 'number_of_paintings'):
        assert table[name].null_count == table.num_rows, name
    assert table.schema.field('spot_size_x').type == pa.float64()


def test_spot_rows_and_mu(tmp_path):
    rows = rows_of(tmp_path)
    assert len(rows) == 16
    for number, meterset in ((1, 100.0), (2, 300.0)):
        own = beam_rows(rows, number)
        assert len(own) == 8 and {r['beam_meterset'] for r in own} == {meterset}
        assert math.isclose(sum(r['spot_mu'] for r in own), meterset, rel_tol=1e-9)
        assert [r['layer_index'] for r in own] == [0, 0, 1, 1, 1, 2, 2, 2]
        assert [r['control_point_index'] for r in own] == [0, 0, 2, 2, 2, 4, 4, 4]
        assert [r['spot_index'] for r in own] == [0, 2, 0, 1, 2, 0, 1, 2]  # the zero spot of layer 0 has no row, its index a gap
    first = beam_rows(rows, 1)
    assert first[2]['spot_weight'] == 0.125 and first[2]['spot_mu'] == 100.0 * 0.125 / 1.0  # layer 1, spot 0
    assert (first[1]['spot_x'], first[1]['spot_y']) == (-10.0, -2.0)  # the stored pair of spot 2, layer 0
    assert (first[3]['spot_x'], first[3]['spot_y']) == (-14.0, 1.0)  # layer 1, spot 1: (-20 + 5 + 1, 2 - 1)
    assert [r['cumulative_meterset_weight'] for r in first] == [0, 0, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5]
    assert first[0]['scan_mode'] == 'MODULATED' and first[0]['radiation_type'] == 'PROTON' and first[0]['number_of_control_points'] == 6


def test_float32_weights_and_decimal_string_cumulative_weights_convert(tmp_path):
    beam = sd.pbs_beam(spot_weights=[[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.05, 0.05]])
    for point in points_of(beam):
        point['CumulativeMetersetWeight'] = str(point['CumulativeMetersetWeight'])
    beam['FinalCumulativeMetersetWeight'] = str(beam['FinalCumulativeMetersetWeight'])
    rows = rows_of(tmp_path, [beam])
    assert len(rows) == 9 and math.isclose(sum(r['spot_mu'] for r in rows), 100.0, rel_tol=1e-4)
    assert not math.isclose(rows[0]['spot_weight'], 0.1, rel_tol=1e-12)  # stored as float32


def test_spot_mu_divides_by_a_final_weight_other_than_one(tmp_path):
    weights = [[37 * w for w in layer] for layer in ([0.125, 0.0, 0.125], [0.125, 0.0625, 0.0625], [0.25, 0.125, 0.125])]
    rows = rows_of(tmp_path, [sd.pbs_beam(spot_weights=weights)])
    assert rows[0]['final_cumulative_meterset_weight'] == 37.0
    assert math.isclose(rows[2]['spot_mu'], 100.0 * rows[2]['spot_weight'] / 37.0, rel_tol=1e-12)
    assert math.isclose(sum(r['spot_mu'] for r in rows), 100.0, rel_tol=1e-9)


def test_meterset_is_matched_by_beam_number(tmp_path):
    def reorder(plan):
        plan.FractionGroupSequence[0].ReferencedBeamSequence.reverse()
    rows = rows_of(tmp_path, edit=reorder)
    assert math.isclose(sum(r['spot_mu'] for r in beam_rows(rows, 1)), 100.0, rel_tol=1e-9)
    assert math.isclose(sum(r['spot_mu'] for r in beam_rows(rows, 2)), 300.0, rel_tol=1e-9)


def _layer_sum_off(beam):
    points_of(beam)[2]['ScanSpotMetersetWeights'][0] += 0.01


def _last_cp_weights(beam):
    points_of(beam)[-1]['ScanSpotMetersetWeights'][0] = 0.25


def _layer_shifted(beam):  # CP 0 is 0.05 over its step, CP 2 0.05 under: the beam total is unchanged
    points_of(beam)[0]['ScanSpotMetersetWeights'][0] += 0.05
    points_of(beam)[2]['ScanSpotMetersetWeights'][0] -= 0.05


def _last_cp_moved(beam):  # spot weight moved from CP 4 onto the last CP: the beam total is unchanged
    points_of(beam)[4]['ScanSpotMetersetWeights'][0] = 0.0
    points_of(beam)[-1]['ScanSpotMetersetWeights'][0] = 0.25


def _total_drifts(beam):  # each CP is 0.6e-4 off its step (inside the tolerance) but the three together are 1.8e-4 off
    for k in (0, 2, 4):
        points_of(beam)[k]['ScanSpotMetersetWeights'][0] += 0.6e-4


def _negative(beam):
    points_of(beam)[0]['ScanSpotMetersetWeights'] = [0.375, -0.125, 0.0]  # the CP still sums to its step


def _odd_map(beam):
    points_of(beam)[0]['ScanSpotPositionMap'] = points_of(beam)[0]['ScanSpotPositionMap'] + [1.0]


def _spot_count(beam):
    points_of(beam)[0]['NumberOfScanSpotPositions'] = 4


def _no_meterset(beam):
    beam['BeamMeterset'] = None


def _energy_nan(beam):
    points_of(beam)[2]['NominalBeamEnergy'] = 'nan'


def _energy_gone(beam):
    points_of(beam)[0]['NominalBeamEnergy'] = ''


def _gantry_gone(beam):
    points_of(beam)[0]['GantryAngle'] = None


def _isocenter_gone(beam):
    points_of(beam)[0]['IsocenterPosition'] = None


def _first_weight(beam):
    points_of(beam)[0]['CumulativeMetersetWeight'] = 0.1


def _last_weight(beam):
    points_of(beam)[-1]['CumulativeMetersetWeight'] = 0.9


def _decreasing(beam):  # a step back of 0.5e-4 leaves every CP sum and the total inside the tolerance
    points_of(beam)[2]['CumulativeMetersetWeight'] = 0.25 - 0.5e-4


def _final_zero(beam):
    beam['FinalCumulativeMetersetWeight'] = 0.0


def _no_unit(beam):
    beam['PrimaryDosimeterUnit'] = None


def _shifter_without_setting(beam):
    del points_of(beam)[0]['RangeShifterSettingsSequence']


def _weights_absent(beam):  # CP 2 would sum to 0 against a step of 0.25
    points_of(beam)[2]['ScanSpotMetersetWeights'] = None
    points_of(beam)[2]['ScanSpotPositionMap'] = None
    points_of(beam)[2]['NumberOfScanSpotPositions'] = None


def _bad_spot_size(beam):
    points_of(beam)[0]['ScanningSpotSize'] = [7.0, 8.0, 9.0]


def _negative_meterset(beam):  # #297
    beam['BeamMeterset'] = -100.0


def _duplicate_shifter(beam):  # #297
    beam['RangeShifterSequence'].append(dict(beam['RangeShifterSequence'][0]))


def _truncated(beam):  # #325: NumberOfControlPoints says one more than the sequence holds
    beam['NumberOfControlPoints'] = len(points_of(beam)) + 1


def _index_out_of_order(beam):  # #325
    points_of(beam)[1]['ControlPointIndex'] = 5


def _undeclared_shifter_setting(beam):  # #297
    points_of(beam)[0]['RangeShifterSettingsSequence'].append(dict(points_of(beam)[0]['RangeShifterSettingsSequence'][0], ReferencedRangeShifterNumber=9))


@pytest.mark.filterwarnings('ignore:Invalid value for VR DS')  # the 'nan' energy case
@pytest.mark.parametrize('mutate', [
    _layer_sum_off, _last_cp_weights, _layer_shifted, _last_cp_moved, _total_drifts, _negative, _odd_map, _spot_count, _no_meterset, _energy_nan, _energy_gone,
    _gantry_gone, _isocenter_gone, _first_weight, _last_weight, _decreasing, _final_zero, _no_unit, _shifter_without_setting,
    _weights_absent, _bad_spot_size, _negative_meterset, _duplicate_shifter, _undeclared_shifter_setting, _truncated, _index_out_of_order])
def test_malformed_pbs_plan_is_refused(tmp_path, mutate):
    beam = sd.pbs_beam(range_shifters=1)
    mutate(beam)
    refused(tmp_path, [beam, sd.pbs_beam(2, 300.0)])


def test_duplicate_beam_number_is_refused(tmp_path):
    def renumber(plan):
        plan.IonBeamSequence[1].BeamNumber = 1
    refused(tmp_path, edit=renumber)


def test_duplicate_referenced_beam_number_is_refused(tmp_path):
    def repeat(plan):
        references = plan.FractionGroupSequence[0].ReferencedBeamSequence
        references.append(copy.deepcopy(references[0]))
    refused(tmp_path, edit=repeat)


def test_vendor_variation_single_spot_layers_and_paintings(tmp_path):
    layers = [[0.25], [0.25], [0.5]]
    eclipse = rows_of(tmp_path, [sd.pbs_beam(spot_weights=layers)])
    raystation = rows_of(tmp_path, [sd.pbs_beam(spot_weights=layers, paintings=2)])
    plan = header_of(tmp_path, [sd.pbs_beam(spot_weights=layers, paintings=2)])
    weights = plan.IonBeamSequence[0].IonControlPointSequence[0].ScanSpotMetersetWeights
    assert isinstance(weights, float) and not isinstance(weights, MultiValue)
    assert len(raystation) == 3 and {r['number_of_paintings'] for r in raystation} == {2}
    assert {r['number_of_paintings'] for r in eclipse} == {None}
    assert [r['spot_mu'] for r in raystation] == [25.0, 25.0, 50.0]
    drop = {'number_of_paintings'}
    assert [{k: v for k, v in r.items() if k not in drop} for r in raystation] == [
        {k: v for k, v in r.items() if k not in drop} for r in eclipse]
    five = rows_of(tmp_path, [sd.pbs_beam(spot_weights=layers, paintings=5)])
    assert [r['spot_mu'] for r in five] == [r['spot_mu'] for r in raystation]


def test_range_shifter_set_at_cp0_is_filled_forward(tmp_path):
    rows = rows_of(tmp_path, [sd.pbs_beam(range_shifters=1)])
    assert {tuple(r['range_shifter_ids']) for r in rows} == {('SYNTHETIC_RS1',)}
    assert {tuple(r['range_shifter_settings']) for r in rows} == {('IN',)}
    assert {tuple(r['range_shifter_wet']) for r in rows} == {(10.0,)}


def test_two_range_shifters_keep_sequence_order_and_join_on_number(tmp_path):
    beam = sd.pbs_beam(range_shifters=2)
    beam['RangeShifterSequence'] = [dict(RangeShifterNumber=7, RangeShifterID='SYNTHETIC_A', RangeShifterType='BINARY'),
                                    dict(RangeShifterNumber=3, RangeShifterID='SYNTHETIC_B', RangeShifterType='BINARY')]
    points_of(beam)[0]['RangeShifterSettingsSequence'] = [  # listed in the other order
        dict(ReferencedRangeShifterNumber=3, RangeShifterSetting='OUT', RangeShifterWaterEquivalentThickness=30.0),
        dict(ReferencedRangeShifterNumber=7, RangeShifterSetting='IN', RangeShifterWaterEquivalentThickness=70.0)]
    points_of(beam)[4]['RangeShifterSettingsSequence'] = [dict(ReferencedRangeShifterNumber=3, RangeShifterSetting='IN')]
    rows = rows_of(tmp_path, [beam])
    assert {tuple(r['range_shifter_ids']) for r in rows} == {('SYNTHETIC_A', 'SYNTHETIC_B')}
    assert [r['range_shifter_settings'] for r in rows if r['control_point_index'] < 4] == [['IN', 'OUT']] * 5
    assert [r['range_shifter_settings'] for r in rows if r['control_point_index'] == 4] == [['IN', 'IN']] * 3
    assert {tuple(r['range_shifter_wet']) for r in rows if r['control_point_index'] < 4} == {(70.0, 30.0)}
    assert {r['range_shifter_wet'] for r in rows if r['control_point_index'] == 4} == {None}  # a new setting drops the old value


def test_a_re_listed_unchanged_setting_keeps_its_water_equivalent_thickness(tmp_path):  # #298
    beam = sd.pbs_beam(range_shifters=1)
    points_of(beam)[4]['RangeShifterSettingsSequence'] = [dict(ReferencedRangeShifterNumber=1, RangeShifterSetting='IN')]  # IN again, no thickness
    rows = rows_of(tmp_path, [beam])
    assert {tuple(r['range_shifter_wet']) for r in rows} == {(10.0,)}


def test_missing_water_equivalent_thickness_makes_the_list_null(tmp_path):
    beam = sd.pbs_beam(range_shifters=2)
    del points_of(beam)[0]['RangeShifterSettingsSequence'][1]['RangeShifterWaterEquivalentThickness']
    rows = rows_of(tmp_path, [beam])
    assert {r['range_shifter_wet'] for r in rows} == {None}
    assert {tuple(r['range_shifter_settings']) for r in rows} == {('IN', 'IN')}
    assert ion_plan_table.to_table(rows).schema.equals(ION_SCHEMA)


def _write(path, writer, **kwargs):
    writer(path, sd.generate_uid(), **kwargs)


def _tweak(**changes):
    return lambda path: _write(path, sd.write_pbs_rtplan, beams=[sd.pbs_beam(1, tags=changes), sd.pbs_beam(2, 300.0)])


NO_TABLE = {
    'no ion beams': lambda path: _write(path, sd.write_photon_rtplan),
    'brachy': lambda path: _write(path, sd.write_rtplan, positions_mm=[(0, 0, 0)]),
    'both sequences': lambda path: _write(path, sd.write_pbs_rtplan, tags=dict(BeamSequence=[dict(BeamNumber=9)])),
    'scan mode none': _tweak(ScanMode='NONE'),
    'uniform': _tweak(ScanMode='UNIFORM'),
    'modulated spec': _tweak(ScanMode='MODULATED_SPEC'),
    'no scan mode': _tweak(ScanMode=None),
    'np': _tweak(PrimaryDosimeterUnit='NP'),
    'minute': _tweak(PrimaryDosimeterUnit='MINUTE'),
    'carbon': _tweak(RadiationType='ION'),
    'no radiation type': _tweak(RadiationType=None),
    'continuation': _tweak(TreatmentDeliveryType='CONTINUATION'),
    'only setup': lambda path: _write(path, sd.write_pbs_rtplan, beams=[sd.pbs_beam(1, tags=dict(TreatmentDeliveryType='SETUP'))]),
    'old fixture': lambda path: _write(path, sd.write_ion_rtplan),
    'no unit, then np': lambda path: _write(path, sd.write_pbs_rtplan, beams=[
        sd.pbs_beam(1, tags=dict(PrimaryDosimeterUnit=None)), sd.pbs_beam(2, 300.0, tags=dict(PrimaryDosimeterUnit='NP'))]),
    'np, then no unit': lambda path: _write(path, sd.write_pbs_rtplan, beams=[
        sd.pbs_beam(1, tags=dict(PrimaryDosimeterUnit='NP')), sd.pbs_beam(2, 300.0, tags=dict(PrimaryDosimeterUnit=None))]),
}


@pytest.mark.parametrize('name', NO_TABLE)
def test_plans_without_a_table_return_none_silently(tmp_path, caplog, name):
    path = tmp_path / 'p.dcm'
    NO_TABLE[name](path)
    caplog.set_level(0)
    assert ion_plan_table.ion_plan_rows(read_header(path), ct_image()) is None
    assert caplog.records == []


def test_setup_beam_without_meterset_is_ignored(tmp_path):
    setup = sd.pbs_beam(3, tags=dict(TreatmentDeliveryType='SETUP', BeamMeterset=None, ScanMode=None, PrimaryDosimeterUnit=None))
    rows = rows_of(tmp_path, [sd.pbs_beam(1, 100.0), sd.pbs_beam(2, 300.0), setup])
    assert {r['beam_number'] for r in rows} == {1, 2}


def test_delivery_type_absent_counts_as_treatment(tmp_path):
    rows = rows_of(tmp_path, [sd.pbs_beam(tags=dict(TreatmentDeliveryType=None))])
    assert len(rows) == 8


def test_isocenter_as_voxel_index(tmp_path):
    ct = ct_image()
    rows = rows_of(tmp_path, ct=ct)
    expected = ct.TransformPhysicalPointToContinuousIndex((10.0, 20.0, 30.0))
    assert expected == (7.0, 7.0, 13.0)
    assert {tuple(r[k] for k in ('isocenter_i', 'isocenter_j', 'isocenter_k')) for r in rows} == {expected}
    assert {tuple(r[k] for k in ('isocenter_x', 'isocenter_y', 'isocenter_z')) for r in rows} == {(10.0, 20.0, 30.0)}


def test_isocenter_index_is_computed_once_per_control_point(tmp_path):
    class Counting:
        calls = 0

        def TransformPhysicalPointToContinuousIndex(self, point):  # noqa: N802 (SimpleITK's name)
            Counting.calls += 1
            return point

    rows_of(tmp_path, [sd.pbs_beam()], ct=Counting())
    assert Counting.calls == 6  # six control points, eight spots


def test_scalars_fill_forward_from_earlier_control_points(tmp_path):
    rows = rows_of(tmp_path, [sd.pbs_beam(paintings=2)])
    assert [r['nominal_energy'] for r in rows] == [100.0] * 2 + [110.0] * 3 + [120.0] * 3
    for key, value in dict(gantry_angle=90.0, couch_angle=0.0, snout_position=300.0, scan_spot_tune_id='SYNTHETIC_TUNE',
                           spot_size_x=7.0, spot_size_y=8.0, number_of_paintings=2, isocenter_x=10.0).items():
        assert {r[key] for r in rows} == {value}, key
    assert {(r['virtual_sad_x'], r['virtual_sad_y'], r['snout_id']) for r in rows} == {(2300.0, 2000.0, 'SYNTHETIC_SNOUT')}
    beam = sd.pbs_beam()
    points_of(beam)[2]['NominalBeamEnergy'] = ''  # present but empty: carried from CP 0
    points_of(beam)[4]['GantryAngle'] = 120.0
    rows = rows_of(tmp_path, [beam])
    assert [r['nominal_energy'] for r in rows] == [100.0] * 5 + [120.0] * 3
    assert [r['gantry_angle'] for r in rows] == [90.0] * 5 + [120.0] * 3


def test_spot_map_and_weights_are_never_filled_forward(tmp_path):
    beam = sd.pbs_beam()
    for key in ('ScanSpotMetersetWeights', 'ScanSpotPositionMap', 'NumberOfScanSpotPositions'):
        points_of(beam)[1][key] = None  # an odd CP that carries no spots: no copy of CP 0's spots
    assert len(rows_of(tmp_path, [beam])) == 8


def test_parquet_round_trips(tmp_path):
    table = ion_plan_table.to_table(rows_of(tmp_path))
    out = tmp_path / 'ion_plan.parquet'
    ion_plan_table.write_ion_plan_table(table, out)
    assert pq.read_table(out).equals(table) and pq.read_table(out).schema.equals(ION_SCHEMA)


def test_reason_is_the_spec_string():
    assert REASON_ION_PLAN == 'the proton RTPLAN has a missing or inconsistent value; no ion plan table written'


def test_ion_plan_table_catches_quiets_and_imports_nothing_of_the_photon_modules():
    tree = ast.parse(inspect.getsource(ion_plan_table))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {'quiet_call', '_pydicom_quiet'}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert [ast.unparse(n.exc) for n in ast.walk(tree) if isinstance(n, ast.Raise)] == ['CaseRefused(REASON_ION_PLAN)']
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert [a for a in attributes if tag_for_keyword(a) is not None] == []  # tags are read with .get, never as attributes
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in imported if m.split('.')[-1] in ('plan_table', 'beam_volume', 'fluence')}


@pytest.mark.slow
def test_fifty_thousand_spots_build_in_under_five_seconds(tmp_path):
    spots = [[1 / 5000] * 5000 for _ in range(5)]
    beams = [sd.pbs_beam(1, 100.0, spot_weights=spots), sd.pbs_beam(2, 300.0, spot_weights=spots)]
    plan = header_of(tmp_path, beams)
    ct = ct_image()
    start = time.perf_counter()
    rows = ion_plan_table.ion_plan_rows(plan, ct)
    built = time.perf_counter()
    table = ion_plan_table.to_table(rows)
    done = time.perf_counter()
    print(f'\nion_plan_rows {built - start:.2f} s, to_table {done - built:.2f} s, total {done - start:.2f} s')
    assert table.num_rows == 50_000 and done - start < 5.0
