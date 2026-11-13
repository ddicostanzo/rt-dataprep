import ast
import copy
import inspect
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import SimpleITK as sitk
from pydicom.datadict import tag_for_keyword
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence

import synthetic_dicom as sd
from rt_dataprep import dwell_table
from rt_dataprep.dicom_io import CaseRefused, read_header_core
from rt_dataprep.dwell_table import DWELL_SCHEMA, DWELL_WEIGHT_TOLERANCE, PAIR_ATOL_MM, REASON_DWELL

NAMES = [
    'brachy_treatment_type', 'setup_number', 'setup_name', 'setup_type', 'channel_number', 'source_movement_type',
    'source_applicator_id', 'source_applicator_type', 'source_applicator_step_size', 'source_applicator_length',
    'transfer_tube_length', 'referenced_roi_number', 'channel_total_time', 'final_cumulative_time_weight',
    'number_of_control_points', 'number_of_pulses', 'pulse_repetition_interval', 'source_number', 'source_isotope_name',
    'source_isotope_half_life', 'reference_air_kerma_rate', 'source_strength_reference_date', 'source_strength_reference_time',
    'dwell_index', 'control_point_index', 'relative_position', 'position_x', 'position_y', 'position_z', 'position_i',
    'position_j', 'position_k', 'cumulative_time_weight', 'dwell_weight', 'dwell_time_s']
NULLABLE = {'setup_name', 'setup_type', 'source_applicator_id', 'source_applicator_type', 'source_applicator_step_size',
            'source_applicator_length', 'transfer_tube_length', 'referenced_roi_number', 'number_of_pulses',
            'pulse_repetition_interval', 'source_isotope_name', 'source_isotope_half_life', 'reference_air_kerma_rate',
            'source_strength_reference_date', 'source_strength_reference_time', 'relative_position'}
INTS = {'setup_number', 'channel_number', 'referenced_roi_number', 'number_of_control_points', 'number_of_pulses',
        'source_number', 'dwell_index', 'control_point_index'}
STRINGS = {'brachy_treatment_type', 'setup_name', 'setup_type', 'source_movement_type', 'source_applicator_id',
           'source_applicator_type', 'source_isotope_name', 'source_strength_reference_date', 'source_strength_reference_time'}


def ct_image():
    image = sitk.Image([10, 12, 8], sitk.sitkInt16)
    image.SetSpacing((2.0, 2.0, 3.0))
    image.SetOrigin((-4.0, 6.0, -9.0))
    return image


def default_channels():
    """Two STEPWISE channels, three dwells each; channel 1 holds a zero-time dwell. Weights are binary-exact."""
    common = dict(source_number=1, movement_type='STEPWISE')
    return [dict(dwells_mm=[(0, 0, 0), (0, 0, 5), (0, 0, 10)], times=[0.25, 0, 0.5], total_time=30, final_weight=0.75, **common),
            dict(dwells_mm=[(10, 10, 0), (10, 10, 5), (10, 10, 10)], times=[0.5, 0.25, 0.25], total_time=40, final_weight=1.0,
                 **common)]


def plan_of(tmp_path, edit=None, channels=None, **kwargs):
    """A written-and-read-back brachy RTPLAN; `edit(plan)` changes the read-back header first."""
    path = tmp_path / 'plan.dcm'
    kwargs.setdefault('treatment_type', 'HDR')
    kwargs.setdefault('sources', [sd.brachy_source()])
    sd.write_rtplan(path, sd.generate_uid(), [(0, 0, 0)], channels=channels or default_channels(), **kwargs)
    plan = read_header_core(path)
    if edit:
        edit(plan)
    return plan


def rows_of(tmp_path, edit=None, ct=None, **kwargs):
    rows = dwell_table.dwell_rows(plan_of(tmp_path, edit, **kwargs), ct or ct_image())
    if rows is not None:
        required_complete(dwell_table.to_table(rows))
    return rows


def refused(tmp_path, edit=None, **kwargs):
    with pytest.raises(CaseRefused) as raised:
        rows_of(tmp_path, edit, **kwargs)
    assert str(raised.value) == REASON_DWELL


def required_complete(table):
    assert all(table[name].null_count == 0 for name in table.schema.names if name not in NULLABLE)


def chan(plan, n=0):
    return plan.ApplicationSetupSequence[0].ChannelSequence[n]


def cps(plan, n=0):
    return chan(plan, n).BrachyControlPointSequence


def test_schema_columns_types_nullability():
    assert DWELL_SCHEMA.names == NAMES
    for field in DWELL_SCHEMA:
        expected = pa.int64() if field.name in INTS else pa.string() if field.name in STRINGS else pa.float64()
        assert field.type == expected, field.name
        assert field.nullable == (field.name in NULLABLE), field.name


def test_constants_are_the_spec_values():
    assert REASON_DWELL == 'the brachy RTPLAN has a missing or inconsistent value; no dwell table written'
    assert DWELL_WEIGHT_TOLERANCE == 1e-4 and PAIR_ATOL_MM == 0.1


def test_hdr_plan_gives_one_row_per_dwell_in_order_with_the_zero_dwell(tmp_path):
    rows = rows_of(tmp_path)
    assert len(rows) == 6
    assert [(r['channel_number'], r['dwell_index']) for r in rows] == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert [r['dwell_weight'] for r in rows] == [0.25, 0.0, 0.5, 0.5, 0.25, 0.25]
    zero = rows[1]
    assert zero['dwell_weight'] == 0 and zero['dwell_time_s'] == 0 and zero['position_z'] == 5.0
    assert [r['dwell_time_s'] for r in rows] == [10.0, 0.0, 20.0, 20.0, 10.0, 10.0]  # channel_total_time x weight / final
    assert rows[5]['dwell_time_s'] == 40 * 0.25 / 1.0


def test_channel_sums_equal_channel_total_time_for_exact_weights(tmp_path):
    rows = rows_of(tmp_path)
    for number, total in ((1, 30.0), (2, 40.0)):
        assert math.isclose(sum(r['dwell_time_s'] for r in rows if r['channel_number'] == number), total, rel_tol=1e-9)


def test_channel_sums_hold_for_ds_rounded_weights(tmp_path):
    channel = dict(dwells_mm=[(0, 0, 0), (0, 0, 5), (0, 0, 10)], times=[1 / 3] * 3, total_time=17.3, final_weight=1.0,
                   source_number=1, movement_type='STEPWISE')  # 1/3 and its running sums are written with 10 digits
    rows = rows_of(tmp_path, channels=[channel])
    assert not math.isclose(rows[0]['dwell_weight'], 1 / 3, rel_tol=1e-12)
    assert math.isclose(sum(r['dwell_time_s'] for r in rows), 17.3, rel_tol=1e-4)


def test_positions_indices_and_control_point_index(tmp_path):
    ct = ct_image()
    rows = rows_of(tmp_path, ct=ct)
    for r in rows:
        point = (r['position_x'], r['position_y'], r['position_z'])
        assert (r['position_i'], r['position_j'], r['position_k']) == ct.TransformPhysicalPointToContinuousIndex(point)
    assert (rows[1]['position_x'], rows[1]['position_y'], rows[1]['position_z']) == (0.0, 0.0, 5.0)
    assert (rows[1]['position_i'], rows[1]['position_j']) == (2.0, -3.0) and math.isclose(rows[1]['position_k'], 14.0 / 3.0)
    assert [r['control_point_index'] for r in rows] == [0, 2, 4, 0, 2, 4]
    assert [r['cumulative_time_weight'] for r in rows[:3]] == [0.0, 0.25, 0.25]
    assert {r['number_of_control_points'] for r in rows} == {6}


def test_setup_channel_source_and_pulse_columns_hold_the_plans_values(tmp_path):
    def fill(plan):
        setup = plan.ApplicationSetupSequence[0]
        setup.ApplicationSetupName, setup.ApplicationSetupType = 'SYNTHETIC_SETUP', 'FLETCHER_SUIT'
        setup.ApplicationSetupNumber = 7
        first = chan(plan)
        first.ChannelNumber = 4
        first.SourceApplicatorID, first.SourceApplicatorType = 'SYNTHETIC_TANDEM', 'INTRALUMENATL'
        first.SourceApplicatorStepSize, first.SourceApplicatorLength, first.TransferTubeLength = 2.5, 150.0, 1000.0
        first.ReferencedROINumber = 3
        cps(plan)[0].ControlPointRelativePosition = 118.5
    rows = rows_of(tmp_path, fill)
    first = rows[0]
    assert (first['brachy_treatment_type'], first['setup_number'], first['setup_name'], first['setup_type']) == (
        'HDR', 7, 'SYNTHETIC_SETUP', 'FLETCHER_SUIT')
    assert (first['channel_number'], first['source_movement_type'], first['source_applicator_id'],
            first['source_applicator_type']) == (4, 'STEPWISE', 'SYNTHETIC_TANDEM', 'INTRALUMENATL')
    assert (first['source_applicator_step_size'], first['source_applicator_length'], first['transfer_tube_length'],
            first['referenced_roi_number']) == (2.5, 150.0, 1000.0, 3)
    assert (first['channel_total_time'], first['final_cumulative_time_weight'], first['relative_position']) == (30.0, 0.75, 118.5)
    assert (first['source_number'], first['source_isotope_name'], first['source_isotope_half_life'],
            first['reference_air_kerma_rate']) == (1, 'SYNTHETIC_Ir-192', 73.81, 40000.0)
    assert (first['source_strength_reference_date'], first['source_strength_reference_time']) == ('20240105', '101515')
    assert first['number_of_pulses'] is None and first['pulse_repetition_interval'] is None
    assert rows[1]['relative_position'] is None  # only the first control point carries one


def test_each_channel_takes_its_own_source(tmp_path):
    sources = [sd.brachy_source(1), sd.brachy_source(2, SourceIsotopeName='SYNTHETIC_Co-60', ReferenceAirKermaRate='25000')]

    def second(plan):
        chan(plan, 1).ReferencedSourceNumber = 2
    rows = rows_of(tmp_path, second, sources=sources)
    assert {r['source_isotope_name'] for r in rows[:3]} == {'SYNTHETIC_Ir-192'}
    assert {(r['source_number'], r['source_isotope_name'], r['reference_air_kerma_rate']) for r in rows[3:]} == {
        (2, 'SYNTHETIC_Co-60', 25000.0)}


def test_table_without_optional_tags_has_the_declared_schema(tmp_path):
    source = sd.brachy_source(SourceIsotopeName=None, SourceIsotopeHalfLife=None, ReferenceAirKermaRate=None,
                              SourceStrengthReferenceDate=None, SourceStrengthReferenceTime=None)
    table = dwell_table.to_table(rows_of(tmp_path, sources=[source]))
    assert table.schema.equals(DWELL_SCHEMA)
    for name in NULLABLE:
        assert table[name].null_count == table.num_rows, name


def test_pdr_plan_fills_pulse_columns_and_keeps_the_time_unchanged(tmp_path):
    channels = [dict(c, pulses=(24, 3600)) for c in default_channels()]
    rows = rows_of(tmp_path, channels=channels, treatment_type='PDR')
    assert {(r['brachy_treatment_type'], r['number_of_pulses'], r['pulse_repetition_interval']) for r in rows} == {
        ('PDR', 24, 3600.0)}
    assert [r['dwell_time_s'] for r in rows] == [10.0, 0.0, 20.0, 20.0, 10.0, 10.0]  # not multiplied by the pulse count


def test_hdr_channel_with_pulse_tags_keeps_them(tmp_path):
    rows = rows_of(tmp_path, channels=[dict(c, pulses=(2, 60)) for c in default_channels()])
    assert {(r['number_of_pulses'], r['pulse_repetition_interval']) for r in rows} == {(2, 60.0)}


def test_setups_and_channels_keep_sequence_order(tmp_path):
    def two_setups(plan):
        first = plan.ApplicationSetupSequence[0]
        first.ApplicationSetupNumber = 7
        second = copy.deepcopy(first)
        second.ApplicationSetupNumber = 3
        second.ChannelSequence[0].ChannelNumber, second.ChannelSequence[1].ChannelNumber = 5, 2
        plan.ApplicationSetupSequence.append(second)
    rows = rows_of(tmp_path, two_setups)
    assert [(r['setup_number'], r['channel_number'], r['dwell_index']) for r in rows] == (
        [(7, 1, k) for k in range(3)] + [(7, 2, k) for k in range(3)] + [(3, 5, k) for k in range(3)] + [(3, 2, k) for k in range(3)])


def test_a_channel_without_control_points_contributes_nothing(tmp_path):
    def add_empty(plan):
        extra = copy.deepcopy(chan(plan))
        extra.ChannelNumber = 9
        del extra.BrachyControlPointSequence
        plan.ApplicationSetupSequence[0].ChannelSequence.append(extra)
    assert len(rows_of(tmp_path, add_empty)) == 6


def test_zero_final_zero_time_channel_gives_zero_time_rows_without_dividing(tmp_path):
    channel = dict(dwells_mm=[(0, 0, 0), (0, 0, 5)], times=[0, 0], total_time=0, final_weight=0, source_number=1,
                   movement_type='FIXED')
    rows = rows_of(tmp_path, channels=[channel])
    assert [(r['dwell_weight'], r['dwell_time_s']) for r in rows] == [(0.0, 0.0)] * 2


def test_zero_weights_with_channel_time_are_refused(tmp_path):
    # every weight 0 passes the weight checks alone: only the final-0-with-time rule keeps 30 s from vanishing
    channel = dict(dwells_mm=[(0, 0, 0), (0, 0, 5)], times=[0, 0], total_time=30, final_weight=0, source_number=1,
                   movement_type='FIXED')
    refused(tmp_path, channels=[channel])


def _no_time(plan):
    del chan(plan).ChannelTotalTime


def _odd(plan):
    cps(plan).append(copy.deepcopy(cps(plan)[-1]))  # the last control point twice: the total is still right


def _pair_apart(plan):
    cps(plan)[1].ControlPoint3DPosition = [0.0, 0.2, 0.0]


def _rising(plan):  # time passes while moving to dwell 1, and dwell 1 starts late
    for i in (2, 3, 4):
        cps(plan)[i].CumulativeTimeWeight = 0.3


def _decreasing(plan):
    cps(plan)[3].CumulativeTimeWeight = 0.2498


def _last_not_final(plan):
    chan(plan).FinalCumulativeTimeWeight = 0.8


def _unknown_source(plan):
    chan(plan).ReferencedSourceNumber = 9


def _no_source_number(plan):
    del chan(plan).ReferencedSourceNumber


def _no_position(plan):
    del cps(plan)[2].ControlPoint3DPosition


def _no_movement(plan):
    del chan(plan).SourceMovementType


def _unknown_movement(plan):
    chan(plan).SourceMovementType = 'BOGUS'


def _empty_movement(plan):
    chan(plan).SourceMovementType = ''


def _duplicate_channel(plan):
    chan(plan, 1).ChannelNumber = 1


def _duplicate_setup(plan):
    plan.ApplicationSetupSequence.append(copy.deepcopy(plan.ApplicationSetupSequence[0]))


def _duplicate_source(plan):
    plan.SourceSequence.append(copy.deepcopy(plan.SourceSequence[0]))


def _count_mismatch(plan):
    chan(plan).NumberOfControlPoints = 8


def _final_zero_with_time(plan):
    chan(plan).FinalCumulativeTimeWeight = 0.0


def _negative_time(plan):
    chan(plan).ChannelTotalTime = -30.0


def _negative_final(plan):
    chan(plan).FinalCumulativeTimeWeight = -0.75


def _first_weight(plan):
    cps(plan)[0].CumulativeTimeWeight = 0.1


def _no_weight(plan):
    del cps(plan)[3].CumulativeTimeWeight


def _no_final(plan):
    del chan(plan).FinalCumulativeTimeWeight


def _no_setup_number(plan):
    del plan.ApplicationSetupSequence[0].ApplicationSetupNumber


def _index_out_of_order(plan):  # #325: the dwell pairs follow the sequence, so its indices must too
    cps(plan)[2].ControlPointIndex = 5


def _no_control_point_index(plan):
    del cps(plan)[2].ControlPointIndex


def _pdr(plan, pulses=(), interval=()):
    """Make the plan PDR; `pulses` and `interval` list the channels that keep NumberOfPulses and PulseRepetitionInterval."""
    plan.BrachyTreatmentType = 'PDR'
    for n in range(2):
        if n in pulses:
            chan(plan, n).NumberOfPulses = 24
        if n in interval:
            chan(plan, n).PulseRepetitionInterval = 3600


def _pdr_neither(plan):
    _pdr(plan)


def _pdr_no_interval(plan):
    _pdr(plan, pulses=(0, 1), interval=(0,))


def _pdr_no_pulses(plan):
    _pdr(plan, pulses=(0,), interval=(0, 1))


CASES = [_no_time, _odd, _pair_apart, _rising, _decreasing, _last_not_final, _unknown_source, _no_source_number, _no_position,
         _no_movement, _unknown_movement, _empty_movement, _duplicate_channel, _duplicate_setup, _duplicate_source,
         _count_mismatch, _final_zero_with_time, _negative_time, _negative_final, _first_weight, _no_weight, _no_final,
         _no_setup_number, _no_control_point_index, _index_out_of_order, _pdr_neither, _pdr_no_interval, _pdr_no_pulses]


@pytest.mark.parametrize('mutate', CASES)
def test_malformed_plan_is_refused(tmp_path, mutate):
    refused(tmp_path, mutate)


@pytest.mark.filterwarnings('ignore:Invalid value for VR DS')
def test_non_finite_position_is_refused(tmp_path):
    channels = default_channels()
    channels[0]['point_tags'] = {2: dict(ControlPoint3DPosition=['nan', '0', '5'])}
    refused(tmp_path, channels=channels)


def test_channel_sum_drift_of_sub_tolerance_gaps_is_refused_alone(tmp_path):
    n, step, gap = 20, 0.05, 0.9e-4  # each gap is under the tolerance, 19 of them are not
    final = round(n * step + (n - 1) * gap, 8)
    weights = [round(k * (step + gap) + part, 8) for k in range(n) for part in (0.0, step)]

    def drift(plan):
        for point, weight in zip(cps(plan), weights, strict=True):
            point.CumulativeTimeWeight = weight
        chan(plan).FinalCumulativeTimeWeight = final
    channel = dict(dwells_mm=[(0, 0, k) for k in range(n)], times=[step] * n, total_time=60, final_weight=final, source_number=1,
                   movement_type='STEPWISE')
    tolerance = DWELL_WEIGHT_TOLERANCE * final
    assert all(0 < b - a <= tolerance for a, b in zip(weights[1::2], weights[2::2], strict=False))  # every gap alone passes
    assert abs(weights[-1] - final) <= tolerance
    assert sum(weights[2 * k + 1] - weights[2 * k] for k in range(n)) < final - tolerance  # only the sum is off
    refused(tmp_path, drift, channels=[channel])


def test_a_decrease_within_a_dwell_is_refused_alone(tmp_path):
    # zero-time dwell 1 (control points 2 and 3): 0.25 -> 0.25 + 0.7e-4 -> 0.25 - 0.7e-4 -> 0.25; each step is inside the tolerance
    # of 0.75e-4 for the flat-between-pairs check, but the drop of 1.4e-4 inside the pair is not
    weights = [0, 0.25, 0.25 + 0.7e-4, 0.25 - 0.7e-4, 0.25, 0.75]

    def dip(plan):
        for point, weight in zip(cps(plan), weights, strict=True):
            point.CumulativeTimeWeight = weight
    tolerance = DWELL_WEIGHT_TOLERANCE * 0.75
    assert weights[2] - weights[3] > tolerance  # the decrease check fails
    assert all(abs(weights[i + 1] - weights[i]) <= tolerance for i in (1, 3))  # the flat-between-pairs check passes
    assert abs(math.fsum(max(0.0, weights[i + 1] - weights[i]) for i in (0, 2, 4)) - 0.75) <= tolerance  # the channel-sum check passes
    refused(tmp_path, dip)


@pytest.mark.parametrize(('index', 'delta'), [(0, 0.5e-4), (3, -0.5e-4), (2, 0.5e-4), (5, -0.5e-4), (4, -0.5e-4)])
def test_a_deviation_within_the_tolerance_passes(tmp_path, index, delta):
    def nudge(plan):
        point = cps(plan)[index]
        point.CumulativeTimeWeight = float(point.CumulativeTimeWeight) + delta
    rows = rows_of(tmp_path, nudge)
    assert len(rows) == 6 and all(r['dwell_weight'] >= 0 for r in rows)


@pytest.mark.parametrize(('index', 'delta'), [(0, 1e-4), (3, -1e-4), (2, 1e-4), (5, -1e-4), (4, -1e-4)])
def test_a_deviation_past_the_tolerance_is_refused(tmp_path, index, delta):
    def nudge(plan):
        point = cps(plan)[index]
        point.CumulativeTimeWeight = float(point.CumulativeTimeWeight) + delta
    refused(tmp_path, nudge)


def test_a_small_decrease_within_a_dwell_gives_dwell_weight_zero(tmp_path):
    def dip(plan):
        cps(plan)[3].CumulativeTimeWeight = 0.25 - 0.5e-4  # the zero-time dwell's second control point
    rows = rows_of(tmp_path, dip)
    assert rows[1]['dwell_weight'] == 0 and rows[1]['dwell_time_s'] == 0


def test_pair_distance_tolerance(tmp_path):
    def apart(distance):
        def edit(plan):
            cps(plan)[1].ControlPoint3DPosition = [10.0 + distance, 0.0, 0.0]
            cps(plan)[0].ControlPoint3DPosition = [10.0, 0.0, 0.0]
        return edit
    assert len(rows_of(tmp_path, apart(0.1))) == 6
    refused(tmp_path, apart(0.1001))


NO_TABLE = {
    'ldr': dict(treatment_type='LDR'), 'mdr': dict(treatment_type='MDR'), 'manual': dict(treatment_type='MANUAL'),
    'no type': dict(treatment_type=None),
    'oscillating': dict(edit=lambda p: setattr(chan(p, 1), 'SourceMovementType', 'OSCILLATING')),
    'unidirectional': dict(edit=lambda p: setattr(chan(p), 'SourceMovementType', 'UNIDIRECTIONAL')),
    'moving beside a malformed channel': dict(edit=lambda p: (setattr(chan(p, 1), 'SourceMovementType', 'OSCILLATING'),
                                                               delattr(chan(p), 'ChannelTotalTime'))),
    'beam sequence': dict(edit=lambda p: setattr(p, 'BeamSequence', Sequence([sd.Dataset()]))),
    'ion beam sequence': dict(edit=lambda p: setattr(p, 'IonBeamSequence', Sequence([sd.Dataset()]))),
    'no control points': dict(edit=lambda p: [delattr(c, 'BrachyControlPointSequence') for c in
                                              p.ApplicationSetupSequence[0].ChannelSequence]),
    'no application setups': dict(edit=lambda p: delattr(p, 'ApplicationSetupSequence')),
}


@pytest.mark.parametrize('name', NO_TABLE)
def test_plans_without_a_table_return_none_silently(tmp_path, caplog, name):
    spec = dict(NO_TABLE[name])
    edit = spec.pop('edit', None)
    plan = plan_of(tmp_path, edit, **spec)
    caplog.set_level(0)
    assert dwell_table.dwell_rows(plan, ct_image()) is None
    assert caplog.records == []


def test_photon_and_ion_plans_get_no_table(tmp_path, caplog):
    caplog.set_level(0)
    for writer in (sd.write_photon_rtplan, sd.write_pbs_rtplan):
        path = tmp_path / 'p.dcm'
        writer(path, sd.generate_uid())
        assert dwell_table.dwell_rows(read_header_core(path), ct_image()) is None
    assert caplog.records == []


def test_a_valid_plan_logs_nothing(tmp_path, caplog):
    caplog.set_level(0)
    assert rows_of(tmp_path)
    assert caplog.records == []


def test_the_existing_brachy_fixture_writes_no_treatment_type(tmp_path):
    path = tmp_path / 'p.dcm'
    sd.write_rtplan(path, sd.generate_uid(), [(0, 0, 0), (0, 0, 5)])
    plan = read_header_core(path)
    assert 'BrachyTreatmentType' not in plan and 'SourceSequence' not in plan
    assert dwell_table.dwell_rows(plan, ct_image()) is None


class Listy:
    """A header node whose multi-values read back as plain lists, as pydicom 3 reads a multi-valued FL/FD."""

    def __init__(self, node):
        self.node = node

    def get(self, keyword, default=None):
        value = self.node.get(keyword, default)
        if isinstance(value, Sequence):
            return [Listy(item) for item in value]
        return list(value) if isinstance(value, MultiValue) else value


def test_multi_values_are_accepted_as_multivalue_and_as_list(tmp_path):
    plan = plan_of(tmp_path)
    assert isinstance(cps(plan)[0].ControlPoint3DPosition, MultiValue)
    assert dwell_table.dwell_rows(plan, ct_image()) == dwell_table.dwell_rows(Listy(plan), ct_image())


def test_parquet_round_trips(tmp_path):
    table = dwell_table.to_table(rows_of(tmp_path))
    out = tmp_path / 'dwell.parquet'
    dwell_table.write_dwell_table(table, out)
    assert pq.read_table(out).equals(table) and pq.read_table(out).schema.equals(DWELL_SCHEMA)


def test_dwell_table_catches_quiets_and_imports_nothing_of_the_other_plan_modules():
    tree = ast.parse(inspect.getsource(dwell_table))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {'quiet_call', '_pydicom_quiet'}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert [ast.unparse(n.exc) for n in ast.walk(tree) if isinstance(n, ast.Raise)] == ['CaseRefused(REASON_DWELL)']
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert [a for a in attributes if tag_for_keyword(a) is not None] == []  # tags are read with .get, never as attributes
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in imported if m.split('.')[-1] in ('plan_table', 'ion_plan_table', 'beam_volume', 'fluence')}
