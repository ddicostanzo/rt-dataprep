"""
HDR/PDR brachy RTPLAN header -> one row per dwell (a pair of control points 2k and 2k+1 at one position), as a schema-typed
pyarrow Table (dwell.parquet).

Spec: docs/specs/2026-11-11-brachy-dwells.md. `dwell_rows` returns None for a plan that gets no table (anything but an
HDR/PDR plan with stepwise or fixed sources; the None pass covers every channel before a row is built) and raises
CaseRefused(REASON_DWELL) for a malformed one. Nothing here quiets pydicom or catches an exception: read_header_core already
converted every element, and every refusal is the one fixed reason (no UID, label, name or value). Dataset values are read with
`.get`, never as attributes, so an absent tag is a refusal of ours rather than a pydicom AttributeError. The small helpers
below are copies of ion_plan_table's: the plan modules stay independent.
"""
import math

import pyarrow as pa
import pyarrow.parquet as pq
from pydicom.multival import MultiValue

from rt_dataprep.dicom_io import CaseRefused

REASON_DWELL = 'the brachy RTPLAN has a missing or inconsistent value; no dwell table written'
DWELL_WEIGHT_TOLERANCE = 1e-4  # relative to the channel's final cumulative time weight
PAIR_ATOL_MM = 0.1  # the two control points of a dwell may differ by this much on each axis
MULTI = (MultiValue, list)  # pydicom 3 reads a multi-valued FL/FD as a plain list, a DS/IS/CS one as a MultiValue
TREATED = ('HDR', 'PDR')
MOVING = ('OSCILLATING', 'UNIDIRECTIONAL')  # deferred: no table
SUPPORTED_MOVEMENT = ('STEPWISE', 'FIXED')


def _columns():
    f64, i64, text = pa.float64(), pa.int64(), pa.string()
    columns = [
        ('brachy_treatment_type', text), ('setup_number', i64), ('setup_name?', text), ('setup_type?', text),
        ('channel_number', i64), ('source_movement_type', text), ('source_applicator_id?', text),
        ('source_applicator_type?', text), ('source_applicator_step_size?', f64), ('source_applicator_length?', f64),
        ('transfer_tube_length?', f64), ('referenced_roi_number?', i64), ('channel_total_time', f64),
        ('final_cumulative_time_weight', f64), ('number_of_control_points', i64), ('number_of_pulses?', i64),
        ('pulse_repetition_interval?', f64), ('source_number', i64), ('source_isotope_name?', text),
        ('source_isotope_half_life?', f64), ('reference_air_kerma_rate?', f64), ('source_strength_reference_date?', text),
        ('source_strength_reference_time?', text),
        ('dwell_index', i64), ('control_point_index', i64), ('relative_position?', f64),
        ('position_x', f64), ('position_y', f64), ('position_z', f64), ('position_i', f64), ('position_j', f64),
        ('position_k', f64), ('cumulative_time_weight', f64), ('dwell_weight', f64), ('dwell_time_s', f64)]
    return [pa.field(name.rstrip('?'), kind, nullable=name.endswith('?')) for name, kind in columns]


DWELL_SCHEMA = pa.schema(_columns())  # '?' in _columns marks the nullable ones; types are never inferred


def _refuse():
    raise CaseRefused(REASON_DWELL)


def _check(condition):
    if not condition:
        _refuse()


def _missing(value):
    """Absent, or present but empty ('' or None as read_header_core leaves it, an empty multi-value)."""
    return value in ('', None) or (isinstance(value, MULTI) and len(value) == 0)


def _need(value):
    _check(not _missing(value))
    return value


def _num(value):
    number = float(value)
    _check(math.isfinite(number))
    return number


def _nums(value):
    """A list of finite floats; a single value (FL and DS read back as a bare float) is a list of one, an absent one is []."""
    if _missing(value):
        return []
    return [_num(v) for v in (value if isinstance(value, MULTI) else [value])]


def _text(value):
    return None if _missing(value) else str(value)


def _optional_num(value):
    return None if _missing(value) else _num(value)


def _optional_int(value):
    return None if _missing(value) else int(value)


def _channels(plan):
    """(setup, channel) of every channel with control points, None when the plan gets no table."""
    setups = plan.get('ApplicationSetupSequence') or []
    if not setups or plan.get('BeamSequence') or plan.get('IonBeamSequence'):
        return None
    if plan.get('BrachyTreatmentType') not in TREATED:
        return None
    channels = [(s, c) for s in setups for c in (s.get('ChannelSequence') or []) if c.get('BrachyControlPointSequence')]
    if not channels or any(c.get('SourceMovementType') in MOVING for _, c in channels):
        return None
    return channels


def _sources(plan):
    """SourceSequence items by SourceNumber; a repeated number is malformed."""
    items = plan.get('SourceSequence') or []
    numbers = [int(_need(i.get('SourceNumber'))) for i in items]
    _check(len(set(numbers)) == len(numbers))
    return dict(zip(numbers, items, strict=True))


def _channel_rows(fixed, channel, ct):
    """The rows of one channel; `fixed` holds the plan, setup and source columns."""
    movement = str(_need(channel.get('SourceMovementType')))
    _check(movement in SUPPORTED_MOVEMENT)
    total = _num(_need(channel.get('ChannelTotalTime')))
    final = _num(_need(channel.get('FinalCumulativeTimeWeight')))
    _check(total >= 0 and final >= 0)
    _check(final > 0 or total == 0)
    points = channel.get('BrachyControlPointSequence')
    declared = channel.get('NumberOfControlPoints')
    _check(len(points) % 2 == 0 and (_missing(declared) or int(declared) == len(points)))
    if fixed['brachy_treatment_type'] == 'PDR':
        _need(channel.get('NumberOfPulses')), _need(channel.get('PulseRepetitionInterval'))
    tolerance = DWELL_WEIGHT_TOLERANCE * final
    weights = [_num(_need(p.get('CumulativeTimeWeight'))) for p in points]
    positions = [_nums(_need(p.get('ControlPoint3DPosition'))) for p in points]
    _check(all(len(p) == 3 for p in positions))
    _check(abs(weights[0]) <= tolerance and abs(weights[-1] - final) <= tolerance)
    _check(all(b - a >= -tolerance for a, b in zip(weights, weights[1:], strict=False)))  # a decrease is a refusal
    _check(all(abs(weights[i + 1] - weights[i]) <= tolerance for i in range(1, len(points) - 1, 2)))  # no time while moving
    for i in range(0, len(points), 2):
        _check(all(abs(a - b) <= PAIR_ATOL_MM + 1e-9 for a, b in zip(positions[i], positions[i + 1], strict=True)))  # 1e-9: float noise
    dwell_weights = [max(0.0, weights[i + 1] - weights[i]) for i in range(0, len(points), 2)]
    _check(abs(math.fsum(dwell_weights) - final) <= tolerance)  # per-gap errors could add up past the tolerance

    number = int(_need(channel.get('ReferencedSourceNumber')))
    source = fixed['sources'].get(number)
    _check(source is not None)
    base = dict(
        {k: v for k, v in fixed.items() if k != 'sources'},
        channel_number=int(_need(channel.get('ChannelNumber'))), source_movement_type=movement,
        source_applicator_id=_text(channel.get('SourceApplicatorID')),
        source_applicator_type=_text(channel.get('SourceApplicatorType')),
        source_applicator_step_size=_optional_num(channel.get('SourceApplicatorStepSize')),
        source_applicator_length=_optional_num(channel.get('SourceApplicatorLength')),
        transfer_tube_length=_optional_num(channel.get('TransferTubeLength')),
        referenced_roi_number=_optional_int(channel.get('ReferencedROINumber')),
        channel_total_time=total, final_cumulative_time_weight=final, number_of_control_points=len(points),
        number_of_pulses=_optional_int(channel.get('NumberOfPulses')),
        pulse_repetition_interval=_optional_num(channel.get('PulseRepetitionInterval')),
        source_number=number, source_isotope_name=_text(source.get('SourceIsotopeName')),
        source_isotope_half_life=_optional_num(source.get('SourceIsotopeHalfLife')),
        reference_air_kerma_rate=_optional_num(source.get('ReferenceAirKermaRate')),
        source_strength_reference_date=_text(source.get('SourceStrengthReferenceDate')),
        source_strength_reference_time=_text(source.get('SourceStrengthReferenceTime')))
    rows = []
    for k, weight in enumerate(dwell_weights):
        point = points[2 * k]
        index = ct.TransformPhysicalPointToContinuousIndex(tuple(positions[2 * k]))
        row = dict(base, dwell_index=k, control_point_index=int(_need(point.get('ControlPointIndex'))),
                   relative_position=_optional_num(point.get('ControlPointRelativePosition')),
                   cumulative_time_weight=weights[2 * k], dwell_weight=weight,
                   dwell_time_s=total * weight / final if final > 0 else 0.0)  # final 0: every weight and the time are 0
        row.update(zip(('position_x', 'position_y', 'position_z'), positions[2 * k], strict=True))
        row.update(zip(('position_i', 'position_j', 'position_k'), index, strict=True))
        rows.append(row)
    return rows


def dwell_rows(plan, ct):
    """
    The dwell rows (dicts keyed by DWELL_SCHEMA's names) of the HDR/PDR brachy plan header `plan`, with the position also as
    a continuous index on the SimpleITK image `ct`; None when the plan gets no table.
    """
    channels = _channels(plan)  # all channels, before any row is built
    if channels is None:
        return None
    sources = _sources(plan)
    numbers = [int(_need(s.get('ApplicationSetupNumber'))) for s in plan.get('ApplicationSetupSequence')]
    _check(len(set(numbers)) == len(numbers))
    seen = set()
    rows = []
    for setup, channel in channels:
        setup_number = int(_need(setup.get('ApplicationSetupNumber')))
        key = (setup_number, int(_need(channel.get('ChannelNumber'))))
        _check(key not in seen)
        seen.add(key)
        fixed = dict(brachy_treatment_type=str(plan.get('BrachyTreatmentType')), setup_number=setup_number,
                     setup_name=_text(setup.get('ApplicationSetupName')), setup_type=_text(setup.get('ApplicationSetupType')),
                     sources=sources)
        rows += _channel_rows(fixed, channel, ct)
    return rows


def to_table(rows):
    return pa.Table.from_pylist(rows, schema=DWELL_SCHEMA)


def write_dwell_table(table, path):
    pq.write_table(table, path)
