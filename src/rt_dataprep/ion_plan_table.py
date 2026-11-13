"""
Proton pencil-beam-scanning RTPLAN header -> one row per non-zero spot, as a schema-typed pyarrow Table (ion_plan.parquet).

Spec: docs/specs/2026-11-11-proton-plans.md. `ion_plan_rows` returns None for a plan that gets no table (anything but a PBS
proton MU plan; the None pass covers every treatment beam before a row is built) and raises CaseRefused(REASON_ION_PLAN) for
a malformed PBS plan. Nothing here quiets pydicom or catches an exception: read_header_core already converted every element,
and every refusal is the one fixed reason (no UID, label, name or value). Dataset values are read with `.get`, never as
attributes, so an absent tag is a refusal of ours rather than a pydicom AttributeError. The small helpers below are copies of
plan_table's: the two modules stay independent.
"""
import math

import pyarrow as pa
import pyarrow.parquet as pq
from pydicom.multival import MultiValue

from rt_dataprep.dicom_io import CaseRefused

REASON_ION_PLAN = 'the proton RTPLAN has a missing or inconsistent value; no ion plan table written'
ION_WEIGHT_TOLERANCE = 1e-4  # relative to the beam's final cumulative weight
MULTI = (MultiValue, list)  # pydicom 3 reads a multi-valued FL/FD as a plain list, a DS/IS/CS one as a MultiValue


def _columns():
    f64, i64, text, texts, floats = pa.float64(), pa.int64(), pa.string(), pa.list_(pa.string()), pa.list_(pa.float64())
    columns = [
        ('beam_number', i64), ('beam_name?', text), ('beam_type', text), ('radiation_type', text),
        ('treatment_machine?', text), ('scan_mode', text), ('beam_meterset', f64), ('final_cumulative_meterset_weight', f64),
        ('number_of_control_points', i64), ('virtual_sad_x?', f64), ('virtual_sad_y?', f64), ('snout_id?', text),
        ('range_shifter_ids', texts),
        ('control_point_index', i64), ('layer_index', i64), ('cumulative_meterset_weight', f64), ('nominal_energy', f64),
        ('gantry_angle', f64), ('couch_angle', f64), ('snout_position?', f64), ('scan_spot_tune_id?', text),
        ('spot_size_x?', f64), ('spot_size_y?', f64), ('number_of_paintings?', i64), ('range_shifter_settings', texts),
        ('range_shifter_wet?', floats), ('isocenter_x', f64), ('isocenter_y', f64), ('isocenter_z', f64),
        ('isocenter_i', f64), ('isocenter_j', f64), ('isocenter_k', f64),
        ('spot_index', i64), ('spot_x', f64), ('spot_y', f64), ('spot_weight', f64), ('spot_mu', f64)]
    return [pa.field(name.rstrip('?'), kind, nullable=name.endswith('?')) for name, kind in columns]


ION_SCHEMA = pa.schema(_columns())  # '?' in _columns marks the nullable ones; types are never inferred
# Control-point tags that carry over from the last control point that listed them.
SCALARS = ('NominalBeamEnergy', 'GantryAngle', 'PatientSupportAngle', 'SnoutPosition', 'ScanSpotTuneID', 'ScanningSpotSize',
           'NumberOfPaintings', 'IsocenterPosition')


def _refuse():
    raise CaseRefused(REASON_ION_PLAN)


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


def _pair(value):
    """Two finite floats, or (None, None) when absent."""
    pair = _nums(value) or [None, None]
    _check(len(pair) == 2)
    return pair


def _text(value):
    return None if _missing(value) else str(value)


def _optional_num(value):
    return None if _missing(value) else _num(value)


def _metersets(plan):
    """BeamMeterset (raw) per ReferencedBeamNumber of the first fraction group; a repeated number is malformed."""
    groups = plan.get('FractionGroupSequence') or []
    references = (groups[0].get('ReferencedBeamSequence') or []) if groups else []
    references = [r for r in references if not _missing(r.get('ReferencedBeamNumber'))]
    numbers = [int(r.get('ReferencedBeamNumber')) for r in references]
    _check(len(set(numbers)) == len(numbers))
    return {n: r.get('BeamMeterset') for n, r in zip(numbers, references, strict=True)}


def _spots(point):
    """(weights, positions) of a control point, [] for what it does not carry (never filled forward)."""
    weights, positions = _nums(point.get('ScanSpotMetersetWeights')), _nums(point.get('ScanSpotPositionMap'))
    declared = point.get('NumberOfScanSpotPositions')
    _check(len(positions) == 2 * len(weights) and (_missing(declared) or int(declared) == len(weights)))
    _check(all(w >= 0 for w in weights))
    return weights, positions


def _beam_rows(beam, meterset, ct):
    final = _num(_need(beam.get('FinalCumulativeMetersetWeight')))
    _check(final > 0)
    tolerance = ION_WEIGHT_TOLERANCE * final
    points = beam.get('IonControlPointSequence') or []
    _check(len(points) > 0)
    _check(int(_need(beam.get('NumberOfControlPoints'))) == len(points))  # a truncated sequence, as plan_table (#325)
    cumulative = [_num(_need(p.get('CumulativeMetersetWeight'))) for p in points]  # never filled forward
    _check(abs(cumulative[0]) <= tolerance and abs(cumulative[-1] - final) <= tolerance)
    _check(all(b >= a for a, b in zip(cumulative, cumulative[1:], strict=False)))
    spots = [_spots(p) for p in points]
    steps = [b - a for a, b in zip(cumulative, cumulative[1:], strict=False)] + [0.0]  # the last CP carries nothing
    sums = [math.fsum(weights) for weights, _ in spots]
    _check(all(abs(s - step) <= tolerance for s, step in zip(sums, steps, strict=True)))
    _check(abs(math.fsum(sums) - final) <= tolerance)  # per-CP errors could add up past the tolerance
    beam_meterset = _num(_need(meterset))
    _check(beam_meterset >= 0)  # else negative spot_mu (#297)

    snouts = beam.get('SnoutSequence') or []
    shifters = beam.get('RangeShifterSequence') or []
    numbers = [int(_need(s.get('RangeShifterNumber'))) for s in shifters]
    _check(len(set(numbers)) == len(numbers))  # a repeat would collapse in the settings dict (#297)
    sad = _pair(beam.get('VirtualSourceAxisDistances'))
    _check(_need(beam.get('PrimaryDosimeterUnit')) == 'MU')
    fixed = dict(
        beam_number=int(_need(beam.get('BeamNumber'))), beam_name=_text(beam.get('BeamName')),
        beam_type=str(_need(beam.get('BeamType'))), radiation_type=str(_need(beam.get('RadiationType'))),
        treatment_machine=_text(beam.get('TreatmentMachineName')), scan_mode=str(_need(beam.get('ScanMode'))),
        beam_meterset=beam_meterset, final_cumulative_meterset_weight=final,
        number_of_control_points=int(_need(beam.get('NumberOfControlPoints'))), virtual_sad_x=sad[0], virtual_sad_y=sad[1],
        snout_id=_text(snouts[0].get('SnoutID')) if snouts else None,
        range_shifter_ids=[str(_need(s.get('RangeShifterID'))) for s in shifters])

    scalars, settings, thicknesses, rows, layer = {}, {}, {}, [], 0
    for i, point in enumerate(points):
        _check(int(_need(point.get('ControlPointIndex'))) == i)  # the rows' order is the sequence's, as plan_table (#325)
        for keyword in SCALARS:
            if not _missing(point.get(keyword)):
                scalars[keyword] = point.get(keyword)
        for item in point.get('RangeShifterSettingsSequence') or []:  # per referenced shifter: a later CP lists only changes
            number = item.get('ReferencedRangeShifterNumber')
            if _missing(number):
                continue
            _check(int(number) in numbers)  # a setting for an undeclared shifter (#297)
            setting = item.get('RangeShifterSetting')
            if not _missing(setting) and settings.get(int(number)) != str(setting):  # a changed setting replaces the whole entry: an old thickness never carries over; a re-listed one keeps it (#298)
                settings[int(number)] = str(setting)
                thicknesses.pop(int(number), None)
            if not _missing(item.get('RangeShifterWaterEquivalentThickness')):
                thicknesses[int(number)] = _num(item.get('RangeShifterWaterEquivalentThickness'))
        if i == 0:  # a declared range shifter must have its setting at CP 0
            _check(all(n in settings for n in numbers))
        isocenter = _nums(_need(scalars.get('IsocenterPosition')))
        _check(len(isocenter) == 3)
        index = ct.TransformPhysicalPointToContinuousIndex(tuple(isocenter))  # once per control point, not per spot
        size = _pair(scalars.get('ScanningSpotSize'))
        paintings = scalars.get('NumberOfPaintings')
        base = dict(
            fixed, control_point_index=i, layer_index=layer,
            cumulative_meterset_weight=cumulative[i], nominal_energy=_num(_need(scalars.get('NominalBeamEnergy'))),
            gantry_angle=_num(_need(scalars.get('GantryAngle'))), couch_angle=_num(_need(scalars.get('PatientSupportAngle'))),
            snout_position=_optional_num(scalars.get('SnoutPosition')), scan_spot_tune_id=_text(scalars.get('ScanSpotTuneID')),
            spot_size_x=size[0], spot_size_y=size[1], number_of_paintings=None if _missing(paintings) else int(paintings),
            range_shifter_settings=[settings[n] for n in numbers],
            range_shifter_wet=None if any(n not in thicknesses for n in numbers) else [thicknesses[n] for n in numbers])
        base.update(zip(('isocenter_x', 'isocenter_y', 'isocenter_z'), isocenter, strict=True))
        base.update(zip(('isocenter_i', 'isocenter_j', 'isocenter_k'), index, strict=True))
        weights, positions = spots[i]
        for k, weight in enumerate(weights):
            if weight > 0:  # zero-weight spots have no row, the next spot keeps its index
                rows.append(dict(base, spot_index=k, spot_x=positions[2 * k], spot_y=positions[2 * k + 1], spot_weight=weight,
                                 spot_mu=beam_meterset * weight / final))
        layer += any(w > 0 for w in weights)
    return rows


def _treatment_beams(plan):
    """The treatment beams of a PBS proton MU plan, None when the plan gets no table; SETUP beams drop first."""
    beams = plan.get('IonBeamSequence')
    if not beams or plan.get('BeamSequence'):
        return None
    if any(b.get('TreatmentDeliveryType') not in (None, '', 'TREATMENT', 'SETUP') for b in beams):
        return None  # a CONTINUATION or port-film beam delivers dose the table would leave out (owner, 2026-11-11)
    treatment = [b for b in beams if b.get('TreatmentDeliveryType') != 'SETUP']
    # A missing PrimaryDosimeterUnit passes here and is refused while building; NP or any other unit is no table.
    if not treatment or any(b.get('RadiationType') != 'PROTON' or b.get('ScanMode') != 'MODULATED'
                            or b.get('PrimaryDosimeterUnit') not in (None, '', 'MU') for b in treatment):
        return None
    return treatment


def ion_plan_rows(plan, ct):
    """
    The spot rows (dicts keyed by ION_SCHEMA's names) of the proton PBS plan header `plan`, with the isocenter also as a
    continuous index on the SimpleITK image `ct`; None when the plan gets no table.
    """
    treatment = _treatment_beams(plan)  # all beams, before any row is built
    if treatment is None:
        return None
    numbers = [int(_need(b.get('BeamNumber'))) for b in treatment]
    _check(len(set(numbers)) == len(numbers))
    metersets = _metersets(plan)
    rows = []
    for beam, number in zip(treatment, numbers, strict=True):
        rows += _beam_rows(beam, metersets.get(number), ct)
    return rows


def to_table(rows):
    return pa.Table.from_pylist(rows, schema=ION_SCHEMA)


def write_ion_plan_table(table, path):
    pq.write_table(table, path)
