"""
Photon EBRT RTPLAN header -> one row per control point, as a schema-typed pyarrow Table (plan.parquet).

Spec: docs/specs/2026-11-11-rtplan-parquet.md. `plan_rows` returns None for a plan that gets no table (brachy, RT Ion, a
non-photon treatment beam, no treatment beam) and raises CaseRefused(REASON_PLAN_TABLE) for a malformed photon plan.
Nothing here quiets pydicom or catches an exception: read_header_core already converted every element, and every
refusal is the one fixed reason (no UID, label, name or value). Dataset values are read with `.get`, never as attributes,
so an absent tag is a refusal of ours rather than a pydicom AttributeError.
"""
import math

import pyarrow as pa
import pyarrow.parquet as pq
from pydicom.multival import MultiValue

from rt_dataprep.dicom_io import CaseRefused

REASON_PLAN_TABLE = 'the photon RTPLAN has a missing or inconsistent value; no plan table written'
WEIGHT_TOLERANCE = 1e-6  # relative to the beam's final cumulative weight


def _columns():
    f64, i64, text, floats = pa.float64(), pa.int64(), pa.string(), pa.list_(pa.float64())
    required = [('beam_number', i64), ('beam_name?', text), ('beam_type', text), ('treatment_machine?', text),
                ('beam_meterset', f64), ('final_cumulative_meterset_weight', f64), ('number_of_control_points', i64),
                ('fluence_mode?', text), ('mlc_type?', text), ('leaf_position_boundaries?', floats),
                ('number_of_wedges', i64), ('wedge_id?', text), ('wedge_type?', text), ('wedge_orientation?', text),
                ('wedge_angle?', f64), ('number_of_blocks', i64), ('number_of_boli', i64),
                ('control_point_index', i64), ('cumulative_meterset_weight', f64), ('mu', f64), ('mu_cumulative', f64),
                ('nominal_energy', f64), ('dose_rate?', f64), ('gantry_angle', f64),
                ('gantry_rotation_direction?', text), ('collimator_angle', f64), ('couch_angle', f64),
                ('jaw_x1?', f64), ('jaw_x2?', f64), ('jaw_y1?', f64), ('jaw_y2?', f64), ('mlc_positions?', floats),
                ('ssd?', f64), ('wedge_position?', text), ('isocenter_x', f64), ('isocenter_y', f64),
                ('isocenter_z', f64), ('isocenter_i', f64), ('isocenter_j', f64), ('isocenter_k', f64)]
    return [pa.field(name.rstrip('?'), kind, nullable=name.endswith('?')) for name, kind in required]


SCHEMA = pa.schema(_columns())  # '?' in _columns marks the nullable ones; types are never inferred
JAW_TYPES = dict(x=('ASYMX', 'X'), y=('ASYMY', 'Y'))
# Control-point tags that carry over from the last control point that listed them.
SCALARS = ('NominalBeamEnergy', 'DoseRateSet', 'GantryAngle', 'GantryRotationDirection', 'BeamLimitingDeviceAngle',
           'PatientSupportAngle', 'SourceToSurfaceDistance', 'IsocenterPosition')


def _refuse():
    raise CaseRefused(REASON_PLAN_TABLE)


def _check(condition):
    if not condition:
        _refuse()


def _missing(value):
    """Absent, or present but empty ('' or None as read_header_core leaves it, an empty multi-value)."""
    return value in ('', None) or (isinstance(value, MultiValue) and len(value) == 0)


def _need(value):
    _check(not _missing(value))
    return value


def _num(value):
    number = float(value)
    _check(math.isfinite(number))
    return number


def _nums(value):
    return [_num(v) for v in (value if isinstance(value, MultiValue) else [value])]


def _text(value):
    return None if _missing(value) else str(value)


def _optional_num(value):
    return None if _missing(value) else _num(value)


def _count(value):
    return 0 if _missing(value) else int(value)


def _metersets(plan):
    """BeamMeterset (raw) per ReferencedBeamNumber of the first fraction group."""
    groups = plan.get('FractionGroupSequence') or []
    references = (groups[0].get('ReferencedBeamSequence') or []) if groups else []
    return {int(r.get('ReferencedBeamNumber')): r.get('BeamMeterset') for r in references
            if not _missing(r.get('ReferencedBeamNumber'))}


def _device_type(devices, kinds):
    return next((d.get('RTBeamLimitingDeviceType') for d in devices if d.get('RTBeamLimitingDeviceType') in kinds), None)


def _weights(points, final):
    weights = [_num(_need(p.get('CumulativeMetersetWeight'))) for p in points]  # never filled forward
    _check(final != 0)
    _check(abs(weights[0]) <= WEIGHT_TOLERANCE * abs(final))
    _check(abs(weights[-1] - final) <= WEIGHT_TOLERANCE * abs(final))
    _check(all(b >= a for a, b in zip(weights, weights[1:], strict=False)))
    return weights


def _beam_rows(beam, meterset, ct):
    final = _num(_need(beam.get('FinalCumulativeMetersetWeight')))
    points = beam.get('ControlPointSequence') or []
    _check(len(points) > 0)
    weights = _weights(points, final)
    beam_meterset = _num(_need(meterset))

    devices = beam.get('BeamLimitingDeviceSequence') or []
    mlcs = [d for d in devices if str(d.get('RTBeamLimitingDeviceType')).startswith('MLC')]
    _check(len(mlcs) <= 1)  # dual-layer MLC is deferred
    mlc = mlcs[0] if mlcs else None
    mlc_type = _text(mlc.get('RTBeamLimitingDeviceType')) if mlc else None
    pairs = int(_need(mlc.get('NumberOfLeafJawPairs'))) if mlc else 0
    boundaries = None
    if mlc and not _missing(mlc.get('LeafPositionBoundaries')):
        boundaries = _nums(mlc.get('LeafPositionBoundaries'))
        _check(len(boundaries) == pairs + 1)
    jaw_type = {axis: _device_type(devices, kinds) for axis, kinds in JAW_TYPES.items()}

    wedges = beam.get('WedgeSequence') or []
    wedge = (wedges[0] if wedges else None) or None  # an empty item is falsy: no wedge, never the Dataset itself
    wedge_number = None if wedge is None or _missing(wedge.get('WedgeNumber')) else int(wedge.get('WedgeNumber'))
    fluence = (beam.get('PrimaryFluenceModeSequence') or [None])[0]

    fixed = dict(
        beam_number=int(_need(beam.get('BeamNumber'))), beam_name=_text(beam.get('BeamName')),
        beam_type=str(_need(beam.get('BeamType'))), treatment_machine=_text(beam.get('TreatmentMachineName')),
        beam_meterset=beam_meterset, final_cumulative_meterset_weight=final,
        number_of_control_points=int(_need(beam.get('NumberOfControlPoints'))),
        fluence_mode=None if fluence is None else _text(fluence.get('FluenceModeID')) or _text(fluence.get('FluenceMode')),
        mlc_type=mlc_type, leaf_position_boundaries=boundaries, number_of_wedges=_count(beam.get('NumberOfWedges')),
        wedge_id=wedge and _text(wedge.get('WedgeID')), wedge_type=wedge and _text(wedge.get('WedgeType')),
        wedge_orientation=wedge and _text(wedge.get('WedgeOrientation')),
        wedge_angle=wedge and _optional_num(wedge.get('WedgeAngle')),
        number_of_blocks=_count(beam.get('NumberOfBlocks')), number_of_boli=_count(beam.get('NumberOfBoli')))

    scalars, positions, wedge_positions, rows = {}, {}, {}, []
    for i, point in enumerate(points):
        for keyword in SCALARS:
            if not _missing(point.get(keyword)):
                scalars[keyword] = point.get(keyword)
        for item in point.get('BeamLimitingDevicePositionSequence') or []:  # per device type: a later CP lists only movers
            if not _missing(item.get('RTBeamLimitingDeviceType')) and not _missing(item.get('LeafJawPositions')):
                positions[str(item.get('RTBeamLimitingDeviceType'))] = _nums(item.get('LeafJawPositions'))
        for item in point.get('WedgePositionSequence') or []:
            if not _missing(item.get('ReferencedWedgeNumber')) and not _missing(item.get('WedgePosition')):
                wedge_positions[int(item.get('ReferencedWedgeNumber'))] = str(item.get('WedgePosition'))
        if i == 0:  # a declared jaw/MLC device must be listed at CP 0
            _check(all(t is None or t in positions for t in (*jaw_type.values(), mlc_type)))
        row = dict(fixed, control_point_index=int(_need(point.get('ControlPointIndex'))),
                   cumulative_meterset_weight=weights[i],
                   mu=0.0 if i == 0 else beam_meterset * (weights[i] - weights[i - 1]) / final,
                   mu_cumulative=beam_meterset * weights[i] / final,
                   nominal_energy=_num(_need(scalars.get('NominalBeamEnergy'))),
                   dose_rate=_optional_num(scalars.get('DoseRateSet')),
                   gantry_angle=_num(_need(scalars.get('GantryAngle'))),
                   gantry_rotation_direction=_text(scalars.get('GantryRotationDirection')),
                   collimator_angle=_num(_need(scalars.get('BeamLimitingDeviceAngle'))),
                   couch_angle=_num(_need(scalars.get('PatientSupportAngle'))),
                   ssd=_optional_num(scalars.get('SourceToSurfaceDistance')),
                   wedge_position=wedge_positions.get(wedge_number))
        for axis, jaw in jaw_type.items():
            pair = positions[jaw] if jaw else [None, None]
            _check(len(pair) == 2)
            row[f'jaw_{axis}1'], row[f'jaw_{axis}2'] = pair
        row['mlc_positions'] = positions[mlc_type] if mlc_type else None
        _check(row['mlc_positions'] is None or len(row['mlc_positions']) == 2 * pairs)
        isocenter = _nums(_need(scalars.get('IsocenterPosition')))
        _check(len(isocenter) == 3)
        index = ct.TransformPhysicalPointToContinuousIndex(tuple(isocenter))
        row.update(zip(('isocenter_x', 'isocenter_y', 'isocenter_z'), isocenter, strict=True))
        row.update(zip(('isocenter_i', 'isocenter_j', 'isocenter_k'), index, strict=True))
        rows.append(row)
    return rows


def plan_rows(plan, ct):
    """
    The control-point rows (dicts keyed by SCHEMA's names) of the photon plan header `plan`, with the isocenter also as
    a continuous index on the SimpleITK image `ct`; None when the plan gets no table.
    """
    beams = plan.get('BeamSequence')  # a brachy plan has none, an RT Ion plan has IonBeamSequence instead
    if not beams:
        return None
    if any(b.get('TreatmentDeliveryType') not in (None, '', 'TREATMENT', 'SETUP') for b in beams):
        return None  # a CONTINUATION or port-film beam delivers dose the table would leave out (owner, 2026-11-11)
    treatment = [b for b in beams if b.get('TreatmentDeliveryType') != 'SETUP']  # SETUP beams drop first
    if not treatment or any(b.get('RadiationType') != 'PHOTON' for b in treatment):
        return None
    metersets = _metersets(plan)
    rows = []
    for beam in treatment:
        rows += _beam_rows(beam, metersets.get(int(_need(beam.get('BeamNumber')))), ct)
    return rows


def to_table(rows):
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def write_plan_table(table, path):
    pq.write_table(table, path)
