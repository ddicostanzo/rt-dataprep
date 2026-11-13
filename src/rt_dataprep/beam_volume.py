"""
Photon plan rows -> beam_mu: the MU-weighted sum of every control point's divergent aperture, on the planning-CT grid.

Spec: docs/specs/2026-11-11-beam-volumes.md ("Geometry" is IEC 61217 / DICOM PS3.3 C.8.8.25.6). Pure geometry: no
attenuation, no inverse square, no dependence on CT values. Each segment's MU (`mu` of row i) goes half to control point
i-1 and half to control point i; a control point of weight 0 is skipped.

`beam_mu` raises CaseRefused(REASON_BEAM_VOLUME) for a plan it does not model and nothing else on purpose: no try/except,
header values read with `.get`, one fixed reason (no UID, name or value). The caller decides what a refusal costs.
"""
import itertools
import math

import numpy as np
import SimpleITK as sitk

from rt_dataprep.dicom_io import CaseRefused

REASON_BEAM_VOLUME = 'the photon plan has a value the beam volume does not model; no beam volume written'
TABLE_TOP_ANGLES = ('TableTopPitchAngle', 'TableTopRollAngle', 'TableTopEccentricAngle')
CHUNK_VOXELS = 1 << 21  # voxels per z-chunk: bounds the float32 work arrays
# Patient-position map P of spec step 1: DICOM patient d -> IEC patient-support frame, as a matrix.
POSITIONS = {'HFS': ((1, 0, 0), (0, 0, 1), (0, -1, 0)), 'HFP': ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
             'FFS': ((-1, 0, 0), (0, 0, -1), (0, -1, 0)), 'FFP': ((1, 0, 0), (0, 0, -1), (0, 1, 0))}


def _refuse():
    raise CaseRefused(REASON_BEAM_VOLUME)


def _check(condition):
    if not condition:
        _refuse()


def _rz(degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _sads(plan):
    """SourceAxisDistance per BeamNumber over the treatment beams (the plan_rows filter); each must be finite, > 0."""
    beams = plan.get('BeamSequence') or []
    sads = {int(b.get('BeamNumber')): b.get('SourceAxisDistance') for b in beams if b.get('TreatmentDeliveryType') != 'SETUP'}
    for value in sads.values():
        _check(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0)
    return sads


def _table_top_flat(plan):
    """Refuse a non-zero table-top pitch, roll or eccentric angle on any treatment control point (6-DoF couch, not modelled).

    Checking every control point that lists one covers the filled-forward value too; absent or empty counts as 0.
    """
    for beam in plan.get('BeamSequence') or []:
        if beam.get('TreatmentDeliveryType') == 'SETUP':
            continue
        for point in beam.get('ControlPointSequence') or []:
            for keyword in TABLE_TOP_ANGLES:
                value = point.get(keyword)
                _check(value in (None, '') or (isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0))


def aperture(row):
    """
    `(jaws, leaves)` of one plan_rows row, or None when the aperture is not modelled; never raises. `jaws` maps 'x'/'y' to a
    (low, high) pair or None; `leaves` is None without an MLC, else (boundaries, x1, x2, opening) as numpy arrays.
    Not modelled: blocks, a non-MLCX MLC, null, fewer than two or non-strictly-increasing leaf boundaries, an axis bounded by neither.
    """
    mlc = row['mlc_type']
    jaws = {axis: None if row[f'jaw_{axis}1'] is None or row[f'jaw_{axis}2'] is None
            else (row[f'jaw_{axis}1'], row[f'jaw_{axis}2']) for axis in 'xy'}
    if row['number_of_blocks'] > 0 or mlc not in (None, 'MLCX'):
        return None
    if not all(jaws[axis] is not None or mlc is not None for axis in 'xy'):  # an unbounded axis is not modelled
        return None
    if mlc is None:
        return jaws, None
    if row['leaf_position_boundaries'] is None or row['mlc_positions'] is None:
        return None
    boundaries = np.array(row['leaf_position_boundaries'])
    if len(boundaries) < 2 or not np.all(np.diff(boundaries) > 0):  # else leaves land on the wrong pair
        return None
    pairs = len(boundaries) - 1
    x1, x2 = np.array(row['mlc_positions'][:pairs]), np.array(row['mlc_positions'][pairs:])
    return jaws, (boundaries, x1, x2, x1 < x2)  # X1 >= X2 is a closed pair


def _point(row, sad, weight, position, grid):
    """What the chunk loop needs of one control point: linear map of the voxel index to the beam frame, and the aperture."""
    modelled = aperture(row)
    _check(modelled is not None)
    jaws, leaves = modelled
    # b = M (p - c), p = origin + D diag(spacing) index  =>  b = t + A index
    to_beam = _rz(row['collimator_angle']).T @ _ry(row['gantry_angle']).T @ _rz(row['couch_angle']) @ position
    isocenter = np.array([row['isocenter_x'], row['isocenter_y'], row['isocenter_z']])
    origin, direction, spacing = grid
    return dict(sad=float(sad), weight=weight, jaws=jaws, leaves=leaves, t=to_beam @ (origin - isocenter),
                a=to_beam @ direction * spacing)


def _points(plan, rows, patient_position, grid):
    _check(isinstance(patient_position, str) and patient_position in POSITIONS)
    position = np.array(POSITIONS[patient_position], dtype=float)
    # A stated setup position other than the CT's would mirror the volume (#288). shortcut: checks every setup item, not
    # only those the treatment beams reference; refusing loses only beam_mu. Narrow it if a real plan is refused for this.
    # An item with no PatientPosition but a PatientAdditionalPosition states a position that cannot be checked (#335).
    _check(all(s.get('PatientPosition') == patient_position
               or (s.get('PatientPosition') in ('', None) and not s.get('PatientAdditionalPosition'))
               for s in plan.get('PatientSetupSequence') or []))
    sads = _sads(plan)
    _table_top_flat(plan)
    points = []
    for _, group in itertools.groupby(rows, key=lambda r: r['beam_number']):
        group = list(group)
        _check(group[0]['beam_number'] in sads)
        for i, row in enumerate(group):
            weight = 0.5 * row['mu'] + (0.5 * group[i + 1]['mu'] if i + 1 < len(group) else 0.0)
            point = _point(row, sads[row['beam_number']], weight, position, grid)
            if weight != 0:
                points.append(point)
    return points


def _axis(t, a, x, y, z):
    """float32 t + a . (i, j, k) over the (z, y, x) block: the in-plane part once, then one add per slice."""
    f = np.float32
    plane = (f(t) + f(a[0]) * x)[None, :] + (f(a[1]) * y)[:, None]
    return plane[None] + (f(a[2]) * z)[:, None, None]


def _add(chunk, point, x, y, z):
    sad = np.float32(point['sad'])
    depth = sad - _axis(point['t'][2], point['a'][2], x, y, z)  # SAD - b_z
    ahead = depth > 0  # b_z >= SAD (at or beyond the source) gets nothing
    scale = np.divide(sad, depth, out=np.zeros_like(depth), where=ahead)
    u = _axis(point['t'][0], point['a'][0], x, y, z) * scale  # X, Y at the isocenter plane
    v = _axis(point['t'][1], point['a'][1], x, y, z) * scale
    for coord, axis in ((u, 'x'), (v, 'y')):
        if point['jaws'][axis] is not None:
            low, high = point['jaws'][axis]
            ahead &= (coord >= low) & (coord <= high)
    weight = np.float32(point['weight'])
    if point['leaves'] is None:
        np.add(chunk, weight, out=chunk, where=ahead)
        return
    boundaries, x1, x2, opening = point['leaves']
    pairs = len(x1)
    hit = np.nonzero(ahead)  # the leaf lookup only for voxels already inside the jaws
    xs, ys = u[hit], v[hit]
    n = np.searchsorted(boundaries, ys, side='right') - 1
    n[(n == pairs) & (ys == boundaries[-1])] = pairs - 1  # the last pair is closed at its far boundary
    inside = (n >= 0) & (n < pairs)
    n = np.clip(n, 0, pairs - 1)
    inside &= opening[n] & (x1[n] <= xs) & (xs <= x2[n])
    chunk[tuple(h[inside] for h in hit)] += weight


def beam_mu(plan, rows, ct, patient_position):
    """
    The float32 SimpleITK image (geometry of `ct`) of MU per fraction: for each control point in `rows` (plan_table.plan_rows
    of `plan`), its weight in every voxel inside its divergent jaw/MLC aperture. `patient_position` is the CT's
    PatientPosition. Raises CaseRefused(REASON_BEAM_VOLUME) for anything the spec's "not modelled" list names.
    """
    grid = (np.array(ct.GetOrigin()), np.array(ct.GetDirection()).reshape(3, 3), np.array(ct.GetSpacing()))
    points = _points(plan, rows, patient_position, grid)
    nx, ny, nz = ct.GetSize()
    out = np.zeros((nz, ny, nx), dtype=np.float32)
    x, y = np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32)
    step = max(1, CHUNK_VOXELS // (nx * ny))
    for start in range(0, nz, step):
        z = np.arange(start, min(start + step, nz), dtype=np.float32)
        chunk = out[start:start + len(z)]  # a view: the points add into it
        for point in points:
            _add(chunk, point, x, y, z)
    image = sitk.GetImageFromArray(out)
    image.CopyInformation(ct)
    return image
