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
        _check(isinstance(value, (int, float)) and math.isfinite(value) and value > 0)
    return sads


def _point(row, sad, weight, position, grid):
    """What the chunk loop needs of one control point: linear map of the voxel index to the beam frame, and the aperture."""
    mlc = row['mlc_type']
    jaws = {axis: None if row[f'jaw_{axis}1'] is None or row[f'jaw_{axis}2'] is None
            else (row[f'jaw_{axis}1'], row[f'jaw_{axis}2']) for axis in 'xy'}
    _check(not row['number_of_blocks'] > 0)
    _check(mlc in (None, 'MLCX'))
    _check(mlc is None or (row['leaf_position_boundaries'] is not None and row['mlc_positions'] is not None))
    _check(all(jaws[axis] is not None or mlc is not None for axis in 'xy'))  # an unbounded axis is not modelled
    leaves = None
    if mlc:
        pairs = len(row['leaf_position_boundaries']) - 1
        x1, x2 = np.array(row['mlc_positions'][:pairs]), np.array(row['mlc_positions'][pairs:])
        leaves = (np.array(row['leaf_position_boundaries']), x1, x2, x1 < x2)  # X1 >= X2 is a closed pair
    # b = M (p - c), p = origin + D diag(spacing) index  =>  b = t + A index
    to_beam = _rz(row['collimator_angle']).T @ _ry(row['gantry_angle']).T @ _rz(row['couch_angle']) @ position
    isocenter = np.array([row['isocenter_x'], row['isocenter_y'], row['isocenter_z']])
    origin, direction, spacing = grid
    return dict(sad=float(sad), weight=weight, jaws=jaws, leaves=leaves, t=to_beam @ (origin - isocenter),
                a=to_beam @ direction * spacing)


def _points(plan, rows, patient_position, grid):
    _check(isinstance(patient_position, str) and patient_position in POSITIONS)
    position = np.array(POSITIONS[patient_position], dtype=float)
    sads = _sads(plan)
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
