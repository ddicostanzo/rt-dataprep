"""
Photon plan rows -> beam_mu: the MU-weighted sum of every control point's divergent aperture, on the planning-CT grid.

Spec: docs/specs/2026-11-11-beam-volumes.md ("Geometry" is IEC 61217 / DICOM PS3.3 C.8.8.25.6). Pure geometry: no
attenuation, no inverse square, no dependence on CT values. Each segment's MU (`mu` of row i) goes half to control point
i-1 and half to control point i; a control point of weight 0 is skipped.

`beam_mu` raises CaseRefused(REASON_BEAM_VOLUME) for a plan it does not model and nothing else on purpose: no try/except,
header values read with `.get`, one fixed reason (no UID, name or value). The caller decides what a refusal costs.

Speed (docs/specs/2026-11-13-beam-mu-perf.md): along one voxel row every aperture condition is linear in the x index, so
`_add_rows` solves each row's intervals in float64, credits voxels that clear every edge by more than EDGE_MARGIN_MM,
and runs the exact per-voxel test of `_add` only on the edge band. The output is bit-identical to `_add` everywhere.
"""
import itertools
import math

import numpy as np
import SimpleITK as sitk

from rt_dataprep.dicom_io import CaseRefused

REASON_BEAM_VOLUME = 'the photon plan has a value the beam volume does not model; no beam volume written'
TABLE_TOP_ANGLES = ('TableTopPitchAngle', 'TableTopRollAngle', 'TableTopEccentricAngle')
CHUNK_VOXELS = 1 << 21  # voxels per z-chunk: bounds the float32 work arrays
EDGE_MARGIN_MM = 0.01  # at the isocenter plane; float32 error of X, Y is a few 1e-3 mm within the bounds below
FALLBACK_DEPTH_FRACTION = 0.5  # a point with any grid corner nearer the source than this x SAD uses `_add`
MAX_COORD_MM = 5000.0  # ... or with any grid corner this far out in beam coordinates
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


def _rows_ready(point, size):
    """True when `_add_rows` is exact for this point on a grid of `size` (x, y, z): finite values, every voxel well
    downstream and within MAX_COORD_MM (depth and b are linear in the index, so the 8 corners bound them), and a float32
    error bound for X and Y within half of EDGE_MARGIN_MM (`_float32_error`)."""
    jaws, leaves = point['jaws'], point['leaves']
    values = [point['t'], point['a']] + [v for v in jaws.values() if v is not None] + list(leaves[:3] if leaves else [])
    if not all(np.isfinite(np.asarray(v, dtype=float)).all() for v in values):
        return False
    if any((np.abs(np.asarray(v, dtype=float)) > MAX_COORD_MM).any() for v in values[2:]):  # jaw/leaf: no overflow
        return False
    corners = np.array(list(itertools.product(*((0, n - 1) for n in size))), dtype=float)
    b = point['t'] + corners @ point['a'].T
    depth = point['sad'] - b[:, 2]
    if not ((depth >= FALLBACK_DEPTH_FRACTION * point['sad']).all() and (np.abs(b) <= MAX_COORD_MM).all()):
        return False
    return _float32_error(point, size, values[2:], depth.min()) <= EDGE_MARGIN_MM / 2


def _float32_error(point, size, edges, least):
    """First-order bound on how far `_add`'s float32 X or Y can sit from the exact value at any voxel near an edge.

    Each float32 b = ((t + a0 x) + a1 y) + a2 z is off by at most 5u M (M: the largest |partial sum|), depth = sad - b_z
    adds cancellation relative to `least` (the smallest depth), and the jaw is rounded to float32 for the comparison.
    A voxel at distance d >= EDGE_MARGIN_MM from every edge has |X| <= edge + d, so error(edge + d) < d holds for all
    such d once it holds at d = EDGE_MARGIN_MM (the relative part grows with slope < 1).
    """
    u = 2.0 ** -24  # float32 unit roundoff
    sad = point['sad']
    m = np.abs(point['t']) + np.abs(point['a']) @ (np.array(size, dtype=float) - 1)
    edge = max((float(np.abs(np.asarray(v, dtype=float)).max()) for v in edges if np.size(v)), default=0.0)
    relative = (6 * u * m[2] + 2 * u * sad) / least + 3 * u
    return 5 * u * max(m[0], m[1]) * sad / least + (edge + EDGE_MARGIN_MM) * relative + u * edge


def _half(g0, g1, last):
    """(lo, hi) real x index where a linear g >= 0 on [0, last], from its values at the two ends; empty as lo > hi.

    The crossing divides by g0 - g1 only where the signs differ, so a near-zero slope never decides membership.
    """
    split = (g0 >= 0) != (g1 >= 0)
    cross = np.divide(g0 * last, g0 - g1, out=np.zeros(np.shape(g0)), where=split)
    lo = np.where(g0 >= 0, 0.0, np.where(g1 >= 0, cross, np.inf))
    hi = np.where(g1 >= 0, float(last), np.where(g0 >= 0, cross, -np.inf))
    return lo, hi


def _window(ends, sad, last, bounds):
    """Intersection over `bounds` of (axis q, value, upper): X_q >= value, or X_q <= value when upper, as (lo, hi).

    With depth = sad - b_z > 0, X_q >= value is sad*b_q - value*depth >= 0, linear in x along a row.
    """
    lo, hi = np.zeros(ends[0].shape[1]), np.full(ends[0].shape[1], float(last))  # one per row, even with no bounds
    depths = [sad - b[2] for b in ends]
    for q, value, upper in bounds:
        g0, g1 = (sad * b[q] - value * depth for b, depth in zip(ends, depths, strict=True))
        low, high = _half(-g0, -g1, last) if upper else _half(g0, g1, last)
        lo, hi = np.maximum(lo, low), np.minimum(hi, high)
    return lo, hi


def _jaw_bounds(jaws, margin):
    return [(q, value, upper) for q, axis in enumerate('xy') if jaws[axis] is not None
            for value, upper in ((jaws[axis][0] - margin, False), (jaws[axis][1] + margin, True))]


def _runs(rows, starts, ends, nx):
    """Flat chunk indices (int32: a chunk is far below 2**31 voxels) of the inclusive voxel runs [starts, ends] on
    `rows`; empty runs are dropped."""
    keep = starts <= ends
    rows, starts, ends = (v[keep].astype(np.int32) for v in (rows, starts, ends))
    lengths = ends - starts + 1
    return np.repeat(rows * nx + starts - (np.cumsum(lengths) - lengths), lengths) + np.arange(lengths.sum(), dtype=np.int32)


def _exact(point, x, y, z):
    """`_add`'s per-voxel test on scattered voxels (float32 index arrays): the same operations in the same order."""
    f = np.float32
    t, a, sad = point['t'], point['a'], f(point['sad'])
    axis = [(f(t[c]) + f(a[c][0]) * x) + f(a[c][1]) * y + f(a[c][2]) * z for c in range(3)]
    depth = sad - axis[2]
    ahead = depth > 0
    scale = np.divide(sad, depth, out=np.zeros_like(depth), where=ahead)
    u, v = axis[0] * scale, axis[1] * scale
    for coord, name in ((u, 'x'), (v, 'y')):
        if point['jaws'][name] is not None:
            low, high = point['jaws'][name]
            ahead &= (coord >= low) & (coord <= high)
    if point['leaves'] is None:
        return ahead
    boundaries, x1, x2, opening = point['leaves']
    pairs = len(x1)
    hit = np.nonzero(ahead)[0]
    xs, ys = u[hit], v[hit]
    n = np.searchsorted(boundaries, ys, side='right') - 1
    n[(n == pairs) & (ys == boundaries[-1])] = pairs - 1
    inside = (n >= 0) & (n < pairs)
    n = np.clip(n, 0, pairs - 1)
    inside &= opening[n] & (x1[n] <= xs) & (xs <= x2[n])
    lit = np.zeros_like(ahead)
    lit[hit[inside]] = True
    return lit


def _add_rows(chunk, point, z):
    """`_add` for a `_rows_ready` point, bit-identical: per row, voxels inside every edge by more than EDGE_MARGIN_MM
    (and a voxel) get the weight; voxels within EDGE_MARGIN_MM (and a voxel) of an edge run `_exact`; the rest nothing.
    """
    nz, ny, nx = chunk.shape
    last, sad, a = nx - 1, point['sad'], point['a']
    jaws, leaves, eps = point['jaws'], point['leaves'], EDGE_MARGIN_MM
    k, j = np.meshgrid(z.astype(float), np.arange(ny, dtype=float), indexing='ij')
    start = point['t'][:, None] + a[:, 1:] @ np.stack([j.ravel(), k.ravel()])  # b at x = 0, per row
    outer = _window((start, start + a[:, :1] * last), sad, last, _jaw_bounds(jaws, eps))
    rows = np.flatnonzero(outer[0] <= outer[1])
    start, outer = start[:, rows], (outer[0][rows], outer[1][rows])
    inner = _window((start, start + a[:, :1] * last), sad, last, _jaw_bounds(jaws, -eps))
    if leaves is not None:  # one entry per (row, open leaf pair the row's Y range can reach)
        boundaries, x1, x2, opening = leaves
        pairs = len(x1)
        ys = [sad * (start[1] + a[1, 0] * i) / (sad - start[2] - a[2, 0] * i) for i in outer]  # Y is monotone on a row
        first = np.clip(np.searchsorted(boundaries, np.minimum(*ys) - eps, side='left') - 1, 0, pairs - 1)
        count = np.clip(np.searchsorted(boundaries, np.maximum(*ys) + eps, side='right') - 1, 0, pairs - 1) - first + 1
        entry = np.repeat(np.arange(len(rows)), count)
        n = np.repeat(first - (np.cumsum(count) - count), count) + np.arange(count.sum())
        keep = opening[n]
        entry, n = entry[keep], n[keep]
        rows, start = rows[entry], start[:, entry]
        ends = (start, start + a[:, :1] * last)
        windows = []
        for window, margin in ((outer, eps), (inner, -eps)):
            lo, hi = _window(ends, sad, last, [(1, boundaries[n] - margin, False), (1, boundaries[n + 1] + margin, True),
                                               (0, x1[n] - margin, False), (0, x2[n] + margin, True)])
            windows.append((np.maximum(lo, window[0][entry]), np.minimum(hi, window[1][entry])))
        outer, inner = windows

    def voxels(window, pad):  # whole-voxel run of the real interval, widened by `pad` voxels (narrowed when < 0)
        lo, hi = window
        valid = lo <= hi
        first = np.where(valid, np.ceil(np.where(valid, lo, 0.0)) - pad, 0.0)
        final = np.where(valid, np.floor(np.where(valid, hi, 0.0)) + pad, -1.0)
        return np.clip(first, 0, last).astype(np.intp), np.clip(final, -1, last).astype(np.intp)

    outer_s, outer_e = voxels(outer, 1)
    inner_s, inner_e = voxels(inner, -1)
    flat = chunk.reshape(-1)  # a view: chunk is a contiguous slab of out
    weight = np.float32(point['weight'])
    full = inner_s <= inner_e  # solid runs are disjoint: leaf bands and jaw-only rows do not overlap
    np.add.at(flat, _runs(rows, inner_s, inner_e, nx), weight)  # the same float32 add; faster than flat[...] +=
    band = np.concatenate([_runs(rows, outer_s, np.where(full, inner_s - 1, outer_e), nx),
                           _runs(rows, np.where(full, inner_e + 1, outer_e + 1), outer_e, nx)])
    order = np.argsort(rows[full] * nx + inner_s[full])  # drop band voxels another entry's solid run already holds
    firsts, lasts = (np.concatenate([[-1], (rows[full] * nx + v[full])[order]]) for v in (inner_s, inner_e))  # -1: none
    band = band[band > lasts[np.searchsorted(firsts, band, side='right') - 1]]
    f = np.float32
    lit = band[_exact(point, (band % nx).astype(f), (band // nx % ny).astype(f), (z[0] + band // (ny * nx)).astype(f))]
    flat[lit] += weight  # adjacent pairs' bands can repeat a voxel; a fancy-index += still adds it once


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
    ready = [_rows_ready(point, (nx, ny, nz)) for point in points]
    for start in range(0, nz, step):
        z = np.arange(start, min(start + step, nz), dtype=np.float32)
        chunk = out[start:start + len(z)]  # a view: the points add into it
        for point, rows_ready in zip(points, ready, strict=True):  # in point order: float32 sums are order-sensitive
            if rows_ready:
                _add_rows(chunk, point, z)
            else:
                _add(chunk, point, x, y, z)
    image = sitk.GetImageFromArray(out)
    image.CopyInformation(ct)
    return image
