"""
Frozen copy of the 324e2b2 beam_mu algorithm (per-voxel _axis/_add and the z-chunk loop), the oracle that the
rewritten beam_volume.beam_mu must match bit for bit (spec 2026-11-13-beam-mu-perf, AC1/AC3). Do not edit it, and do not
import rt_dataprep.beam_volume here. It reads only the point keys sad, weight, jaws, leaves, t, a.
"""
import numpy as np

CHUNK_VOXELS = 1 << 21  # voxels per z-chunk, as in 324e2b2


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


def reference_beam_mu(points, ct):
    """The (nz, ny, nx) float32 array 324e2b2's beam_mu computes from `points` (beam_volume._points) on SimpleITK image `ct`."""
    nx, ny, nz = ct.GetSize()
    out = np.zeros((nz, ny, nx), dtype=np.float32)
    x, y = np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32)
    step = max(1, CHUNK_VOXELS // (nx * ny))
    for start in range(0, nz, step):
        z = np.arange(start, min(start + step, nz), dtype=np.float32)
        chunk = out[start:start + len(z)]  # a view: the points add into it
        for point in points:
            _add(chunk, point, x, y, z)
    return out
