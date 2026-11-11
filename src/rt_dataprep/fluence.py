"""
Photon plan rows -> fluence_mu: one 400 x 400 MU map per treatment beam, 1 mm pixels, in the collimator (BEV) frame.

Spec: docs/specs/2026-11-11-fluence-maps.md. Pure geometry: each control point's open region is a union of rectangles (one
per open leaf pair intersected with the jaws, or the jaw rectangle alone), and a pixel holds weight x covered area (mm2).
Weights are beam_mu's half-segment rule. Pixel j covers X in [j - 200, j - 199] mm, row i covers Y in [i - 200, i - 199].

`fluence_mu` raises CaseRefused(REASON_FLUENCE) for a plan it does not model and nothing else on purpose: no try/except,
header values read with `.get`, one fixed reason (no UID, name or value). The caller decides what a refusal costs.
"""
import itertools

import numpy as np
import SimpleITK as sitk

from rt_dataprep.beam_volume import aperture
from rt_dataprep.dicom_io import CaseRefused

REASON_FLUENCE = 'the photon plan has a value the fluence map does not model; no fluence map written'
SIZE = 400
EDGES = np.arange(-SIZE // 2, SIZE // 2 + 1, dtype=np.float64)  # pixel edges in mm, 401 of them
LIMIT = SIZE / 2


def _refuse():
    raise CaseRefused(REASON_FLUENCE)


def _rectangles(jaws, leaves):
    """(xlo, xhi, ylo, yhi) arrays of the open rectangles: jaws intersected, non-positive sizes dropped."""
    jx, jy = jaws['x'] or (-np.inf, np.inf), jaws['y'] or (-np.inf, np.inf)
    if leaves is None:
        xlo, xhi, ylo, yhi = (np.array([v], dtype=np.float64) for v in (jx[0], jx[1], jy[0], jy[1]))
    else:
        boundaries, x1, x2, opening = leaves
        xlo, xhi = np.maximum(x1, jx[0]), np.minimum(x2, jx[1])
        ylo, yhi = np.maximum(boundaries[:-1], jy[0]), np.minimum(boundaries[1:], jy[1])
        xlo, xhi, ylo, yhi = (a[opening] for a in (xlo, xhi, ylo, yhi))
    keep = (xhi - xlo > 0) & (yhi - ylo > 0)  # also drops NaN
    return xlo[keep], xhi[keep], ylo[keep], yhi[keep]


def _overlap(low, high):
    """(rectangles, 400) overlap in mm of each [low, high] with each pixel."""
    return np.maximum(0.0, np.minimum(high[:, None], EDGES[None, 1:]) - np.maximum(low[:, None], EDGES[None, :-1]))


def _add(image, weight, jaws, leaves):
    xlo, xhi, ylo, yhi = _rectangles(jaws, leaves)
    if xlo.size and (min(xlo.min(), ylo.min()) < -LIMIT or max(xhi.max(), yhi.max()) > LIMIT):
        _refuse()  # refuse, never clip
    image += weight * _overlap(ylo, yhi).T @ _overlap(xlo, xhi)


def fluence_mu(plan, rows):
    """
    `(image, beam_numbers)`: the float32 SimpleITK image (beams, 400, 400) of MU per fraction, one slice per treatment beam in
    `rows` order (plan_table.plan_rows of `plan`), and those beams' BeamNumbers as a list of ints. Raises
    CaseRefused(REASON_FLUENCE) for anything the spec's Refusals list names.
    """
    numbers = [int(b.get('BeamNumber')) for b in plan.get('BeamSequence') or [] if b.get('TreatmentDeliveryType') != 'SETUP']
    if len(set(numbers)) != len(numbers):
        _refuse()
    slices, beam_numbers = [], []
    for number, group in itertools.groupby(rows, key=lambda r: r['beam_number']):
        group = list(group)
        out = np.zeros((SIZE, SIZE), dtype=np.float64)
        for i, row in enumerate(group):
            modelled = aperture(row)
            if modelled is None:
                _refuse()
            weight = 0.5 * row['mu'] + (0.5 * group[i + 1]['mu'] if i + 1 < len(group) else 0.0)
            if weight != 0:
                _add(out, weight, *modelled)
        slices.append(out)
        beam_numbers.append(int(number))
    image = sitk.GetImageFromArray(np.array(slices, dtype=np.float32).reshape(len(slices), SIZE, SIZE))
    image.SetOrigin((-SIZE / 2 + 0.5, -SIZE / 2 + 0.5, 0.0))
    return image, beam_numbers
