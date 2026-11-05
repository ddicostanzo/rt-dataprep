"""
Parity of a preprocessed tree against rnt's legacy pooled folders (phase1.md, "Parity check on real data").

    uv run python scripts/parity.py PREPROCESSED POOLED [--json OUT.json]

A pooled folder <mrn>_<x>_<fx> (rnt's per_fraction_dataframe naming) pairs with PREPROCESSED/<mrn>/plan_<fx>/; the
new files are found through that folder's series.csv. Per pair: the CT (ct.nii.gz) by geometry, max abs difference
and Pearson r; each legacy structure mask (<structure>.nii.gz) against the new mask of the same format_name, by Dice
after thresholding both at 0.5; dose_gy.nii.gz against the new dose when the grids are equal, as information only (the
new dose is resampled twice, legacy once, so no tolerance applies). Prints a JSON summary of counts and distributions only: no folder
name, path, UID or exception text.
"""
import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from rt_dataprep.masks import format_name

STRUCTURES = ('hrctv', 'bladder', 'rectum', 'sigmoid', 'bowel', 'body')
CT_TOL, DICE_MIN = 1e-3, 0.98
SPACING_TOL, ORIGIN_TOL_MM = 1e-4, 1e-3

REASON_NEW_MISSING = 'no new plan folder with a series.csv for the pooled folder'
REASON_SHARED = 'two pooled folders map to the same new plan folder'
REASON_NO_CT = 'the pooled or the new CT is missing'
REASON_UNREADABLE = 'a file of the pair is unreadable or malformed'


def pooled_folders(pooled):
    """[(mrn, fx, folder)] for every <mrn>_<x>_<fx> folder holding files, never descending into one (the rnt rule).
    A three-part name without files (the root, a grouping folder) is descended into."""
    out = []
    for root, dirs, files in os.walk(pooled):
        leaf = Path(root).name.split('_')
        if len(leaf) == 3 and files:
            dirs[:] = []
            out.append((leaf[0], leaf[2], Path(root)))
    return sorted(out, key=lambda t: str(t[2]))


def new_files(folder):
    """{'ct': path, 'dose': path, 'mask': {format_name: path}} from series.csv."""
    found = {'mask': {}}
    with open(folder / 'series.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            path = folder / row['file']
            if row['kind'] in ('ct', 'dose'):
                found[row['kind']] = path
            elif row['kind'] == 'mask':
                found['mask'][format_name(row['roi_name'])] = path
    return found


def geometry(image):
    return image.GetSize(), np.array(image.GetSpacing()), np.array(image.GetOrigin()), np.array(image.GetDirection())


def geometry_equal(a, b):
    (sa, pa, oa, da), (sb, pb, ob, db) = geometry(a), geometry(b)
    return (sa == sb and np.allclose(pa, pb, atol=SPACING_TOL, rtol=0) and np.allclose(oa, ob, atol=ORIGIN_TOL_MM, rtol=0)
            and np.allclose(da, db, atol=SPACING_TOL, rtol=0))


def array(image):
    return sitk.GetArrayViewFromImage(image).astype(np.float32, copy=False)


def pearson(a, b):
    a, b = a[::2, ::2, ::2].ravel().astype(np.float64), b[::2, ::2, ::2].ravel().astype(np.float64)  # 1/8: memory
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def dice(a, b):
    a, b = a >= 0.5, b >= 0.5
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else 2.0 * int(np.logical_and(a, b).sum()) / total


class Tally:
    def __init__(self):
        self.counts = collections.Counter()
        self.refused = collections.Counter()
        self.size_delta = collections.Counter()  # new size - legacy size per axis (x, y, z), for #100
        self.values = collections.defaultdict(list)
        self.masks = {s: collections.Counter() for s in STRUCTURES}

    def merge(self, other):
        self.counts.update(other.counts)
        self.size_delta.update(other.size_delta)
        for k, v in other.values.items():
            self.values[k].extend(v)
        for s, c in other.masks.items():
            self.masks[s].update(c)

    def summary(self):
        def dist(v):
            return {'n': len(v), 'min': min(v), 'median': float(np.median(v)), 'max': max(v)} if v else {'n': 0}
        return {'counts': dict(self.counts), 'refused': dict(self.refused),
                'ct_size_delta_xyz': {str(k): n for k, n in sorted(self.size_delta.items())},
                'masks': {s: dict(c) for s, c in self.masks.items() if c},
                'distributions': {k: dist(v) for k, v in sorted(self.values.items())},
                'tolerances': {'ct_max_abs': CT_TOL, 'dice_min': DICE_MIN}}


def compare_pair(old, new, tally):
    """Compares one pooled folder with its new plan folder into tally (a fresh one per pair); returns a refusal reason or None."""
    files = new_files(new)
    if 'ct' not in files or not (old / 'ct.nii.gz').is_file():
        return REASON_NO_CT
    old_ct, new_ct = sitk.ReadImage(str(old / 'ct.nii.gz')), sitk.ReadImage(str(files['ct']))
    same = geometry_equal(old_ct, new_ct)
    tally.counts['ct_geometry_equal' if same else 'ct_geometry_differs'] += 1
    if not same:
        tally.size_delta[tuple(int(n) - int(o) for n, o in zip(new_ct.GetSize(), old_ct.GetSize(), strict=True))] += 1
        return None  # no voxel comparison on different grids; the masks and dose share the CT grid
    a, b = array(old_ct), array(new_ct)
    diff = float(np.abs(a - b).max())
    tally.values['ct_max_abs_diff'].append(diff)
    r = pearson(a, b)
    if r is not None:
        tally.values['ct_pearson_r'].append(r)
    tally.counts['ct_pass' if diff <= CT_TOL else 'ct_fail'] += 1
    del a, b
    for s in STRUCTURES:
        path = old / f'{s}.nii.gz'
        if not path.is_file():
            continue
        c = tally.masks[s]
        c['legacy'] += 1
        if s not in files['mask']:
            c['no_new_mask'] += 1
            continue
        old_m, new_m = sitk.ReadImage(str(path)), sitk.ReadImage(str(files['mask'][s]))
        if not geometry_equal(old_m, new_m):
            c['grid_differs'] += 1
            continue
        d = dice(array(old_m), array(new_m))
        tally.values[f'dice_{s}'].append(d)
        c['pass' if d >= DICE_MIN else 'fail'] += 1
    if (old / 'dose_gy.nii.gz').is_file() and 'dose' in files:
        old_d, new_d = sitk.ReadImage(str(old / 'dose_gy.nii.gz')), sitk.ReadImage(str(files['dose']))
        if geometry_equal(old_d, new_d):
            a, b = array(old_d), array(new_d)
            diff = float(np.abs(a - b).max())
            tally.values['dose_max_abs_diff_gy'].append(diff)
            tally.values['dose_legacy_max_gy'].append(float(a.max()))
            r = pearson(a, b)
            if r is not None:
                tally.values['dose_pearson_r'].append(r)
            tally.counts['dose_compared'] += 1
        else:
            tally.counts['dose_grid_differs'] += 1
    else:
        tally.counts['dose_missing'] += 1
    return None


def run(preprocessed, pooled):
    tally = Tally()
    pairs = pooled_folders(pooled)
    tally.counts['pooled_folders'] = len(pairs)
    shared = collections.Counter((mrn, fx) for mrn, fx, _ in pairs)
    for mrn, fx, old in pairs:
        new = Path(preprocessed) / mrn / f'plan_{fx}'
        pair = Tally()
        try:
            if shared[(mrn, fx)] > 1:
                reason = REASON_SHARED
            elif not (new / 'series.csv').is_file():
                reason = REASON_NEW_MISSING
            else:
                reason = compare_pair(old, new, pair)
        except Exception:  # the message could name a path; count it under a fixed reason
            reason = REASON_UNREADABLE
        if reason:
            tally.refused[reason] += 1
        else:
            tally.merge(pair)  # only whole pairs, so every count covers the same pairs
            tally.counts['compared'] += 1
    return tally.summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('preprocessed', help='the preprocessed root: <patient>/plan_<n>/series.csv')
    parser.add_argument('pooled', help="rnt's pooled root of <mrn>_<x>_<fx> folders")
    parser.add_argument('--json', help='also write the summary to this file')
    args = parser.parse_args(argv)
    sitk.ProcessObject.SetGlobalWarningDisplay(False)  # an ITK warning would name the file
    summary = json.dumps(run(args.preprocessed, args.pooled), indent=2)
    print(summary)
    if args.json:
        Path(args.json).write_text(summary + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
