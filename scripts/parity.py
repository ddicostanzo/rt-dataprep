"""
Parity of a preprocessed tree against rnt's legacy pooled folders (phase1.md, "Parity check on real data").

    uv run python scripts/parity.py PREPROCESSED POOLED [--json OUT.json]

A pooled folder <mrn>_<x>_<fx> (rnt's per_fraction_dataframe naming) pairs with PREPROCESSED/<mrn>/plan_<fx>/; the
new files are found through that folder's series.csv. Per pair: the CT (ct.nii.gz) by geometry, max abs difference
and Pearson r; each legacy structure mask (<structure>.nii.gz) against the new mask of the same format_name, by Dice
after thresholding both at 0.5; dose_gy.nii.gz against the new dose when the grids are equal, as information only (the
new dose is resampled twice, legacy once, so no tolerance applies). Prints a JSON summary of counts and distributions only: no folder
name, path, UID or exception text.

When a legacy file's grid differs from the new one (legacy crops per patient, and its 1.25 mm grid is offset by a fraction
of a voxel), the new image is resampled linearly onto the legacy grid and compared over the voxels both cover. Those
results carry a 'resampled_' prefix and are information only: no pass/fail.
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
    for root, dirs, files in os.walk(pooled, followlinks=True):  # a pooled root may be symlinks to train_val and test
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
    return sitk.GetArrayFromImage(image).astype(np.float32, copy=False)  # a copy: a view dies with a temporary image


def on_legacy_grid(old, new):
    """(legacy values, new values, same grid) as 1-D float32 arrays; on another grid, the new image resampled linearly
    onto the legacy grid, over the legacy voxels the new image covers."""
    if geometry_equal(old, new):
        return array(old).ravel(), array(new).ravel(), True
    moved = sitk.Resample(new, old, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    ones = sitk.Image(new.GetSize(), sitk.sitkUInt8)
    ones.CopyInformation(new)
    inside = sitk.GetArrayFromImage(sitk.Resample(ones + 1, old, sitk.Transform(), sitk.sitkNearestNeighbor, 0)) > 0
    return array(old)[inside], sitk.GetArrayFromImage(moved)[inside], False


def pearson(a, b):
    a, b = a.ravel()[::8].astype(np.float64), b.ravel()[::8].astype(np.float64)  # 1/8: memory
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
    a, b, same = on_legacy_grid(old_ct, new_ct)
    tally.counts['ct_geometry_equal' if same else 'ct_geometry_differs'] += 1
    p = '' if same else 'resampled_'
    if not same:
        tally.size_delta[tuple(int(n) - int(o) for n, o in zip(new_ct.GetSize(), old_ct.GetSize(), strict=True))] += 1
    if a.size == 0:
        tally.counts['ct_no_overlap'] += 1
        return None
    diff = np.abs(a - b)
    tally.values[f'{p}ct_max_abs_diff'].append(float(diff.max()))
    tally.values[f'{p}ct_p99_abs_diff'].append(float(np.percentile(diff, 99)))
    r = pearson(a, b)
    if r is not None:
        tally.values[f'{p}ct_pearson_r'].append(r)
    if same:
        tally.counts['ct_pass' if diff.max() <= CT_TOL else 'ct_fail'] += 1
    else:
        tally.counts['ct_resampled'] += 1
    del a, b, diff
    for s in STRUCTURES:
        path = old / f'{s}.nii.gz'
        if not path.is_file():
            continue
        c = tally.masks[s]
        c['legacy'] += 1
        if s not in files['mask']:
            c['no_new_mask'] += 1
            continue
        a, b, same = on_legacy_grid(sitk.ReadImage(str(path)), sitk.ReadImage(str(files['mask'][s])))
        d = dice(a, b)
        p = '' if same else 'resampled_'
        tally.values[f'{p}dice_{s}'].append(d)
        c[('pass' if d >= DICE_MIN else 'fail') if same else 'resampled'] += 1  # resampled: information only
    if (old / 'dose_gy.nii.gz').is_file() and 'dose' in files:
        a, b, same = on_legacy_grid(sitk.ReadImage(str(old / 'dose_gy.nii.gz')), sitk.ReadImage(str(files['dose'])))
        p = '' if same else 'resampled_'
        if a.size:
            tally.values[f'{p}dose_max_abs_diff_gy'].append(float(np.abs(a - b).max()))
            tally.values[f'{p}dose_legacy_max_gy'].append(float(a.max()))
            r = pearson(a, b)
            if r is not None:
                tally.values[f'{p}dose_pearson_r'].append(r)
        tally.counts[f'{p}dose_compared'] += 1
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
