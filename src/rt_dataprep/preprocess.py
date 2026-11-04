"""
Legacy RNT preprocessing of one converted plan: the CT pipeline and the resampling of dose and masks (spec: f2b-preprocess).

The CT runs initial resample -> rescale -> final spacing -> reshape, each resample linear with default value 0. The grid
and the crop/pad follow torchio 1.2.1 (floor size, centre-preserving origin, ceil(d/2) at the start of the axis). The final
CT image is the one reference grid: dose and masks are each resampled once onto it. Direction is never re-derived.
Every compute call in the three process_* entry points runs inside dicom_io.quiet_call; a failure is one fixed REASON_.
"""
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from rt_dataprep import convert as conv
from rt_dataprep.dicom_io import quiet_call

REASON_FILE_UNREADABLE = 'an input NIfTI file is unreadable'
REASON_INTERNAL = 'an unexpected error stopped the preprocessing; the report lists what was finished'


@dataclass(frozen=True)
class PreprocessSteps:
    resample: bool = False  # initial resample to new_spacing, linear
    rescale: bool = False  # CT only: (HU + 1024) / 4096
    harmonize_spacing: bool = False  # final resample to final_spacing, linear
    reshape: bool = False  # centred crop/pad to new_shape
    legacy_linear_masks: bool = False
    new_spacing: tuple = (1, 1, 1)  # mm, (x, y, z)
    final_spacing: tuple = (1.25, 1.25, 1.25)
    new_shape: tuple = (701, 701, 433)  # voxels, (x, y, z)


LEGACY_RNT = PreprocessSteps(resample=True, rescale=True, harmonize_spacing=True, reshape=True)


def read_image(path):
    """The NIfTI at `path`; any failure raises CaseRefused(REASON_FILE_UNREADABLE) (ITK names the path)."""
    return quiet_call(lambda: sitk.ReadImage(str(path)), REASON_FILE_UNREADABLE)


def reference_grid(image, spacing):
    """(size, origin, spacing, direction) of `image` resampled to `spacing`, as torchio 1.2.1 Resample builds it."""
    old_size = np.array(image.GetSize())
    old_spacing = np.array(image.GetSpacing())
    new_spacing = np.array(spacing, float)
    size = np.floor(old_size * old_spacing / new_spacing).astype(int)  # float64 on GetSpacing() values, as torchio
    size[old_size == 1] = 1
    direction = np.array(image.GetDirection()).reshape(3, 3)
    centre = np.array(image.GetOrigin()) + direction @ ((old_size - 1) * old_spacing / 2)
    origin = centre - direction @ ((size - 1) / 2 * new_spacing)
    return [int(n) for n in size], tuple(float(v) for v in origin), tuple(float(v) for v in new_spacing), image.GetDirection()


def resample_to(image, spacing):
    """`image` linearly resampled onto reference_grid(image, spacing), default value 0."""
    size, origin, new_spacing, direction = reference_grid(image, spacing)
    return sitk.Resample(image, size, sitk.Transform(), sitk.sitkLinear, origin, new_spacing, direction, 0.0, image.GetPixelID())


def rescale(image):
    """(HU + 1024) / 4096 as float32."""
    return sitk.ShiftScale(sitk.Cast(image, sitk.sitkFloat32), 1024.0, 1 / 4096)  # arithmetic with a float would promote to float64


def crop_or_pad(image, shape, pad_value):
    """`image` cropped or padded with `pad_value` to `shape` (x, y, z): ceil(d/2) voxels at the start of an axis, floor(d/2) at the end."""
    diff = np.array(shape) - np.array(image.GetSize())
    start, end = np.ceil(np.abs(diff) / 2).astype(int), np.floor(np.abs(diff) / 2).astype(int)
    crop = [(int(a), int(b)) if d < 0 else (0, 0) for d, a, b in zip(diff, start, end, strict=True)]
    pad = [(int(a), int(b)) if d > 0 else (0, 0) for d, a, b in zip(diff, start, end, strict=True)]
    if any(crop[0] + crop[1] + crop[2]):
        image = sitk.Crop(image, [c[0] for c in crop], [c[1] for c in crop])
    if any(pad[0] + pad[1] + pad[2]):
        image = sitk.ConstantPad(image, [p[0] for p in pad], [p[1] for p in pad], float(pad_value))
    return image


def process_ct(ct, steps):
    """The CT through the steps that are on, in legacy order. The result is the reference grid for dose and masks."""
    def run():
        image = ct
        if steps.resample:
            image = resample_to(image, steps.new_spacing)
        if steps.rescale:
            image = rescale(image)
        if steps.harmonize_spacing:
            image = resample_to(image, steps.final_spacing)
        if steps.reshape:
            low = sitk.MinimumMaximumImageFilter()
            low.Execute(image)  # the pad value is the minimum of the CT entering reshape
            image = crop_or_pad(image, steps.new_shape, low.GetMinimum())
        return image
    return quiet_call(run, REASON_INTERNAL)


def process_dose(dose, final_ct):
    """`dose` (Gy) linearly resampled once onto final_ct as float32; 0 outside the dose grid."""
    return quiet_call(lambda: conv.resample_dose(dose, final_ct), REASON_INTERNAL)


def process_mask(mask, final_ct, steps):
    """`mask` resampled once onto final_ct: nearest as uint8, or linear as float32 under legacy_linear_masks; 0 outside."""
    if steps.legacy_linear_masks:  # float32 is explicit: the native mask is uint8 and would otherwise drop the fractions
        interpolator, pixel = sitk.sitkLinear, sitk.sitkFloat32
    else:
        interpolator, pixel = sitk.sitkNearestNeighbor, sitk.sitkUInt8
    return quiet_call(lambda: sitk.Resample(mask, final_ct, sitk.Transform(), interpolator, 0, pixel), REASON_INTERNAL)


def output_name(name, steps, ct=False):
    """`name` with the prefixes of the steps that are on, newest first; dose and masks (ct=False) get the geometric ones only."""
    prefixes = [('reshape', steps.reshape), ('resample', steps.harmonize_spacing),
                ('rescale', ct and steps.rescale), ('initialresample', steps.resample)]
    return ''.join(f'{prefix}_' for prefix, on in prefixes if on) + name
