"""
Preprocessing of a converted tree, one opt-in step flag per step (spec: f2b-preprocess): the CT pipeline, the resampling of dose and masks, and the walk.

The CT runs initial resample -> rescale -> ahe -> final spacing -> reshape, each resample linear with default value 0. The ahe step
(opt-in) min-max scales the volume in float32 to [0, 1] and equalizes it with skimage's CLAHE (default kernel, 128 bins, clip 0.05). The grid
and the crop/pad follow torchio 1.2.1 (floor size, centre-preserving origin, ceil(d/2) at the start of the axis). The final
CT image is the one reference grid: dose and masks are each resampled once onto it. Direction is never re-derived.
Every compute call in the three process_* entry points runs inside dicom_io.quiet_call; a failure is one fixed REASON_.

preprocess() reads each plan's series.json (kinds ct, dose, mask, applicator, line, plan, beam, fluence, ion_plan, dwell, and the registered series, each resampled once onto the final CT) and each study folder's
(<patient>/study_<n>/, one case per series: a CT through every step that is on, an MR or PT through all but the HU rescale),
and writes a separate tree plus preprocess_report.csv in the output root. Log lines and exceptions carry fixed REASON_ constants only; the report
is the one place the patient folder names appear.
"""
import bisect
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import SimpleITK as sitk

from rt_dataprep import convert as conv
from rt_dataprep import dicom_io, dwell_table, ion_plan_table, plan_table, pool
from rt_dataprep.dicom_io import CaseRefused, quiet_call, write_report
from rt_dataprep.series import DIRECTION_ATOL, IDENTITY, IMAGE_MODALITIES, reg_matrix_array

log = logging.getLogger(__name__)

REPORT_NAME = 'preprocess_report.csv'
OUTCOME_PREPROCESSED = 'preprocessed'
KINDS = ('ct', 'dose', 'mask', 'applicator', 'line', 'plan', 'beam', 'fluence', 'ion_plan', 'dwell')

REASON_OUTSIDE_INPUT = 'output root is the converted root or inside it'
REASON_NO_STEPS = 'no preprocessing step is on'
REASON_ALLOW_CROP_STEPS = 'allow_crop needs a geometric step: resample, harmonize_spacing or reshape'
REASON_NO_SERIES_STEPS = 'no preprocessing step that is on applies to an MR or PT series'
REASON_EXISTS = 'output folder already exists'
REASON_NAME_CLAIMED = 'another folder of this run already maps to this output name (case-insensitive)'
REASON_NO_CASES = 'patient folder holds no plan or study folder with a series.json'
STUDY_FOLDER = r'study_[0-9]+'
PREPROCESSED_SERIES_COLUMNS = [*conv.SERIES_COLUMNS, 'retained']  # preprocess's series.json; blank until a step sets it
REASON_NO_CT = 'series.json does not list exactly one ct file'
REASON_DOSE_COUNT = 'series.json lists more than one dose file'
REASON_KIND = 'series.json has a row of an unknown kind'
LOW_RETAINED = 0.9  # the run log counts preprocessed plans whose lowest `retained` (dose in its dose_region, non-EXTERNAL mask) is below this (#226)
DOSE_RETAINED_GY = 1.0  # a dose file's `retained` is the in-grid fraction of its voxels at or above this, or DOSE_RETAINED_FRACTION of its max when lower (inclusive; dose is always Gy)
DOSE_RETAINED_FRACTION = 0.1  # of a dose's max: the threshold of a dose whose max is below 10 Gy, e.g. one EBRT fraction (#307)
REASON_SERIES_CSV = 'series.json is malformed: not valid JSON, not an array of objects of strings with the expected keys, or one file named twice'
LEGACY_SERIES_NAME = 'series.csv'  # what convert v0.1.x wrote; never read
REASON_LEGACY_SERIES = 'converted by an older version (series.csv); reconvert with --overwrite'
REASON_TOO_SMALL = 'the image is shorter than one output voxel along an axis'
REASON_NOT_VOLUME = 'the image is not a 3D volume of one real value per voxel'
REASON_NONFINITE = 'the image contains non-finite voxel values'
REASON_AHE_OPTIONS = 'AHE requires at least 2 bins, a finite clip limit from 0 to 1, and three positive finite kernel widths'
REASON_FILE_MISSING = 'a series.json row names a file that is not in the case folder'
REASON_STUDY_MODALITY = 'a series.json series row has a modality other than CT, MR or PT, or one that differs from its file name'
REASON_STUDY_NAME = 'a series.json series row names a file that is not a series volume'
REASON_GRID = 'a dose, mask or beam volume is not on the planning CT grid'
REASON_PLAN_INDEX = 'a plan table (plan.parquet, ion_plan.parquet or dwell.parquet) is unreadable or not its schema; left out of the output'
REASON_CROP = 'a dose or mask would lose voxels to the final grid'
REASON_MASK_RETENTION = 'a resampled mask retains less than the configured minimum physical volume'
REASON_MASK_RETENTION_LIMIT = 'minimum mask retention must be finite, greater than 0 and at most 1'
REASON_IO_ERROR = 'file system error while reading or writing; nothing kept for this patient'
REASON_FILE_UNREADABLE = 'an input NIfTI file is unreadable'
REASON_INTERNAL = 'an unexpected error stopped the preprocessing; the report lists what was finished'


@dataclass(frozen=True)
class PreprocessSteps:
    resample: bool = False  # initial resample to new_spacing, linear
    rescale: bool = False  # CT only: (HU + 1024) / 4096
    harmonize_spacing: bool = False  # final resample to final_spacing, linear
    reshape: bool = False  # centred crop/pad to new_shape
    ahe: bool = False  # CLAHE of every image volume: float32 min-max to [0, 1], then skimage equalize_adapthist
    ahe_kernel_mm: tuple | None = None  # optional physical window, x/y/z mm; None keeps skimage's voxel default
    ahe_bins: int = 128
    ahe_clip_limit: float = 0.05
    min_mask_retention: float | None = None  # optional minimum output/input physical foreground volume; no clinical default
    legacy_linear_masks: bool = False
    allow_crop: bool = False  # crop a dose or mask that extends past the final grid instead of refusing the plan; `retained` records how much is kept (a dose: its dose_region)
    series_fov_mask: bool = False  # not a step: uint8 support masks for planning CT, study and aligned plan images
    new_spacing: tuple = (1, 1, 1)  # mm, (x, y, z)
    final_spacing: tuple = (1.25, 1.25, 1.25)
    new_shape: tuple = (701, 701, 433)  # voxels, (x, y, z)


def read_image(path):
    """The NIfTI at `path`. A malformed or vanished file raises CaseRefused(REASON_FILE_UNREADABLE) (ITK names the path); a file-system error from the Python re-read propagates as OSError (ENOMEM as MemoryError)."""
    refusal = None
    try:
        return quiet_call(lambda: sitk.ReadImage(str(path)), REASON_FILE_UNREADABLE)
    except CaseRefused as exc:
        refusal = exc
    dicom_io.read_failure(path)
    raise refusal


def read_info(path):
    """(size, origin, spacing, direction) from the header of the NIfTI at `path`, no pixels read. A malformed or vanished file raises CaseRefused(REASON_FILE_UNREADABLE); a file-system error from the Python re-read propagates as OSError (ENOMEM as MemoryError)."""
    def read():
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        return reader.GetSize(), reader.GetOrigin(), reader.GetSpacing(), reader.GetDirection()
    refusal = None
    try:
        return quiet_call(read, REASON_FILE_UNREADABLE)
    except CaseRefused as exc:
        refusal = exc
    dicom_io.read_failure(path)
    raise refusal


def resampled_size(size, spacing, new_spacing):
    """torchio 1.2.1's output size: floor(N*s/s') in float64 on GetSpacing() values; an axis of 1 stays 1."""
    size = np.array(size)
    new = np.floor(size * np.array(spacing) / np.array(new_spacing, float)).astype(int)
    new[size == 1] = 1
    return new


def too_small(info, steps):
    """True when a resample step would leave a CT axis (read_info result) with no voxel."""
    size, spacing = info[0], info[2]
    for on, target in ((steps.resample, steps.new_spacing), (steps.harmonize_spacing, steps.final_spacing)):
        if on:
            size, spacing = resampled_size(size, spacing, target), target
    return bool((np.array(size) == 0).any())


def reference_grid(image, spacing):
    """(size, origin, spacing, direction) of `image` resampled to `spacing`, as torchio 1.2.1 Resample builds it."""
    old_size = np.array(image.GetSize())
    old_spacing = np.array(image.GetSpacing())
    new_spacing = np.array(spacing, float)
    size = resampled_size(old_size, old_spacing, new_spacing)
    direction = np.array(image.GetDirection()).reshape(3, 3)
    centre = np.array(image.GetOrigin()) + direction @ ((old_size - 1) * old_spacing / 2)
    origin = centre - direction @ ((size - 1) / 2 * new_spacing)
    return [int(n) for n in size], tuple(float(v) for v in origin), tuple(float(v) for v in new_spacing), image.GetDirection()


def resample_to(image, spacing):
    """`image` as float32, linearly resampled onto reference_grid(image, spacing), default value 0."""
    size, origin, new_spacing, direction = reference_grid(image, spacing)
    image = sitk.Cast(image, sitk.sitkFloat32)  # an integer CT would be interpolated, then truncated
    return sitk.Resample(image, size, sitk.Transform(), sitk.sitkLinear, origin, new_spacing, direction, 0.0, sitk.sitkFloat32)


def rescale(image):
    """(HU + 1024) / 4096 as float32."""
    return sitk.ShiftScale(sitk.Cast(image, sitk.sitkFloat32), 1024.0, 1 / 4096)  # arithmetic with a float would promote to float64


def equalize_adapthist(*args, **kwargs):
    """Load the optional compute path only when AHE is used; keep a seam for backend failure tests."""
    from skimage.exposure import equalize_adapthist as clahe
    return clahe(*args, **kwargs)


def finite_voxels(image):
    """Whether every voxel is finite, scanning planes to avoid a volume-sized boolean allocation."""
    return quiet_call(lambda: all(np.isfinite(plane).all() for plane in sitk.GetArrayViewFromImage(image)), REASON_INTERNAL)


def check_ahe_options(kernel_mm, nbins, clip_limit):
    valid = False
    try:
        valid = (isinstance(nbins, int) and not isinstance(nbins, bool) and nbins >= 2
                 and np.isfinite(clip_limit) and 0 <= clip_limit <= 1
                 and (kernel_mm is None or (len(kernel_mm) == 3 and all(np.isfinite(v) and v > 0 for v in kernel_mm))))
    except (TypeError, ValueError):
        pass
    if not valid:
        raise ValueError(REASON_AHE_OPTIONS)


def ahe_options(steps):
    return steps.ahe_kernel_mm, steps.ahe_bins, steps.ahe_clip_limit


def equalize(image, kernel_mm=None, nbins=128, clip_limit=0.05):
    """Float32 min-max then CLAHE; preserve normalized contrast if CLAHE collapses a varying volume. Geometry is kept."""
    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    low, high = array.min(), array.max()
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError  # internal compute guard; public process entry points refuse before entering quiet_call
    if high == low:
        equalized = np.zeros_like(array)
    else:
        # Float32 extremes can overflow their subtraction even though both inputs are finite.
        if float(high) - float(low) > np.finfo(np.float32).max:
            normalized = ((array.astype(np.float64) - float(low)) / (float(high) - float(low))).astype(np.float32)
        else:
            normalized = (array - low) / (high - low)
        options = {} if kernel_mm is None else {'kernel_size': tuple(
            max(1, int(np.ceil(width / spacing))) for width, spacing in zip(kernel_mm, image.GetSpacing(), strict=True))[::-1]}
        equalized = equalize_adapthist(normalized, nbins=nbins, clip_limit=clip_limit, **options)
        if not np.isfinite(equalized).all():
            raise ValueError
        if equalized.max() == equalized.min():
            equalized = normalized  # tiny tiles/two-level volumes can otherwise lose all contrast
    result = sitk.GetImageFromArray(equalized)
    result.CopyInformation(image)
    return result


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
    if steps.ahe:
        check_ahe_options(*ahe_options(steps))
        if not finite_voxels(ct):
            raise CaseRefused(REASON_NONFINITE)
    def run():
        image = ct
        if steps.resample:
            image = resample_to(image, steps.new_spacing)
        if steps.rescale:
            image = rescale(image)
        if steps.ahe:
            image = equalize(image, *ahe_options(steps))
        if steps.harmonize_spacing:
            image = resample_to(image, steps.final_spacing)
        if steps.reshape:
            image = crop_or_pad(image, steps.new_shape, minimum(image))  # the pad value is the minimum of the CT entering reshape
        return image
    return quiet_call(run, REASON_INTERNAL)


def minimum(image):
    low = sitk.MinimumMaximumImageFilter()
    low.Execute(image)
    return low.GetMinimum()


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


def mask_retention(native, image, linear=False):
    """Output/input physical foreground volume; linear masks use fractional occupancy, not nonzero voxel counts."""
    def run():
        before = sum(plane.sum(dtype=np.float64) if linear else np.count_nonzero(plane)
                     for plane in sitk.GetArrayViewFromImage(native)) * np.prod(native.GetSpacing())
        if not before:
            return 1.0  # an originally empty mask has no foreground to lose
        values = sitk.GetArrayViewFromImage(image)
        after = sum(plane.sum(dtype=np.float64) if linear else np.count_nonzero(plane) for plane in values)
        return float(after * np.prod(image.GetSpacing()) / before)
    return quiet_call(run, REASON_INTERNAL)


def process_series(native, final_ct, matrix, rescale_hu, ct, ahe, options=(None, 128, 0.05)):
    """`native` (HU-rescaled first when rescale_hu, then equalized on its own grid when ahe) linearly resampled once onto final_ct as float32;
    0 outside its FOV, or its own minimum for a CT that is neither rescaled nor equalized (#125: 0 would be water; the planning CT pads with its minimum too).
    `matrix` is series frame -> CT frame (series_transform)."""
    if ahe:
        check_ahe_options(*options)
        if not finite_voxels(native):
            raise CaseRefused(REASON_NONFINITE)
    transform = series_transform(matrix)
    def run():
        image = rescale(native) if rescale_hu else native
        if ahe:
            image = equalize(image, *options)
        fill = minimum(native) if ct and not rescale_hu and not ahe else 0.0
        return sitk.Resample(image, final_ct, transform, sitk.sitkLinear, fill, sitk.sitkFloat32)
    return quiet_call(run, REASON_INTERNAL)


def series_transform(matrix):
    """Resample's transform for `matrix` (series frame -> CT frame): Resample maps output points to input points, so its inverse."""
    transform = sitk.Transform()
    if not np.array_equal(matrix, np.eye(4)):
        inverse = np.linalg.inv(matrix)
        transform = sitk.AffineTransform(3)
        transform.SetMatrix(inverse[:3, :3].ravel().tolist())
        transform.SetTranslation(inverse[:3, 3].tolist())
    return transform


def covers_ct(native, final_ct, matrix):
    """(the uint8 image on final_ct that is 1 where native's FOV lands under `matrix`, whether any voxel is), whatever native's values (D6)."""
    def run():
        fov = sitk.Image(native.GetSize(), sitk.sitkUInt8) + 1
        fov.CopyInformation(native)
        landed = sitk.Resample(fov, final_ct, series_transform(matrix), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        return landed, has_voxels(landed)
    return quiet_call(run, REASON_INTERNAL)


def image_fov_mask(info, steps):
    """Native support through each geometric step, nearest-neighbour with zero padding; independent of intensity."""
    def run():
        size, origin, spacing, direction = info
        mask = sitk.Image(size, sitk.sitkUInt8) + 1
        mask.SetOrigin(origin)
        mask.SetSpacing(spacing)
        mask.SetDirection(direction)
        for on, target in ((steps.resample, steps.new_spacing), (steps.harmonize_spacing, steps.final_spacing)):
            if on:
                new_size, new_origin, new_spacing, new_direction = reference_grid(mask, target)
                mask = sitk.Resample(mask, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
                                     new_origin, new_spacing, new_direction, 0, sitk.sitkUInt8)
        if steps.reshape:
            mask = crop_or_pad(mask, steps.new_shape, 0)
        return mask
    return quiet_call(run, REASON_INTERNAL)


def is_volume(image):
    """Whether `image` is a 3D volume of one real value per voxel, the only kind convert writes (#127, #138, #139)."""
    return (image.GetDimension() == 3 and image.GetNumberOfComponentsPerPixel() == 1
            and image.GetPixelID() not in (sitk.sitkComplexFloat32, sitk.sitkComplexFloat64))


def read_series(path):
    """The image at `path`, or None when it is unreadable (that series is left out); any other refusal propagates, and so does a
    file-system error (OSError, ENOMEM as MemoryError), which fails the patient."""
    refusal = None
    try:
        return read_image(path)
    except CaseRefused as exc:
        refusal = exc
    if str(refusal) != REASON_FILE_UNREADABLE:  # raised here, after the except block: nothing chained
        raise refusal
    return None


def output_name(name, steps, ct=False, image=False):
    """`name` with the prefixes of the steps that are on, newest first: ahe needs an image volume (CT, MR, PT), rescale a CT; dose and masks get the geometric ones only."""
    prefixes = [('reshape', steps.reshape), ('resample', steps.harmonize_spacing), ('ahe', image and steps.ahe),
                ('rescale', ct and steps.rescale), ('initialresample', steps.resample)]
    return ''.join(f'{prefix}_' for prefix, on in prefixes if on) + name


class PreprocessAborted(Exception):
    """A patient hit an unexpected error: the report is written, the run stops. str() is exactly REASON_INTERNAL."""


@dataclass
class PreprocessRow:
    patient: str  # the converted folder name; preprocess_report.csv only
    folder: str  # plan_<n>, study_<n>/series_<MOD>_<k> (study_<n> when the whole study is refused), or '' for a patient-level row
    outcome: str
    reason: str = ''
    files: list = field(default_factory=list)
    skipped_series: int = 0  # registered series left out of a plan (unusable matrix, no overlap, unreadable file)
    cropped: int | str = ''  # preprocessed plan row of a geometric run: its dose and mask files that extend past the final grid; blank otherwise
    retained: float | None = None  # its lowest `retained` over counts_toward_retained files (run log only, #226); None when no file has one


@dataclass
class PreprocessReport:
    rows: list


def preprocess(converted_root, out_root, steps=PreprocessSteps(), *, workers=1, overwrite=False, _patient_fn=None):  # noqa: B008 (frozen, so the shared default is safe; with no step on it raises)
    """
    Preprocess every <patient>/plan_<n>/ and <patient>/study_<n>/ under converted_root into out_root and write preprocess_report.csv.

    workers: patient processes, one patient per process (about 7 GB RAM each at 701x701x433); the output is the same for any N.
        'auto' runs patients in this process until one is preprocessed, then sizes the pool from that patient's peak RSS.
    overwrite: replace a patient's existing output folder once the new output holds a preprocessed row; otherwise the old one is kept.
    _patient_fn: internal test seam, replaces the per-patient call only (module-level and picklable for workers > 1).
    """
    converted_root, out_root = Path(converted_root), Path(out_root)
    if out_root.resolve().is_relative_to(converted_root.resolve()):  # before any mkdir: nothing is written into the input
        raise ValueError(REASON_OUTSIDE_INPUT)
    if overwrite and converted_root.resolve().is_relative_to(out_root.resolve()):  # a replaced patient folder could hold the input
        raise ValueError(dicom_io.REASON_INPUT_INSIDE_OUTPUT)
    if not (steps.resample or steps.rescale or steps.harmonize_spacing or steps.reshape or steps.ahe):
        raise ValueError(REASON_NO_STEPS)
    if steps.allow_crop and not (steps.resample or steps.harmonize_spacing or steps.reshape):
        raise ValueError(REASON_ALLOW_CROP_STEPS)
    if steps.ahe:
        check_ahe_options(*ahe_options(steps))
    if steps.min_mask_retention is not None:
        if not np.isfinite(steps.min_mask_retention) or not 0 < steps.min_mask_retention <= 1:
            raise ValueError(REASON_MASK_RETENTION_LIMIT)
    folders = sorted(p for p in converted_root.iterdir() if p.is_dir() and not p.name.startswith('.'))  # a missing root fails before the mkdir
    out_root.mkdir(parents=True, exist_ok=True)
    patient_fn = _patient_fn or preprocess_patient
    # The parent owns the claims, as in convert: an undispatched refusal under overwrite, a hold on the casefolded name otherwise.
    dispatched, refused, claimed = [], {}, set()
    for i, folder in enumerate(folders):
        if overwrite and folder.name.casefold() in claimed:
            refused[i] = [PreprocessRow(folder.name, '', conv.OUTCOME_REFUSED, REASON_NAME_CLAIMED)]
        else:
            dispatched.append(i)
        claimed.add(folder.name.casefold())
    tasks = [pool.Task(_preprocess_worker, (patient_fn, folders[i], out_root, steps, overwrite), folders[i].name.casefold())
             for i in dispatched]
    results, aborted = {}, False  # folder index -> (rows, status, internal)

    def handle(k, result):
        nonlocal aborted
        i = dispatched[k]
        if isinstance(result, pool.Failed):  # escaped the worker wrapper (a broken pool...): only the type name is known
            if result.type_name == 'SystemExit':  # as before: exits the run with no row; the report is still written
                raise SystemExit(result.code)  # its sanitized code, as quiet_call keeps it
            log.error('preprocess: %s: %s', REASON_INTERNAL, result.type_name)
            result = [PreprocessRow(folders[i].name, '', conv.OUTCOME_REFUSED, REASON_INTERNAL)], None, True
        results[i] = result
        if result[-1]:
            aborted = True
            return pool.ABORT

    try:
        pool.run_auto(workers, tasks, handle, lambda k: any(r.outcome == OUTCOME_PREPROCESSED for r in results[dispatched[k]][0]))
    finally:  # Ctrl-C too: the report lists what was done
        rows = conv.rows_in_order(len(folders), refused, {i: r[0] for i, r in results.items()})
        write_report(out_root, REPORT_NAME, ['patient', 'folder', 'outcome', 'reason', 'files', 'skipped_series', 'cropped'],
                     ([r.patient, r.folder, r.outcome, r.reason, ';'.join(r.files), r.skipped_series, r.cropped] for r in rows))
    replaced, kept = (sum(r[1] == status for r in results.values()) for status in ('replaced', 'kept'))
    imaging_rows = [r for r in rows if r.folder.startswith('study_')]
    plan_rows = [r for r in rows if r.folder and not r.folder.startswith('study_')]
    patient_rows = [r for r in rows if r.folder == '']
    log.info('preprocess: %d plan rows (%d preprocessed, %d refused), %d imaging rows (%d preprocessed, %d refused), %d patient rows (%d refused, %d skipped); %d registered series skipped; %d plans cropped, %d with retained < %g; %d replaced, %d kept',
             len(plan_rows), *(sum(r.outcome == o for r in plan_rows) for o in (OUTCOME_PREPROCESSED, conv.OUTCOME_REFUSED)),
             len(imaging_rows), *(sum(r.outcome == o for r in imaging_rows) for o in (OUTCOME_PREPROCESSED, conv.OUTCOME_REFUSED)),
             len(patient_rows), *(sum(r.outcome == o for r in patient_rows) for o in (conv.OUTCOME_REFUSED, conv.OUTCOME_SKIPPED)),
             sum(r.skipped_series for r in rows), sum(bool(r.cropped) for r in rows),
             sum(r.retained is not None and r.retained < LOW_RETAINED for r in rows), LOW_RETAINED, replaced, kept)
    if aborted:  # raised here, after the except blocks: nothing of the original is chained
        raise PreprocessAborted(REASON_INTERNAL)
    return PreprocessReport(rows)


def _preprocess_worker(patient_fn, folder, out_root, steps, overwrite):
    """One patient in a worker (or inline): patient_fn's (rows, status) plus an internal flag; failures become data here, as in convert."""
    try:
        return (*patient_fn(folder, out_root, steps, overwrite), False)
    except OSError:  # its message names a path, i.e. the patient: one fixed reason, and no plan row of this patient
        log.warning('preprocess: %s', REASON_IO_ERROR)
        return [PreprocessRow(folder.name, '', conv.OUTCOME_REFUSED, REASON_IO_ERROR)], None, False
    except Exception as exc:  # not KeyboardInterrupt/SystemExit: those keep the flag-based cleanup and add no row
        module, line = conv.crash_site(sys.exc_info()[2])
        log.error('preprocess: %s: %s at %s:%d', REASON_INTERNAL, getattr(type(exc), '__name__'), module, line)  # noqa: B009 (no message)
        return pool.Stop(([PreprocessRow(folder.name, '', conv.OUTCOME_REFUSED, REASON_INTERNAL)], None, True))


def preprocess_patient(folder, out_root, steps, overwrite=False):
    """(rows, status None/'new'/'replaced'/'kept'); a raise adds no status."""
    target = out_root / folder.name
    dicom_io.recover(out_root, folder.name)
    exists = os.path.lexists(target)  # lexists, as publish: a dangling link is an existing target
    if exists and not overwrite:
        return [PreprocessRow(folder.name, '', conv.OUTCOME_SKIPPED, REASON_EXISTS)], None
    plans = sorted((p for p in folder.iterdir() if p.is_dir() and re.fullmatch(conv.PLAN_FOLDER, p.name) and holds_series(p)),
                   key=lambda p: (int(p.name.split('_')[1]), p.name))  # plan_1 and fx_1: a fixed order
    studies = sorted((p for p in folder.iterdir() if p.is_dir() and re.fullmatch(STUDY_FOLDER, p.name) and holds_series(p)),
                     key=lambda p: int(p.name.split('_')[1]))
    if not plans and not studies:
        return [PreprocessRow(folder.name, '', conv.OUTCOME_SKIPPED, REASON_NO_CASES)], 'kept' if exists else None
    # Built under a hidden name and renamed at the end, as convert does: a run that dies part-way leaves no folder that a
    # later run would skip as done. The flag, not an except block, decides the cleanup, so any failure (and Ctrl-C) removes it.
    partial = out_root / f'.{folder.name}.partial'  # recover, above, already removed a leftover
    rows, done, status = [], False, None
    try:
        partial.mkdir()
        rows = [preprocess_one(plan_dir, partial / plan_dir.name, steps) for plan_dir in plans]
        for study_dir in studies:
            rows += preprocess_study(study_dir, partial / study_dir.name, steps)
        # An empty partial (every plan and series refused) is renamed too when there is no old folder: the patient has its outcome.
        status = dicom_io.publish(out_root, folder.name, overwrite=overwrite,
                                  success=any(row.outcome == OUTCOME_PREPROCESSED for row in rows))
        done = True
    finally:
        if not done:
            shutil.rmtree(partial, ignore_errors=True)
    return rows, status


def preprocess_one(plan_dir, out_dir, steps):
    """The PreprocessRow of one plan. A refusal leaves no `out_dir`; an OSError or REASON_INTERNAL is the patient's and propagates."""
    reason = None
    try:
        files, skipped, cropped, retained = preprocess_plan(plan_dir, out_dir, steps)
    except CaseRefused as exc:
        if str(exc) == REASON_INTERNAL:
            raise
        reason = str(exc)
    if reason is not None:  # raised here, after the except block: nothing of the original is chained
        if out_dir.exists():  # a failure is an OSError, the patient's: a refused plan's files must never be renamed into place
            shutil.rmtree(out_dir)
        log.warning('preprocess: plan refused: %s', reason)
        return PreprocessRow(plan_dir.parent.name, plan_dir.name, conv.OUTCOME_REFUSED, reason)
    return PreprocessRow(plan_dir.parent.name, plan_dir.name, OUTCOME_PREPROCESSED, files=files, skipped_series=skipped, cropped=cropped,
                         retained=retained)


def preprocess_study(study_dir, out_dir, steps):
    """The PreprocessRows of one study, one per series (one row `study_<n>` when check_study_rows refuses it). `out_dir` exists only if a series was written."""
    patient, study = study_dir.parent.name, study_dir.name
    refusal = None
    try:
        series = check_study_rows(study_dir)
    except CaseRefused as exc:
        refusal = exc
    if refusal is not None:  # an OSError is the patient's and propagates
        log.warning('preprocess: study refused: %s', refusal)
        return [PreprocessRow(patient, study, conv.OUTCOME_REFUSED, str(refusal))]
    rows, written = [], []
    for row in series:
        folder = f'{study}/{series_id(row["file"])}'
        name, refusal = None, None
        try:
            name = preprocess_series(study_dir, out_dir, row, steps)
        except CaseRefused as exc:
            refusal = exc
        if refusal is not None and str(refusal) == REASON_INTERNAL:
            raise refusal  # to the patient, as for a plan
        if refusal is not None:
            log.warning('preprocess: series refused: %s', refusal)
            rows.append(PreprocessRow(patient, folder, conv.OUTCOME_REFUSED, str(refusal)))
            continue
        written.append(row | {'file': name, 'retained': ''})
        rows.append(PreprocessRow(patient, folder, OUTCOME_PREPROCESSED,
                                  files=[name, *([f'fovmask_{name}'] if steps.series_fov_mask else [])]))
    if written:
        conv.write_series_json(written, out_dir / conv.SERIES_NAME)
        next(r for r in rows if r.outcome == OUTCOME_PREPROCESSED).files.append(conv.SERIES_NAME)
    return rows


def preprocess_series(study_dir, out_dir, row, steps):
    """Write one study series into `out_dir` (made on the first write) and return its output name; a CaseRefused refuses this series only and writes nothing."""
    is_ct = row['modality'] == 'CT'
    steps = steps if is_ct else replace(steps, rescale=False)  # the HU rescale is for CT only
    if not (steps.resample or steps.rescale or steps.harmonize_spacing or steps.reshape or steps.ahe):
        raise CaseRefused(REASON_NO_SERIES_STEPS)  # a rescale-only run has nothing to do to an MR or PT: no read, no copy under another name
    source = study_dir / row['file']
    info = read_info(source)  # header first: a too-small series is refused without a pixel read (#141)
    if len(info[0]) == 3 and too_small(info, steps):  # too_small assumes three axes; other shapes fail is_volume below
        raise CaseRefused(REASON_TOO_SMALL)
    image = read_image(source)
    if not is_volume(image):
        raise CaseRefused(REASON_NOT_VOLUME)
    image = process_ct(image, steps)
    name = output_name(row['file'], steps, ct=is_ct, image=True)
    out_dir.mkdir(exist_ok=True)
    conv.write_atomic(image, out_dir / name)
    if steps.series_fov_mask:
        conv.write_atomic(image_fov_mask(info, steps), out_dir / f'fovmask_{name}')
    return name


def same_grid(a, b):
    """Two read_info results on one grid: equal size; origin, spacing and direction within ORIGIN_ATOL_MM and DIRECTION_ATOL."""
    return (a[0] == b[0] and np.allclose(a[1], b[1], rtol=0, atol=conv.ORIGIN_ATOL_MM)
            and np.allclose(a[2], b[2], rtol=0, atol=conv.ORIGIN_ATOL_MM) and np.allclose(a[3], b[3], rtol=0, atol=DIRECTION_ATOL))


def check_rows(plan_dir):
    """(kept rows, ct first; the CT's read_info; the kind=series rows in input order). Every row is checked here, the CT and dose/mask/beam headers only, before the plan's first write."""
    if is_legacy(plan_dir):
        raise CaseRefused(REASON_LEGACY_SERIES)
    rows, malformed = read_series_json(plan_dir)
    if malformed:  # raised here, after the except block: nothing chained
        raise CaseRefused(REASON_SERIES_CSV)
    if any(row['kind'] not in (*KINDS, 'series') for row in rows):
        raise CaseRefused(REASON_KIND)
    kept = [row for row in rows if row['kind'] != 'series']
    series = [row for row in rows if row['kind'] == 'series']
    if sum(row['kind'] == 'ct' for row in kept) != 1:
        raise CaseRefused(REASON_NO_CT)
    if sum(row['kind'] == 'dose' for row in kept) > 1:
        raise CaseRefused(REASON_DOSE_COUNT)
    if len({row['file'] for row in rows}) != len(rows):  # after the counts: two ct or dose rows are refused as such
        raise CaseRefused(REASON_SERIES_CSV)
    check_series_rows(series)
    present = {p.name for p in plan_dir.iterdir() if p.is_file() and not p.is_symlink()}  # a value with a separator or '..' is never in it
    if any(row['file'] not in present for row in (*kept, *series)):
        raise CaseRefused(REASON_FILE_MISSING)
    kept.sort(key=lambda row: row['kind'] != 'ct')
    ct = read_info(plan_dir / kept[0]['file'])
    if any(not same_grid(ct, read_info(plan_dir / row['file'])) for row in kept if row['kind'] in ('dose', 'mask', 'beam')):
        raise CaseRefused(REASON_GRID)
    return kept, ct, series


STUDY_FILE = re.compile(r'(reshape_)?(resample_)?(ahe_)?(rescale_)?(initialresample_)?(?P<id>series_(CT|MR|PT)_[0-9]+)\.nii\.gz')  # output_name's prefixes, in order


def series_id(name):
    """`series_<MOD>_<k>` of a study volume's file name (a full match, so no newline tail), or None when the name is not one."""
    match = STUDY_FILE.fullmatch(name)
    return match.group('id') if match else None


def check_series_rows(series):
    """The kind=series row checks plans and studies share: an image modality, a series_<MOD>_<k> name whose MOD matches it, no id twice."""
    if any(row['modality'] not in IMAGE_MODALITIES for row in series):
        raise CaseRefused(REASON_STUDY_MODALITY)
    ids = [series_id(row['file']) for row in series]
    if None in ids:
        raise CaseRefused(REASON_STUDY_NAME)
    if any(i.split('_')[1] != row['modality'] for i, row in zip(ids, series, strict=True)):
        raise CaseRefused(REASON_STUDY_MODALITY)
    if len(set(ids)) != len(ids):
        raise CaseRefused(REASON_SERIES_CSV)


def holds_series(folder):
    """A plan or study folder that holds a series.json, or a legacy series.csv (check_rows and check_study_rows refuse that one)."""
    return (folder / conv.SERIES_NAME).is_file() or (folder / LEGACY_SERIES_NAME).is_file()


def is_legacy(folder):
    """A series.csv and no series.json: refused with REASON_LEGACY_SERIES, before any read, and the CSV is never parsed."""
    return (folder / LEGACY_SERIES_NAME).is_file() and not (folder / conv.SERIES_NAME).is_file()


def _no_duplicate_keys(pairs):
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError  # no content in it: read_series_json maps it to the fixed reason
    return dict(pairs)


def read_series_json(folder):
    """(rows, malformed) of a plan's or study's series.json. malformed: not UTF-8 JSON (a BOM, duplicate keys or deep nesting included), not an array,
    an item that is not an object, key sets other than SERIES_COLUMNS or SERIES_COLUMNS + `retained`, each with or without `dicom` (one set for the whole file: on every item or none),
    a value that is not a string, or a `dicom` that is not an object (convert's metadata; carried unread).
    rows is the array when it is one, else []; an OSError (the file cannot be opened) propagates."""
    try:
        with open(folder / conv.SERIES_NAME, encoding='utf-8') as handle:
            rows = json.load(handle, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, RecursionError):  # JSONDecodeError and UnicodeDecodeError are ValueErrors; no `from`: they quote the file
        return [], True
    if not isinstance(rows, list):
        return [], True
    first = set(rows[0]) if rows and isinstance(rows[0], dict) else None
    malformed = bool(rows) and (first is None or first - {'dicom'} not in (set(conv.SERIES_COLUMNS), set(PREPROCESSED_SERIES_COLUMNS)) or not all(
        isinstance(row, dict) and set(row) == first and all(isinstance(value, dict if key == 'dicom' else str) for key, value in row.items())
        for row in rows))
    return rows, malformed


def study_refusal_count(rows):
    """What a refused study counts in the manifest: its series.json items (read_series_json), or 1 when it cannot be parsed or holds no row."""
    return max(1, len(rows))


def check_study_rows(study_dir, parsed=None):
    """The series.json rows of a study folder, every one checked here and no NIfTI header read. A failure refuses the whole study.
    `parsed` is read_series_json's result when the caller has already read it."""
    if is_legacy(study_dir):
        raise CaseRefused(REASON_LEGACY_SERIES)
    rows, malformed = parsed or read_series_json(study_dir)
    if malformed or not rows or len({row['file'] for row in rows}) != len(rows):
        raise CaseRefused(REASON_SERIES_CSV)
    if any(row['kind'] != 'series' for row in rows):
        raise CaseRefused(REASON_KIND)
    check_series_rows(rows)
    present = {p.name for p in study_dir.iterdir() if p.is_file() and not p.is_symlink()}
    if any(row['file'] not in present for row in rows):
        raise CaseRefused(REASON_FILE_MISSING)
    return rows


def has_voxels(image):
    values = sitk.GetArrayViewFromImage(image)
    return bool(values.any())


def crosses_final(native, final):
    """True when a nonzero voxel centre of `native` falls outside final's buffer, continuous index outside [-0.5, n-0.5)."""
    values = sitk.GetArrayViewFromImage(native)
    nonzero = values != 0
    if not nonzero.any():
        return False
    # Same direction as the CT, so each final axis follows one native axis: the extreme voxels sit at the nonzero bounding box.
    hits = [np.flatnonzero(nonzero.any(axis=tuple(a for a in range(3) if a != axis))) for axis in range(3)]  # array axes z, y, x
    low = [int(hit[0]) for hit in hits][::-1]
    high = [int(hit[-1]) for hit in hits][::-1]
    first = final.TransformPhysicalPointToContinuousIndex(native.TransformIndexToPhysicalPoint(low))
    last = final.TransformPhysicalPointToContinuousIndex(native.TransformIndexToPhysicalPoint(high))
    return bool(min(first) < -0.5 or any(v >= n - 0.5 for v, n in zip(last, final.GetSize(), strict=True)))


def check_crop(native, final):
    """Refuse (REASON_CROP) when a nonzero voxel centre of `native` falls outside final's buffer (crosses_final)."""
    if crosses_final(native, final):
        raise CaseRefused(REASON_CROP)


def counts_toward_retained(row):
    """Whether a series.json row sets a plan's lowest `retained`: a dose or mask with a value, except a body outline (mask typed exactly 'EXTERNAL')."""
    external = row['kind'] == 'mask' and row.get('dicom', {}).get('roi_interpreted_type') == 'EXTERNAL'
    return row['kind'] in ('dose', 'mask') and bool(row.get('retained')) and not external


def dose_region(dose):
    """The 0/1 voxels a dose's `retained` counts: at or above the lesser of DOSE_RETAINED_GY and DOSE_RETAINED_FRACTION of its max (#307); never empty for a dose with a positive voxel."""
    return dose >= min(DOSE_RETAINED_GY, DOSE_RETAINED_FRACTION * float(sitk.GetArrayViewFromImage(dose).max()))


def native_retained(native, final):
    """The part of `native` (summed values, float64; a 0/1 image gives a voxel fraction) whose voxel centres fall inside final's buffer, over the whole; 1.0 for an empty image.
    Same direction as the CT, so each native axis maps increasingly onto one final axis: bisect the first and last in-grid index per axis."""
    def run():
        total = sitk.GetArrayViewFromImage(native).sum(dtype=np.float64)
        if not total:
            return 1.0
        def centre(axis, k):  # continuous final index along `axis` of native index k (the other axes at 0)
            index = [0, 0, 0]
            index[axis] = k
            return final.TransformPhysicalPointToContinuousIndex(native.TransformIndexToPhysicalPoint(index))[axis]
        bounds = []
        for axis, (n, length) in enumerate(zip(final.GetSize(), native.GetSize(), strict=True)):
            start = bisect.bisect_left(range(length), True, key=lambda k: centre(axis, k) >= -0.5)
            stop = bisect.bisect_left(range(length), True, key=lambda k: centre(axis, k) >= n - 0.5)
            bounds.append(slice(start, stop))
        inside = sitk.GetArrayViewFromImage(native)[bounds[2], bounds[1], bounds[0]]  # array axes z, y, x
        return float(inside.sum(dtype=np.float64) / total)
    return quiet_call(run, REASON_INTERNAL)


def index_plan_table(source, path, final, schema=plan_table.SCHEMA, prefix='isocenter'):
    """Write `source` (plan.parquet, ion_plan.parquet with schema=ion_plan_table.ION_SCHEMA, or dwell.parquet with schema=dwell_table.DWELL_SCHEMA and prefix='position') to `path` with <prefix>_i/j/k replaced by the continuous index of
    <prefix>_x/y/z on `final`; False, nothing written, when it is unreadable or not `schema` (compared as fields: file metadata is not part of it). An OSError propagates."""
    try:
        table = pq.read_table(source)
    except pa.ArrowInvalid:
        return False
    if table.schema != schema:
        return False
    points = zip(*(table[name].to_pylist() for name in (f'{prefix}_x', f'{prefix}_y', f'{prefix}_z')), strict=True)
    indices = list(zip(*(final.TransformPhysicalPointToContinuousIndex(p) for p in points), strict=True))
    for name, values in zip((f'{prefix}_i', f'{prefix}_j', f'{prefix}_k'), indices or [[]] * 3, strict=True):
        table = table.set_column(table.schema.get_field_index(name), schema.field(name), pa.array(values, pa.float64()))
    conv._atomic(lambda temp: pq.write_table(table, temp), path)
    return True


def preprocess_plan(plan_dir, out_dir, steps):
    """Write the plan into `out_dir` (which must not exist); returns (the output file names, series.json last; the number of series left out; the dose and mask files cut by the final grid, '' on a non-geometric run; the lowest `retained` of the rows counts_toward_retained accepts, None when none is set). CaseRefused on a refusal."""
    kept, ct, series = check_rows(plan_dir)
    if too_small(ct, steps):
        raise CaseRefused(REASON_TOO_SMALL)
    geometric = steps.resample or steps.harmonize_spacing or steps.reshape
    out_dir.mkdir()
    files, series_rows, cropped = [], [], 0
    for row in kept:  # the CT first: its final grid is the reference of the rest
        kind, source = row['kind'], plan_dir / row['file']
        cell = ''  # series.json `retained`: only a dose or mask of a geometric run
        name = row['file'] if kind in ('applicator', 'line', 'plan', 'fluence', 'ion_plan', 'dwell') else output_name(row['file'], steps, ct=kind == 'ct', image=kind == 'ct')
        if kind == 'ct':
            final = process_ct(read_image(source), steps)
            conv.write_atomic(final, out_dir / name)
        elif kind in ('plan', 'ion_plan', 'dwell') and geometric:  # isocenter_i/j/k (position_i/j/k of a dwell) always index the CT of the folder
            schema, prefix = {'plan': (plan_table.SCHEMA, 'isocenter'), 'ion_plan': (ion_plan_table.ION_SCHEMA, 'isocenter'), 'dwell': (dwell_table.DWELL_SCHEMA, 'position')}[kind]
            if not index_plan_table(source, out_dir / name, final, schema, prefix):
                log.warning('preprocess: %s', REASON_PLAN_INDEX)
                continue
        elif kind in ('applicator', 'line', 'plan', 'fluence', 'ion_plan', 'dwell') or not geometric:  # point text, fluence (a collimator-frame map, no CT grid) and (non-geometric) plan tables: copied unchanged
            shutil.copyfile(source, out_dir / name)
        else:  # one image at a time: read, resampled, written, released
            native = read_image(source)
            if kind != 'beam':  # a beam volume is a field footprint: a reshape that cuts it is neither refused nor counted, and it has no `retained`
                if steps.allow_crop:
                    crossed = crosses_final(native, final)
                else:
                    check_crop(native, final)
                    crossed = False
                cropped += crossed
                # a dose is judged on its dose_region (a 0/1 temporary, released at once); crosses_final, process_dose and has_voxels keep the unthresholded native
                cell = repr(native_retained(dose_region(native) if kind == 'dose' else native, final) if crossed else 1.0)  # exactly 1.0 when nothing lies outside
            image = process_dose(native, final) if kind in ('dose', 'beam') else process_mask(native, final, steps)
            if kind != 'beam' and has_voxels(native) and not has_voxels(image):
                raise CaseRefused(REASON_CROP)
            if kind == 'mask' and steps.min_mask_retention is not None:
                ratio = mask_retention(native, image, steps.legacy_linear_masks)
                if ratio < steps.min_mask_retention and not np.isclose(ratio, steps.min_mask_retention, rtol=1e-6, atol=0):
                    raise CaseRefused(REASON_MASK_RETENTION)
            conv.write_atomic(image, out_dir / name)
            del native, image
        files.append(name)
        if kind == 'ct' and steps.series_fov_mask:
            mask_name = f'fovmask_{name}'
            conv.write_atomic(image_fov_mask(ct, steps), out_dir / mask_name)
            files.append(mask_name)
        series_rows.append(row | {'file': name, 'retained': cell})
    skipped = 0
    for row in series:  # one series at a time, after the CT: read, resampled, written, released
        matrix = reg_matrix_array(row['reg_matrix'])
        native = None if matrix is None else read_series(plan_dir / row['file'])
        if native is None or not is_volume(native) or (steps.ahe and not finite_voxels(native)):  # D8
            skipped += 1
            continue
        is_ct = row['modality'] == 'CT'
        fov, covered = covers_ct(native, final, matrix)
        if not steps.series_fov_mask:
            fov = None  # released right after the check: a default run's peak memory is unchanged (#47)
        if not covered:  # entirely outside the CT: never resampled
            skipped += 1
        else:
            image = process_series(native, final, matrix, is_ct and steps.rescale, is_ct, steps.ahe, ahe_options(steps))
            name = output_name(row['file'], steps, ct=is_ct, image=True)
            conv.write_atomic(image, out_dir / name)
            files.append(name)
            if fov is not None:  # the mask is listed after its series, not in series.json
                conv.write_atomic(fov, out_dir / f'fovmask_{name}')
                files.append(f'fovmask_{name}')
            series_rows.append(row | {'file': name, 'retained': '', 'reg_matrix': IDENTITY, 'frame_of_reference_uid': kept[0]['frame_of_reference_uid']})
            del image
        del native, fov
    conv.write_series_json(series_rows, out_dir / conv.SERIES_NAME)
    retained = [float(row['retained']) for row in series_rows if counts_toward_retained(row)]
    return [*files, conv.SERIES_NAME], skipped, cropped if geometric else '', min(retained, default=None)
