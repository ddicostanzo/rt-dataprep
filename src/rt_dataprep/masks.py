"""
RTSTRUCT -> training-label masks and brachy applicator point files (spec: f2a-convert, steps 2.4-2.6).

`classify` reads the headers only: which ROIs become masks and which become applicator files, and every output name.
`build_masks` runs DicomRTTool 6.2 once on a temp folder of hard links (the planning CT slices and the RTSTRUCT) and
verifies the result, because DicomRTTool fails silently with all-zero masks or masks on another grid. Nothing here puts a
path, ROI name, UID or tag value in a message: every refusal is one fixed REASON_ constant.
"""
import contextlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
from DicomRTTool import DicomReaderWriter

from rt_dataprep.dicom_io import CaseRefused, quiet_call

SUFFIX = '.nii.gz'  # also on the applicator text files: numpy gzips by the suffix

REASON_ROI_NAMES = 'ROI names give an empty or clashing output file name'
REASON_MASK_RUN = 'DicomRTTool could not turn the RTSTRUCT into masks'
REASON_MASK_LINK = 'DicomRTTool did not link the RTSTRUCT to exactly one planning CT series'
REASON_MASK_MISSING = 'a kept ROI is missing from the DicomRTTool result or maps to another ROI'
REASON_MASK_EMPTY = 'a kept ROI came back with no voxel'
REASON_MASK_GRID = 'a mask is not on the planning CT grid'
REASON_MASK_SLICES = 'a mask does not lie on exactly the CT slices its contours lie on'


@dataclass
class RoiPlan:
    kept: list = field(default_factory=list)  # [(ROINumber, ROIName, '<format_name>.nii.gz', [(N, 3) non-POINT contours in patient mm])] in RTSTRUCT order
    applicators: list = field(default_factory=list)  # [('<name>.nii.gz', (N, 3) points in patient mm, ROIName)] in RTSTRUCT order


def format_name(text):
    """`text` lowercased with every non-alphanumeric character removed (legacy utils.py)."""
    return ''.join(c for c in str(text).lower() if c.isalnum())


def classify(struct, plan, taken=()):
    """
    The RoiPlan of an RTSTRUCT (and the RTPLAN naming its applicator channels).

    Not kept: ROIs with no contour data, POINT ROIs (every contour POINT) and names containing 'dose'. Of the rest, ROIs
    with an OPEN_NONPLANAR contour or RTROIInterpretedType BRACHY_CHANNEL are applicators, the others masks. An empty
    output name, or two names equal ignoring case (also with `taken`, names already used in the plan folder), refuse.
    """
    contours = {int(c.ReferencedROINumber): c for c in struct.get('ROIContourSequence', [])}
    interpreted = {int(o.ReferencedROINumber): str(o.get('RTROIInterpretedType', ''))
                   for o in struct.get('RTROIObservationsSequence', [])}
    channel_ids = {}
    for setup in plan.get('ApplicationSetupSequence', []):
        for channel in setup.get('ChannelSequence', []):
            if 'ReferencedROINumber' in channel and 'SourceApplicatorID' in channel:
                channel_ids.setdefault(int(channel.ReferencedROINumber), str(channel.SourceApplicatorID))
    result, names = RoiPlan(), [str(n).lower() for n in taken]
    for roi in struct.get('StructureSetROISequence', []):
        number, name = int(roi.ROINumber), str(roi.ROIName)
        items = list(contours[number].get('ContourSequence', [])) if number in contours else []
        points = [np.asarray([float(v) for v in item.ContourData]).reshape(-1, 3) for item in items]
        types = {str(item.ContourGeometricType) for item in items}
        if not points or sum(len(p) for p in points) == 0 or types == {'POINT'} or 'dose' in name.lower():
            continue
        if 'OPEN_NONPLANAR' in types or interpreted.get(number) == 'BRACHY_CHANNEL':
            stem = ''.join(c for c in channel_ids.get(number, name) if c.isalnum())
            result.applicators.append((stem + SUFFIX, np.concatenate(points), name))
            names.append((stem + SUFFIX).lower())
        else:
            planar = [p for p, item in zip(points, items, strict=True) if str(item.ContourGeometricType) != 'POINT']
            result.kept.append((number, name, format_name(name) + SUFFIX, planar))
            names.append((format_name(name) + SUFFIX).lower())
    if any(name == SUFFIX for name in names) or len(set(names)) != len(names):
        raise CaseRefused(REASON_ROI_NAMES)
    return result


def write_applicator(points, path):
    """`points` (N, 3) as gzip text (np.savetxt, '%.4f') at `path`, through a temp name that also ends .nii.gz."""
    temp = path.with_name(f'.{os.getpid()}.tmp.{path.name}')
    try:
        np.savetxt(temp, points, fmt='%.4f')
        os.replace(temp, path)
    finally:
        with contextlib.suppress(OSError):
            temp.unlink()


def _link_or_copy(source, target):
    linked = True
    try:
        os.link(source, target)
    except OSError:
        linked = False
    if not linked:
        shutil.copyfile(source, target)


def build_masks(kept, ct_paths, struct_path, struct_series_uid, ct_series_uid, ct, scratch_parent,
                origin_atol, direction_atol):
    """
    {file name: uint8 {0,1} image on `ct`'s grid} for the `kept` ROIs, or {} (DicomRTTool not run) when none is kept.

    The temp folder is made inside `scratch_parent` and removed in every outcome. Any failed check raises CaseRefused.
    """
    if not kept:
        return {}
    scratch = tempfile.mkdtemp(prefix='.rttool.', dir=scratch_parent)
    try:
        for path in [*ct_paths, struct_path]:
            _link_or_copy(path, os.path.join(scratch, os.path.basename(path)))
        reader, hits = quiet_call(lambda: _run(scratch, [name for _, name, _, _ in kept], ct_series_uid), REASON_MASK_RUN)
        # Linked: the RTSTRUCT's own series UID is a key of the CT index's RTs (DicomRTTool links by the series it references).
        if len(hits) != 1 or struct_series_uid not in reader.series_instances_dictionary[hits[0]].RTs:
            raise CaseRefused(REASON_MASK_LINK)
        rois = reader.series_instances_dictionary[hits[0]].RTs[struct_series_uid].ROIs_In_Structure
        masks = {}
        for number, name, file_name, contours in kept:
            key = name.lower()
            image = reader.mask_dictionary.get(key)
            if image is None or key not in rois or int(rois[key].ROINumber) != number:
                raise CaseRefused(REASON_MASK_MISSING)
            masks[file_name] = _checked(image, ct, contours, origin_atol, direction_atol)
        return masks
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _run(folder, names, ct_series_uid):
    reader = DicomReaderWriter(arg_max=False, verbose=False)
    reader.walk_through_folders(folder)
    reader.set_contour_names_and_associations(contour_names=names)
    hits = [i for i, entry in reader.series_instances_dictionary.items() if entry.SeriesInstanceUID == ct_series_uid]
    if len(hits) == 1:  # a bad index is judged by the caller, with its own reason
        reader.set_index(hits[0])
        reader.get_images_and_mask()
    return reader, hits


def _checked(image, ct, contours, origin_atol, direction_atol):
    """
    `image` as {0,1} uint8 with `ct`'s geometry; it must have a voxel, match the CT grid within tolerance, and have voxels
    on exactly the slices its `contours` lie on.
    """
    values = sitk.GetArrayFromImage(image) > 0
    if not values.any():
        raise CaseRefused(REASON_MASK_EMPTY)
    if not (image.GetSize() == ct.GetSize()
            and np.allclose(image.GetOrigin(), ct.GetOrigin(), rtol=0, atol=origin_atol)
            and np.allclose(image.GetSpacing(), ct.GetSpacing(), rtol=0, atol=origin_atol)
            and np.allclose(image.GetDirection(), ct.GetDirection(), rtol=0, atol=direction_atol)):
        raise CaseRefused(REASON_MASK_GRID)
    # DicomRTTool 6.2 picks a contour's slice by patient z alone; the CT's full geometry says where each contour belongs.
    slices = set()
    for points in contours:
        index = np.rint([ct.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in p))[2] for p in points])
        if index.min() != index.max():
            raise CaseRefused(REASON_MASK_SLICES)
        slices.add(int(index[0]))
    if slices != set(np.flatnonzero(values.any(axis=(1, 2))).tolist()):
        raise CaseRefused(REASON_MASK_SLICES)
    out = sitk.GetImageFromArray(values.astype(np.uint8))
    out.CopyInformation(ct)
    return out
