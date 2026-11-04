"""
A non-planning image series (MR, PT, an unplanned CT) as one float32 volume, any orientation (spec: f2a-pr2-other-series).

check_volume(slices) validates the headers of one series and returns the slices sorted along the normal; read_series(slices)
reads the pixels and sets origin, spacing and direction from those headers. Both return None for a series that is skipped
(a bad or missing tag, not a uniform single-frame volume, an unreadable file): no exception escapes, and nothing here
logs or raises text that holds a path, UID, date or description.

registration(ct_frame, series_frame, regs, headers_by_sop) says whether a series is registered to the planning CT and
gives its reg_matrix text (series frame -> CT frame). candidates(headers, exclude_models) picks the series and REGs of a
patient from its headers.
"""
from collections import defaultdict

import numpy as np
import SimpleITK as sitk

from rt_dataprep.dicom_io import CaseRefused, quiet_call

SERIES_ATOL_MM = 0.01  # slice positions and pixel spacing (scanners round positions to ~0.01 mm)
ORTHO_ATOL = 1e-4  # unit length and orthogonality of the row and column cosines
DIRECTION_ATOL = 1e-6  # equality of ImageOrientationPatient across slices

REASON_SERIES_UNREADABLE = 'the series pixel data is unreadable'


def _or_none(fn, *args):
    """fn(*args), or None when a tag is missing or odd (AttributeError, KeyError, IndexError, TypeError, ValueError)."""
    failed, result = False, None
    try:
        result = fn(*args)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):  # not quoted
        failed = True
    return None if failed else result


def _cosines(ds):
    """(row cosine, column cosine, normal), each unit length, from one header."""
    iop = [float(v) for v in ds.ImageOrientationPatient]
    if len(iop) != 6:
        raise ValueError
    row, column = np.asarray(iop[:3]), np.asarray(iop[3:])
    row, column = row / np.linalg.norm(row), column / np.linalg.norm(column)
    normal = np.cross(row, column)
    return row, column, normal / np.linalg.norm(normal)


def _position(ds):
    position = [float(v) for v in ds.ImagePositionPatient]
    if len(position) != 3:
        raise ValueError
    return np.asarray(position)


def _pixel_spacing(ds):
    spacing = [float(v) for v in ds.PixelSpacing]
    if len(spacing) != 2:
        raise ValueError
    return np.asarray(spacing)


def _sorted_volume(slices):
    heads = [ds for _, ds in slices]
    if len(heads) < 2 or any(int(ds.get('NumberOfFrames', 1)) != 1 for ds in heads):
        return None
    iops = np.asarray([[float(v) for v in ds.ImageOrientationPatient] for ds in heads])
    row, column = iops[0][:3], iops[0][3:]
    if not (np.allclose(iops, iops[0], rtol=0, atol=DIRECTION_ATOL)
            and abs(np.linalg.norm(row) - 1) <= ORTHO_ATOL and abs(np.linalg.norm(column) - 1) <= ORTHO_ATOL
            and abs(np.dot(row, column)) <= ORTHO_ATOL):
        return None
    pixel_spacing = np.asarray([_pixel_spacing(ds) for ds in heads])
    rescale = np.asarray([[float(ds.get('RescaleSlope', 1)), float(ds.get('RescaleIntercept', 0))] for ds in heads])
    # GDCM reads an inf spacing as 1.0 and a non-finite rescale as 1 or 0; allclose takes inf == inf as close.
    if not (np.isfinite(pixel_spacing).all() and np.isfinite(rescale).all()
            and (pixel_spacing > 0).all()  # SetSpacing raises on a zero or negative spacing
            and np.allclose(pixel_spacing, pixel_spacing[0], rtol=0, atol=SERIES_ATOL_MM)
            and len({(int(ds.Rows), int(ds.Columns)) for ds in heads}) == 1
            and len({str(ds.FrameOfReferenceUID) for ds in heads}) == 1):
        return None
    normal = _cosines(heads[0])[2]
    positions = np.asarray([_position(ds) for ds in heads])
    order = np.argsort(positions @ normal, kind='stable')
    positions = positions[order]
    n = len(heads)
    step = (positions[-1] - positions[0]) @ normal / (n - 1)
    expected = positions[0] + np.outer(np.arange(n) * step, normal)  # distance to the line's k-th point: sideways counts
    if not (step > SERIES_ATOL_MM and np.linalg.norm(positions - expected, axis=1).max() <= SERIES_ATOL_MM):
        return None
    return [slices[i] for i in order]


def check_volume(slices):
    """
    `slices`: [(path, header Dataset)] of ONE series. The slices sorted ascending along the normal (row x column), or None
    (skip) unless: single-frame files; one orientation, image size, pixel spacing and FrameOfReferenceUID; unit, orthogonal
    cosines; at least two slices; uniform step along the normal with every position (sideways too) within SERIES_ATOL_MM.
    """
    return _or_none(_sorted_volume, slices)


def _geometry(ordered):
    first = ordered[0][1]
    row, column, normal = _cosines(first)
    row_spacing, column_spacing = _pixel_spacing(first)
    n = len(ordered)
    step = (_position(ordered[-1][1]) - _position(first)) @ normal / (n - 1)
    size = (int(first.Columns), int(first.Rows), n)
    direction = tuple(np.column_stack([row, column, normal]).ravel())
    return size, tuple(_position(first)), (column_spacing, row_spacing, step), direction


def read_series(slices):
    """
    The float32 image of one series (stored x RescaleSlope + RescaleIntercept, each slice's own), origin = first position,
    spacing = (column, row, step), direction = (row cosine, column cosine, normal) as columns; all from the headers.
    None (skip) when check_volume skips it, a pixel read fails or the image size differs from the headers.
    """
    ordered = check_volume(slices)
    geometry = None if ordered is None else _or_none(_geometry, ordered)
    if geometry is None:
        return None
    size, origin, spacing, direction = geometry
    paths = [str(path) for path, _ in ordered]

    def read():
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(paths)
        reader.SetOutputPixelType(sitk.sitkFloat32)  # before Execute, so each slice's rescale applies
        return reader.Execute()
    image, failed = None, False
    try:
        image = quiet_call(read, REASON_SERIES_UNREADABLE)
    except CaseRefused:
        failed = True
    if failed or image.GetSize() != size:
        return None
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    return image


IDENTITY = 'identity'
RIGID_ATOL = 1e-4  # max |R^T R - I| of a registration matrix's 3x3 part
LAST_ROW_ATOL = 1e-6  # last row of a registration matrix against 0 0 0 1


def _rigid_matrix(item):
    """The 4x4 matrix of one REG item, or None unless its MatrixSequence holds exactly one usable RIGID matrix."""
    entries = [m for registration in item.MatrixRegistrationSequence for m in registration.MatrixSequence]
    if len(entries) != 1 or str(entries[0].FrameOfReferenceTransformationMatrixType) != 'RIGID':
        return None
    values = [float(v) for v in entries[0].FrameOfReferenceTransformationMatrix]
    if len(values) != 16:
        return None
    matrix = np.asarray(values).reshape(4, 4)
    rotation = matrix[:3, :3]
    usable = (np.isfinite(matrix).all()  # a NaN or inf translation would pass the rotation checks below
              and np.abs(matrix[3] - (0, 0, 0, 1)).max() <= LAST_ROW_ATOL
              and np.abs(rotation.T @ rotation - np.eye(3)).max() <= RIGID_ATOL and np.linalg.det(rotation) > 0)
    return matrix if usable else None


def _item_frame(item, headers_by_sop):
    """An item's own FrameOfReferenceUID, else the one frame shared by the headers its ReferencedImageSequence names."""
    if 'FrameOfReferenceUID' in item and str(item.FrameOfReferenceUID):
        return str(item.FrameOfReferenceUID)
    sops = [str(ref.ReferencedSOPInstanceUID) for ref in item.ReferencedImageSequence] if 'ReferencedImageSequence' in item else []
    frames = {str(headers_by_sop[sop].FrameOfReferenceUID) for sop in sops
              if sop in headers_by_sop and 'FrameOfReferenceUID' in headers_by_sop[sop]}
    return frames.pop() if len(frames) == 1 else None


def _reg_text(reg, ct_frame, series_frame, headers_by_sop):
    """The reg_matrix text of one REG for the pair ('' for no usable matrix), or None when the REG has no item for a frame."""
    own = str(reg.FrameOfReferenceUID) if 'FrameOfReferenceUID' in reg else None
    items = [(_item_frame(item, headers_by_sop), item) for item in reg.RegistrationSequence]
    matrices = {}
    for frame in (ct_frame, series_frame):
        found = [_or_none(_rigid_matrix, item) for item_frame, item in items if item_frame == frame]
        matrices[frame] = found or ([np.eye(4)] if own == frame else [])  # symmetric: the REG's own frame has an identity item
    if not (matrices[ct_frame] and matrices[series_frame]):
        return None
    ct_matrix, series_matrix = matrices[ct_frame], matrices[series_frame]
    if len(ct_matrix) > 1 or len(series_matrix) > 1 or ct_matrix[0] is None or series_matrix[0] is None:
        return ''
    matrix = np.round(np.linalg.inv(ct_matrix[0]) @ series_matrix[0], 10) + 0.0  # a computed zero prints 0, not -0 or 1e-17
    return ' '.join('%.10g' % v for v in matrix.ravel())


def registration(ct_frame, series_frame, regs, headers_by_sop):
    """
    (registered, reg_matrix) of a series on `series_frame` against the planning CT on `ct_frame`.

    `regs`: the patient's REG headers, already deduplicated by SOPInstanceUID and without those in an excluded study (the
    caller does both). `headers_by_sop`: {SOPInstanceUID: header} of every readable file of the patient, excluded studies
    and localizers included; it resolves an item that has no FrameOfReferenceUID from its ReferencedImageSequence.
    reg_matrix: 'identity' for the same frame (no REG is read); else, from the first qualifying REG by SOPInstanceUID
    (string order, no fallback), 16 values row-major '%.10g' = inv(M_ct) . M_series (series frame -> CT frame), or '' when
    that REG's items do not give exactly one usable rigid matrix per frame, or when no REG qualifies (registered False).
    A REG qualifies with an item (real or implied by its own frame) for each frame, whatever its matrices hold; a REG with a
    malformed tag does not qualify. Nothing here raises for a bad header.
    """
    if not ct_frame:  # no FrameOfReferenceUID on the CT: an empty series frame is not the same frame
        return False, ''
    if ct_frame == series_frame:
        return True, IDENTITY
    keyed = [(sop, reg) for reg in regs if (sop := _or_none(lambda r: str(r.SOPInstanceUID), reg)) is not None]
    for _, reg in sorted(keyed, key=lambda pair: pair[0]):
        text = _or_none(_reg_text, reg, ct_frame, series_frame, headers_by_sop)
        if text is not None:
            return True, text
    return False, ''


IMAGE_MODALITIES = ('CT', 'MR', 'PT')


def _file_keys(ds):
    """(SOPInstanceUID, StudyInstanceUID, Modality, ManufacturerModelName) of one header, each as text ('' when absent)."""
    return tuple(str(ds.get(tag, '')) for tag in ('SOPInstanceUID', 'StudyInstanceUID', 'Modality', 'ManufacturerModelName'))


def _candidate_series(ds):
    """The SeriesInstanceUID of a non-localizer CT/MR/PT file, else None."""
    if ds.get('Modality') in IMAGE_MODALITIES and ds.get('SeriesInstanceUID') and 'LOCALIZER' not in ds.get('ImageType', []):
        return str(ds.SeriesInstanceUID)
    return None


def tainted_studies(headers):
    """
    (StudyInstanceUIDs, blind) of the files candidates() drops for a bad tag (#44): every other series of such a study is
    skipped, since the dropped file may be its lost slice or its REG or exclusion context. blind: one had no readable study.
    """
    studies, blind = set(), False
    for _, ds in headers:
        # candidates() also drops a file whose _candidate_series fails, a bad SeriesInstanceUID or ImageType (#69): '' stands
        # for its normal None, so None here means it raised
        if _or_none(_file_keys, ds) is None or _or_none(lambda d: _candidate_series(d) or '', ds) is None:
            study = _or_none(lambda d: str(d.StudyInstanceUID), ds)
            if study:
                studies.add(study)
            else:
                blind = True
    return studies, blind


def candidates(headers, exclude_models):
    """
    (series, regs, headers_by_sop) of one patient. `headers`: [(path, Dataset)] of every readable file, in folder order
    (the caller's; the first copy of a SOP is the one kept). A file whose SOPInstanceUID, StudyInstanceUID, Modality or
    ManufacturerModelName cannot be read is dropped first, and so is the series it belongs to (when that UID reads).
    series: [(SeriesInstanceUID, [(path, header)])] of the CT/MR/PT series, first-seen order; localizer files are dropped one
    by one, files without a SeriesInstanceUID are dropped, and so is every file of a study that holds a file whose
    ManufacturerModelName contains an excluded model (case-sensitive substring) or that has no StudyInstanceUID.
    regs: the REG headers outside those studies. headers_by_sop: {SOPInstanceUID: header} of all kept files, excluded
    studies and localizers included. The caller still removes the planning CTs of converted plans from `series`.
    """
    keyed, broken = [], set()
    for path, ds in headers:
        keys = _or_none(_file_keys, ds)
        if keys is None:  # a dropped slice may be an end slice, which check_volume cannot see: drop its whole series
            # shortcut: a file whose SeriesInstanceUID is unreadable too taints nothing, upgrade if one shows up in real data
            broken.add(_or_none(lambda d: str(d.SeriesInstanceUID), ds))
        else:
            keyed.append((path, ds, keys))
    excluded = {study for _, _, (_, study, _, model) in keyed if any(m in model for m in exclude_models)}
    excluded.add('')
    unique, seen = [], set()
    for path, ds, (sop, study, modality, _) in keyed:
        if sop and sop in seen:
            continue
        seen.add(sop)
        unique.append((path, ds, study, modality))
    series = defaultdict(list)
    for path, ds, study, _ in unique:
        if study not in excluded and (uid := _or_none(_candidate_series, ds)):
            series[uid].append((path, ds))
    regs = [ds for _, ds, study, modality in unique if modality == 'REG' and study not in excluded]
    by_sop = {}
    for _, ds, (sop, *_) in keyed:
        by_sop.setdefault(sop, ds)
    return [(uid, slices) for uid, slices in series.items() if uid not in broken], regs, by_sop
