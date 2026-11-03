"""
Synthetic DICOM for the rt_dataprep tests: CT slices, RTSTRUCT, brachy RTPLAN, multi-frame RTDOSE, REG. No patient data.
Ported from rnt_transformer tests/synthetic_dicom.py.

Every UID comes from generate_uid() at run time, PatientID and PatientName start with SYNTHETIC_, and a date or time tag
is set only when asked for. Files carry file meta and the explicit-VR little-endian transfer syntax.

Every writer takes the shared metadata keywords patient_id=, study_uid=, series_date=, series_time= and
manufacturer_model_name= (all default to "not set" / a fresh study UID / SYNTHETIC_ID), forwarded to `_dataset`.
"""
import contextlib
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

PATIENT_NAME = 'SYNTHETIC_Patient'
PATIENT_ID = 'SYNTHETIC_ID'
# Long enough to be an invalid PN (a component over 64 characters) and not a DA, and holding the SYNTHETIC_ marker.
INVALID_PN = 'SYNTHETIC_' + 'x' * 70
INVALID_DA = 'SYNTHETIC_notadate'


def _dataset(path, sop_class_uid, modality, sop_uid=None, frame_uid=None, series_uid=None, invalid_values=False,
             patient_id=None, study_uid=None, series_date=None, series_time=None, manufacturer_model_name=None):
    ds = Dataset()
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_uid or generate_uid()
    ds.Modality = modality
    with _writing_unvalidated():  # assigning the deliberately invalid values would warn
        ds.PatientName = INVALID_PN if invalid_values else PATIENT_NAME
        if invalid_values:
            ds.StudyDate = INVALID_DA
    ds.PatientID = patient_id or PATIENT_ID
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    if series_date:
        ds.SeriesDate = series_date
    if series_time:
        ds.SeriesTime = series_time
    if manufacturer_model_name:
        ds.ManufacturerModelName = manufacturer_model_name
    if frame_uid:
        ds.FrameOfReferenceUID = frame_uid
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.path = Path(path)  # not a DICOM element: a plain attribute for _save
    return ds


def _save(ds):
    with _writing_unvalidated():
        pydicom.dcmwrite(ds.path, ds, enforce_file_format=True)
    return ds.path


@contextlib.contextmanager
def _writing_unvalidated():
    """Both validation modes off (assigning a PN or DA validates with the reading mode), then back as they were."""
    settings = pydicom.config.settings
    modes = settings.reading_validation_mode, settings.writing_validation_mode
    settings.reading_validation_mode = settings.writing_validation_mode = pydicom.config.IGNORE
    try:
        yield
    finally:
        settings.reading_validation_mode, settings.writing_validation_mode = modes


def write_ct_series(folder, values, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), series_uid=None, frame_uid=None,
                    prefix='ct', image_type=None, **meta):
    """
    One CT slice file per z of `values` (a (z, y, x) int array; HU as stored, slope 1, intercept 0), axial, slice k at
    origin z + k * spacing z, in files named <prefix>_<k>.dcm (a second series in the folder needs its own prefix).
    image_type: the ImageType list (None omits it), e.g. ['ORIGINAL', 'PRIMARY', 'LOCALIZER']. All slices share one study
    (study_uid= or a fresh one). Returns {'series_uid', 'frame_uid', 'study_uid', 'sop_uids', 'paths'}.
    """
    series_uid, frame_uid = series_uid or generate_uid(), frame_uid or generate_uid()
    values = np.asarray(values, dtype=np.int16)
    meta['study_uid'] = meta.get('study_uid') or generate_uid()
    sop_uids, paths = [], []
    for k, plane in enumerate(values):
        ds = _dataset(Path(folder) / f'{prefix}_{k:03d}.dcm', pydicom.uid.CTImageStorage, 'CT', frame_uid=frame_uid,
                      series_uid=series_uid, **meta)
        if image_type:
            ds.ImageType = list(image_type)
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + k * spacing[2]]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [spacing[1], spacing[0]]  # [between rows, between columns]
        ds.SliceThickness = spacing[2]
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        _set_pixels(ds, plane, bits=16, signed=True)
        sop_uids.append(ds.SOPInstanceUID)
        paths.append(_save(ds))
    return dict(series_uid=series_uid, frame_uid=frame_uid, study_uid=meta['study_uid'], sop_uids=sop_uids, paths=paths)


def write_rtstruct(path, ct, sop_uid=None, extra_series_uids=(), **meta):
    """
    An RTSTRUCT referencing the CT series dict `ct` (and the series UIDs in `extra_series_uids`) through the study and
    series chain. Returns its SOPInstanceUID.
    """
    ds = _dataset(path, pydicom.uid.RTStructureSetStorage, 'RTSTRUCT', sop_uid=sop_uid, frame_uid=ct['frame_uid'], **meta)
    referenced = []
    for uid in (ct['series_uid'], *extra_series_uids):
        series = Dataset()
        series.SeriesInstanceUID = uid
        referenced.append(series)
    study = Dataset()
    study.ReferencedSOPInstanceUID = generate_uid()
    study.RTReferencedSeriesSequence = referenced
    frame = Dataset()
    frame.FrameOfReferenceUID = ct['frame_uid']
    frame.RTReferencedStudySequence = [study]
    ds.ReferencedFrameOfReferenceSequence = [frame]
    _save(ds)
    return ds.SOPInstanceUID


def write_rtplan(path, struct_sop_uid, positions_mm, times=None, fractions=1, sop_uid=None, frame_uid=None,
                  dose_references=None, channels=None, plan_date=None, plan_time=None, **meta):
    """
    A brachy RTPLAN: ReferencedStructureSetSequence -> `struct_sop_uid`, one fraction group with `fractions`
    NumberOfFractionsPlanned (None writes it empty), one channel whose control points are `positions_mm` (N, 3) patient
    mm with a CumulativeTimeWeight (`times`, default 1 each). dose_references: [(description, point_mm or None)], a
    DoseReferenceSequence of COORDINATES entries, or of SITE entries when the point is None (None omits the sequence).
    channels: a list of dicts that replaces the single channel (positions_mm is then ignored): applicator_id (the
    SourceApplicatorID), dwells_mm (N, 3), times (default 1 each), roi (ReferencedROINumber, default the channel
    number), step_size (SourceApplicatorStepSize, omitted when None), paired (default True: control points 2k and 2k+1
    both at dwell k, the time passing between them). plan_date / plan_time set RTPlanDate / RTPlanTime (omitted when None).
    Returns its SOPInstanceUID.
    """
    ds = _dataset(path, pydicom.uid.RTPlanStorage, 'RTPLAN', sop_uid=sop_uid, frame_uid=frame_uid, **meta)
    if plan_date:
        ds.RTPlanDate = plan_date
    if plan_time:
        ds.RTPlanTime = plan_time
    reference = Dataset()
    reference.ReferencedSOPClassUID = pydicom.uid.RTStructureSetStorage
    reference.ReferencedSOPInstanceUID = struct_sop_uid
    ds.ReferencedStructureSetSequence = [reference]
    group = Dataset()
    group.FractionGroupNumber = 1
    group.NumberOfFractionsPlanned = fractions
    ds.FractionGroupSequence = [group]
    if channels is None:
        channels = [dict(dwells_mm=positions_mm, times=times, paired=False)]
    channel_datasets = [_channel(number, **spec) for number, spec in enumerate(channels, start=1)]
    setup = Dataset()
    setup.ApplicationSetupNumber = 1
    setup.ChannelSequence = channel_datasets
    ds.ApplicationSetupSequence = [setup]
    if dose_references is not None:
        references = []
        for i, (description, point) in enumerate(dose_references):
            reference = Dataset()
            reference.DoseReferenceNumber = i + 1
            reference.DoseReferenceStructureType = 'SITE' if point is None else 'COORDINATES'
            reference.DoseReferenceDescription = description
            if point is not None:
                reference.DoseReferencePointCoordinates = [float(v) for v in point]
            references.append(reference)
        ds.DoseReferenceSequence = references
    _save(ds)
    return ds.SOPInstanceUID


def _channel(number, dwells_mm, times=None, applicator_id=None, roi=None, step_size=None, paired=True):
    positions = np.asarray(dwells_mm, dtype=float).reshape(-1, 3)
    times = np.ones(len(positions)) if times is None else np.asarray(times, dtype=float)
    if paired:
        weights = np.cumsum(times)
        # even control point: the weight on arrival; odd: after the dwell
        rows = [(position, weight) for k, position in enumerate(positions)
                for weight in (weights[k] - times[k], weights[k])]
    else:
        rows = list(zip(positions, np.cumsum(times), strict=True))
    points = []
    for i, (position, weight) in enumerate(rows):
        point = Dataset()
        point.ControlPointIndex = i
        point.ControlPoint3DPosition = [float(v) for v in position]
        point.CumulativeTimeWeight = float(weight)
        points.append(point)
    channel = Dataset()
    channel.ChannelNumber = number
    channel.BrachyControlPointSequence = points
    if applicator_id is not None:
        channel.SourceApplicatorID = applicator_id
        channel.ReferencedROINumber = number if roi is None else roi
    if step_size is not None:
        channel.SourceApplicatorStepSize = float(step_size)
    return channel


def write_rtstruct_contours(path, contours, sop_uid=None, brachy=True, **meta):
    """
    An RTSTRUCT of open contours: `contours` is [(roi_number, name, points_mm (N, 3))]. Each ROI gets its ContourData in
    ContourSequence items of at most 10 points, and with brachy=True the OPEN_NONPLANAR type and a BRACHY_CHANNEL
    observation (False: a CLOSED_PLANAR contour and no observation). Returns its SOPInstanceUID.
    """
    ds = _dataset(path, pydicom.uid.RTStructureSetStorage, 'RTSTRUCT', sop_uid=sop_uid, frame_uid=generate_uid(), **meta)
    rois, roi_contours, observations = [], [], []
    for number, name, points in contours:
        roi = Dataset()
        roi.ROINumber, roi.ROIName = number, name
        rois.append(roi)
        items = []
        for k, chunk in enumerate(np.array_split(np.asarray(points, dtype=float), -(-len(points) // 10))):
            item = Dataset()
            item.ContourNumber = k + 1
            item.ContourGeometricType = 'OPEN_NONPLANAR' if brachy else 'CLOSED_PLANAR'
            item.NumberOfContourPoints = len(chunk)
            item.ContourData = [float(v) for v in chunk.ravel()]
            items.append(item)
        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = number
        roi_contour.ContourSequence = items
        roi_contours.append(roi_contour)
        if brachy:
            observation = Dataset()
            observation.ObservationNumber = number
            observation.ReferencedROINumber = number
            observation.RTROIInterpretedType = 'BRACHY_CHANNEL'
            observations.append(observation)
    ds.StructureSetROISequence = rois
    ds.ROIContourSequence = roi_contours
    ds.RTROIObservationsSequence = observations
    _save(ds)
    return ds.SOPInstanceUID


def write_rtdose(path, stored, scaling=1e-4, position=(0.0, 0.0, 0.0), orientation=(1, 0, 0, 0, 1, 0),
                 pixel_spacing=(2.0, 3.0), offsets=None, plan_sop_uid=None, frame_uid=None, units='GY',
                 dose_type='PHYSICAL', summation='PLAN', number_of_frames=None, invalid_values=False, sop_uid=None, **meta):
    """
    A multi-frame RTDOSE. stored: (frames, rows, columns) unsigned ints, the dose is stored x `scaling` Gy.
    position: ImagePositionPatient of the first frame. offsets: GridFrameOffsetVector (default 0, 1, 2 ... mm);
    pass the first frame's position along the slice axis for the absolute convention. plan_sop_uid=None omits
    ReferencedRTPlanSequence. number_of_frames overrides the NumberOfFrames tag. invalid_values writes an invalid PN
    and DA that contain a SYNTHETIC_ string. Returns its SOPInstanceUID.
    """
    stored = np.asarray(stored, dtype=np.uint32)
    frames, rows, columns = stored.shape
    ds = _dataset(path, pydicom.uid.RTDoseStorage, 'RTDOSE', sop_uid=sop_uid, frame_uid=frame_uid or generate_uid(),
                  invalid_values=invalid_values, **meta)
    ds.DoseUnits, ds.DoseType, ds.DoseSummationType = units, dose_type, summation
    ds.DoseGridScaling = scaling
    if plan_sop_uid:
        reference = Dataset()
        reference.ReferencedSOPClassUID = pydicom.uid.RTPlanStorage
        reference.ReferencedSOPInstanceUID = plan_sop_uid
        ds.ReferencedRTPlanSequence = [reference]
    ds.ImagePositionPatient = [float(v) for v in position]
    ds.ImageOrientationPatient = [float(v) for v in orientation]
    ds.PixelSpacing = [float(v) for v in pixel_spacing]
    ds.NumberOfFrames = frames if number_of_frames is None else number_of_frames
    ds.FrameIncrementPointer = pydicom.tag.Tag('GridFrameOffsetVector')
    ds.GridFrameOffsetVector = [float(v) for v in (range(frames) if offsets is None else offsets)]
    _set_pixels(ds, stored, bits=32, signed=False)
    _save(ds)
    return ds.SOPInstanceUID


def write_reg(path, study_uid, other_study_uid, sop_uid=None, **meta):
    """
    A spatial registration (REG) in study `study_uid` whose StudiesContainingOtherReferencedInstancesSequence names
    `other_study_uid` (a study UID, or a list of them). Returns its SOPInstanceUID.
    """
    ds = _dataset(path, pydicom.uid.SpatialRegistrationStorage, 'REG', sop_uid=sop_uid, study_uid=study_uid, **meta)
    others = [other_study_uid] if isinstance(other_study_uid, str) else other_study_uid
    items = []
    for uid in others:
        item = Dataset()
        item.StudyInstanceUID = uid
        items.append(item)
    ds.StudiesContainingOtherReferencedInstancesSequence = items
    _save(ds)
    return ds.SOPInstanceUID


def _set_pixels(ds, array, bits, signed):
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows, ds.Columns = array.shape[-2:]
    ds.BitsAllocated = ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = int(signed)
    ds.PixelData = np.ascontiguousarray(array).tobytes()
