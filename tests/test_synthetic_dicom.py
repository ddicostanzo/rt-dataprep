import numpy as np
import pydicom
from DicomRTTool import DicomReaderWriter

import synthetic_dicom as sd

SPACING, ORIGIN = (0.5, 2.0, 3.0), (10.0, 20.0, 30.0)
BOX = ((1, 2), (1, 3), (2, 5))  # inclusive (z, y, x) voxel index ranges
POINT_MM, CHANNEL_MM = (11.0, 22.0, 33.0), [(10.0, 20.0, 30.0 + 3 * k) for k in range(12)]
ROIS = [
    dict(number=3, name='SYNTHETICBOX', kind='box', box=BOX, interpreted='ORGAN'),
    dict(number=5, name='SYNTHETICPOINT', kind='point', point=POINT_MM, interpreted='MARKER'),
    dict(number=7, name='SYNTHETICEMPTY', kind='empty', interpreted='PTV'),
    dict(number=9, name='SYNTHETICCHANNEL', kind='channel', points=CHANNEL_MM, interpreted='BRACHY_CHANNEL'),
]


def make(tmp_path, **kwargs):
    ct = sd.write_ct_series(tmp_path, np.zeros((4, 6, 8)), spacing=SPACING, origin=ORIGIN, patient_id='SYNTHETICPTA')
    path = tmp_path / 'struct.dcm'
    sop = sd.write_rtstruct_rois(path, ct, ROIS, patient_id='SYNTHETICPTA', **kwargs)
    return ct, pydicom.dcmread(path), sop


def test_rois_frame_and_series_chain(tmp_path):
    ct, ds, sop = make(tmp_path)
    assert ds.SOPInstanceUID == sop
    assert ds.Modality == 'RTSTRUCT'
    assert ds.FrameOfReferenceUID == ct['frame_uid']
    frame = ds.ReferencedFrameOfReferenceSequence[0]
    assert frame.FrameOfReferenceUID == ct['frame_uid']
    series = frame.RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
    assert series.SeriesInstanceUID == ct['series_uid']
    assert [i.ReferencedSOPInstanceUID for i in series.ContourImageSequence] == ct['sop_uids']
    assert {i.ReferencedSOPClassUID for i in series.ContourImageSequence} == {pydicom.uid.CTImageStorage}


def test_rois_without_slice_list(tmp_path):
    _, ds, _ = make(tmp_path, contour_images=False)
    series = ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
    assert 'ContourImageSequence' not in series
    # the contours still reference their slice
    assert 'ContourImageSequence' in ds.ROIContourSequence[0].ContourSequence[0]


def test_rois_numbers_names_types(tmp_path):
    ct, ds, _ = make(tmp_path)
    assert [(r.ROINumber, r.ROIName) for r in ds.StructureSetROISequence] == [(r['number'], r['name']) for r in ROIS]
    assert {r.ReferencedFrameOfReferenceUID for r in ds.StructureSetROISequence} == {ct['frame_uid']}
    assert [(o.ReferencedROINumber, o.RTROIInterpretedType) for o in ds.RTROIObservationsSequence] == \
        [(r['number'], r['interpreted']) for r in ROIS]
    assert [c.ReferencedROINumber for c in ds.ROIContourSequence] == [3, 5, 7, 9]


def test_box_roi_polygons_on_each_slice(tmp_path):
    ct, ds, _ = make(tmp_path)
    items = ds.ROIContourSequence[0].ContourSequence
    assert len(items) == 2  # z index 1 and 2
    for item, k in zip(items, (1, 2), strict=True):
        assert item.ContourGeometricType == 'CLOSED_PLANAR'
        assert item.NumberOfContourPoints == 4
        assert [i.ReferencedSOPInstanceUID for i in item.ContourImageSequence] == [ct['sop_uids'][k]]
        xyz = np.array(item.ContourData, dtype=float).reshape(-1, 3)
        assert set(xyz[:, 2]) == {ORIGIN[2] + k * SPACING[2]}
        assert set(xyz[:, 0]) == {ORIGIN[0] + 2 * SPACING[0], ORIGIN[0] + 5 * SPACING[0]}  # x index 2 and 5
        assert set(xyz[:, 1]) == {ORIGIN[1] + 1 * SPACING[1], ORIGIN[1] + 3 * SPACING[1]}  # y index 1 and 3


def test_point_empty_and_channel_rois(tmp_path):
    _, ds, _ = make(tmp_path)
    point, empty, channel = ds.ROIContourSequence[1], ds.ROIContourSequence[2], ds.ROIContourSequence[3]
    assert point.ContourSequence[0].ContourGeometricType == 'POINT'
    assert point.ContourSequence[0].NumberOfContourPoints == 1
    assert list(point.ContourSequence[0].ContourData) == list(POINT_MM)
    assert 'ContourSequence' not in empty
    assert {i.ContourGeometricType for i in channel.ContourSequence} == {'OPEN_NONPLANAR'}
    assert [i.NumberOfContourPoints for i in channel.ContourSequence] == [6, 6]  # array_split balances 12 points
    data = np.concatenate([np.array(i.ContourData, dtype=float) for i in channel.ContourSequence]).reshape(-1, 3)
    assert np.allclose(data, CHANNEL_MM)


def test_dicomrttool_rasterizes_box_exactly(tmp_path):
    ct, _, _ = make(tmp_path)
    reader = DicomReaderWriter(Contour_Names=['SYNTHETICBOX'], arg_max=False, verbose=False)
    reader.walk_through_folders(str(tmp_path))
    reader.get_images_and_mask()
    mask = np.asarray(reader.mask)
    assert mask.shape[1:3] == (6, 8) and mask[..., 1].sum() == 2 * 3 * 4  # channel 0 is background


def test_rtplan_label(tmp_path):
    plan = tmp_path / 'plan.dcm'
    sd.write_rtplan(plan, sd.generate_uid(), [(0, 0, 0)], label='SYNTHETICLABEL')
    assert pydicom.dcmread(plan).RTPlanLabel == 'SYNTHETICLABEL'
    sd.write_rtplan(plan, sd.generate_uid(), [(0, 0, 0)])
    assert 'RTPlanLabel' not in pydicom.dcmread(plan)


def test_rtdose_parameters_round_trip(tmp_path):
    path = tmp_path / 'dose.dcm'
    plan_uid, frame_uid = sd.generate_uid(), sd.generate_uid()
    sd.write_rtdose(path, np.ones((2, 2, 2)), units='RELATIVE', summation='FRACTION', plan_sop_uid=plan_uid,
                    frame_uid=frame_uid, patient_id='SYNTHETICPTA')
    ds = pydicom.dcmread(path)
    assert (ds.DoseUnits, ds.DoseSummationType, ds.FrameOfReferenceUID) == ('RELATIVE', 'FRACTION', frame_uid)
    assert ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID == plan_uid
