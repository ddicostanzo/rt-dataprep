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


def test_reg_defaults_have_no_registration_sequence(tmp_path):
    path = tmp_path / 'reg.dcm'
    sd.write_reg(path, sd.generate_uid(), sd.generate_uid())
    ds = pydicom.dcmread(path)
    assert 'RegistrationSequence' not in ds and 'FrameOfReferenceUID' not in ds


def test_reg_registration_sequence_round_trip(tmp_path):
    path = tmp_path / 'reg.dcm'
    reg_frame, ct_frame, mr_frame, image = (sd.generate_uid() for _ in range(4))
    shift = sd.transformation_matrix(translation=(10.0, -2.5, 0.125))
    items = [
        dict(frame_uid=ct_frame, matrices=[('RIGID', sd.transformation_matrix())]),
        dict(frame_uid=mr_frame, images=[image], matrices=[('RIGID', shift), ('AFFINE', [2.0] + [0.0] * 15)]),
        dict(matrices=[('RIGID_SCALE', shift)]),  # no frame, no images
    ]
    sd.write_reg(path, sd.generate_uid(), sd.generate_uid(), frame_uid=reg_frame, registrations=items)
    ds = pydicom.dcmread(path)
    assert ds.FrameOfReferenceUID == reg_frame
    first, second, third = ds.RegistrationSequence
    assert first.FrameOfReferenceUID == ct_frame and 'ReferencedImageSequence' not in first
    assert second.FrameOfReferenceUID == mr_frame
    assert [i.ReferencedSOPInstanceUID for i in second.ReferencedImageSequence] == [image]
    assert 'FrameOfReferenceUID' not in third
    matrices = [[(m.FrameOfReferenceTransformationMatrixType, [float(v) for v in m.FrameOfReferenceTransformationMatrix])
                 for m in item.MatrixRegistrationSequence[0].MatrixSequence] for item in (first, second, third)]
    assert matrices[0] == [('RIGID', [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])]
    assert matrices[1] == [('RIGID', [1, 0, 0, 10, 0, 1, 0, -2.5, 0, 0, 1, 0.125, 0, 0, 0, 1]),
                           ('AFFINE', [2.0] + [0.0] * 15)]
    assert matrices[2][0][0] == 'RIGID_SCALE'


def test_transformation_matrix_maps_item_frame_to_reg_frame():
    rotation = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90 degrees about z
    values = sd.transformation_matrix(rotation=rotation, translation=(10, 20, 30))
    assert values == [0, -1, 0, 10, 1, 0, 0, 20, 0, 0, 1, 30, 0, 0, 0, 1]  # M11 M12 M13 M14 M21 ...
    point = np.array(values).reshape(4, 4) @ [1, 0, 0, 1]  # (1, 0, 0) in the item frame
    assert list(point[:3]) == [10, 21, 30]


def test_ct_series_defaults_unchanged(tmp_path):
    ct = sd.write_ct_series(tmp_path, np.zeros((2, 3, 4)), spacing=(0.5, 2.0, 3.0), origin=(10.0, 20.0, 30.0))
    ds = [pydicom.dcmread(p) for p in ct['paths']]
    assert [list(d.ImagePositionPatient) for d in ds] == [[10.0, 20.0, 30.0], [10.0, 20.0, 33.0]]
    assert list(ds[0].ImageOrientationPatient) == [1, 0, 0, 0, 1, 0]
    assert list(ds[0].PixelSpacing) == [2.0, 0.5] and ds[0].Modality == 'CT'
    assert (ds[0].RescaleSlope, ds[0].RescaleIntercept) == (1, 0)
    assert 'NumberOfFrames' not in ds[0]


def test_ct_series_sagittal_orientation(tmp_path):
    orientation = (0, 1, 0, 0, 0, -1)  # normal = row x column = (-1, 0, 0)
    ct = sd.write_ct_series(tmp_path, np.zeros((3, 4, 5)), spacing=(1.0, 1.0, 2.5), origin=(10.0, 20.0, 30.0),
                            orientation=orientation)
    ds = [pydicom.dcmread(p) for p in ct['paths']]
    assert list(ds[0].ImageOrientationPatient) == list(orientation)
    assert [list(d.ImagePositionPatient) for d in ds] == [[10.0, 20.0, 30.0], [7.5, 20.0, 30.0], [5.0, 20.0, 30.0]]


def test_ct_series_explicit_positions(tmp_path):
    positions = [(0, 0, 0), (0, 0, 1.01), (0, 0, 1.01), (0, 0, 5)]  # jitter, a duplicate, a gap
    ct = sd.write_ct_series(tmp_path, np.zeros((4, 2, 2)), positions=positions)
    ds = [pydicom.dcmread(p) for p in ct['paths']]
    assert [list(d.ImagePositionPatient) for d in ds] == [list(p) for p in positions]
    assert [d.InstanceNumber for d in ds] == [1, 2, 3, 4]


def test_ct_series_pet_modality_and_rescale(tmp_path):
    ct = sd.write_ct_series(tmp_path, np.zeros((2, 2, 2)), modality='PT', rescales=[(0.5, 1), (2.25, -3)])
    ds = [pydicom.dcmread(p) for p in ct['paths']]
    assert {d.Modality for d in ds} == {'PT'}
    assert {d.SOPClassUID for d in ds} == {pydicom.uid.PositronEmissionTomographyImageStorage}
    assert [(float(d.RescaleSlope), float(d.RescaleIntercept)) for d in ds] == [(0.5, 1), (2.25, -3)]


def test_ct_series_number_of_frames_and_no_pixel_spacing(tmp_path):
    ct = sd.write_ct_series(tmp_path, np.zeros((2, 2, 2)), number_of_frames=4, pixel_spacing=False)
    ds = [pydicom.dcmread(p) for p in ct['paths']]
    assert {d.NumberOfFrames for d in ds} == {4}
    assert all('PixelSpacing' not in d for d in ds)
