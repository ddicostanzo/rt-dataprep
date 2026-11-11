import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

import synthetic_dicom as sd
from rt_dataprep import metadata
from rt_dataprep.dicom_io import read_header
from rt_dataprep.metadata import ALL, IMAGES, RTDOSE, RTPLAN, RTSTRUCT, collect, keys, reg_keys

# Every tag of the All group (patient_age comes from PatientAge). All values are synthetic.
ALL_TAGS = dict(
    PatientAge='063Y', PatientSex='O', PatientSize='1.7', PatientWeight='70', StudyDate='20240101', StudyTime='101010',
    StudyDescription='SYNTHETIC_STUDY', AccessionNumber='SYNTHETIC_ACC', StudyID='S1', SeriesNumber='2',
    SeriesTime='101112', AcquisitionDate='20240102', AcquisitionTime='101213', ContentDate='20240103',
    ContentTime='101314', Manufacturer='SYNTHETIC_MAKER', ManufacturerModelName='SYNTHETIC_MODEL',
    SoftwareVersions='SYNTHETIC_SW', InstitutionName='SYNTHETIC_HOSPITAL', StationName='SYNTHETIC_STATION',
    DeviceSerialNumber='SYNTHETIC_SN', BodyPartExamined='PELVIS', PatientPosition='HFS', ProtocolName='SYNTHETIC_PROTO',
)
PT_SEQUENCE = [dict(
    Radiopharmaceutical='SYNTHETIC_TRACER', RadiopharmaceuticalCodeSequence=[dict(CodeMeaning='SYNTHETIC_TRACER_CODE')],
    RadionuclideCodeSequence=[dict(CodeMeaning='SYNTHETIC_NUCLIDE')], RadionuclideTotalDose='370000000',
    RadiopharmaceuticalStartDateTime='20240101090000', RadiopharmaceuticalStartTime='090000', RadionuclideHalfLife='6586',
)]
# Every tag of the Images group that the synthetic CT writer does not already set (Rows, Columns, PixelSpacing, SliceThickness).
IMAGES_TAGS = dict(
    ImageType=['ORIGINAL', 'PRIMARY'], SpacingBetweenSlices='3', ContrastBolusAgent='SYNTHETIC_AGENT',
    KVP='120', XRayTubeCurrent='200', Exposure='300', ExposureTime='400', ConvolutionKernel='SYNTHETIC_B30',
    ReconstructionDiameter='500', CTDIvol=10.5,
    MagneticFieldStrength='3', ScanningSequence='SE', SequenceVariant='SK', SequenceName='SYNTHETIC_SEQ',
    MRAcquisitionType='2D', RepetitionTime='500', EchoTime='10', InversionTime='100', FlipAngle='90', EchoTrainLength='4',
    DiffusionBValue=800.0, ReceiveCoilName='SYNTHETIC_COIL',
    RadiopharmaceuticalInformationSequence=PT_SEQUENCE, Units='BQML', DecayCorrection='START',
    CorrectedImage=['DECY', 'ATTN'], ReconstructionMethod='SYNTHETIC_RECON',
)
PLAN_TAGS = dict(
    RTPlanTime='120000', PlanIntent='CURATIVE', ApprovalStatus='APPROVED', RTPlanDescription='SYNTHETIC_PLAN',
    DoseReferenceSequence=[dict(TargetPrescriptionDose='50')],
    BeamSequence=[dict(RadiationType='PHOTON', BeamType='STATIC', TreatmentMachineName='SYNTHETIC_LINAC')],
    SourceSequence=[dict(SourceIsotopeName='SYNTHETIC_IR192', ReferenceAirKermaRate='40000',
                         SourceStrengthReferenceDate='20240101', SourceStrengthReferenceTime='080000')],
    FractionGroupSequence=[dict(FractionGroupNumber=1, NumberOfFractionsPlanned=5, NumberOfBeams=3)],
    BrachyTreatmentTechnique='INTRACAVITARY', BrachyTreatmentType='HDR',
)


def ct_heads(folder, n=1, modality='CT', per_slice=None, **tags):
    """n slice headers, written by the synthetic writer and read back the way convert reads them."""
    Path(folder).mkdir(exist_ok=True)
    series = sd.write_ct_series(folder, np.zeros((n, 2, 2)), modality=modality, per_slice=per_slice, tags=tags)
    return [read_header(p) for p in series['paths']]


def plan_head(folder, name='plan.dcm', **tags):
    Path(folder).mkdir(exist_ok=True)
    sd.write_rtplan(Path(folder) / name, sd.generate_uid(), [(0, 0, 0)], tags=tags)
    return read_header(Path(folder) / name)


def plan_collect(tmp_path, **tags):
    return collect([plan_head(tmp_path, **tags)], (RTPLAN,))


def one(tmp_path, group, **tags):
    return collect(ct_heads(tmp_path, **tags), (group,))


# ---- the catalog ----------------------------------------------------------------------------------------------------

def test_keys_in_table_order_and_unique():
    everything = [k for g in (ALL, IMAGES, RTPLAN, RTSTRUCT, RTDOSE, metadata.REG) for k in keys(g)]
    assert len(everything) == len(set(everything))
    assert keys(ALL)[:3] == ('patient_age', 'patient_sex', 'patient_size')
    assert keys(ALL)[-1] == 'protocol_name'
    assert keys(IMAGES)[:4] == ('image_type', 'rows', 'columns', 'slices')
    assert keys(RTDOSE) == ('dose_units', 'dose_type')
    assert 'patient_birth_date' not in everything and 'patient_name' not in everything


def test_all_and_images_key_sets(tmp_path):
    heads = ct_heads(tmp_path, **ALL_TAGS, **IMAGES_TAGS)  # a CT header carrying the MR and PT tags too
    assert set(collect(heads, (ALL,))) == set(keys(ALL))
    assert set(collect(heads, (IMAGES,))) == set(keys(IMAGES))
    assert set(collect(heads, (ALL, IMAGES))) == set(keys(ALL)) | set(keys(IMAGES))
    got = collect(heads, (ALL, IMAGES))
    assert got['patient_age'] == 63 and got['image_type'] == ['ORIGINAL', 'PRIMARY']
    assert got['radionuclide'] == 'SYNTHETIC_NUCLIDE' and got['radiopharmaceutical_code'] == 'SYNTHETIC_TRACER_CODE'
    assert got['injection_time'] == '20240101090000' and got['corrected_image'] == ['DECY', 'ATTN']
    assert got['ctdi_vol'] == '10.5' and got['rows'] == '2' and got['pixel_spacing'] == ['1.0', '1.0']


def test_plan_key_set(tmp_path):
    got = plan_collect(tmp_path, **PLAN_TAGS)
    assert set(got) == set(keys(RTPLAN))
    assert got['prescription_dose'] == '50' and got['number_of_beams'] == '3' and got['source_strength'] == '40000'
    assert got['treatment_machine'] == 'SYNTHETIC_LINAC' and got['radiation_type'] == 'PHOTON'


def test_struct_key_set_and_roi_lookup_by_number(tmp_path):
    ct = sd.write_ct_series(tmp_path, np.zeros((2, 6, 8)))
    rois = [dict(number=3, name='SYNTHETIC_A', kind='empty', interpreted='ORGAN', algorithm='AUTOMATIC'),
            dict(number=5, name='SYNTHETIC_B', kind='empty', interpreted='PTV', algorithm='MANUAL')]
    path = tmp_path / 'struct.dcm'
    sd.write_rtstruct_rois(path, ct, rois, tags=dict(StructureSetLabel='SYNTHETIC_SET', StructureSetDate='20240105',
                                                    StructureSetTime='101515'))
    head = read_header(path)
    assert set(collect([head], (RTSTRUCT,), roi_number=3)) == set(keys(RTSTRUCT))
    assert collect([head], (RTSTRUCT,), roi_number=5) == dict(
        structure_set_label='SYNTHETIC_SET', structure_set_date='20240105', structure_set_time='101515',
        roi_interpreted_type='PTV', roi_generation_algorithm='MANUAL')
    assert collect([head], (RTSTRUCT,), roi_number=3)['roi_interpreted_type'] == 'ORGAN'
    none_given = collect([head], (RTSTRUCT,))
    assert 'roi_interpreted_type' not in none_given and 'roi_generation_algorithm' not in none_given
    assert 'roi_interpreted_type' not in collect([head], (RTSTRUCT,), roi_number=99)


def test_dose_key_set_and_two_files_with_different_type(tmp_path):
    paths = [tmp_path / 'a.dcm', tmp_path / 'b.dcm']
    sd.write_rtdose(paths[0], np.ones((1, 2, 2)), dose_type='PHYSICAL')
    sd.write_rtdose(paths[1], np.ones((1, 2, 2)), dose_type='EFFECTIVE')
    heads = [read_header(p) for p in paths]
    assert set(collect(heads, (RTDOSE,))) == set(keys(RTDOSE))
    assert collect(heads, (RTDOSE,)) == dict(dose_units='GY', dose_type={'distinct': ['PHYSICAL', 'EFFECTIVE']})
    assert collect(heads[:1], (RTDOSE,))['dose_type'] == 'PHYSICAL'


def test_groups_do_not_leak(tmp_path):
    heads = ct_heads(tmp_path, **ALL_TAGS)
    assert set(collect(heads, (ALL,))).isdisjoint(keys(IMAGES))
    assert 'slices' not in collect(heads, (ALL,))
    assert collect(heads, ()) == {}


# ---- value rules ----------------------------------------------------------------------------------------------------

def test_one_value_is_a_stripped_string_and_multi_value_is_a_list(tmp_path):
    got = one(tmp_path, IMAGES, ImageType=['DERIVED', 'SECONDARY'], SpacingBetweenSlices='3', ContrastBolusAgent='  SYNTHETIC_AGENT ')
    assert got['image_type'] == ['DERIVED', 'SECONDARY']
    assert got['pixel_spacing'] == ['1.0', '1.0']
    assert got['spacing_between_slices'] == '3'
    assert got['contrast_agent'] == 'SYNTHETIC_AGENT'
    assert got['rows'] == '2' and isinstance(got['rows'], str)


def test_absent_blank_and_empty_values_are_omitted(tmp_path):
    got = one(tmp_path, ALL, PatientSex='', StudyDate='', StudyDescription='   ', Manufacturer='SYNTHETIC_MAKER')
    assert got == {'manufacturer': 'SYNTHETIC_MAKER'}


# ---- aggregation ----------------------------------------------------------------------------------------------------

def test_numeric_and_time_elements_give_min_max_with_first_seen_text(tmp_path):
    slices = [dict(XRayTubeCurrent='200', AcquisitionTime='101500'), dict(XRayTubeCurrent='80', AcquisitionTime='101000'),
              dict(XRayTubeCurrent='150.5', AcquisitionTime='101700')]
    got = collect(ct_heads(tmp_path, 3, per_slice=slices), (ALL, IMAGES))
    assert got['tube_current'] == {'min': '80', 'max': '200'}  # numeric order: '80' < '150.5' < '200'
    assert got['acquisition_time'] == {'min': '101000', 'max': '101700'}
    assert got['slices'] == 3 and isinstance(got['slices'], int)


def test_multi_value_and_text_elements_give_distinct_in_first_seen_order(tmp_path):
    slices = [dict(ImageType=['ORIGINAL', 'PRIMARY'], PixelSpacing=[1.0, 1.0], Manufacturer='SYNTHETIC_B'),
              dict(ImageType=['DERIVED', 'PRIMARY'], PixelSpacing=[0.5, 0.5], Manufacturer='SYNTHETIC_A'),
              dict(ImageType=['ORIGINAL', 'PRIMARY'], PixelSpacing=[1.0, 1.0], Manufacturer='SYNTHETIC_B')]
    got = collect(ct_heads(tmp_path, 3, per_slice=slices), (ALL, IMAGES))
    assert got['image_type'] == {'distinct': [['ORIGINAL', 'PRIMARY'], ['DERIVED', 'PRIMARY']]}
    assert got['pixel_spacing'] == {'distinct': [['1.0', '1.0'], ['0.5', '0.5']]}
    assert got['manufacturer'] == {'distinct': ['SYNTHETIC_B', 'SYNTHETIC_A']}


def test_missing_on_some_slices_gives_the_value_of_the_rest(tmp_path):
    slices = [None, dict(KVP='120'), None]
    got = collect(ct_heads(tmp_path, 3, per_slice=slices), (IMAGES,))
    assert got['kvp'] == '120'
    assert 'tube_current' not in got


def test_numerically_equal_values_are_one_value_first_seen_text(tmp_path):
    got = collect(ct_heads(tmp_path, 3, per_slice=[dict(KVP='3'), dict(KVP='3.0'), dict(KVP='3.00')]), (IMAGES,))
    assert got['kvp'] == '3'
    got = collect(ct_heads(tmp_path / 'b', 2, per_slice=[dict(KVP='3.0'), dict(KVP='3')]), (IMAGES,))
    assert got['kvp'] == '3.0'


def test_unparseable_number_falls_back_to_distinct():
    # A DS or IS that does not parse cannot survive the real read (read_header refuses the file), so the branch is
    # exercised on the aggregator itself.
    assert metadata._aggregate([('abc', 'DS'), ('5', 'DS')], True) == {'distinct': ['abc', '5']}
    assert metadata._aggregate([('7', 'DS'), ('5', 'DS')], True) == {'min': '5', 'max': '7'}
    assert metadata._aggregate([], True) is None
    assert metadata._aggregate([('12', 'LO'), (['1', '2'], 'LO')], True) == {'distinct': ['12', ['1', '2']]}  # PR 251 Codex


# ---- patient_age ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('tags, expected', [
    (dict(PatientAge='063Y'), 63),
    (dict(PatientAge='000Y'), 0),
    (dict(PatientAge='006M'), 0.5),
    (dict(PatientAge='010W'), 0.19),
    (dict(PatientAge='030D'), 0.08),
    (dict(PatientBirthDate='19600615', StudyDate='20230610'), 62),
    (dict(PatientBirthDate='19600615', StudyDate='20230615'), 63),
    (dict(PatientBirthDate='19600615', StudyDate='', SeriesDate='20230701'), 63),
    (dict(PatientBirthDate='19600615', AcquisitionDate='20240101'), 63),
    (dict(PatientBirthDate='19600615', StudyDate='20230610', SeriesDate='20300101'), 62),  # StudyDate comes first
    (dict(PatientAge='abc', PatientBirthDate='19600615', StudyDate='20230610'), 62),
    (dict(PatientAge='63Y', PatientBirthDate='19600615', StudyDate='20230610'), 62),  # not nnn: the dates are used
    (dict(PatientAge='063Y', PatientBirthDate='19600615', StudyDate='20230610'), 63),  # PatientAge wins
    (dict(PatientAge='abc'), None),
    (dict(PatientAge='063X'), None),
    (dict(PatientBirthDate='19600615', StudyDate='20230229'), None),
    (dict(PatientBirthDate='20230229', StudyDate='20230610'), None),
    (dict(PatientBirthDate='19600615'), None),
    (dict(StudyDate='20230610'), None),
    (dict(), None),
    (dict(PatientBirthDate='20300101', StudyDate='20230610'), None),
])
def test_patient_age(tmp_path, tags, expected):
    got = one(tmp_path, ALL, **tags)
    if expected is None:
        assert 'patient_age' not in got
    else:
        assert got['patient_age'] == expected and type(got['patient_age']) is type(expected)


def test_patient_age_is_a_json_number_aggregated_and_birth_date_never_written(tmp_path):
    slices = [dict(PatientAge='063Y', PatientBirthDate='19600615'), dict(PatientAge='064Y', PatientBirthDate='19600615'),
              dict(PatientAge='063Y', PatientBirthDate='19600615')]
    got = collect(ct_heads(tmp_path, 3, per_slice=slices), (ALL,))
    assert got['patient_age'] == {'min': 63, 'max': 64}
    assert '19600615' not in json.dumps(got)
    assert json.loads(json.dumps(collect(ct_heads(tmp_path / 'b', PatientAge='006M'), (ALL,))))['patient_age'] == 0.5


# ---- PT, RTPLAN and REG sources -------------------------------------------------------------------------------------

def test_injection_time_prefers_start_date_time_chosen_once_for_all_heads(tmp_path):
    with_both = [dict(RadiopharmaceuticalInformationSequence=PT_SEQUENCE),
                 dict(RadiopharmaceuticalInformationSequence=[dict(RadiopharmaceuticalStartTime='091500')])]
    got = collect(ct_heads(tmp_path, 2, modality='PT', per_slice=with_both), (IMAGES,))
    assert got['injection_time'] == '20240101090000'  # the second head's time is not mixed in
    only_time = [dict(RadiopharmaceuticalInformationSequence=[dict(RadiopharmaceuticalStartTime='090000')]),
                 dict(RadiopharmaceuticalInformationSequence=[dict(RadiopharmaceuticalStartTime='091500')])]
    got = collect(ct_heads(tmp_path / 'b', 2, modality='PT', per_slice=only_time), (IMAGES,))
    assert got['injection_time'] == {'min': '090000', 'max': '091500'}


def test_plan_sequence_keys_are_distinct_never_min_max(tmp_path):
    got = plan_collect(tmp_path, DoseReferenceSequence=[dict(TargetPrescriptionDose='50'), dict(TargetPrescriptionDose='60'),
                                                        dict(TargetPrescriptionDose='50.0')],
                       BeamSequence=[dict(RadiationType='PHOTON', BeamType='STATIC', TreatmentMachineName='SYNTHETIC_L1'),
                                     dict(RadiationType='ELECTRON', BeamType='STATIC', TreatmentMachineName='SYNTHETIC_L1')])
    assert got['prescription_dose'] == {'distinct': ['50', '60']}
    assert got['radiation_type'] == {'distinct': ['PHOTON', 'ELECTRON']}
    assert got['beam_type'] == 'STATIC' and got['treatment_machine'] == 'SYNTHETIC_L1'


def test_treatment_machine_falls_back_to_treatment_machine_sequence(tmp_path):
    got = plan_collect(tmp_path, TreatmentMachineSequence=[dict(TreatmentMachineName='SYNTHETIC_AFTERLOADER')])
    assert got['treatment_machine'] == 'SYNTHETIC_AFTERLOADER'
    got = plan_collect(tmp_path / 'b', BeamSequence=[dict(TreatmentMachineName='SYNTHETIC_LINAC')],
                       TreatmentMachineSequence=[dict(TreatmentMachineName='SYNTHETIC_AFTERLOADER')])
    assert got['treatment_machine'] == 'SYNTHETIC_LINAC'


def test_reg_keys_one_and_two_matrix_types(tmp_path):
    matrix = sd.transformation_matrix()
    tags = dict(Manufacturer='SYNTHETIC_REGMAKER', ContentDate='20240102', ContentTime='111111')

    def reg(name, matrices):
        path = tmp_path / name
        sd.write_reg(path, sd.generate_uid(), sd.generate_uid(), tags=tags, registrations=[dict(matrices=matrices)])
        return read_header(path)

    got = reg_keys(reg('two.dcm', [('RIGID', matrix), ('AFFINE', matrix), ('RIGID', matrix)]))
    assert got == dict(reg_type={'distinct': ['RIGID', 'AFFINE']}, reg_date='20240102', reg_time='111111',
                       reg_manufacturer='SYNTHETIC_REGMAKER')
    assert set(got) == set(keys(metadata.REG))
    assert reg_keys(reg('one.dcm', [('RIGID', matrix)]))['reg_type'] == 'RIGID'
    bare = tmp_path / 'bare.dcm'
    sd.write_reg(bare, sd.generate_uid(), sd.generate_uid())  # no RegistrationSequence at all
    assert reg_keys(read_header(bare)) == {}


# ---- the per-key guard ----------------------------------------------------------------------------------------------

def test_empty_pet_sequence_omits_only_the_pet_keys(tmp_path):
    got = collect(ct_heads(tmp_path, modality='PT', Manufacturer='SYNTHETIC_MAKER', RadiopharmaceuticalInformationSequence=[]),
                  (ALL, IMAGES))
    assert got['manufacturer'] == 'SYNTHETIC_MAKER' and got['slices'] == 1
    assert not {'radiopharmaceutical', 'radionuclide', 'injected_dose', 'injection_time', 'half_life'} & set(got)


def test_plan_without_fraction_group_omits_number_of_beams_only(tmp_path):
    got = plan_collect(tmp_path, FractionGroupSequence=None, **{k: v for k, v in PLAN_TAGS.items()
                                                                  if k != 'FractionGroupSequence'})
    assert 'number_of_beams' not in got
    assert set(got) == set(keys(RTPLAN)) - {'number_of_beams'}


def test_malformed_patient_age_omits_only_patient_age(tmp_path):
    got = collect(ct_heads(tmp_path, PatientAge='abc', Manufacturer='SYNTHETIC_MAKER'), (ALL,))
    assert got == {'manufacturer': 'SYNTHETIC_MAKER'}


def test_an_empty_sequence_on_one_slice_is_skipped_like_a_missing_tag(tmp_path):
    slices = [dict(RadiopharmaceuticalInformationSequence=[]), dict(RadiopharmaceuticalInformationSequence=PT_SEQUENCE)]
    alone = collect(ct_heads(tmp_path / 'one', 1, modality='PT', per_slice=slices[1:]), (IMAGES,))
    got = collect(ct_heads(tmp_path / 'two', 2, modality='PT', per_slice=slices), (IMAGES,))
    assert got['radiopharmaceutical'] == alone['radiopharmaceutical'] and got['slices'] == 2


def test_roi_lookup_skips_an_item_with_an_odd_roi_number():
    items = [Dataset(), Dataset()]
    items[0].ROINumber, items[1].ROINumber = '', 4  # the empty one: int('') raised and hid the match
    for item, algorithm in zip(items, ('AUTOMATIC', 'MANUAL'), strict=True):
        item.ROIGenerationAlgorithm = algorithm
    head = Dataset()
    head.StructureSetROISequence = Sequence(items)
    lookup = metadata._roi_lookup('StructureSetROISequence', 'ROINumber', 'ROIGenerationAlgorithm')
    assert lookup([head], 4) == 'MANUAL'


def test_tm_extremes_compare_as_times_across_precision():
    samples = [('090000.5', 'TM'), ('0900', 'TM'), ('10', 'TM'), ('0830', 'TM')]
    assert metadata._aggregate(samples, True) == {'min': '0830', 'max': '10'}


# ---- quiet ----------------------------------------------------------------------------------------------------------

def test_collecting_emits_no_warning_and_no_log_record(tmp_path, caplog):
    caplog.set_level(0)
    plan = tmp_path / 'p'
    heads = ct_heads(tmp_path / 'c', 2, modality='PT', **ALL_TAGS, **IMAGES_TAGS)
    plan_header = plan_head(plan, **PLAN_TAGS)
    caplog.clear()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        heads = [read_header(h.filename) for h in heads]
        plan_header = read_header(plan_header.filename)
        collect(heads, (ALL, IMAGES))
        collect([plan_header], (RTPLAN,))
        collect([plan_header], (RTSTRUCT, RTDOSE), roi_number=1)
        reg_keys(plan_header)
    assert caplog.records == []


def test_module_has_no_logging_print_or_raise():
    source = Path(metadata.__file__).read_text()
    for forbidden in ('logging', 'print(', 'raise ', 'warnings'):
        assert forbidden not in source
