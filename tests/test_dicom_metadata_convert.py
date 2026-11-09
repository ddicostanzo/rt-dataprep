"""
convert writes `dicom` on every series.json object (spec docs/specs/2026-11-09-dicom-metadata.md, Acceptance). Every tree is synthetic,
written to disk by synthetic_dicom and converted; the values are SYNTHETIC_ strings or numbers that no log line can contain.
"""
import csv
import json
import logging
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
import series_files as sf
import synthetic_dicom as sd
from test_convert import RING, RING_CHANNEL, SHAPE, dose_spec, make_plan
from test_metadata import ALL_TAGS, IMAGES_TAGS, PLAN_TAGS
from test_series import VALUES, rigid

from rt_dataprep import convert as conv
from rt_dataprep import metadata
from rt_dataprep.metadata import ALL, IMAGES, REG, RTDOSE, RTPLAN, RTSTRUCT

PID = 'SYNTHETICPTM'
# Values no log line holds: a long age string, size and weight with four decimals, site strings with the marker.
LOUD = dict(ALL_TAGS, PatientAge='073Y', PatientSize='1.8765', PatientWeight='70.321', StudyID='SYNTHETIC_SID')
CT_TAGS = LOUD | IMAGES_TAGS
STRUCT_TAGS = LOUD | dict(StructureSetLabel='SYNTHETIC_SET', StructureSetDate='20240105', StructureSetTime='101515')
REG_TAGS = dict(ContentDate='20240110', ContentTime='111111', Manufacturer='SYNTHETIC_REG_MAKER')
LIVER = dict(number=1, name='Liver', kind='box', box=((0, 2), (0, 2), (0, 2)), interpreted='ORGAN', algorithm='AUTOMATIC')
PTV = dict(number=4, name='PTV', kind='box', box=((1, 3), (1, 3), (1, 3)), interpreted='PTV', algorithm='MANUAL')
RING_ROI = dict(RING, interpreted='BRACHY_CHANNEL', algorithm='MANUAL')
LINE_ROI = dict(number=3, name='RefLine', kind='line', lines=[np.array([[-1.0, 8.0, 7.0], [-1.0, 8.0, 10.0], [-1.0, 8.0, 13.0]])],
                interpreted='MARKER', algorithm='SEMIAUTOMATIC')  # spans the 4 slices of test_convert's CT (z 7..13)
SYNTHETIC_VALUES = [LOUD[k] for k in ('PatientAge', 'PatientSize', 'PatientWeight', 'InstitutionName', 'StationName', 'DeviceSerialNumber',
                                      'AccessionNumber', 'StudyID', 'Manufacturer', 'ManufacturerModelName')] + ['SYNTHETIC_REG_MAKER']


def own(*groups):
    return [key for group in groups for key in metadata.keys(group)]


EXPECTED = {
    'ct': own(ALL, IMAGES, RTPLAN),
    'dose': own(ALL, RTDOSE, RTPLAN),
    'mask': own(ALL, RTSTRUCT, RTPLAN),
    'applicator': own(ALL, RTSTRUCT, RTPLAN),
    'line': own(ALL, RTSTRUCT, RTPLAN),
    'registered': own(ALL, IMAGES, RTPLAN, REG),
    'identity': own(ALL, IMAGES, RTPLAN),
    'study': own(ALL, IMAGES),
}


def add_registered_mr(folder, ct, patient, **kwargs):
    """An MR series on its own frame in the plan folder and a rigid REG that registers it to the plan's CT."""
    mr = sd.write_ct_series(folder, VALUES, spacing=(1.5, 2.0, 3.0), origin=(7.0, -4.0, 12.0), patient_id=patient, prefix='mr',
                            modality='MR', **kwargs)
    sd.write_reg(folder / 'reg.dcm', ct['study_uid'], mr['study_uid'], patient_id=patient, tags=REG_TAGS,
                 registrations=[dict(frame_uid=ct['frame_uid'], matrices=rigid()),
                                dict(frame_uid=mr['frame_uid'], matrices=rigid(translation=(10, 0, 0)))])
    return mr


def add_same_frame(folder, ct, patient, **kwargs):
    return sd.write_ct_series(folder, VALUES, patient_id=patient, prefix='same', modality='MR', frame_uid=ct['frame_uid'], **kwargs)


def build_patient(org, patient=PID):
    """plan_1 (CT, RTSTRUCT with two masks, RTPLAN, RTDOSE, a registered MR, a same-frame MR), plan_2 (brachy: applicator and line ROI
    and a source), and a PT series free of any plan. Returns the uids of the series by name."""
    one = make_plan(org / patient / 'plan_1', doses=[dose_spec(10, tags=LOUD)], rois=[LIVER, PTV], patient=patient,
                    ct_meta=dict(tags=CT_TAGS), plan_meta=dict(tags=PLAN_TAGS), struct_meta=dict(tags=STRUCT_TAGS))
    folder = one['folder']
    mr = add_registered_mr(folder, one['ct'], patient, tags=dict(CT_TAGS, SeriesNumber='7'))
    same = add_same_frame(folder, one['ct'], patient, tags=dict(CT_TAGS, SeriesNumber='8'))
    make_plan(org / patient / 'plan_2', rois=[RING_ROI, LINE_ROI], patient=patient, ct_meta=dict(tags=CT_TAGS),
              plan_meta=dict(tags=PLAN_TAGS, channels=RING_CHANNEL), struct_meta=dict(tags=STRUCT_TAGS))
    misc = org / patient / 'misc_files'
    misc.mkdir()
    pt = sd.write_ct_series(misc, VALUES, patient_id=patient, prefix='pt', modality='PT', tags=dict(CT_TAGS, SeriesNumber='9'))
    return dict(mr=mr['series_uid'], same=same['series_uid'], pt=pt['series_uid'], one=one)


def convert_tree(tmp_path, build=build_patient):
    org, out = tmp_path / 'org', tmp_path / 'out'
    org.mkdir(parents=True)
    built = build(org)
    return built, conv.convert(org, out).rows, out


def study_rows(out, patient=PID):
    return [row for folder in sorted((out / patient).glob('study_*')) for row in sf.read(folder)]


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    built, rows, out = convert_tree(tmp_path_factory.mktemp('metadata'))
    return built, rows, out


def by_kind(out, plan):
    rows = sf.read(out / PID / plan)
    return {row['kind']: row for row in rows if row['kind'] != 'series'} | {'series': [row for row in rows if row['kind'] == 'series']}


# ---------------------------------------------------------------- AC4: the key set per row kind (catches keyword typos)

def test_every_row_kind_has_exactly_the_keys_of_its_groups_in_catalog_order(run):
    built, rows, out = run
    assert [(r.folder, r.outcome, r.reason) for r in rows if r.folder.startswith('plan')] == [
        ('plan_1', 'converted', ''), ('plan_2', 'converted', '')]
    one, two = by_kind(out, 'plan_1'), by_kind(out, 'plan_2')
    mask_rows = [r for r in sf.read(out / PID / 'plan_1') if r['kind'] == 'mask']
    series = {r['series_instance_uid']: r for r in one['series']}
    seen = {
        'ct': one['ct'], 'dose': one['dose'], 'applicator': two['applicator'], 'line': two['line'], 'mask': mask_rows[0],
        'registered': series[built['mr']], 'identity': series[built['same']],
        'study': next(r for r in study_rows(out) if r['series_instance_uid'] == built['pt']),
    }
    assert len(mask_rows) == 2 and len(one['series']) == 2
    for kind, row in seen.items():
        assert list(row) == [*conv.SERIES_COLUMNS, 'dicom'], kind
        assert list(row['dicom']) == EXPECTED[kind], kind  # the full key set, in the order AC1 names
    assert all(list(row['dicom']) == EXPECTED['study'] for row in study_rows(out))  # MR, same-frame MR and PT in study folders


def test_the_values_of_the_main_keys_are_the_header_values(run):
    built, _, out = run
    ct = by_kind(out, 'plan_1')['ct']['dicom']
    assert ct['patient_age'] == 73 and ct['slices'] == SHAPE[0] and ct['patient_size'] == '1.8765'
    assert ct['image_type'] == ['ORIGINAL', 'PRIMARY'] and ct['institution_name'] == 'SYNTHETIC_HOSPITAL'
    assert ct['prescription_dose'] == '50' and ct['plan_intent'] == 'CURATIVE' and ct['number_of_beams'] == '3'
    assert by_kind(out, 'plan_1')['dose']['dicom']['dose_units'] == 'GY'
    assert by_kind(out, 'plan_2')['ct']['dicom']['source_isotope'] == 'SYNTHETIC_IR192'


def test_the_existing_fields_are_unchanged_and_ahead_of_dicom(run):
    _, _, out = run
    for folder in ('plan_1', 'plan_2', *(p.name for p in (out / PID).glob('study_*'))):
        for row in sf.read(out / PID / folder):
            assert list(row)[:-1] == conv.SERIES_COLUMNS and list(row)[-1] == 'dicom' and isinstance(row['dicom'], dict)
            assert all(isinstance(row[column], str) for column in conv.SERIES_COLUMNS)


# ---------------------------------------------------------------- AC6, AC7: ROI types and the chosen REG

def test_each_mask_applicator_and_line_row_has_its_own_rois_type_and_algorithm(run):
    _, _, out = run
    masks = {r['roi_name']: r['dicom'] for r in sf.read(out / PID / 'plan_1') if r['kind'] == 'mask'}
    assert (masks['Liver']['roi_interpreted_type'], masks['Liver']['roi_generation_algorithm']) == ('ORGAN', 'AUTOMATIC')
    assert (masks['PTV']['roi_interpreted_type'], masks['PTV']['roi_generation_algorithm']) == ('PTV', 'MANUAL')
    two = by_kind(out, 'plan_2')
    assert (two['applicator']['dicom']['roi_interpreted_type'], two['applicator']['dicom']['roi_generation_algorithm']) == (
        'BRACHY_CHANNEL', 'MANUAL')
    assert (two['line']['dicom']['roi_interpreted_type'], two['line']['dicom']['roi_generation_algorithm']) == ('MARKER', 'SEMIAUTOMATIC')


def test_a_registered_series_carries_the_chosen_regs_keys_and_the_ct_row_none(run):
    built, _, out = run
    one = by_kind(out, 'plan_1')
    series = {r['series_instance_uid']: r for r in one['series']}
    assert {k: v for k, v in series[built['mr']]['dicom'].items() if k in metadata.keys(REG)} == dict(
        reg_type='RIGID', reg_date='20240110', reg_time='111111', reg_manufacturer='SYNTHETIC_REG_MAKER')
    assert series[built['mr']]['dicom']['manufacturer'] == 'SYNTHETIC_MAKER'  # the image's own, not the REG's
    for row in (one['ct'], one['dose'], series[built['same']]):
        assert not set(metadata.keys(REG)) & set(row['dicom'])
    assert not set(metadata.keys(REG)) & set(by_kind(out, 'plan_2')['ct']['dicom'])


# ---------------------------------------------------------------- AC3: series rows

def test_series_rows_in_a_plan_folder_get_the_plans_keys_and_a_study_row_never(run):
    built, _, out = run
    one = by_kind(out, 'plan_1')
    plan_keys = {k: one['ct']['dicom'][k] for k in metadata.keys(RTPLAN)}
    assert plan_keys == {k: one['ct']['dicom'][k] for k in own(RTPLAN)} and len(plan_keys) == len(own(RTPLAN))
    in_study = {r['series_instance_uid']: r['dicom'] for r in study_rows(out)}
    for row in one['series']:
        uid = row['series_instance_uid']
        assert {k: row['dicom'][k] for k in metadata.keys(RTPLAN)} == plan_keys
        # the series' own aggregated keys are the study row's, byte for byte; the plan row only adds RTPLAN and REG keys
        assert {k: v for k, v in row['dicom'].items() if k in in_study[uid]} == in_study[uid]
        assert set(row['dicom']) - set(in_study[uid]) == set(own(RTPLAN)) | (set(own(REG)) if uid == built['mr'] else set())
        assert not set(own(RTPLAN, REG)) & set(in_study[uid])


def test_each_folder_gets_its_own_dicom_dict(tmp_path, monkeypatch):
    written = []
    real = conv.write_series_json
    monkeypatch.setattr(conv, 'write_series_json', lambda rows, path: (written.append((path.parent.name, rows)), real(rows, path))[1])
    built, _, out = convert_tree(tmp_path)
    plan_row = next(r for name, rows in written if name == 'plan_1' for r in rows if r['series_instance_uid'] == built['mr'])
    study_row = next(r for name, rows in written if name.startswith('study_') for r in rows if r['series_instance_uid'] == built['mr'])
    assert plan_row['dicom'] is not study_row['dicom'] and set(plan_row['dicom']) > set(study_row['dicom'])
    dicts = [r['dicom'] for _, rows in written for r in rows]
    assert len({id(d) for d in dicts}) == len(dicts)


def test_a_registration_without_a_usable_matrix_still_gives_the_chosen_regs_keys_and_an_identity_row_none(tmp_path):
    def build(org):
        one = make_plan(org / PID / 'plan_1', ct_meta=dict(tags=CT_TAGS), plan_meta=dict(tags=PLAN_TAGS), patient=PID)
        mr = sd.write_ct_series(one['folder'], VALUES, patient_id=PID, prefix='mr', modality='MR', tags=LOUD)
        # no matrix at all: the REG qualifies (an item for each frame) but gives reg_matrix ''
        sd.write_reg(one['folder'] / 'reg.dcm', one['ct']['study_uid'], mr['study_uid'], patient_id=PID, tags=REG_TAGS,
                     registrations=[dict(frame_uid=one['ct']['frame_uid'], matrices=rigid()),
                                    dict(frame_uid=mr['frame_uid'], matrices=[('RIGID', [0] * 16), ('RIGID', [0] * 16)])])
        return mr
    mr, _, out = convert_tree(tmp_path, build)
    [row] = [r for r in sf.read(out / PID / 'plan_1') if r['kind'] == 'series']
    assert row['reg_matrix'] == '' and row['series_instance_uid'] == mr['series_uid']
    assert {k: v for k, v in row['dicom'].items() if k in metadata.keys(REG)}['reg_manufacturer'] == 'SYNTHETIC_REG_MAKER'


# ---------------------------------------------------------------- AC5: variation over the slices and the files

def per_slice(**by_tag):
    """per_slice= for 4 slices from {tag: [value of slice 0..3]}."""
    return [{tag: values[k] for tag, values in by_tag.items()} for k in range(SHAPE[0])]


VARIATION = per_slice(
    XRayTubeCurrent=['100', '200', '150', '200'],
    AcquisitionTime=['100000', '100500', '101000', '101500'],
    ImageType=[['ORIGINAL', 'PRIMARY'], ['DERIVED', 'SECONDARY'], ['ORIGINAL', 'PRIMARY'], ['ORIGINAL', 'PRIMARY']],
    PixelSpacing=[['1.5', '1.0'], ['1.5004', '1.0'], ['1.5', '1.0'], ['1.5', '1.0']],  # within the volume check's 1e-3 mm
    KVP=['120', None, '120', None],
    Exposure=[None, None, '300', None],
    SliceThickness=['3', '3.0', '3.0', '3.0'],
    SeriesDescription=['SYNTHETIC_FIRST', 'SYNTHETIC_OTHER', 'SYNTHETIC_OTHER', 'SYNTHETIC_OTHER'],
)


def check_variation(dicom, slices=SHAPE[0]):
    assert dicom['tube_current'] == {'min': '100', 'max': '200'}
    assert dicom['acquisition_time'] == {'min': '100000', 'max': '101500'}
    assert dicom['image_type'] == {'distinct': [['ORIGINAL', 'PRIMARY'], ['DERIVED', 'SECONDARY']]}
    assert dicom['pixel_spacing'] == {'distinct': [['1.5', '1.0'], ['1.5004', '1.0']]}
    assert dicom['kvp'] == '120' and dicom['exposure'] == '300'  # a tag missing on some slices: the value of the rest
    assert dicom['slice_thickness'] == '3'  # 3 and 3.0 are one value, the first text
    assert dicom['slices'] == slices


def test_per_slice_variation_on_the_ct_row_while_its_series_columns_come_from_the_first_slice(tmp_path):
    def build(org):
        return make_plan(org / PID / 'plan_1', patient=PID, ct_meta=dict(tags=CT_TAGS, per_slice=VARIATION), plan_meta=dict(tags=PLAN_TAGS))
    _, rows, out = convert_tree(tmp_path, build)
    assert rows[0].outcome == 'converted'
    [ct] = sf.read(out / PID / 'plan_1')
    check_variation(ct['dicom'])
    assert ct['series_description'] == 'SYNTHETIC_FIRST'  # SERIES_COLUMNS: the first slice only, as before


def test_a_series_aggregates_over_every_slice_in_its_plan_row_and_its_study_row(tmp_path):
    def build(org):
        one = make_plan(org / PID / 'plan_1', patient=PID, ct_meta=dict(tags=CT_TAGS), plan_meta=dict(tags=PLAN_TAGS))
        return add_registered_mr(one['folder'], one['ct'], PID, tags=CT_TAGS, per_slice=VARIATION[:3] + [VARIATION[3] | {'KVP': None}])
    mr, _, out = convert_tree(tmp_path, build)
    [plan_row] = [r for r in sf.read(out / PID / 'plan_1') if r['kind'] == 'series']
    [study_row] = study_rows(out)
    assert mr['series_uid'] == plan_row['series_instance_uid'] == study_row['series_instance_uid']
    for dicom in (plan_row['dicom'], study_row['dicom']):
        check_variation(dicom, slices=len(VALUES))


def test_a_two_file_dose_row_aggregates_over_both_files(tmp_path):
    def build(org):
        return make_plan(org / PID / 'plan_1', patient=PID, doses=[
            dose_spec(5, 'BEAM', tags=dict(Manufacturer='SYNTHETIC_MAKER_A', SeriesNumber='2', DoseComment='SYNTHETIC_ONE')),
            dose_spec(6, 'BEAM', tags=dict(Manufacturer='SYNTHETIC_MAKER_B', SeriesNumber='10'))])
    _, rows, out = convert_tree(tmp_path, build)
    assert rows[0].outcome == 'converted'
    dose = next(r for r in sf.read(out / PID / 'plan_1') if r['kind'] == 'dose')
    assert dose['dicom']['manufacturer'] == {'distinct': ['SYNTHETIC_MAKER_A', 'SYNTHETIC_MAKER_B']}
    assert dose['dicom']['series_number'] == {'min': '2', 'max': '10'}  # numeric order
    assert dose['dicom']['dose_type'] == 'PHYSICAL'


def test_a_two_file_dose_with_different_dose_types_is_distinct_on_the_dose_row(tmp_path, monkeypatch):
    # convert refuses any DoseType but PHYSICAL (REASON_DOSE_TYPE), so no plan reaches the row with two types: the check is let through
    # here on a copy whose DoseType is PHYSICAL, while the header the row reads keeps EFFECTIVE.
    real = conv._check_dose_header

    def lenient(ds, frame):
        copy = deepcopy(ds)
        copy.DoseType = 'PHYSICAL'
        return real(copy, frame)
    monkeypatch.setattr(conv, '_check_dose_header', lenient)

    def build(org):
        return make_plan(org / PID / 'plan_1', patient=PID, doses=[dose_spec(5, 'BEAM', dose_type='PHYSICAL'),
                                                                   dose_spec(6, 'BEAM', dose_type='EFFECTIVE')])
    _, rows, out = convert_tree(tmp_path, build)
    assert rows[0].outcome == 'converted'
    dose = next(r for r in sf.read(out / PID / 'plan_1') if r['kind'] == 'dose')
    assert dose['dicom']['dose_type'] == {'distinct': ['PHYSICAL', 'EFFECTIVE']} and dose['dicom']['dose_units'] == 'GY'


# ---------------------------------------------------------------- AC8: the guard, end to end

def guard_plan(case):
    """(make_plan kwargs with the bad value, those without it, the keys that must then be absent from the ct row, from the dose row)."""
    good = dict(ct_meta=dict(tags=CT_TAGS), plan_meta=dict(tags=PLAN_TAGS), doses=[dose_spec(10, tags=LOUD)])
    if case == 'empty PT sequence':
        bad = dict(good, ct_meta=dict(tags=CT_TAGS | dict(RadiopharmaceuticalInformationSequence=[])))
        return bad, good, {'radiopharmaceutical', 'radiopharmaceutical_code', 'radionuclide', 'injected_dose', 'injection_time',
                           'half_life'}, set()
    if case == 'plan without FractionGroupSequence':
        bad = dict(good, plan_meta=dict(tags=PLAN_TAGS | dict(FractionGroupSequence=None)))
        return bad, good, {'number_of_beams'}, {'number_of_beams'}
    if case == 'age abc':
        bad = dict(good, ct_meta=dict(tags=CT_TAGS | dict(PatientAge='abc')), doses=[dose_spec(10, tags=LOUD | dict(PatientAge='abc'))])
        return bad, good, {'patient_age'}, {'patient_age'}
    bad = dict(good, ct_meta=dict(tags=CT_TAGS | dict(PatientAge=None, PatientBirthDate='20230229')),
               doses=[dose_spec(10, tags=LOUD | dict(PatientAge=None, PatientBirthDate='20230229'))])
    return bad, good, {'patient_age'}, {'patient_age'}


GUARD_CASES = ['empty PT sequence', 'plan without FractionGroupSequence', 'age abc', 'no age, impossible birth date']


def outcomes(rows):
    return [(r.folder, r.outcome, r.reason, r.files, r.skipped_series) for r in rows]


@pytest.mark.parametrize('case', GUARD_CASES)
def test_a_bad_value_omits_its_keys_and_changes_no_outcome(tmp_path, case):
    bad, good, ct_absent, dose_absent = guard_plan(case)
    results = {}
    for name, kwargs in (('bad', bad), ('good', good)):
        _, rows, out = convert_tree(tmp_path / name, lambda org, kwargs=kwargs: make_plan(org / PID / 'plan_1', patient=PID, **kwargs))
        results[name] = (outcomes(rows), {r['kind']: r['dicom'] for r in sf.read(out / PID / 'plan_1')})
    assert results['bad'][0] == results['good'][0] and results['bad'][0][0][1:3] == ('converted', '')
    for kind, absent in (('ct', ct_absent), ('dose', dose_absent)):
        got, expected = results['bad'][1][kind], results['good'][1][kind]
        assert not absent & set(got) and absent <= set(expected), (kind, absent)
        assert got == {k: v for k, v in expected.items() if k not in absent}  # nothing else changed


# ---------------------------------------------------------------- AC9: nothing reaches a log line or the report

def test_no_synthetic_value_reaches_a_log_line_or_the_report_in_a_full_run_the_guard_cases_and_a_refused_plan(tmp_path, caplog):
    org, out = tmp_path / 'org', tmp_path / 'out'
    org.mkdir()
    build_patient(org)
    for k, case in enumerate(GUARD_CASES):
        make_plan(org / f'SYNTHETICPTG{k}' / 'plan_1', patient=f'SYNTHETICPTG{k}', **guard_plan(case)[0])
    refused = make_plan(org / 'SYNTHETICPTR' / 'plan_1', patient='SYNTHETICPTR', ct_meta=dict(tags=CT_TAGS), plan_meta=dict(tags=PLAN_TAGS),
                        struct_meta=dict(tags=STRUCT_TAGS), doses=[dose_spec(10, dose_type='EFFECTIVE', tags=LOUD)])
    assert refused['folder'].exists()
    with caplog.at_level(logging.DEBUG):
        rows = conv.convert(org, out).rows
    outcome_of = {r.patient: r.outcome for r in rows if r.folder in ('plan_1', '')}
    assert outcome_of['SYNTHETICPTR'] == 'refused' and outcome_of[PID] == 'converted'
    assert Counter(outcome_of.values()) == {'converted': 5, 'refused': 1}
    assert not [r for r in caplog.records if r.name == 'rt_dataprep.metadata']
    reports = [(out / conv.REPORT_NAME).read_text(encoding='utf-8'), *(p.read_text(encoding='utf-8') for p in out.glob('convert_report_*.csv'))]
    logged = '\n'.join(r.getMessage() for r in caplog.records) + caplog.text
    for text in (logged, *reports):
        for value in SYNTHETIC_VALUES:
            assert value not in text, 'a synthetic metadata value reached a log line or the report'
    assert logged  # the run did log (the refusal), so the check above looked at something
    with open(out / conv.REPORT_NAME, newline='', encoding='utf-8') as handle:
        assert next(csv.reader(handle)) == ['patient', 'folder', 'outcome', 'reason', 'files', 'skipped_series']  # no column gained
    assert json.loads((out / PID / 'plan_1' / conv.SERIES_NAME).read_text(encoding='utf-8'))[0]['dicom']['institution_name'] == 'SYNTHETIC_HOSPITAL'
