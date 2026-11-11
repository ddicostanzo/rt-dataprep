"""manifest: one row per plan folder or study series of a converted or preprocessed tree (task-01, task-04); patient folder names are SYNTHETIC-prefixed."""
import dataclasses
import json
import logging
import os
import re
import shutil

import numpy as np
import pytest
import series_files as sf
import SimpleITK as sitk
from test_convert import BOX, PID, dose_spec, make_plan
from test_preprocess import (AHE_MR_OUT, AHE_STUDY_CT, CUT, DOSE_SPLIT, HALF_OUT, MR_OUT, MR_RAMP, MR_SIZE, SMALL, SMALL_AHE, box, grid, landmark,
                             mr_spec, rigid, run_plan, save_rows, series_row, state, write_plan, write_study_tree)
from test_series import STUDY_D1, STUDY_D2, STUDY_T1, STUDY_T2, put

from rt_dataprep import convert as conv
from rt_dataprep import manifest as man
from rt_dataprep import preprocess as pre
from rt_dataprep import split as sp

COLUMNS = ['case_dir', 'patient', 'fx', 'ct_size', 'ct_spacing', 'dose', 'masks', 'rtplan_uid', 'ct_series_uid', 'study_uid',
           'case_type', 'modality', 'image', 'series']
OPTIONAL = ['patient_age', 'patient_sex', 'patient_weight', 'study_date', 'series_date', 'manufacturer', 'model', 'institution_name']
PATIENT_A, PATIENT_B = 'SYNTHETIC_PTA_2', 'SYNTHETICPTB'
ONE_MASK = {'ptv': box((4, 4, 4), (8, 8, 8))}


def plan(root, patient, folder, uid, **kwargs):
    """write_plan, then real UIDs on the ct row: write_plan leaves rtplan_uid and study_instance_uid blank."""
    path, rows = write_plan(root, patient, folder, **kwargs)
    rows[0].update(series_instance_uid=f'SYNTHETICCT{uid}', rtplan_uid=f'SYNTHETICRTPLAN{uid}', study_instance_uid=f'SYNTHETICSTUDY{uid}')
    save_rows(path, rows)
    return path


def tree(root):
    """Two patients; A has two plans (the second without dose, one mask); B is a larger grid, and a fx_ folder."""
    plan(root, PATIENT_A, 'plan_1', 'A1')
    plan(root, PATIENT_A, 'plan_2', 'A2', dose=None, masks=ONE_MASK)
    plan(root, PATIENT_B, 'fx_1', 'B1', n=14, dose=box((3, 3, 3), (9, 9, 9), n=14, value=2.0), masks={})
    return root


def read(root):
    rows = [json.loads(line) for line in (root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    for row in rows:  # the optional keys (task-04) follow the fixed ones, in order; this helper's callers compare the fixed part
        base = [*COLUMNS, 'fov_mask'] if 'fov_mask' in row else COLUMNS
        assert list(row) == [*base, *(key for key in [*OPTIONAL, 'retained'] if key in row)]
    return [{key: value for key, value in row.items() if key not in [*OPTIONAL, 'retained']} for row in rows]


def test_rows_columns_and_values_on_a_converted_tree(tmp_path):
    root = tree(tmp_path / 'converted')
    assert man.manifest(root) == (3, 0)
    rows = read(root)
    assert [r['case_dir'] for r in rows] == [f'{PATIENT_B}/fx_1', f'{PATIENT_A}/plan_1', f'{PATIENT_A}/plan_2']  # '_' sorts after 'P'
    assert rows[1] == dict(case_dir=f'{PATIENT_A}/plan_1', patient=PATIENT_A, fx='plan_1', ct_size='12;12;12', ct_spacing='1.0;1.0;1.0',
                           dose='1', masks=['bladder', 'ptv'], rtplan_uid='SYNTHETICRTPLANA1', ct_series_uid='SYNTHETICCTA1',
                           study_uid='SYNTHETICSTUDYA1', case_type='plan', modality='CT', image='', series=[])
    assert (rows[2]['dose'], rows[2]['masks'], rows[2]['rtplan_uid']) == ('0', ['ptv'], 'SYNTHETICRTPLANA2')
    assert (rows[0]['patient'], rows[0]['fx'], rows[0]['ct_size'], rows[0]['masks']) == (PATIENT_B, 'fx_1', '14;14;14', [])
    assert {r['case_type'] for r in rows} == {'plan'} and {r['modality'] for r in rows} == {'CT'} and {r['image'] for r in rows} == {''}


def test_patient_is_the_folder_name_not_parsed_from_the_case_dir(tmp_path):
    root = tree(tmp_path / 'converted')
    man.manifest(root)
    assert {r['patient'] for r in read(root)} == {PATIENT_A, PATIENT_B}  # PATIENT_A holds '_', as does a plan folder name


def test_converted_and_preprocessed_trees_list_the_same_cases(tmp_path):
    converted, out = tree(tmp_path / 'converted'), tmp_path / 'out'
    pre.preprocess(converted, out, SMALL)
    man.manifest(converted)
    assert man.manifest(out) == (3, 0)
    before, after = read(converted), read(out)
    assert [r['case_dir'] for r in after] == [r['case_dir'] for r in before]
    assert {r['ct_size'] for r in after} == {'10;10;10'} and {r['ct_spacing'] for r in after} == {'1.25;1.25;1.25'}
    assert {r['ct_size'] for r in before} == {'12;12;12', '14;14;14'}
    for kept, done in zip(before, after, strict=True):
        assert {k: v for k, v in done.items() if k not in ('ct_size', 'ct_spacing')} == {k: v for k, v in kept.items() if k not in ('ct_size', 'ct_spacing')}


def test_the_walk_gives_no_row_to_images_misc_hidden_or_series_less_folders(tmp_path):
    root = tree(tmp_path / 'converted')
    for name in ('images', 'misc_files'):  # even with a series.json inside, not a plan folder
        folder = root / PATIENT_A / name
        folder.mkdir()
        (folder / conv.SERIES_NAME).write_bytes((root / PATIENT_A / 'plan_1' / conv.SERIES_NAME).read_bytes())
    (root / PATIENT_A / 'plan_3').mkdir()  # a plan folder without series.json
    write_plan(root, '.SYNTHETICPTC.partial', 'plan_1')  # a hidden patient folder
    (root / 'stray.txt').write_text('x')
    assert man.manifest(root) == (3, 0)
    assert len(read(root)) == 3


def test_a_malformed_series_json_is_left_out_and_counted_without_a_name_in_the_log(tmp_path, caplog, capfd):
    root = tree(tmp_path / 'converted')
    bad = plan(root, 'SYNTHETIC_DISTINCTIVE_NAME', 'plan_1', 'D1')
    (bad / conv.SERIES_NAME).write_text('[{"not": "the keys"}]')
    with caplog.at_level(logging.DEBUG):
        assert man.manifest(root) == (3, 1)
    assert 'SYNTHETIC_DISTINCTIVE_NAME' not in [r['case_dir'].split('/')[0] for r in read(root)]
    logged = capfd.readouterr()
    for text in (caplog.text, logged.out, logged.err):
        for secret in ('SYNTHETIC', str(tmp_path), 'plan_1'):
            assert secret not in text
    assert [r.getMessage() for r in caplog.records if r.name == 'rt_dataprep.manifest'] == ['manifest: 3 written, 1 refused']


def test_only_manifest_jsonl_and_its_dated_copy_are_written(tmp_path):
    root = tree(tmp_path / 'converted')
    before = state(root)
    man.manifest(root)
    after = state(root)
    new = {p for p in after if p not in before}
    assert {p.name for p in new} >= {'manifest.jsonl'} and all(re.fullmatch(r'manifest(_\d{8}T\d{6}Z)?\.jsonl', p.name) for p in new)
    assert len(new) == 2
    assert all(after[p] == data for p, data in before.items())



def test_a_semicolon_roi_name_and_float32_spacing_read_back_exactly(tmp_path):
    root = tmp_path / 'converted'
    plan(root, PATIENT_A, 'plan_1', 'A1', spacing=0.8, masks={'ptv;boost': box((4, 4, 4), (8, 8, 8))})
    man.manifest(root)
    row = read(root)[0]
    assert (row['masks'], row['ct_spacing']) == (['ptv;boost'], '0.8;0.8;0.8')


def test_a_plan_whose_series_row_names_a_missing_file_is_counted_refused(tmp_path):
    root = tree(tmp_path / 'converted')
    path = plan(root, 'SYNTHETICPTC', 'plan_1', 'C1')
    rows = pre.read_series_json(path)[0]
    save_rows(path, [*rows, series_row('series_MR_1.nii.gz', 'series', modality='MR', reg_matrix='identity')])  # no such file
    assert man.manifest(root) == (3, 1)
    assert 'SYNTHETICPTC' not in {r['patient'] for r in read(root)}


def test_an_unreadable_series_json_is_one_refused_plan_not_the_end_of_the_run(tmp_path, monkeypatch):
    root = tree(tmp_path / 'converted')
    check = man.check_rows

    def flaky(plan_dir):
        if plan_dir.name == 'plan_2':
            raise OSError('SYNTHETIC_PATH')
        return check(plan_dir)

    monkeypatch.setattr(man, 'check_rows', flaky)
    assert man.manifest(root) == (2, 1)


# ---------------------------------------------------------------- task-04: imaging rows of the study folders

BLANK_IMAGING = dict(fx='', dose='0', masks=[], rtplan_uid='', series=[])


@pytest.fixture(scope='module')
def master(tmp_path_factory):
    """A converted tree of PID built from synthetic DICOM: plan_1; study_1 one CT (0.8 mm, so float32 text shows);
    study_2 two MR series (series times order them). The plan's own CT is undated and never in a study."""
    base = tmp_path_factory.mktemp('manifest_imaging')
    org, out = base / 'org', base / 'out'
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10)], rois=[BOX])
    misc = org / PID / 'misc_files'
    ct = put(misc, 'ct', 'CT', spacing=(0.8, 1.5, 3.0), study_date=STUDY_D1)
    mr_a = put(misc, 'mra', 'MR', study_date=STUDY_D2, series_time=STUDY_T1)
    mr_b = put(misc, 'mrb', 'MR', study_uid=mr_a['study_uid'], study_date=STUDY_D2, series_time=STUDY_T2)
    conv.convert(org, out)
    assert sorted(p.name for p in (out / PID).iterdir()) == ['plan_1', 'study_1', 'study_2']
    return dict(out=out, ct=ct, mr_a=mr_a, mr_b=mr_b)


@pytest.fixture
def converted(master, tmp_path):
    shutil.copytree(master['out'], tmp_path / 'out')
    return tmp_path / 'out'


def imaging(series, k, modality, n, size, spacing):
    return dict(case_dir=f'{PID}/study_{n}/series_{modality}_{k}', patient=PID, ct_size=size, ct_spacing=spacing, ct_series_uid=series['series_uid'],
                study_uid=series['study_uid'], case_type='imaging', modality=modality, image=f'series_{modality}_{k}.nii.gz', **BLANK_IMAGING)


def test_a_converted_tree_gives_one_imaging_row_per_study_series_and_a_plan_row(converted, master):
    assert man.manifest(converted) == (4, 0)
    rows = read(converted)
    assert [r['case_dir'] for r in rows] == [f'{PID}/plan_1', f'{PID}/study_1/series_CT_1', f'{PID}/study_2/series_MR_1', f'{PID}/study_2/series_MR_2']
    assert rows[0]['case_type'] == 'plan' and rows[0]['modality'] == 'CT' and rows[0]['image'] == ''
    assert rows[1:] == [imaging(master['ct'], 1, 'CT', 1, '5;3;4', '0.8;1.5;3.0'),  # np.float32 text, not 0.800000011920929
                        imaging(master['mr_a'], 1, 'MR', 2, '5;3;4', '1.0;1.0;1.0'),
                        imaging(master['mr_b'], 2, 'MR', 2, '5;3;4', '1.0;1.0;1.0')]
    for row in rows[1:]:  # the image is a file of the study folder, the case_dir an id
        assert (converted / row['patient'] / row['case_dir'].split('/')[1] / row['image']).is_file()
        assert not (converted / row['case_dir']).exists()


def test_an_imaging_size_and_spacing_are_formatted_as_a_plan_rows(converted):
    man.manifest(converted)
    rows = read(converted)
    plan_row = rows[0]
    plan_spacing = ';'.join(str(np.float32(v)) for v in (1.0, 1.5, 2.0))
    assert (plan_row['ct_size'], plan_row['ct_spacing']) == ('6;5;4', plan_spacing)  # test_convert's SHAPE and SPACING
    assert re.fullmatch(r'\d+(;\d+){2}', rows[1]['ct_size']) and re.fullmatch(r'[\d.]+(;[\d.]+){2}', rows[1]['ct_spacing'])


def study_edit(change):
    def run(study):
        rows = sf.read(study)
        change(rows)
        (study / conv.SERIES_NAME).write_text(json.dumps(rows), encoding='utf-8')  # raw: a change may break the keys
    return run


def set_cell(row, column, value):  # row 1 is the first item
    def change(rows):
        rows[row - 1][column] = value
    return study_edit(change)


def drop_data_rows(rows):
    del rows[:]


def rename_first_column(rows):
    rows[0]['filename'] = rows[0].pop('file')


BROKEN_STUDIES = {
    'unparseable': (lambda study: (study / conv.SERIES_NAME).write_bytes(b'\xff\xfe\x00\x80'), 1),
    'empty file': (lambda study: (study / conv.SERIES_NAME).write_bytes(b''), 1),
    'empty array': (study_edit(drop_data_rows), 1),
    'wrong keys': (study_edit(rename_first_column), 2),
    'missing key': (study_edit(lambda rows: rows[0].popitem()), 2),
    'kind': (set_cell(1, 'kind', 'ct'), 2),
    'modality': (set_cell(2, 'modality', 'RTDOSE'), 2),
    'bad name': (set_cell(1, 'file', 'weird.nii.gz'), 2),
    'duplicate id': (set_cell(1, 'file', 'resample_series_MR_2.nii.gz'), 2),
    'missing file': (lambda study: (study / 'series_MR_2.nii.gz').unlink(), 2),
}


@pytest.mark.parametrize('case', list(BROKEN_STUDIES))
def test_a_broken_study_is_left_out_and_counted_while_the_plan_and_other_studies_keep_their_rows(converted, case):
    change, count = BROKEN_STUDIES[case]
    change(converted / PID / 'study_2')
    assert man.manifest(converted) == (2, count)
    assert [r['case_dir'] for r in read(converted)] == [f'{PID}/plan_1', f'{PID}/study_1/series_CT_1']


@pytest.mark.parametrize('how', ['garbage', 'oserror'])
def test_a_series_whose_header_cannot_be_read_is_refused_and_its_siblings_are_kept(converted, monkeypatch, how):
    if how == 'garbage':
        (converted / PID / 'study_2' / 'series_MR_1.nii.gz').write_bytes(b'not a nifti')
    else:
        real = man.read_info

        def flaky(path):
            if path.name == 'series_MR_1.nii.gz':
                raise OSError('SYNTHETIC_PATH')
            return real(path)

        monkeypatch.setattr(man, 'read_info', flaky)
    assert man.manifest(converted) == (3, 1)
    assert [r['case_dir'] for r in read(converted)] == [f'{PID}/plan_1', f'{PID}/study_1/series_CT_1', f'{PID}/study_2/series_MR_2']


def test_an_unreadable_study_series_json_is_one_refused_study_and_the_plan_row_is_kept(converted, monkeypatch):
    real = man.read_series_json

    def flaky(folder):
        if folder.name == 'study_2':
            raise OSError('SYNTHETIC_PATH')
        return real(folder)

    monkeypatch.setattr(man, 'read_series_json', flaky)
    assert man.manifest(converted) == (2, 1)
    assert [r['case_dir'] for r in read(converted)] == [f'{PID}/plan_1', f'{PID}/study_1/series_CT_1']


def test_a_study_whose_check_fails_after_series_json_reads_counts_its_rows_from_that_single_read(converted, monkeypatch):
    real_read, real_check, reads = man.read_series_json, man.check_study_rows, []

    def counted(folder):
        reads.append(folder.name)
        return real_read(folder)

    def flaky(study, parsed):
        if study.name == 'study_2':
            raise OSError('SYNTHETIC_PATH')
        return real_check(study, parsed)

    monkeypatch.setattr(man, 'read_series_json', counted)
    monkeypatch.setattr(man, 'check_study_rows', flaky)
    assert man.manifest(converted) == (2, 2)  # study_2's two series rows
    assert reads.count('study_2') == 1


def test_only_study_n_folders_with_a_series_json_are_walked(converted):
    patient = converted / PID
    (patient / 'study_3').mkdir()  # no series.json: not a study, not a refusal
    for name in ('study_x', 'study_1b', '.study_0', 'images'):
        (patient / name).mkdir()
        (patient / name / conv.SERIES_NAME).write_bytes((patient / 'study_2' / conv.SERIES_NAME).read_bytes())
    assert man.manifest(converted) == (4, 0)


def test_the_log_and_exceptions_of_the_imaging_path_carry_counts_only(converted, caplog, capfd, monkeypatch):
    (converted / PID / 'study_2' / 'series_MR_1.nii.gz').write_bytes(b'not a nifti')
    (converted / PID / 'study_1' / conv.SERIES_NAME).write_bytes(b'\xff\xfe\x00\x80')
    with caplog.at_level(logging.DEBUG):
        assert man.manifest(converted) == (2, 2)
    logged = capfd.readouterr()
    for text in (caplog.text, logged.out, logged.err):
        for secret in ('SYNTHETIC', PID, str(converted), 'study_', 'series_', '.nii'):
            assert secret not in text
    assert [r.getMessage() for r in caplog.records if r.name == 'rt_dataprep.manifest'] == ['manifest: 2 written, 2 refused']


# ---- task-04: the series column lists the plan's aligned series (reg-resampling D10, AC11) ----

def series_column(root, case='plan_1'):
    man.manifest(root)
    return next(r for r in read(root) if r['case_dir'].endswith(f'/{case}'))['series']


CT_GRID_SIZE = (12, 12, 12)  # write_plan's native CT grid


def test_ac11_a_preprocessed_tree_lists_the_aligned_mr_by_output_name_and_the_converted_tree_lists_none(tmp_path):
    run_plan(tmp_path, [mr_spec(landmark(MR_SIZE, (3, 4, 2)), rigid(0, (1.25, 0, 0)))])
    assert series_column(tmp_path / 'out') == [{'file': MR_OUT, 'modality': 'MR'}]
    assert series_column(tmp_path / 'converted') == []  # a translated MR on its own grid


def test_fov_mask_key_is_listed_when_the_file_exists_on_the_ct_grid(tmp_path):
    spec = mr_spec(landmark(MR_SIZE, (3, 4, 2)), rigid(0, (1.25, 0, 0)))
    run_plan(tmp_path / 'off', [spec])
    run_plan(tmp_path / 'on', [spec], steps=dataclasses.replace(SMALL, series_fov_mask=True))
    assert series_column(tmp_path / 'on' / 'out') == [{'file': MR_OUT, 'modality': 'MR', 'fov_mask': f'fovmask_{MR_OUT}'}]
    assert series_column(tmp_path / 'off' / 'out') == [{'file': MR_OUT, 'modality': 'MR'}]  # exactly two keys
    assert series_column(tmp_path / 'on' / 'converted') == [] == series_column(tmp_path / 'off' / 'converted')


@pytest.mark.parametrize('how', ['off_grid', 'unreadable'])
def test_a_fov_mask_off_the_ct_grid_or_unreadable_is_not_listed_and_its_series_is(tmp_path, how):
    _, out, _ = run_plan(tmp_path, [mr_spec(landmark(MR_SIZE, (3, 4, 2)), rigid(0, (1.25, 0, 0)))],
                         steps=dataclasses.replace(SMALL, series_fov_mask=True))
    mask = out / f'fovmask_{MR_OUT}'
    assert mask.is_file()
    if how == 'off_grid':
        sitk.WriteImage(sitk.Image([3, 3, 3], sitk.sitkUInt8), str(mask))
    else:
        mask.write_bytes(b'not a nifti')
    assert series_column(tmp_path / 'out') == [{'file': MR_OUT, 'modality': 'MR'}]


def test_ac11_a_translated_mr_with_the_ct_geometry_is_not_listed_but_an_identity_one_is(tmp_path):
    ones = np.ones(CT_GRID_SIZE)
    translated = mr_spec(ones, rigid(0, (2.0, 0, 0)), grid(12))
    on_grid = mr_spec(ones, 'identity', grid(12), file='series_PT_1.nii.gz', modality='PT')
    run_plan(tmp_path, [translated, on_grid])
    assert series_column(tmp_path / 'converted') == [{'file': 'series_PT_1.nii.gz', 'modality': 'PT'}]


def test_the_series_column_keeps_series_json_order_and_skips_an_unreadable_header_without_refusing_the_plan(tmp_path):
    ones = np.ones(CT_GRID_SIZE)
    run_plan(tmp_path, [mr_spec(ones, 'identity', grid(12), file='series_PT_1.nii.gz', modality='PT'),
                        mr_spec(b'not a nifti', 'identity', file='series_MR_2.nii.gz'),
                        mr_spec(ones, 'identity', grid(12))])
    assert man.manifest(tmp_path / 'converted') == (1, 0)
    assert series_column(tmp_path / 'converted') == [{'file': 'series_PT_1.nii.gz', 'modality': 'PT'}, {'file': 'series_MR_1.nii.gz', 'modality': 'MR'}]


def test_imaging_rows_hold_an_empty_series_list(converted):
    man.manifest(converted)
    assert [r['series'] for r in read(converted)[1:]] == [[]] * 3


# ---- AHE task-05: the ahe_ names go through series_id, so the manifest reads them for study series and aligned plan series (spec AC5) ----

def test_ac5_an_ahe_study_series_is_listed_under_its_series_id_with_its_ahe_file_name(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted, series=(('CT', 1), ('MR', 1)), n=24)  # 24^3 ramps: smaller volumes equalize to all 1
    pre.preprocess(converted, out, SMALL_AHE)
    assert man.manifest(out) == (2, 0)
    rows = read(out)
    assert [(r['case_dir'], r['case_type'], r['modality'], r['image'], r['series']) for r in rows] == [
        ('SYNTHETICPTA/study_1/series_CT_1', 'imaging', 'CT', AHE_STUDY_CT, []),
        ('SYNTHETICPTA/study_1/series_MR_1', 'imaging', 'MR', AHE_MR_OUT, [])]
    assert all((out / 'SYNTHETICPTA' / 'study_1' / r['image']).is_file() for r in rows)


def test_ac5_an_ahe_only_study_series_is_listed_under_its_series_id(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_study_tree(converted, series=(('MR', 1),), n=24)
    pre.preprocess(converted, out, pre.PreprocessSteps(ahe=True))
    assert man.manifest(out) == (1, 0)
    assert [(r['case_dir'], r['image']) for r in read(out)] == [('SYNTHETICPTA/study_1/series_MR_1', 'ahe_series_MR_1.nii.gz')]


def test_ac5_an_ahe_aligned_plan_series_is_listed_under_its_ahe_output_name(tmp_path):
    run_plan(tmp_path, [mr_spec(MR_RAMP, rigid(0, (4, 0, 0)))], steps=SMALL_AHE)
    assert series_column(tmp_path / 'out') == [{'file': AHE_MR_OUT, 'modality': 'MR'}]
    assert AHE_MR_OUT != MR_OUT


def test_a_converted_tree_and_a_preprocessed_tree_both_pass_the_checks_and_the_manifest(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted)
    write_study_tree(converted)
    pre.preprocess(converted, out, SMALL)
    for root, columns in ((converted, conv.SERIES_COLUMNS), (out, pre.PREPROCESSED_SERIES_COLUMNS)):
        pre.check_rows(root / 'SYNTHETICPTA' / 'plan_1')
        study = pre.check_study_rows(root / 'SYNTHETICPTA' / 'study_1')
        assert list(study[0]) == columns
        assert man.manifest(root) == (3, 0)


def test_a_series_json_with_keys_matching_neither_list_is_malformed_and_left_out(tmp_path):
    root = tmp_path / 'converted'
    path, rows = write_plan(root)
    sf.write([row | {'other': ''} for row in rows], path / 'series.json')
    assert pre.read_series_json(path)[1] is True
    assert man.manifest(root) == (0, 1)


# ---------------------------------------------------------------- json-outputs task-01: legacy series.csv folders are counted refused

def make_legacy(folder):
    (folder / conv.SERIES_NAME).unlink()
    (folder / pre.LEGACY_SERIES_NAME).write_text('file,kind\n', encoding='utf-8')


def test_a_legacy_plan_and_a_legacy_study_count_one_refusal_each(converted):
    patient = converted / PID
    make_legacy(patient / 'plan_1')
    make_legacy(patient / 'study_2')  # two series rows in the file it replaces: still 1, the CSV is never parsed
    assert man.manifest(converted) == (1, 2)
    assert [r['case_dir'] for r in read(converted)] == [f'{PID}/study_1/series_CT_1']


def test_a_legacy_study_is_counted_without_reading_anything(converted, monkeypatch):
    make_legacy(converted / PID / 'study_2')
    monkeypatch.setattr(man, 'read_series_json', lambda *a: pytest.fail('series.json was read'))
    assert man._study_rows(converted / PID, converted / PID / 'study_2') == ([], 1)


def test_images_and_misc_files_with_a_series_csv_are_not_counted(converted):
    for name in ('images', 'misc_files'):
        (converted / PID / name).mkdir()
        (converted / PID / name / pre.LEGACY_SERIES_NAME).write_text('file,kind\n', encoding='utf-8')
    assert man.manifest(converted) == (4, 0)


# ---------------------------------------------------------------- json-outputs task-02: manifest.jsonl

def raw_lines(root, name='manifest.jsonl'):
    """The file's bytes split on b'\\n' only, so a stray U+2028 would not be hidden by str.splitlines."""
    data = (root / name).read_bytes()
    assert data.endswith(b'\n') and b'\r' not in data
    return data.decode('utf-8').split('\n')[:-1]


def test_manifest_jsonl_has_one_ascii_json_object_per_line_in_column_order_with_native_lists(tmp_path):
    root = tree(tmp_path / 'converted')
    assert man.MANIFEST_NAME == 'manifest.jsonl' and man.manifest(root) == (3, 0)
    lines = raw_lines(root)
    assert len(lines) == 3 and all(line.isascii() for line in lines)
    objs = [json.loads(line) for line in lines]
    assert [list(o) for o in objs] == [COLUMNS] * 3 and lines == [json.dumps(o) for o in objs]
    assert [o['case_dir'] for o in objs] == sorted(o['case_dir'] for o in objs)
    assert objs[1] == dict(case_dir=f'{PATIENT_A}/plan_1', patient=PATIENT_A, fx='plan_1', ct_size='12;12;12', ct_spacing='1.0;1.0;1.0',
                           dose='1', masks=['bladder', 'ptv'], rtplan_uid='SYNTHETICRTPLANA1', ct_series_uid='SYNTHETICCTA1',
                           study_uid='SYNTHETICSTUDYA1', case_type='plan', modality='CT', image='', series=[])
    assert all(isinstance(o[k], str) for o in objs for k in COLUMNS if k not in ('masks', 'series'))


def test_fov_mask_is_on_every_line_when_any_case_has_one_and_on_none_otherwise(tmp_path):
    root = tree(tmp_path / 'converted')
    man.manifest(root)
    assert all('fov_mask' not in json.loads(line) for line in raw_lines(root))
    plan_dir = root / PATIENT_A / 'plan_1'
    shutil.copyfile(plan_dir / conv.CT_NAME, plan_dir / f'fovmask_{conv.CT_NAME}')  # a mask on the CT grid
    man.manifest(root)
    objs = [json.loads(line) for line in raw_lines(root)]
    assert [list(o) for o in objs] == [[*COLUMNS, 'fov_mask']] * 3
    assert {o['case_dir']: o['fov_mask'] for o in objs} == {f'{PATIENT_B}/fx_1': '', f'{PATIENT_A}/plan_1': f'fovmask_{conv.CT_NAME}',
                                                              f'{PATIENT_A}/plan_2': ''}


def test_a_root_without_cases_gets_a_zero_byte_manifest_and_a_dated_copy(tmp_path):
    root = tmp_path / 'empty'
    root.mkdir()
    assert man.manifest(root) == (0, 0)
    names = sorted(p.name for p in root.iterdir())
    assert len(names) == 2 and names[0] == 'manifest.jsonl' and re.fullmatch(r'manifest_\d{8}T\d{6}Z\.jsonl', names[1])
    assert all((root / name).stat().st_size == 0 for name in names)


def test_an_roi_name_with_u2028_stays_on_its_line_and_splits(tmp_path):
    root = tmp_path / 'converted'
    name = 'ptv boost'
    plan(root, PATIENT_A, 'plan_1', 'A1', masks={name: box((4, 4, 4), (8, 8, 8))})
    plan(root, PATIENT_B, 'plan_1', 'B1')
    assert man.manifest(root) == (2, 0)
    data = (root / 'manifest.jsonl').read_bytes()
    assert ' '.encode() not in data and b'\\u2028' in data
    assert len(data.decode('utf-8').splitlines()) == 2
    assert [json.loads(line)['masks'] for line in raw_lines(root)] == [['bladder', 'ptv'], [name]]  # B first: 'P' sorts before '_'
    obj = sp.split(root / 'manifest.jsonl', test_size=0.0, val_size=0.0, seed=0)
    assert sorted(obj['train']) == [PATIENT_B, PATIENT_A]


# task-04: the 8 optional keys, from the primary image's `dicom` (series_date from its own field)
FULL_TAGS = dict(PatientAge='063Y', PatientSex='F', PatientWeight='70.5', Manufacturer='SYNTHETIC_MAKER',
                 ManufacturerModelName='SYNTHETIC_MODEL', InstitutionName='SYNTHETIC_SITE')


@pytest.fixture(scope='module')
def meta_master(tmp_path_factory):
    """A converted tree: a plan whose CT carries every optional tag, a study CT carrying only two (and no series date), and a study
    PT whose slices differ in Manufacturer (a {'distinct'} value)."""
    base = tmp_path_factory.mktemp('manifest_meta')
    org, out = base / 'org', base / 'out'
    make_plan(org / PID / 'plan_1', doses=[dose_spec(10)], rois=[BOX],
              ct_meta=dict(study_date=STUDY_D1, series_date=STUDY_D1, tags=FULL_TAGS))
    misc = org / PID / 'misc_files'
    put(misc, 'ct', 'CT', study_date=STUDY_D2, tags=dict(PatientSex='M', Manufacturer='SYNTHETIC_MAKER'))
    put(misc, 'pt', 'PT', study_date=STUDY_D2, series_date=STUDY_D2,
        tags=dict(Manufacturer='SYNTHETIC_A'), per_slice=[{'Manufacturer': 'SYNTHETIC_B'}, None, None, None])
    conv.convert(org, out)
    return out


def meta_lines(out, tmp_path):
    shutil.copytree(out, tmp_path / 'out')
    root = tmp_path / 'out'
    man.manifest(root)
    return root, {row['case_dir']: row for row in [json.loads(line) for line in (root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]}


def test_the_optional_keys_follow_the_fixed_keys_in_order_and_come_from_the_primary_image(meta_master, tmp_path):
    root, lines = meta_lines(meta_master, tmp_path)
    plan_line = lines[f'{PID}/plan_1']
    assert list(plan_line) == [*COLUMNS, *OPTIONAL]
    assert (plan_line['patient_age'], plan_line['patient_sex'], plan_line['patient_weight']) == (63, 'F', '70.5')
    assert (plan_line['study_date'], plan_line['series_date']) == (STUDY_D1, STUDY_D1)
    assert (plan_line['manufacturer'], plan_line['model'], plan_line['institution_name']) == ('SYNTHETIC_MAKER', 'SYNTHETIC_MODEL', 'SYNTHETIC_SITE')
    ct_line = next(row for row in lines.values() if row['modality'] == 'CT' and row['case_type'] == 'imaging')
    assert list(ct_line) == [*COLUMNS, 'patient_sex', 'study_date', 'manufacturer']  # absent tags and an empty series_date: no key, no null
    assert (ct_line['patient_sex'], ct_line['study_date'], ct_line['manufacturer']) == ('M', STUDY_D2, 'SYNTHETIC_MAKER')
    pt_line = next(row for row in lines.values() if row['modality'] == 'PT')
    assert pt_line['manufacturer'] == {'distinct': ['SYNTHETIC_B', 'SYNTHETIC_A']} and pt_line['series_date'] == STUDY_D2  # copied as is


def test_a_series_json_without_dicom_still_gives_lines_with_only_series_date(meta_master, tmp_path):
    root = tmp_path / 'out'
    shutil.copytree(meta_master, root)
    for folder in (p for patient in root.iterdir() if patient.is_dir() for p in patient.iterdir() if p.is_dir()):
        rows = sf.read(folder)
        for row in rows:
            row.pop('dicom')
        (folder / conv.SERIES_NAME).write_text(json.dumps(rows), encoding='utf-8')
    assert man.manifest(root)[1] == 0
    lines = [json.loads(line) for line in (root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(lines) == 3
    for line in lines:
        assert not set(line) & (set(OPTIONAL) - {'series_date'}) and set(line) - {'series_date'} <= {*COLUMNS}
    assert {line['case_dir'].split('/')[1]: line.get('series_date') for line in lines}['plan_1'] == STUDY_D1


def test_a_plan_line_carries_the_lowest_dose_or_mask_retained_of_a_cut_plan_and_none_on_a_converted_tree(tmp_path):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted, dose=DOSE_SPLIT, masks={'ptv': HALF_OUT, 'bladder': box((5, 5, 5), (8, 8, 8))})  # retained 0.75, 0.5, 1.0
    pre.preprocess(converted, out, CUT)
    assert man.manifest(out) == (1, 0) and man.manifest(converted) == (1, 0)
    line = json.loads((out / 'manifest.jsonl').read_text(encoding='utf-8'))
    assert list(line)[-1] == 'retained' and line['retained'] == 0.5
    assert 'retained' not in json.loads((converted / 'manifest.jsonl').read_text(encoding='utf-8'))
    assert 'retained' not in read(out)[0]


@pytest.mark.parametrize('cell', ['half', 'nan'])
def test_a_plan_whose_retained_cell_is_not_a_finite_number_is_refused(tmp_path, cell):
    converted, out = tmp_path / 'converted', tmp_path / 'out'
    write_plan(converted, dose=DOSE_SPLIT)
    pre.preprocess(converted, out, CUT)
    path = out / 'SYNTHETICPTA' / 'plan_1' / conv.SERIES_NAME
    rows = json.loads(path.read_text(encoding='utf-8'))
    next(row for row in rows if row['retained'])['retained'] = cell
    path.write_text(json.dumps(rows), encoding='utf-8')
    assert man.manifest(out) == (0, 1)


def test_a_patient_folder_whose_name_is_not_utf8_is_refused_and_the_rest_still_splits(tmp_path):
    root = tree(tmp_path / 'converted')
    write_plan(root, 'SYNTHETIC_C')
    (root / 'SYNTHETIC_C').rename(root / os.fsdecode(b'SYNTHETIC_\xff'))  # a surrogate escape; SimpleITK aborts on such a path
    assert man.manifest(root) == (3, 1)
    assert {row['patient'] for row in read(root)} == {PATIENT_A, PATIENT_B}
    sp.split(root / 'manifest.jsonl', test_size=0.5, val_size=0.0, seed=0)  # #249 would refuse the whole manifest
