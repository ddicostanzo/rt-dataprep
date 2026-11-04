"""manifest: one row per plan folder of a converted or preprocessed tree (task-01); patient folder names are SYNTHETIC-prefixed."""
import csv
import logging
import re

from test_preprocess import SMALL, box, save_rows, state, write_plan

from rt_dataprep import convert as conv
from rt_dataprep import manifest as man
from rt_dataprep import preprocess as pre

COLUMNS = ['case_dir', 'patient', 'fx', 'ct_size', 'ct_spacing', 'dose', 'masks', 'rtplan_uid', 'ct_series_uid', 'study_uid']
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
    with open(root / 'manifest.csv', newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == COLUMNS
        return list(reader)


def test_rows_columns_and_values_on_a_converted_tree(tmp_path):
    root = tree(tmp_path / 'converted')
    assert man.manifest(root) == (3, 0)
    rows = read(root)
    assert [r['case_dir'] for r in rows] == [f'{PATIENT_B}/fx_1', f'{PATIENT_A}/plan_1', f'{PATIENT_A}/plan_2']  # '_' sorts after 'P'
    assert rows[1] == dict(case_dir=f'{PATIENT_A}/plan_1', patient=PATIENT_A, fx='plan_1', ct_size='12;12;12', ct_spacing='1.0;1.0;1.0',
                           dose='1', masks='bladder;ptv', rtplan_uid='SYNTHETICRTPLANA1', ct_series_uid='SYNTHETICCTA1',
                           study_uid='SYNTHETICSTUDYA1')
    assert (rows[2]['dose'], rows[2]['masks'], rows[2]['rtplan_uid']) == ('0', 'ptv', 'SYNTHETICRTPLANA2')
    assert (rows[0]['patient'], rows[0]['fx'], rows[0]['ct_size'], rows[0]['masks']) == (PATIENT_B, 'fx_1', '14;14;14', '')


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
    for name in ('images', 'misc_files'):  # even with a series.csv inside, not a plan folder
        folder = root / PATIENT_A / name
        folder.mkdir()
        (folder / conv.SERIES_NAME).write_bytes((root / PATIENT_A / 'plan_1' / conv.SERIES_NAME).read_bytes())
    (root / PATIENT_A / 'plan_3').mkdir()  # a plan folder without series.csv
    write_plan(root, '.SYNTHETICPTC.partial', 'plan_1')  # a hidden patient folder
    (root / 'stray.txt').write_text('x')
    assert man.manifest(root) == (3, 0)
    assert len(read(root)) == 3


def test_a_malformed_series_csv_is_left_out_and_counted_without_a_name_in_the_log(tmp_path, caplog, capfd):
    root = tree(tmp_path / 'converted')
    bad = plan(root, 'SYNTHETIC_DISTINCTIVE_NAME', 'plan_1', 'D1')
    (bad / conv.SERIES_NAME).write_text('not,the,columns\n1,2\n')
    with caplog.at_level(logging.DEBUG):
        assert man.manifest(root) == (3, 1)
    assert 'SYNTHETIC_DISTINCTIVE_NAME' not in [r['case_dir'].split('/')[0] for r in read(root)]
    logged = capfd.readouterr()
    for text in (caplog.text, logged.out, logged.err):
        for secret in ('SYNTHETIC', str(tmp_path), 'plan_1'):
            assert secret not in text
    assert [r.getMessage() for r in caplog.records if r.name == 'rt_dataprep.manifest'] == ['manifest: 3 written, 1 refused']


def test_only_manifest_csv_and_its_dated_copy_are_written(tmp_path):
    root = tree(tmp_path / 'converted')
    before = state(root)
    man.manifest(root)
    after = state(root)
    new = {p for p in after if p not in before}
    assert {p.name for p in new} >= {'manifest.csv'} and all(re.fullmatch(r'manifest(_\d{8}T\d{6}Z)?\.csv', p.name) for p in new)
    assert len(new) == 2
    assert all(after[p] == data for p, data in before.items())

