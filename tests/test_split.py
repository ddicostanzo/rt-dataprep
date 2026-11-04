"""split: splits.json from a hand-written manifest.csv, whole patients per split (task-02); patient names are SYNTHETIC-prefixed except the bare '00123'."""
import ast
import csv
import hashlib
import inspect
import json
import os
import threading
import time

import pytest

from rt_dataprep import convert as conv
from rt_dataprep import split as sp
from rt_dataprep import manifest as man
from rt_dataprep.manifest import COLUMNS

SEED_A, SEED_B = 0, 1
TEN = [f'SYNTHETIC_P{i:02d}' for i in range(9)] + ['00123']


def write_manifest(folder, patients, name='manifest.csv', two_plans=('SYNTHETIC_P00',)):
    """One plan per patient, two for those in `two_plans`; only the case_dir and patient columns matter to split."""
    folder.mkdir(exist_ok=True)
    path = folder / name
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for patient in patients:
            for fx in (['plan_1', 'plan_2'] if patient in two_plans else ['plan_1']):
                writer.writerow([f'{patient}/{fx}', patient, fx, '', '', '1', '', '', '', ''])
    return path


def run(path, test_size=0.2, val_size=0.2, seed=SEED_A, **kwargs):
    sp.split(path, test_size=test_size, val_size=val_size, seed=seed, **kwargs)
    return json.loads((path.parent / 'splits.json').read_text())


def patients_of(obj):
    return {name: set(obj[name]) for name in ('train', 'val', 'test')}


def assignment(obj):
    return {patient: name for name in ('train', 'val', 'test') for patient in obj[name]}


def test_seed_is_required_and_keyword_only(tmp_path):
    path = write_manifest(tmp_path, TEN)
    with pytest.raises(TypeError):
        sp.split(path, test_size=0.1, val_size=0.1)
    with pytest.raises(TypeError):
        sp.split(path, 0.1, 0.1, 0)


@pytest.mark.parametrize(('n', 'test_size', 'val_size', 'counts'), [
    (10, 0.1, 0.1, (8, 1, 1)), (10, 0.15, 0.1, (7, 1, 2)), (100, 0.07, 0.1, (83, 10, 7)), (10, 0.0, 0.0, (10, 0, 0))])
def test_split_sizes_are_ceil_of_rounded_product(tmp_path, n, test_size, val_size, counts):
    obj = run(write_manifest(tmp_path, [f'SYNTHETIC_P{i:03d}' for i in range(n)]), test_size, val_size)
    assert tuple(len(obj[name]) for name in ('train', 'val', 'test')) == counts


def test_float_noise_does_not_round_up(tmp_path):
    # 100 * 0.07 = 7.000000000000001 in float: an unrounded ceil would give 8
    obj = run(write_manifest(tmp_path, [f'SYNTHETIC_P{i:03d}' for i in range(100)]), 0.07, 0.0)
    assert len(obj['test']) == 7


@pytest.mark.parametrize(('test_size', 'val_size'), [(-0.1, 0.1), (1.0, 0.0), (0.1, 1.0), (0.1, -0.5), (1.5, 0.1)])
def test_a_size_outside_the_unit_interval_is_refused_before_writing(tmp_path, test_size, val_size):
    path = write_manifest(tmp_path, TEN)
    with pytest.raises(ValueError, match=sp.REASON_SIZE):
        sp.split(path, test_size=test_size, val_size=val_size, seed=SEED_A)
    assert not (tmp_path / 'splits.json').exists()


@pytest.mark.parametrize(('test_size', 'val_size'), [(0.5, 0.5), (0.9, 0.9), (0.6, 0.4)])
def test_an_empty_train_is_refused(tmp_path, test_size, val_size):
    path = write_manifest(tmp_path, TEN)
    with pytest.raises(ValueError, match=sp.REASON_NO_TRAIN):
        sp.split(path, test_size=test_size, val_size=val_size, seed=SEED_A)
    assert not (tmp_path / 'splits.json').exists()


def test_whole_patients_per_split_and_every_case_placed(tmp_path):
    path = write_manifest(tmp_path, TEN)
    obj = run(path)
    sets = patients_of(obj)
    assert sets['train'] | sets['val'] | sets['test'] == set(TEN)
    assert sum(len(s) for s in sets.values()) == len(TEN)  # no patient in two splits
    cases = [case for name in ('train', 'val', 'test') for plans in obj[name].values() for case in plans]
    assert len(cases) == len(set(cases)) == 11
    for name in ('train', 'val', 'test'):
        for patient, plans in obj[name].items():
            assert plans == sorted(f'{patient}/{fx}' for fx in ('plan_1', 'plan_2') if f'{patient}/{fx}' in cases)
    holder = next(name for name in ('train', 'val', 'test') if 'SYNTHETIC_P00' in obj[name])
    assert obj[holder]['SYNTHETIC_P00'] == ['SYNTHETIC_P00/plan_1', 'SYNTHETIC_P00/plan_2']


def test_same_seed_is_byte_identical_and_two_seeds_differ_as_assignments(tmp_path):
    path = write_manifest(tmp_path, TEN)
    run(path, seed=SEED_A)
    first = (tmp_path / 'splits.json').read_bytes()
    run(path, seed=SEED_A)
    assert (tmp_path / 'splits.json').read_bytes() == first
    other = tmp_path / 'other'
    write_manifest(other, TEN)
    assert assignment(run(other / 'manifest.csv', seed=SEED_B)) != assignment(json.loads(first))


def test_file_format_and_case_id_hash(tmp_path):
    path = write_manifest(tmp_path, TEN)
    obj = run(path)
    text = (tmp_path / 'splits.json').read_text()
    assert text == json.dumps(obj, indent=1, sort_keys=True) + '\n'
    assert set(obj) == {'train', 'val', 'test', 'case_id_hash'}
    for name in ('train', 'val', 'test'):
        ids = sorted(case for plans in obj[name].values() for case in plans)
        assert obj['case_id_hash'][name] == hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def test_a_bare_numeric_patient_keeps_its_leading_zeros(tmp_path):
    obj = run(write_manifest(tmp_path, TEN))
    placed = [plans for name in ('train', 'val', 'test') for patient, plans in obj[name].items() if patient == '00123']
    assert placed == [['00123/plan_1']]


def test_external_overlap_is_refused_and_nothing_is_written(tmp_path, caplog, capfd):
    path = write_manifest(tmp_path, TEN)
    external = write_manifest(tmp_path / 'ext', ['SYNTHETIC_OTHER_ONLY', 'SYNTHETIC_P03'], two_plans=())
    caplog.set_level('DEBUG')
    with pytest.raises(ValueError, match=sp.REASON_OVERLAP) as excinfo:
        sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A, external=external)
    assert not (tmp_path / 'splits.json').exists()
    out, err = capfd.readouterr()
    for text in (str(excinfo.value), caplog.text, out, err):
        assert 'SYNTHETIC_P03' not in text
        assert 'SYNTHETIC_OTHER_ONLY' not in text


def test_a_disjoint_external_manifest_passes(tmp_path):
    path = write_manifest(tmp_path, TEN)
    external = write_manifest(tmp_path / 'ext', ['SYNTHETIC_ELSEWHERE'], two_plans=())
    assert 'train' in run(path, external=external)


def test_check_disjoint_on_the_split_dict(tmp_path):
    external = write_manifest(tmp_path, ['SYNTHETIC_X'], two_plans=())
    splits = {'train': {'SYNTHETIC_A': ['SYNTHETIC_A/plan_1']}, 'val': {}, 'test': {}, 'case_id_hash': {'train': 'SYNTHETIC_X'}}
    sp.check_disjoint(splits, external)  # a hash value is not a patient
    splits['test'] = {'SYNTHETIC_X': ['SYNTHETIC_X/plan_1']}
    with pytest.raises(ValueError, match=sp.REASON_OVERLAP):
        sp.check_disjoint(splits, external)


def test_an_existing_splits_json_with_different_content_is_refused_and_untouched(tmp_path):
    path = write_manifest(tmp_path, TEN)
    run(path, seed=SEED_A)
    before = (tmp_path / 'splits.json').read_bytes()
    with pytest.raises(ValueError, match=sp.REASON_EXISTS):
        sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_B)
    assert (tmp_path / 'splits.json').read_bytes() == before
    run(path, seed=SEED_A)  # identical content: rewritten without error
    assert (tmp_path / 'splits.json').read_bytes() == before


def test_the_write_is_atomic_a_failed_replace_leaves_no_splits_json_and_no_temp_file(tmp_path, monkeypatch):
    path = write_manifest(tmp_path, TEN)

    def fail(*args):
        raise OSError('SYNTHETIC_PATH')

    monkeypatch.setattr(os, 'link', fail)  # no hard links, so the rename fallback runs, and fails too
    monkeypatch.setattr(os, 'replace', fail)
    with pytest.raises(OSError, match=conv.REASON_WRITE):
        sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.csv']
    monkeypatch.undo()
    run(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.csv', 'splits.json']


def test_without_hard_links_a_ctrl_c_during_the_replace_leaves_no_splits_json(tmp_path, monkeypatch):
    path = write_manifest(tmp_path, TEN)

    def unsupported(*args):
        raise PermissionError('SYNTHETIC_PATH')

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, 'link', unsupported)
    monkeypatch.setattr(os, 'replace', interrupt)
    with pytest.raises(KeyboardInterrupt):
        sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.csv']  # no empty reservation blocks a rerun


def test_a_concurrent_run_that_writes_first_wins_and_this_one_is_refused(tmp_path, monkeypatch):
    path = write_manifest(tmp_path, TEN)
    other = '{"SYNTHETIC_OTHER_RUN": 1}\n'
    link = os.link

    def race(source, target):
        (tmp_path / 'splits.json').write_text(other)  # the other run lands between the existence check and the link
        link(source, target)

    monkeypatch.setattr(os, 'link', race)
    with pytest.raises(ValueError, match=sp.REASON_EXISTS):
        sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A)
    assert (tmp_path / 'splits.json').read_text() == other
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.csv', 'splits.json']


def test_without_hard_links_a_concurrent_run_that_writes_first_is_kept_and_this_one_is_refused(tmp_path, monkeypatch):
    path = write_manifest(tmp_path, TEN)
    other = '{"SYNTHETIC_OTHER_RUN": 1}\n'

    def race(source, target):
        (tmp_path / 'splits.json').write_text(other)  # the other run lands between the existence check and the write
        raise PermissionError('SYNTHETIC_PATH')  # no hard links here

    monkeypatch.setattr(os, 'link', race)
    with pytest.raises(ValueError, match=sp.REASON_EXISTS):
        sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A)
    assert (tmp_path / 'splits.json').read_text() == other
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.csv', 'splits.json']


def test_an_identical_split_is_not_refused_by_another_runs_empty_reservation(tmp_path):
    path = write_manifest(tmp_path, TEN)
    text = json.dumps(sp.split(write_manifest(tmp_path / 'other', TEN), test_size=0.2, val_size=0.2, seed=SEED_A), indent=1,
                      sort_keys=True) + '\n'
    (tmp_path / 'splits.json').write_text('')  # the other run's O_EXCL reservation, not yet renamed over
    timer = threading.Timer(0.2, (tmp_path / 'splits.json').write_text, [text])
    timer.start()
    try:
        obj = sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A)
    finally:
        timer.cancel()
        timer.join()
    assert json.dumps(obj, indent=1, sort_keys=True) + '\n' == text


def test_an_empty_splits_json_that_is_never_filled_is_still_refused_after_about_a_second(tmp_path):
    path = write_manifest(tmp_path, TEN)
    (tmp_path / 'splits.json').write_text('')
    started = time.monotonic()
    with pytest.raises(ValueError, match=sp.REASON_EXISTS):
        sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A)
    assert time.monotonic() - started < 1.5
    assert (tmp_path / 'splits.json').read_text() == ''


def test_without_hard_links_the_write_falls_back_to_a_rename(tmp_path, monkeypatch):
    path = write_manifest(tmp_path, TEN)

    def unsupported(*args):
        raise PermissionError('SYNTHETIC_PATH')

    monkeypatch.setattr(os, 'link', unsupported)
    obj = run(path)
    assert sum(map(len, (obj[name] for name in ('train', 'val', 'test')))) == len(TEN)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.csv', 'splits.json']


def test_the_order_is_the_documented_hash_so_no_library_version_moves_it(tmp_path):
    obj = run(write_manifest(tmp_path, TEN), seed=-7)  # any int is a seed, negative included
    order = sorted(TEN, key=lambda patient: hashlib.sha256(f'-7\n{patient}'.encode()).hexdigest())
    assert (set(obj['test']), set(obj['val'])) == (set(order[:2]), set(order[2:4]))  # splits.json sorts its keys


@pytest.mark.parametrize('dropped', ['patient', 'case_dir'])
def test_a_manifest_without_patient_or_case_dir_is_refused(tmp_path, dropped):
    columns = [c for c in COLUMNS if c != dropped]
    bad = tmp_path / 'bad'
    bad.mkdir()
    (bad / 'manifest.csv').write_text(','.join(columns) + '\n' + ','.join('SYNTHETIC_X' for _ in columns) + '\n')
    with pytest.raises(ValueError, match=sp.REASON_COLUMNS):
        sp.split(bad / 'manifest.csv', test_size=0.2, val_size=0.2, seed=SEED_A)
    with pytest.raises(ValueError, match=sp.REASON_COLUMNS):
        sp.split(write_manifest(tmp_path, TEN), test_size=0.2, val_size=0.2, seed=SEED_A, external=bad / 'manifest.csv')
    assert not (bad / 'splits.json').exists() and not (tmp_path / 'splits.json').exists()


@pytest.mark.parametrize('module', [sp, man])
def test_split_and_manifest_import_no_pandas_sklearn_and_split_no_numpy(module):
    tree = ast.parse(inspect.getsource(module))
    imported = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    banned = {'pandas', 'sklearn'} | ({'numpy'} if module is sp else set())
    assert not imported & banned


@pytest.mark.parametrize('body', [
    'SYNTHETIC_P00/plan_1\n',  # a short row: patient reads None
    'SYNTHETIC_P00/plan_1,\n',  # an empty patient
    'SYNTHETIC_P00/plan_1,  \n',  # a whitespace-only patient
    'SYNTHETIC_P00/plan_1,SYNTHETIC_\xe9\n',  # not UTF-8 once written as latin-1
])
def test_a_malformed_row_or_a_non_utf8_manifest_is_refused(tmp_path, capfd, body):
    path = tmp_path / 'manifest.csv'
    path.write_bytes(('case_dir,patient\n' + body).encode('latin-1'))
    with pytest.raises(ValueError) as excinfo:
        sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A)
    assert str(excinfo.value) == sp.REASON_ROWS
    assert not (tmp_path / 'splits.json').exists()
    out, err = capfd.readouterr()
    assert 'SYNTHETIC' not in out + err


def test_two_writes_in_one_process_use_different_temp_names(tmp_path, monkeypatch):
    names = []
    real = os.link

    def spy(source, target):
        names.append(os.path.basename(source))
        real(source, target)
    monkeypatch.setattr(os, 'link', spy)
    for folder in ('a', 'b'):
        run(write_manifest(tmp_path / folder, TEN))
    assert len(set(names)) == 2


def test_a_manifest_with_a_utf8_bom_is_read(tmp_path):
    path = write_manifest(tmp_path, TEN)
    path.write_bytes(b'\xef\xbb\xbf' + path.read_bytes())
    obj = run(path)
    assert sum(len(obj[name]) for name in ('train', 'val', 'test')) == len(TEN)
