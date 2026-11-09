"""split: splits.json from a hand-written manifest.jsonl, whole patients per split (task-02); patient names are SYNTHETIC-prefixed except the bare '00123'."""
import ast
import hashlib
import inspect
import json
import operator
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


def write_manifest(folder, patients, name='manifest.jsonl', two_plans=('SYNTHETIC_P00',), imaging=()):
    """One plan per patient, two for those in `two_plans`; a patient in `imaging` also gets two study series rows (a patient
    only in `imaging` has no plan). Only the case_dir and patient keys matter to split."""
    folder.mkdir(exist_ok=True)
    path = folder / name
    rows = []
    for patient in patients:
        for fx in (['plan_1', 'plan_2'] if patient in two_plans else ['plan_1']):
            rows.append([f'{patient}/{fx}', patient, fx, '', '', '1', [], '', '', '', 'plan', 'CT', '', []])
    for patient in imaging:
        for study, series in (('study_1', 'series_CT_1'), ('study_2', 'series_MR_1')):
            rows.append([f'{patient}/{study}/{series}', patient, '', '', '', '0', [], '', '', '', 'imaging', series.split('_')[1], f'{series}.nii.gz', []])
    path.write_text(''.join(json.dumps(dict(zip(COLUMNS, row, strict=True))) + '\n' for row in rows), encoding='utf-8')
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


def test_a_patients_plan_and_imaging_cases_share_one_split(tmp_path):
    imaging = [*TEN[:5], 'SYNTHETIC_IMG_ONLY']  # five patients with plan and imaging cases, one with imaging cases only
    obj = run(write_manifest(tmp_path, TEN, imaging=imaging))
    placed = {patient: name for name in ('train', 'val', 'test') for patient in obj[name]}
    assert set(placed) == {*TEN, 'SYNTHETIC_IMG_ONLY'}  # each patient is in exactly one split
    for patient in imaging:
        cases = obj[placed[patient]][patient]
        assert {c for c in cases if '/study_' in c} == {f'{patient}/study_1/series_CT_1', f'{patient}/study_2/series_MR_1'}
        assert patient == 'SYNTHETIC_IMG_ONLY' or f'{patient}/plan_1' in cases
    every = [case for name in ('train', 'val', 'test') for cases in obj[name].values() for case in cases]
    assert len(every) == len(set(every)) == 11 + 2 * len(imaging)


def test_same_seed_is_byte_identical_and_two_seeds_differ_as_assignments(tmp_path):
    path = write_manifest(tmp_path, TEN)
    run(path, seed=SEED_A)
    first = (tmp_path / 'splits.json').read_bytes()
    run(path, seed=SEED_A)
    assert (tmp_path / 'splits.json').read_bytes() == first
    other = tmp_path / 'other'
    write_manifest(other, TEN)
    assert assignment(run(other / 'manifest.jsonl', seed=SEED_B)) != assignment(json.loads(first))


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
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.jsonl']
    monkeypatch.undo()
    run(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.jsonl', 'splits.json']


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
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.jsonl']  # no empty reservation blocks a rerun


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
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.jsonl', 'splits.json']


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
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.jsonl', 'splits.json']


def test_an_identical_split_is_not_refused_by_another_runs_empty_reservation(tmp_path):
    path = write_manifest(tmp_path, TEN)
    text = json.dumps(sp.split(write_manifest(tmp_path / 'other', TEN), test_size=0.2, val_size=0.2, seed=SEED_A), indent=1,
                      sort_keys=True) + '\n'
    (tmp_path / 'splits.json').write_text('')  # the other run's O_EXCL reservation, not yet renamed over
    def fill():  # atomic, as the real code fills the reservation
        temp = tmp_path / 'SYNTHETIC_fill.tmp'
        temp.write_text(text)
        os.replace(temp, tmp_path / 'splits.json')

    timer = threading.Timer(0.2, fill)
    timer.start()
    try:
        obj = sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A)
    finally:
        timer.cancel()
        timer.join()
    assert json.dumps(obj, indent=1, sort_keys=True) + '\n' == text


def test_a_reservation_released_during_the_wait_is_taken_over(tmp_path):
    path = write_manifest(tmp_path, TEN)
    (tmp_path / 'splits.json').write_text('')  # another run's reservation, released after its rename failed
    timer = threading.Timer(0.2, (tmp_path / 'splits.json').unlink)
    timer.start()
    try:
        obj = sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A)
    finally:
        timer.cancel()
        timer.join()
    assert (tmp_path / 'splits.json').read_text() == json.dumps(obj, indent=1, sort_keys=True) + '\n'


def test_an_empty_splits_json_that_is_never_filled_is_still_refused_after_about_a_second(tmp_path):
    path = write_manifest(tmp_path, TEN)
    (tmp_path / 'splits.json').write_text('')
    started = time.monotonic()
    with pytest.raises(ValueError, match=sp.REASON_EXISTS):
        sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A)
    assert time.monotonic() - started < 3
    assert (tmp_path / 'splits.json').read_text() == ''


def test_without_hard_links_the_write_falls_back_to_a_rename(tmp_path, monkeypatch):
    path = write_manifest(tmp_path, TEN)

    def unsupported(*args):
        raise PermissionError('SYNTHETIC_PATH')

    monkeypatch.setattr(os, 'link', unsupported)
    obj = run(path)
    assert sum(map(len, (obj[name] for name in ('train', 'val', 'test')))) == len(TEN)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.jsonl', 'splits.json']


def test_the_order_is_the_documented_hash_so_no_library_version_moves_it(tmp_path):
    obj = run(write_manifest(tmp_path, TEN), seed=-7)  # any int is a seed, negative included
    order = sorted(TEN, key=lambda patient: hashlib.sha256(f'-7\n{patient}'.encode()).hexdigest())
    assert (set(obj['test']), set(obj['val'])) == (set(order[:2]), set(order[2:4]))  # splits.json sorts its keys


@pytest.mark.parametrize('dropped', ['patient', 'case_dir'])
def test_a_manifest_without_patient_or_case_dir_is_refused(tmp_path, dropped):
    columns = [c for c in COLUMNS if c != dropped]
    bad = tmp_path / 'bad'
    bad.mkdir()
    (bad / 'manifest.jsonl').write_text(json.dumps({c: 'SYNTHETIC_X' for c in columns}) + '\n')
    with pytest.raises(ValueError, match=sp.REASON_COLUMNS):
        sp.split(bad / 'manifest.jsonl', test_size=0.2, val_size=0.2, seed=SEED_A)
    with pytest.raises(ValueError, match=sp.REASON_COLUMNS):
        sp.split(write_manifest(tmp_path, TEN), test_size=0.2, val_size=0.2, seed=SEED_A, external=bad / 'manifest.jsonl')
    assert not (bad / 'splits.json').exists() and not (tmp_path / 'splits.json').exists()


@pytest.mark.parametrize('module', [sp, man])
def test_split_and_manifest_import_no_pandas_sklearn_and_split_no_numpy(module):
    tree = ast.parse(inspect.getsource(module))
    imported = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    banned = {'pandas', 'sklearn'} | ({'numpy'} if module is sp else set())
    assert not imported & banned


@pytest.mark.parametrize('body', [
    '{"case_dir": "SYNTHETIC_P00/plan_1", "patient": null}\n',  # no patient value
    '{"case_dir": "SYNTHETIC_P00/plan_1", "patient": ""}\n',  # an empty patient
    '{"case_dir": "SYNTHETIC_P00/plan_1", "patient": "  "}\n',  # a whitespace-only patient
    '{"case_dir": "SYNTHETIC_P00/plan_1", "patient": "SYNTHETIC_\xe9"}\n',  # not UTF-8 once written as latin-1
])
def test_a_malformed_row_or_a_non_utf8_manifest_is_refused(tmp_path, capfd, body):
    path = tmp_path / 'manifest.jsonl'
    path.write_bytes(body.encode('latin-1'))
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


# ---------------------------------------------------------------- json-outputs task-02: manifest.jsonl is what split reads

def raw_manifest(folder, body: bytes):
    folder.mkdir(exist_ok=True)
    path = folder / 'manifest.jsonl'
    path.write_bytes(body)
    return path


_chain = operator.attrgetter('__cause__', '__context__')  # (cause, context) of an exception
GOOD = b'{"case_dir": "SYNTHETIC_P0/plan_1", "patient": "SYNTHETIC_P0"}\n'
ROWS_BREAKS = {
    'invalid json': b'{"case_dir": "SYNTHETIC_P0/plan_1", "patient"\n',
    'not utf-8': b'{"case_dir": "SYNTHETIC_P0/plan_1", "patient": "SYNTHETIC_\xe9"}\n',
    'array line': b'["SYNTHETIC_P0/plan_1", "SYNTHETIC_P0"]\n',
    'string line': b'"SYNTHETIC_P0"\n',
    'null line': b'null\n',
    'deep nesting': b'[' * 100000 + b']' * 100000 + b'\n',
    'duplicate key': b'{"case_dir": "SYNTHETIC_P0/plan_1", "patient": "SYNTHETIC_P0", "patient": "SYNTHETIC_P1"}\n',
    'number patient': b'{"case_dir": "SYNTHETIC_P0/plan_1", "patient": 123}\n',
    'null case_dir': b'{"case_dir": null, "patient": "SYNTHETIC_P0"}\n',
    'empty patient': b'{"case_dir": "SYNTHETIC_P0/plan_1", "patient": ""}\n',
    'whitespace case_dir': b'{"case_dir": "  ", "patient": "SYNTHETIC_P0"}\n',
    'bad line after a good one': GOOD + b'{oops\n',
}


@pytest.mark.parametrize('body', ROWS_BREAKS.values(), ids=ROWS_BREAKS)
def test_a_bad_manifest_line_is_refused_with_the_rows_reason_and_nothing_chained(tmp_path, capfd, body):
    path = raw_manifest(tmp_path, body)
    with pytest.raises(ValueError) as excinfo:
        sp.split(path, test_size=0.2, val_size=0.2, seed=SEED_A)
    assert str(excinfo.value) == sp.REASON_ROWS and _chain(excinfo.value) == (None, None)
    external = raw_manifest(tmp_path / 'ext', body)
    with pytest.raises(ValueError, match=sp.REASON_ROWS):
        sp.split(write_manifest(tmp_path / 'main', TEN), test_size=0.2, val_size=0.2, seed=SEED_A, external=external)
    assert not (tmp_path / 'splits.json').exists() and not (tmp_path / 'main' / 'splits.json').exists()
    out, err = capfd.readouterr()
    assert 'SYNTHETIC' not in out + err


@pytest.mark.parametrize('line', [b'{"case_dir": "SYNTHETIC_P0/plan_1"}\n', b'{"patient": "SYNTHETIC_P0"}\n', b'{}\n'])
def test_a_line_without_patient_or_case_dir_is_refused_with_the_columns_reason(tmp_path, line):
    with pytest.raises(ValueError) as excinfo:
        sp.split(raw_manifest(tmp_path, line), test_size=0.2, val_size=0.2, seed=SEED_A)
    assert str(excinfo.value) == sp.REASON_COLUMNS and _chain(excinfo.value) == (None, None)
    with pytest.raises(ValueError, match=sp.REASON_COLUMNS):
        sp.split(write_manifest(tmp_path / 'main', TEN), test_size=0.2, val_size=0.2, seed=SEED_A, external=tmp_path / 'manifest.jsonl')


def test_a_zero_byte_manifest_leaves_no_patient_for_train(tmp_path):
    with pytest.raises(ValueError, match=sp.REASON_NO_TRAIN):
        sp.split(raw_manifest(tmp_path, b''), test_size=0.0, val_size=0.0, seed=SEED_A)


def test_blank_lines_a_bom_and_crlf_are_read(tmp_path):
    path = raw_manifest(tmp_path, b'\xef\xbb\xbf' + GOOD.replace(b'\n', b'\r\n') + b'\n  \n' + GOOD.replace(b'P0', b'P1') + b'\n')
    obj = run(path, test_size=0.0, val_size=0.0)
    assert sorted(obj['train']) == ['SYNTHETIC_P0', 'SYNTHETIC_P1']


def test_a_manifest_reasons_do_not_say_column():
    assert 'column' not in sp.REASON_COLUMNS and 'column' not in sp.REASON_ROWS


def test_optional_manifest_keys_and_lines_that_differ_in_them_do_not_change_the_splits(tmp_path):
    plain = write_manifest(tmp_path / 'plain', TEN)
    rows = [json.loads(line) for line in plain.read_text(encoding='utf-8').splitlines()]
    optional = [{'patient_age': 63, 'patient_sex': 'F', 'series_date': '20240110'}, {}, {'manufacturer': {'distinct': ['SYNTHETIC_A', 'SYNTHETIC_B']}},
                {'study_date': '20240110', 'patient_weight': {'min': '60', 'max': '70'}}]
    rich = tmp_path / 'rich' / 'manifest.jsonl'
    rich.parent.mkdir()
    rich.write_text(''.join(json.dumps(row | optional[i % len(optional)]) + '\n' for i, row in enumerate(rows)), encoding='utf-8')
    assert run(rich) == run(plain)
