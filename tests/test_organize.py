"""organize: acceptance tests A1.1-A1.7 plus the guardrails. Every tree is synthetic and built in tmp_path."""
import csv
import hashlib
import logging
import os
import re
import subprocess
import sys
import warnings
from collections import Counter

import numpy as np
import pydicom
import pytest
import synthetic_dicom as sd

from rt_dataprep import dicom_io
from rt_dataprep import organize as org
from rt_dataprep.organize import organize

PID = 'SYNTHETIC_P1'
EARLY, LATE = '20240105', '20240301'
# Has the preamble but pydicom cannot parse it: a sequence of undefined length whose item ends mid-element.
TRUNCATED = (bytes(128) + b'DICM' + b'\x08\x11\x40\x11SQ\x00\x00\xff\xff\xff\xff' + b'\xfe\xff\x00\xe0\xff\xff\xff\xff'
             + b'\x08\x00\x16\x00UI\x40\x00ab')

def listing(root):
    """Sorted names under root, without the dated per-run copies of the report."""
    return sorted(p.name for p in root.iterdir() if not re.fullmatch(r'\w+_report_\d{8}T\d{6}Z\.csv', p.name))



def planning_study(folder, pid=PID, date=EARLY, slices=3, model=None, tag=''):
    """CT + RTSTRUCT + RTPLAN + RTDOSE sharing one study. `tag` keeps file names apart inside one folder."""
    folder.mkdir(parents=True, exist_ok=True)
    ct = sd.write_ct_series(folder, np.zeros((slices, 4, 4)), patient_id=pid, series_date=date, series_time='100000',
                            manufacturer_model_name=model, prefix=f'ct{tag}')
    study = dict(patient_id=pid, study_uid=ct['study_uid'])
    struct = sd.write_rtstruct(folder / f'struct{tag}.dcm', ct, **study)
    plan = sd.write_rtplan(folder / f'plan{tag}.dcm', struct, np.zeros((2, 3)), plan_date=date, plan_time='110000', **study)
    dose = sd.write_rtdose(folder / f'dose{tag}.dcm', np.ones((1, 2, 2)), plan_sop_uid=plan, **study)
    return dict(ct=ct, struct=struct, plan=plan, dose=dose, study=ct['study_uid'])


def mk(folder):
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def header(path):
    return pydicom.dcmread(path, stop_before_pixels=True)


def modalities(folder):
    return Counter(header(p).Modality for p in folder.iterdir())


def ct_series_uids(folder):
    return {header(p).SeriesInstanceUID for p in folder.iterdir() if header(p).Modality == 'CT'}


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}


def rows(out):
    with open(out / 'organize_report.csv', newline='') as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def tree(tmp_path):
    src, out = tmp_path / 'src', tmp_path / 'out'
    src.mkdir()
    return src, out


def test_a1_1_two_planning_studies_numbered_by_date(tree):
    src, out = tree
    late = planning_study(src / PID / 'a', date=LATE)  # name order disagrees with date order: dates must decide
    early = planning_study(src / PID / 'z', date=EARLY)
    report = organize(src, out)
    assert modalities(out / PID / 'plan_1') == Counter(CT=3, RTSTRUCT=1, RTPLAN=1, RTDOSE=1)
    assert modalities(out / PID / 'plan_2') == Counter(CT=3, RTSTRUCT=1, RTPLAN=1, RTDOSE=1)
    assert ct_series_uids(out / PID / 'plan_1') == {early['ct']['series_uid']}
    assert ct_series_uids(out / PID / 'plan_2') == {late['ct']['series_uid']}
    assert not (out / PID / 'misc_files').exists()
    assert [(r.plans, r.misc, r.unreadable, r.outcome) for r in report.rows] == [(2, 0, 0, org.OUTCOME_ORGANIZED)]


def test_a1_2_reg_referenced_diagnostic_study_is_copied(tree):
    src, out = tree
    study = planning_study(src / PID / 'plan')
    mr = sd.write_ct_series(mk(src / PID / 'mr'), np.zeros((2, 4, 4)), modality='MR', patient_id=PID)
    sd.write_reg(src / PID / 'plan' / 'reg.dcm', study['study'], mr['study_uid'], patient_id=PID)
    organize(src, out)
    assert modalities(out / PID / 'plan_1') == Counter(CT=3, RTSTRUCT=1, RTPLAN=1, RTDOSE=1, REG=1, MR=2)
    assert not (out / PID / 'misc_files').exists()


def test_a1_2_same_file_names_in_one_plan_are_all_kept(tree):
    src, out = tree
    study = planning_study(src / PID / 'plan')
    mr = sd.write_ct_series(mk(src / PID / 'mr'), np.zeros((3, 4, 4)), modality='MR', patient_id=PID)  # prefix ct, like the CT
    sd.write_reg(src / PID / 'plan' / 'reg.dcm', study['study'], mr['study_uid'], patient_id=PID)
    organize(src, out)
    assert modalities(out / PID / 'plan_1')['MR'] == 3


def test_a1_3_reg_to_other_planning_study_copies_nothing(tree):
    src, out = tree
    a = planning_study(src / PID / 'a', date=EARLY)
    b = planning_study(src / PID / 'b', date=LATE)
    sd.write_reg(src / PID / 'a' / 'reg.dcm', a['study'], b['study'], patient_id=PID)
    organize(src, out)
    assert modalities(out / PID / 'plan_1') == Counter(CT=3, RTSTRUCT=1, RTPLAN=1, RTDOSE=1, REG=1)
    assert modalities(out / PID / 'plan_2') == Counter(CT=3, RTSTRUCT=1, RTPLAN=1, RTDOSE=1)
    assert ct_series_uids(out / PID / 'plan_1') == {a['ct']['series_uid']}
    assert ct_series_uids(out / PID / 'plan_2') == {b['ct']['series_uid']}


def test_a1_4_localizer_excluded_model_and_orphan_go_to_misc(tree):
    src, out = tree
    study = planning_study(src / PID / 'plan')
    sd.write_ct_series(src / PID / 'plan', np.zeros((1, 4, 4)), patient_id=PID, study_uid=study['study'], prefix='loc',
                       image_type=['ORIGINAL', 'PRIMARY', 'LOCALIZER'])  # a localizer inside the planning study
    planning_study(src / PID / 'acuity', model='Acuity Sim', date=LATE)  # 3 CT + struct + plan + dose
    sd.write_ct_series(mk(src / PID / 'orphan'), np.zeros((2, 4, 4)), patient_id=PID)
    report = organize(src, out)
    assert modalities(out / PID / 'plan_1') == Counter(CT=3, RTSTRUCT=1, RTPLAN=1, RTDOSE=1)
    assert not (out / PID / 'plan_2').exists()
    assert modalities(out / PID / 'misc_files') == Counter(CT=1 + 3 + 2, RTSTRUCT=1, RTPLAN=1, RTDOSE=1)
    assert report.rows[0].misc == 9


def test_a1_5_two_plans_on_one_ct_and_struct(tree):
    src, out = tree
    first = planning_study(src / PID / 'x', tag='')
    study = dict(patient_id=PID, study_uid=first['study'])
    plan2 = sd.write_rtplan(src / PID / 'x' / 'plan2.dcm', first['struct'], np.zeros((2, 3)), plan_date=EARLY,
                            plan_time='090000', **study)  # earlier than plan.dcm's 110000, against name order
    dose2 = sd.write_rtdose(src / PID / 'x' / 'dose2.dcm', np.ones((1, 2, 2)), plan_sop_uid=plan2, **study)
    organize(src, out)
    for name, plan, dose in (('plan_1', plan2, dose2), ('plan_2', first['plan'], first['dose'])):  # plan time orders them
        folder = out / PID / name
        assert modalities(folder) == Counter(CT=3, RTSTRUCT=1, RTPLAN=1, RTDOSE=1)
        assert {header(p).SOPInstanceUID for p in folder.iterdir() if header(p).Modality in ('RTPLAN', 'RTDOSE')} == {plan, dose}


def test_a1_5b_two_planning_cts_in_one_study(tree):
    src, out = tree
    folder = src / PID / 'x'
    folder.mkdir(parents=True)
    study_uid = pydicom.uid.generate_uid()
    plans = []
    for tag, time in (('a', '090000'), ('b', '100000')):
        ct = sd.write_ct_series(folder, np.zeros((2, 4, 4)), patient_id=PID, study_uid=study_uid, series_date=EARLY,
                                series_time=time, prefix=f'ct{tag}')
        study = dict(patient_id=PID, study_uid=study_uid)
        struct = sd.write_rtstruct(folder / f'struct{tag}.dcm', ct, **study)
        plans.append(sd.write_rtplan(folder / f'plan{tag}.dcm', struct, np.zeros((2, 3)), **study))
    report = organize(src, out)
    assert report.rows[0].plans == 2
    plan_folders = sorted((out / PID).glob('plan_*'))
    assert [f.name for f in plan_folders] == ['plan_1', 'plan_2']
    for folder, plan in zip(plan_folders, plans, strict=True):
        assert any(folder.iterdir())
        assert [header(p).SOPInstanceUID for p in folder.iterdir() if header(p).Modality == 'RTPLAN'] == [plan]


def test_a1_6_hostile_patient_id_stays_inside_out_root(tree, tmp_path):
    src, out = tree
    planning_study(src / 'p', pid='../evil')
    organize(src, out)
    created = out / 'evil'
    assert created.resolve().is_relative_to(out.resolve())
    assert (created / 'plan_1').is_dir()
    assert not (tmp_path / 'evil').exists()


def test_a1_7_source_tree_is_byte_identical(tree):
    src, out = tree
    planning_study(src / PID / 'a', date=EARLY)
    planning_study(src / PID / 'b', date=LATE)
    sd.write_ct_series(mk(src / PID / 'orphan'), np.zeros((2, 4, 4)), patient_id=PID)
    (src / PID / 'bad.dcm').write_bytes(b'not dicom')
    before = snapshot(src)
    organize(src, out)
    assert snapshot(src) == before


def test_incomplete_chain_plan_is_left_over(tree):
    src, out = tree
    study = planning_study(src / PID / 'a')
    sd.write_rtplan(src / PID / 'a' / 'stray.dcm', pydicom.uid.generate_uid(), np.zeros((2, 3)), patient_id=PID,
                    study_uid=study['study'])  # names an RTSTRUCT that does not exist
    organize(src, out)
    assert modalities(out / PID / 'plan_1') == Counter(CT=3, RTSTRUCT=1, RTPLAN=1, RTDOSE=1)
    assert modalities(out / PID / 'misc_files') == Counter(RTPLAN=1)


def test_existing_output_folder_is_skipped_and_counted(tree):
    src, out = tree
    planning_study(src / 'p')
    (out / PID).mkdir(parents=True)
    (out / PID / 'keep.txt').write_text('earlier run')
    report = organize(src, out)
    assert [p.name for p in (out / PID).iterdir()] == ['keep.txt']
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_SKIPPED, org.REASON_EXISTS)]
    assert rows(out) == [dict(patient=PID, plan_count='0', misc_count='0', unreadable_count='0',
                              outcome=org.OUTCOME_SKIPPED, reason=org.REASON_EXISTS)]


def test_an_existing_output_folder_is_skipped_before_grouping(tree, monkeypatch):
    """#180: a skipped patient is never grouped, so a grouping error cannot turn a skip into an abort."""
    src, out = tree
    planning_study(src / 'p')
    (out / PID).mkdir(parents=True)
    monkeypatch.setattr(org, '_group', lambda *a: 1 / 0)
    report = organize(src, out)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_SKIPPED, org.REASON_EXISTS)]


def test_colliding_sanitized_names_refuse_the_later_patient(tree, caplog):
    src, out = tree
    planning_study(src / 'p1', pid='SYNTHETIC_a b', date=EARLY)
    planning_study(src / 'p2', pid='SYNTHETIC_a_b', date=LATE)  # a different PatientID, the same folder name
    caplog.set_level(logging.DEBUG)
    report = organize(src, out)
    assert [r.outcome for r in report.rows] == [org.OUTCOME_ORGANIZED, org.OUTCOME_REFUSED]
    assert report.rows[1].reason == org.REASON_NAME_CLAIMED
    assert (report.rows[1].plans, report.rows[1].misc) == (0, 0)
    assert ct_series_uids(out / 'SYNTHETIC_a_b' / 'plan_1')  # the first patient's tree, with one plan only
    assert [p.name for p in (out / 'SYNTHETIC_a_b').glob('plan_*')] == ['plan_1']
    assert [r['outcome'] for r in rows(out)] == [org.OUTCOME_ORGANIZED, org.OUTCOME_REFUSED]
    text = ' '.join(r.getMessage() for r in caplog.records)
    for secret in ('SYNTHETIC', str(out.parent), EARLY, LATE):
        assert secret not in text


def test_refusal_exceptions_and_logs_carry_no_identifiers(tree, caplog):
    src, out = tree
    planning_study(src / 'p', pid='SYNTHETIC_a')
    (src / 'p' / 'x.dcm').write_bytes(TRUNCATED)
    mixed = src / 'q'
    planning_study(mixed / '1', pid='SYNTHETIC_m1')
    planning_study(mixed / '2', pid='SYNTHETIC_m2')
    caplog.set_level(logging.DEBUG)
    report = organize(src, out)
    assert [r.reason for r in report.rows] == ['', org.REASON_MIXED_IDS]
    assert report.rows[0].unreadable == 1
    assert not (out / 'SYNTHETIC_m1').exists()
    text = ' '.join(r.getMessage() for r in caplog.records)
    assert text  # the run did log
    for secret in ('SYNTHETIC', str(out.parent), EARLY):
        assert secret not in text


def test_folder_without_readable_dicom_is_refused_not_fatal(tree):
    src, out = tree
    (src / 'empty').mkdir()
    (src / 'empty' / 'bad.dcm').write_bytes(TRUNCATED)
    report = organize(src, out)
    assert [(r.outcome, r.reason, r.unreadable) for r in report.rows] == [(org.OUTCOME_REFUSED, org.REASON_NO_DICOM, 1)]
    assert listing(out) == ['organize_report.csv']


def test_cli_runs_organize_and_exits_zero(tree):
    src, out = tree
    planning_study(src / 'p')
    done = subprocess.run([sys.executable, '-m', 'rt_dataprep', 'organize', str(src), str(out)], capture_output=True, text=True)
    assert done.returncode == 0
    assert (out / PID / 'plan_1').is_dir()
    assert PID not in done.stdout + done.stderr


def test_copy_failure_keeps_nothing_leaks_nothing_and_reruns(tree, monkeypatch, caplog):
    src, out = tree
    planning_study(src / PID / 'p')
    real_copy, calls = org.shutil.copy2, []

    def failing_copy(source, target):
        calls.append(target)
        if len(calls) == 3:
            raise OSError(f'disk full writing {target}')  # a real OSError names the path, i.e. the patient
        return real_copy(source, target)

    monkeypatch.setattr(org.shutil, 'copy2', failing_copy)
    with caplog.at_level(logging.DEBUG):
        report = organize(src, out)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_REFUSED, org.REASON_IO_ERROR)]
    assert listing(out) == ['organize_report.csv']  # no target, no partial left behind
    assert PID not in caplog.text and str(out.parent) not in caplog.text
    monkeypatch.setattr(org.shutil, 'copy2', real_copy)
    assert organize(src, out).rows[0].outcome == org.OUTCOME_ORGANIZED  # the failed patient is redone, not skipped


def test_cli_file_system_error_exits_1_without_paths(tmp_path):
    src = tmp_path / 'SYNTHETIC_SRC'
    planning_study(src / PID / 'p')
    out = tmp_path / 'SYNTHETIC_OUT'
    out.write_text('not a directory')
    result = subprocess.run([sys.executable, '-m', 'rt_dataprep', 'organize', str(src), str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert 'SYNTHETIC_' not in result.stdout + result.stderr and str(tmp_path) not in result.stderr


def test_a1_8_rtstruct_in_another_study_goes_to_the_plan_folder(tree):
    src, out = tree
    folder = mk(src / PID / 'p')
    ct_time = '100000'
    ct = sd.write_ct_series(folder, np.zeros((3, 4, 4)), patient_id=PID, series_date=EARLY, series_time=ct_time)
    other_study = pydicom.uid.generate_uid()  # the RTSTRUCT was exported in a study of its own, not the CT's
    struct = sd.write_rtstruct(folder / 'struct.dcm', ct, patient_id=PID, study_uid=other_study)
    plan = sd.write_rtplan(folder / 'plan.dcm', struct, np.zeros((2, 3)), plan_date=EARLY, plan_time='110000',
                           patient_id=PID, study_uid=ct['study_uid'])
    sd.write_rtdose(folder / 'dose.dcm', np.ones((1, 2, 2)), plan_sop_uid=plan, patient_id=PID, study_uid=ct['study_uid'])
    report = organize(src, out)
    assert [(r.plans, r.misc) for r in report.rows] == [(1, 0)]
    assert not (out / PID / 'misc_files').exists()
    assert modalities(out / PID / 'plan_1') == Counter(CT=3, RTSTRUCT=1, RTPLAN=1, RTDOSE=1)


def test_a_file_without_sop_instance_uid_refuses_the_patient_not_the_run(tree):
    src, out = tree
    planning_study(src / PID / 'p')
    path = next((src / PID / 'p').glob('ct*.dcm'))
    ds = pydicom.dcmread(path)
    del ds.SOPInstanceUID
    ds.save_as(path)
    organize(src, out)
    assert (rows(out)[0]['outcome'], rows(out)[0]['reason']) == ('refused', org.REASON_NO_SOP_UID)
    assert not (out / PID).exists()


def test_one_rtplan_in_two_sub_folders_gives_one_plan_folder(tree):
    src, out = tree
    planning_study(src / PID / 'a')
    (src / PID / 'b').mkdir()
    for name in ('plan.dcm', 'dose.dcm'):  # the plan and its dose, exported twice
        (src / PID / 'b' / name).write_bytes((src / PID / 'a' / name).read_bytes())
        copy = pydicom.dcmread(src / PID / 'b' / name)
        copy.SeriesDescription = 'SYNTHETIC_COPY'  # the one byte that tells the second path's copy from the first's
        copy.save_as(src / PID / 'b' / name)
    organize(src, out)
    assert sorted(p.name for p in (out / PID).iterdir()) == ['misc_files', 'plan_1']
    assert modalities(out / PID / 'plan_1') == Counter(CT=3, RTSTRUCT=1, RTPLAN=1, RTDOSE=1)  # one dose, not two to sum
    assert modalities(out / PID / 'misc_files') == Counter(RTPLAN=1, RTDOSE=1)
    for folder, description in (('plan_1', None), ('misc_files', 'SYNTHETIC_COPY')):  # the first path's copy is the plan's
        assert {header(p).get('SeriesDescription') for p in (out / PID / folder).iterdir()
                if header(p).Modality in ('RTPLAN', 'RTDOSE')} == {description}


def test_an_out_root_inside_the_source_root_is_refused_before_any_write(tree):
    src, out = tree
    planning_study(src / PID / 'p')
    before = snapshot(src)
    with pytest.raises(ValueError, match=org.REASON_OUTSIDE_SRC):
        organize(src, src / 'out')
    assert snapshot(src) == before and not (src / 'out').exists()


def blank_id(path):
    ds = pydicom.dcmread(path)
    ds.PatientID = ''
    ds.save_as(path)


def test_a_file_with_a_blank_patient_id_goes_to_misc_and_the_patient_is_organized(tree):
    src, out = tree
    planning_study(src / PID / 'p')
    planning_study(src / PID / 'q', tag='b')  # a second plan whose RTPLAN loses its PatientID
    blank_id(src / PID / 'q' / 'planb.dcm')
    organize(src, out)
    assert (rows(out)[0]['outcome'], rows(out)[0]['plan_count']) == ('organized', '1')
    assert modalities(out / PID / 'misc_files')['RTPLAN'] == 1


def test_a_folder_of_blank_patient_ids_only_is_refused(tree):
    src, out = tree
    planning_study(src / PID / 'p')
    for path in (src / PID / 'p').iterdir():
        blank_id(path)
    organize(src, out)
    assert (rows(out)[0]['outcome'], rows(out)[0]['reason']) == ('refused', org.REASON_NO_ID)


def test_each_run_keeps_its_own_report_and_the_fixed_name_is_the_latest(tree, monkeypatch):
    src, out = tree
    planning_study(src / PID / 'p')
    stamps = iter(['20260101T000000Z', '20260101T000001Z'])
    monkeypatch.setattr('rt_dataprep.dicom_io.time.strftime', lambda fmt, t=None: next(stamps))
    organize(src, out)
    organize(src, out)  # the rerun skips the patient
    first = (out / 'organize_report_20260101T000000Z.csv').read_text(encoding='utf-8')
    second = (out / 'organize_report_20260101T000001Z.csv').read_text(encoding='utf-8')
    assert ',organized,' in first and ',skipped,' in second
    assert (out / 'organize_report.csv').read_text(encoding='utf-8') == second


@pytest.mark.parametrize('where', ['same', 'link_into_src'])
def test_out_root_equal_to_the_source_or_a_symlink_into_it_is_refused_before_any_write(tree, tmp_path, where):
    src, _ = tree
    planning_study(src / 'p', pid='SYNTHETICPTB')
    (src / 'inside').mkdir()
    link = tmp_path / 'link'
    link.symlink_to(src / 'inside', target_is_directory=True)
    before = snapshot(src)
    with pytest.raises(ValueError, match=org.REASON_OUTSIDE_SRC):
        organize(src, src if where == 'same' else link)
    assert snapshot(src) == before and sorted(p.name for p in src.iterdir()) == ['inside', 'p']


def test_files_are_found_by_preamble_not_extension(tree, tmp_path):
    plain, mixed = tmp_path / 'plain', tmp_path / 'mixed'
    planning_study(plain / 'src' / PID / 'a')
    planning_study(mixed / 'src' / PID / 'a')
    for i, path in enumerate(sorted((mixed / 'src' / PID / 'a').iterdir())):
        if i % 3 == 0:
            path.rename(path.with_suffix(''))
        elif i % 3 == 1:
            path.rename(path.with_suffix('.DCM'))
    (mixed / 'src' / PID / 'a' / 'notes.txt').write_text('free text, no preamble')
    (mixed / 'src' / PID / 'a' / 'tiny').write_bytes(b'0123456789')
    reports = [organize(root / 'src', root / 'out') for root in (plain, mixed)]
    assert [(r.plans, r.misc, r.unreadable) for r in reports[1].rows] == [(r.plans, r.misc, r.unreadable) for r in reports[0].rows]
    assert modalities(mixed / 'out' / PID / 'plan_1') == modalities(plain / 'out' / PID / 'plan_1')
    assert not (mixed / 'out' / PID / 'misc_files').exists()
    assert not list((mixed / 'out' / PID).rglob('notes.txt')) and not list((mixed / 'out' / PID).rglob('tiny'))


def redate_ct(study, **dates):
    """Set (or, for None, delete) date elements on every slice of a planning_study's CT."""
    for path in study['ct']['paths']:
        ds = pydicom.dcmread(path)
        for name, value in dates.items():
            if value is None:
                del ds[name]
            else:
                setattr(ds, name, value)
        ds.save_as(path)


def plan_ct_series(out):
    return [ct_series_uids(out / PID / f'plan_{n}') for n in (1, 2)]


def test_ct_without_series_date_sorts_by_study_date(tree):
    src, out = tree
    dated = planning_study(src / PID / 'a', date=EARLY)
    fallback = planning_study(src / PID / 'b', date=EARLY)  # same plan date: only the CT date can order them
    redate_ct(fallback, SeriesDate=None, StudyDate=LATE)
    organize(src, out)
    assert plan_ct_series(out) == [{dated['ct']['series_uid']}, {fallback['ct']['series_uid']}]


def test_ct_without_any_date_is_the_last_plan(tree):
    src, out = tree
    undated = planning_study(src / PID / 'a', date=EARLY)  # the earliest plan date and first by name: still last
    dated = planning_study(src / PID / 'b', date=LATE)
    redate_ct(undated, SeriesDate=None)
    organize(src, out)
    assert plan_ct_series(out) == [{dated['ct']['series_uid']}, {undated['ct']['series_uid']}]


@pytest.mark.parametrize('workers', [1, 4])
def test_unreadable_dicom_is_copied_to_misc_files(tree, workers):
    src, out = tree
    planning_study(src / PID / 'a')
    (src / PID / 'a' / 'cut.dcm').write_bytes(TRUNCATED)
    report = organize(src, out, workers=workers)
    assert (out / PID / 'misc_files' / 'cut.dcm').read_bytes() == TRUNCATED
    assert [f.name for f in (out / PID / 'misc_files').iterdir()] == ['cut.dcm']
    assert [(r.plans, r.misc, r.unreadable) for r in report.rows] == [(1, 1, 1)]
    assert rows(out)[0]['misc_count'] == rows(out)[0]['unreadable_count'] == '1'


def test_a_series_split_between_blank_and_real_patient_ids_refuses_the_patient(tree):
    src, out = tree
    planning_study(src / PID / 'p')
    blank_id(next((src / PID / 'p').glob('ct*.dcm')))  # one planning CT slice loses its id
    organize(src, out)
    assert (rows(out)[0]['outcome'], rows(out)[0]['reason']) == ('refused', org.REASON_SPLIT_SERIES)
    assert not (out / PID).exists()


def test_a_blank_id_duplicate_of_a_present_slice_does_not_refuse_the_patient(tree):
    src, out = tree
    planning_study(src / PID / 'p')
    copy = mk(src / PID / 'dup') / 'ct_copy.dcm'
    copy.write_bytes(next((src / PID / 'p').glob('ct*.dcm')).read_bytes())
    blank_id(copy)  # the same SOPInstanceUID: takes no slice away from the series
    organize(src, out)
    assert rows(out)[0]['outcome'] == 'organized'


def test_a_blank_id_file_whose_sop_collides_with_another_series_still_refuses(tree):
    src, out = tree
    first = planning_study(src / PID / 'p')
    other = planning_study(src / PID / 'q', tag='x')
    ds = pydicom.dcmread(first['ct']['paths'][0])  # its SOP is present, but in the first CT series
    ds.PatientID, ds.SeriesInstanceUID = '', other['ct']['series_uid']
    ds.save_as(mk(src / PID / 'blank') / 'collide.dcm')
    organize(src, out)
    assert (rows(out)[0]['outcome'], rows(out)[0]['reason']) == ('refused', org.REASON_SPLIT_SERIES)


def test_empty_sop_instance_uids_refuse_the_patient(tree):
    src, out = tree
    planning_study(src / PID / 'p')
    for path in (src / PID / 'p').glob('ct*.dcm'):
        ds = pydicom.dcmread(path)
        ds.SOPInstanceUID = ''
        ds.save_as(path)
    organize(src, out)
    assert (rows(out)[0]['outcome'], rows(out)[0]['reason']) == ('refused', org.REASON_NO_SOP_UID)


def test_ctrl_c_still_writes_the_report_of_what_was_done(tree, monkeypatch):
    src, out = tree
    planning_study(src / PID / 'a')
    planning_study(src / 'SYNTHETIC_SECOND' / 'a', pid='SYNTHETIC_SECOND')
    real = getattr(dicom_io, 'read_header_core')  # noqa: B009

    def interrupt(path):
        if 'SYNTHETIC_SECOND' in str(path):
            raise KeyboardInterrupt
        return real(path)

    monkeypatch.setattr(dicom_io, 'read_header_core', interrupt)
    with pytest.raises(KeyboardInterrupt):
        organize(src, out)
    with open(out / org.REPORT_NAME, newline='') as handle:
        assert [row['outcome'] for row in csv.DictReader(handle)] == [org.OUTCOME_ORGANIZED]
    assert not [p for p in out.iterdir() if p.name.endswith('.partial')]


def test_ctrl_c_during_the_copy_leaves_no_partial_folder(tree, monkeypatch):
    src, out = tree
    planning_study(src / PID / 'a')
    planning_study(src / 'SYNTHETIC_SECOND' / 'a', pid='SYNTHETIC_SECOND')
    real, copied = org.shutil.copy2, []

    def interrupt(source, target):
        if 'SYNTHETIC_SECOND' in str(source):
            copied.append(source)
            if len(copied) == 2:  # part-way through the second patient's copy
                raise KeyboardInterrupt
        return real(source, target)

    monkeypatch.setattr(org.shutil, 'copy2', interrupt)
    with pytest.raises(KeyboardInterrupt):
        organize(src, out)
    assert not [p for p in out.iterdir() if p.name.endswith('.partial')]
    assert (out / PID).is_dir() and not (out / 'SYNTHETIC_SECOND').exists()


def test_workers_1_and_4_give_identical_rows_and_copied_trees(tree, tmp_path):
    src, out = tree
    planning_study(src / PID / 'a', slices=6)
    (src / PID / 'a' / 'cut.dcm').write_bytes(TRUNCATED)
    (src / PID / 'a' / 'notes.txt').write_text('not dicom')
    planning_study(src / 'SYNTHETIC_SECOND' / 'a', pid='SYNTHETIC_SECOND')
    reports = [organize(src, tmp_path / f'out{n}', workers=n) for n in (1, 4)]
    assert reports[0].rows == reports[1].rows and reports[0].rows[0].unreadable == 1
    assert snapshot(tmp_path / 'out1' / PID) == snapshot(tmp_path / 'out4' / PID)
    assert snapshot(tmp_path / 'out1' / 'SYNTHETIC_SECOND') == snapshot(tmp_path / 'out4' / 'SYNTHETIC_SECOND')
    assert [r.outcome for r in reports[1].rows] == [org.OUTCOME_ORGANIZED] * 2


def test_a_preamble_oserror_refuses_that_patient_only(tree, monkeypatch):
    src, out = tree
    planning_study(src / PID / 'a')
    planning_study(src / 'SYNTHETIC_SECOND' / 'a', pid='SYNTHETIC_SECOND')
    real = getattr(org, '_has_preamble')  # noqa: B009

    def flaky(path):
        if 'SYNTHETIC_SECOND' in str(path) and path.name.startswith('plan'):
            raise OSError(f'permission denied {path}')
        return real(path)

    monkeypatch.setattr(org, '_has_preamble', flaky)
    report = organize(src, out)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_ORGANIZED, ''), (org.OUTCOME_REFUSED, org.REASON_IO_ERROR)]
    assert (out / PID / 'plan_1').is_dir() and not (out / 'SYNTHETIC_SECOND').exists()
    assert not [p for p in out.iterdir() if p.name.endswith('.partial')]


def test_a_patient_header_read_that_makes_pydicom_warn_prints_nothing_and_restores_state(tree, capfd, caplog):
    src, out = tree
    planning_study(src / PID / 'a')
    sd.write_rtdose(src / PID / 'a' / 'odd.dcm', [[[1]]], patient_id=PID, invalid_values=True)
    odd = src / PID / 'a' / 'odd.dcm'
    state = lambda: (pydicom.config.settings.reading_validation_mode, logging.getLogger('pydicom').disabled)  # noqa: E731
    with warnings.catch_warnings(record=True) as control, caplog.at_level(logging.DEBUG):  # control: it warns unquieted
        warnings.simplefilter('always')
        for element in pydicom.dcmread(odd, stop_before_pixels=True).iterall():
            _ = element.value
    assert control or caplog.records
    caplog.clear()
    capfd.readouterr()
    before, filters = state(), list(warnings.filters)
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.DEBUG):
        warnings.simplefilter('always')
        report = organize(src, out)
    captured = capfd.readouterr()
    assert [r.outcome for r in report.rows] == [org.OUTCOME_ORGANIZED]
    assert caught == [] and [r for r in caplog.records if r.name.startswith('pydicom')] == []
    assert (captured.out, captured.err) == ('', '')
    assert state() == before and list(warnings.filters) == filters


TIMING = re.compile(r'organize: (\d+\.\d) s total, (\d+\.\d) s reading \(summed over patients\), (\d+\.\d) s copying \(summed over patients\), (\d+) files, workers (\d+)')


def test_one_timing_line_per_run_with_only_numbers_and_fixed_words(tree, caplog, monkeypatch):
    src, out = tree
    planning_study(src / PID / 'a')  # 3 CT + struct + plan + dose = 6 files
    (src / PID / 'notes.txt').write_text('SYNTHETIC_ not dicom')  # walked, not in any plan: 7 files
    planning_study(src / 'SYNTHETIC_SECOND' / 'a', pid='SYNTHETIC_SECOND')
    real = getattr(org, '_has_preamble')  # noqa: B009

    def flaky(path):
        if 'SYNTHETIC_SECOND' in str(path):
            raise OSError('x')
        return real(path)

    monkeypatch.setattr(org, '_has_preamble', flaky)
    with caplog.at_level(logging.INFO):
        organize(src, out)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith('organize:') and ' s total' in r.getMessage()]
    assert len(lines) == 1 and TIMING.fullmatch(lines[0])
    assert TIMING.fullmatch(lines[0]).group(4, 5) == ('7', '1')  # every walked file counts; the refused patient adds none
    assert 'SYNTHETIC' not in lines[0] and '/' not in lines[0]


def test_same_day_fallback_dates_are_ordered_by_their_own_time(tree):
    src, out = tree
    late = planning_study(src / PID / 'a', date=EARLY)  # first by name and by SeriesTime, later by StudyTime
    early = planning_study(src / PID / 'b', date=EARLY)
    redate_ct(late, SeriesDate=None, StudyDate=LATE, SeriesTime='080000', StudyTime='110000')
    redate_ct(early, SeriesDate=None, StudyDate=LATE, SeriesTime='090000', StudyTime='100000')
    organize(src, out)
    assert plan_ct_series(out) == [{early['ct']['series_uid']}, {late['ct']['series_uid']}]


@pytest.mark.parametrize('workers', [1, 4])
def test_a_folder_of_only_non_dicom_files_is_no_dicom_with_nothing_unreadable(tree, workers):
    src, out = tree
    mk(src / PID)
    (src / PID / 'notes.txt').write_text('not dicom')
    row = organize(src, out, workers=workers).rows[0]
    assert (row.outcome, row.reason, row.unreadable) == (org.OUTCOME_REFUSED, org.REASON_NO_DICOM, 0)


def test_hidden_folders_are_not_walked(tree):
    src, out = tree
    planning_study(src / PID / 'a')
    planning_study(src / '.SYNTHETIC_OLD.partial' / 'a', pid='SYNTHETIC_OLD')  # an old output root's leftover
    planning_study(src / PID / '.cache', pid='SYNTHETIC_OTHER')  # would be REASON_MIXED_IDS if walked
    rows = organize(src, out).rows
    assert [(r.patient, r.outcome) for r in rows] == [(PID, org.OUTCOME_ORGANIZED)]


@pytest.mark.skipif(os.geteuid() == 0, reason='root lists any folder')
def test_an_unlistable_subfolder_refuses_the_patient(tree):
    src, out = tree
    planning_study(src / PID / 'a')
    locked = src / PID / 'locked'
    planning_study(locked, tag='x')
    locked.chmod(0)
    try:
        row = organize(src, out).rows[0]
    finally:
        locked.chmod(0o755)
    assert (row.outcome, row.reason) == (org.OUTCOME_REFUSED, org.REASON_IO_ERROR)
    assert not (out / PID).exists()


# ---------------------------------------------------------------- hardening B8 (#98): workers, timing line, Ctrl-C

def summary_and_timing(caplog):
    texts = [r.getMessage() for r in caplog.records if r.name == 'rt_dataprep.organize']
    return texts, next(i for i, t in enumerate(texts) if 'patient folders' in t)


def test_the_timing_line_comes_right_after_the_summary_line(tree, caplog):
    src, out = tree
    planning_study(src / PID / 'a')
    caplog.set_level(logging.INFO)
    organize(src, out)
    texts, at = summary_and_timing(caplog)
    assert TIMING.fullmatch(texts[at + 1]) and len(texts) == at + 2


def test_no_timing_line_on_ctrl_c(tree, monkeypatch, caplog):
    src, out = tree
    planning_study(src / PID / 'a')
    real = getattr(dicom_io, 'read_header_core')  # noqa: B009

    def interrupt(path):
        if 'a' == path.parent.name and path.name.startswith('plan'):
            raise KeyboardInterrupt
        return real(path)

    monkeypatch.setattr(dicom_io, 'read_header_core', interrupt)
    caplog.set_level(logging.INFO)
    with pytest.raises(KeyboardInterrupt):
        organize(src, out)
    assert not [r for r in caplog.records if ' s total' in r.getMessage()]



# ---------------------------------------------------------------- --overwrite (spec 2026-11-05-overwrite-flag)

SUMMARY = re.compile(r'organize: \d+ patient folders, .*; (\d+) replaced, (\d+) kept')


def hidden(root):
    return sorted(p.name for p in root.iterdir() if p.name.startswith('.'))


def counts(caplog):
    """(replaced, kept) from the stage's summary line."""
    line = next(r.getMessage() for r in caplog.records if r.name == 'rt_dataprep.organize' and 'patient folders' in r.getMessage())
    return tuple(int(n) for n in SUMMARY.fullmatch(line).groups())


@pytest.fixture
def stale(tree, tmp_path):
    """(src, out, fresh): `out` holds a first run's PID folder, then a marker added and a copied file removed."""
    src, out = tree
    planning_study(src / PID / 'p')
    organize(src, out)
    organize(src, tmp_path / 'fresh')
    (out / PID / 'marker.txt').write_text('stale')
    next((out / PID / 'plan_1').glob('ct*.dcm')).unlink()
    return src, out, tmp_path / 'fresh'


def test_o1_without_overwrite_an_existing_folder_is_skipped_and_untouched(stale, caplog):
    src, out, fresh = stale
    before = snapshot(out / PID)
    caplog.set_level(logging.INFO)
    report = organize(src, out)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_SKIPPED, org.REASON_EXISTS)]
    assert snapshot(out / PID) == before and hidden(out) == []
    assert counts(caplog) == (0, 0)


def test_o2_overwrite_replaces_a_stale_folder_with_a_fresh_run(stale, caplog):
    src, out, fresh = stale
    caplog.set_level(logging.INFO)
    report = organize(src, out, overwrite=True)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_ORGANIZED, '')]
    assert snapshot(out / PID) == snapshot(fresh / PID)
    assert hidden(out) == []
    assert counts(caplog) == (1, 0)


def test_t1d_without_overwrite_a_dangling_link_at_the_output_name_is_an_existing_output(tree, caplog):
    src, out = tree
    planning_study(src / PID / 'p')
    out.mkdir()
    (out / PID).symlink_to(out / 'SYNTHETIC_missing', target_is_directory=True)
    caplog.set_level(logging.INFO)
    report = organize(src, out)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_SKIPPED, org.REASON_EXISTS)]
    assert (out / PID).is_symlink() and hidden(out) == []


def test_o2_overwrite_with_no_existing_folder_is_a_plain_new_run(tree, caplog):
    src, out = tree
    planning_study(src / PID / 'p')
    caplog.set_level(logging.INFO)
    assert organize(src, out, overwrite=True).rows[0].outcome == org.OUTCOME_ORGANIZED
    assert hidden(out) == [] and counts(caplog) == (0, 0)


def test_o3_a_copy_failure_under_overwrite_keeps_the_old_folder(stale, monkeypatch, caplog):
    src, out, fresh = stale
    before, real, calls = snapshot(out / PID), org.shutil.copy2, []

    def failing_copy(source, target):
        calls.append(target)
        if len(calls) == 3:
            raise OSError(f'disk full writing {target}')
        return real(source, target)

    monkeypatch.setattr(org.shutil, 'copy2', failing_copy)
    caplog.set_level(logging.INFO)
    report = organize(src, out, overwrite=True)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_REFUSED, org.REASON_IO_ERROR)]
    assert snapshot(out / PID) == before and hidden(out) == []
    assert counts(caplog) == (0, 0)  # a patient that raised adds nothing


def test_o4_a_failed_move_into_place_rolls_back_and_leaks_nothing(stale, monkeypatch, caplog):
    src, out, fresh = stale
    before, real = snapshot(out / PID), dicom_io.rename_dir

    def failing_rename(source, target):
        if source.name == f'.{PID}.partial':
            raise OSError(f'cannot rename {source} to {target}')
        return real(source, target)

    monkeypatch.setattr(dicom_io, 'rename_dir', failing_rename)
    caplog.set_level(logging.DEBUG)
    report = organize(src, out, overwrite=True)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_REFUSED, org.REASON_IO_ERROR)]
    assert snapshot(out / PID) == before and hidden(out) == []
    assert caplog.text and PID not in caplog.text and str(out.parent) not in caplog.text


def test_o5_recovery_restores_or_deletes_the_old_folder_with_or_without_overwrite(stale):
    src, out, fresh = stale
    before = snapshot(out / PID)
    (out / PID).rename(out / f'.{PID}.old')  # killed between the two swap renames
    (out / f'.{PID}.partial').mkdir()
    report = organize(src, out)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_SKIPPED, org.REASON_EXISTS)]
    assert snapshot(out / PID) == before and hidden(out) == []
    (out / f'.{PID}.old').mkdir()  # killed after the swap, before the delete
    (out / f'.{PID}.old' / 'old.txt').write_text('old')
    organize(src, out)
    assert snapshot(out / PID) == before and hidden(out) == []


@pytest.mark.parametrize('refusal', ['sop', 'split'])
def test_o6_a_patient_refused_after_its_name_is_known_keeps_its_folder_and_counts_as_kept(stale, caplog, refusal):
    src, out, fresh = stale
    first = next((src / PID / 'p').glob('ct*.dcm'))
    if refusal == 'sop':
        ds = pydicom.dcmread(first)
        del ds.SOPInstanceUID
        ds.save_as(first)
    else:
        blank_id(first)
    before = snapshot(out / PID)
    caplog.set_level(logging.INFO)
    report = organize(src, out, overwrite=True)
    assert [(r.outcome, r.reason) for r in report.rows] == [
        (org.OUTCOME_REFUSED, org.REASON_NO_SOP_UID if refusal == 'sop' else org.REASON_SPLIT_SERIES)]
    assert snapshot(out / PID) == before and hidden(out) == []
    assert counts(caplog) == (0, 1)


def test_o6_an_unreadable_rtplan_under_overwrite_keeps_the_plan_folders(stale, monkeypatch, caplog):
    # an unreadable RTPLAN goes to misc_files and the patient is still organized, with no plan: never a replacement
    src, out, _ = stale
    real = dicom_io.read_header_core

    def flaky(path):
        if os.path.basename(path) == 'plan.dcm':
            raise OSError('share hiccup')
        return real(path)

    monkeypatch.setattr(dicom_io, 'read_header_core', flaky)
    before = snapshot(out / PID)
    caplog.set_level(logging.INFO)
    report = organize(src, out, overwrite=True)
    assert [(r.outcome, r.plans) for r in report.rows] == [(org.OUTCOME_ORGANIZED, 0)]
    assert snapshot(out / PID) == before and hidden(out) == []
    assert counts(caplog) == (0, 1)


def test_overwrite_refuses_an_input_root_inside_the_output_root(tmp_path):
    out = tmp_path / 'out'
    src = out / 'SYNTHETICPT' / 'export'
    planning_study(src / PID / 'p')
    before = snapshot(out)
    with pytest.raises(ValueError, match=dicom_io.REASON_INPUT_INSIDE_OUTPUT):
        organize(src, out, overwrite=True)
    assert snapshot(out) == before and hidden(out) == []


def test_o6_a_refusal_with_no_existing_folder_is_not_kept_and_never_recovers(tree, caplog):
    src, out = tree
    planning_study(src / PID / 'p')
    ds = pydicom.dcmread(next((src / PID / 'p').glob('ct*.dcm')))
    del ds.SOPInstanceUID
    ds.save_as(next((src / PID / 'p').glob('ct*.dcm')))
    (out / f'.{PID}.old').mkdir(parents=True)  # refused before the claimed check: recovery never runs
    caplog.set_level(logging.INFO)
    organize(src, out, overwrite=True)
    assert hidden(out) == [f'.{PID}.old'] and not (out / PID).exists()
    assert counts(caplog) == (0, 0)


def test_o7_a_later_same_name_folder_is_refused_and_never_touches_the_first_output(tree, caplog):
    src, out = tree
    name = 'SYNTHETIC_a_b'
    planning_study(src / 'p1', pid='SYNTHETIC_a b', date=EARLY)
    planning_study(src / 'p2', pid=name, date=LATE)  # same folder name, claimed by p1
    planning_study(src / 'p3', pid=name, date=LATE, tag='x')
    ds = pydicom.dcmread(next((src / 'p3').glob('ctx*.dcm')))
    del ds.SOPInstanceUID  # refused before the claimed check
    ds.save_as(next((src / 'p3').glob('ctx*.dcm')))
    (out / name).mkdir(parents=True)
    (out / name / 'marker.txt').write_text('stale')
    caplog.set_level(logging.INFO)
    report = organize(src, out, overwrite=True)
    assert [(r.outcome, r.reason) for r in report.rows[:2]] == [(org.OUTCOME_ORGANIZED, ''), (org.OUTCOME_REFUSED, org.REASON_NAME_CLAIMED)]
    assert report.rows[2].reason == org.REASON_NO_SOP_UID
    assert [p.name for p in (out / name).glob('plan_*')] == ['plan_1'] and not (out / name / 'marker.txt').exists()
    assert hidden(out) == [] and counts(caplog) == (1, 0)  # p3 was refused after p1 claimed the name: not kept


def test_o8_an_output_folder_with_no_input_patient_is_untouched(stale):
    src, out, fresh = stale
    (out / 'SYNTHETIC_other').mkdir()
    (out / 'SYNTHETIC_other' / 'a.txt').write_text('other')
    (out / '.SYNTHETIC_other.old').mkdir()
    before = snapshot(out)
    organize(src, out, overwrite=True)
    after = snapshot(out)
    assert {k: v for k, v in after.items() if 'SYNTHETIC_other' in k} == {k: v for k, v in before.items() if 'SYNTHETIC_other' in k}
    assert (out / '.SYNTHETIC_other.old').is_dir()


def test_o9_the_source_tree_is_unchanged_by_an_overwrite_run(stale):
    src, out, fresh = stale
    before = snapshot(src)
    organize(src, out, overwrite=True)
    assert snapshot(src) == before


# ---------------------------------------------------------------- patient processes (#171): N=2 equals N=1, parent-side claims, abort

def written(root):
    """{relative path: sha256} of every file under root except the report copies (their names carry a timestamp)."""
    return {k: v for k, v in snapshot(root).items() if 'report' not in k}


def call_guarded(fn):
    """Run fn in a daemon thread and fail instead of hanging; return the exception it raised (or None)."""
    import threading
    outcome = {}

    def target():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            outcome['error'] = exc
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(120)
    assert not thread.is_alive(), 'the call did not return'
    return outcome.get('error')


def test_two_patient_processes_give_the_same_tree_rows_and_counts_as_one_under_overwrite(tree, tmp_path, caplog):
    src, _ = tree
    planning_study(src / 'a1' / 'p', pid='SYNTHETIC_A')
    planning_study(src / 'a2' / 'p', pid='SYNTHETIC_A', date=LATE)  # the same PatientID: the second folder is refused by the claim
    planning_study(src / 'b' / 'p', pid='SYNTHETIC_B')  # has an old output: replaced
    planning_study(src / 'c' / 'p', pid='SYNTHETIC_C')  # refused for its missing SOP UID, has an old output: kept
    path = next((src / 'c' / 'p').glob('ct*.dcm'))
    ds = pydicom.dcmread(path)
    del ds.SOPInstanceUID
    ds.save_as(path)
    caplog.set_level(logging.INFO)
    results = {}
    for n in (1, 2):
        out = tmp_path / f'out{n}'
        for name in ('SYNTHETIC_B', 'SYNTHETIC_C'):
            (out / name).mkdir(parents=True)
            (out / name / 'old.txt').write_text('old')
        caplog.clear()
        results[n] = (organize(src, out, workers=n, overwrite=True).rows, written(out), counts(caplog), hidden(out))
    assert results[2] == results[1]
    rows1, tree1, counted, left = results[1]
    assert [(r.patient, r.outcome, r.reason) for r in rows1] == [
        ('SYNTHETIC_A', org.OUTCOME_ORGANIZED, ''), ('SYNTHETIC_A', org.OUTCOME_REFUSED, org.REASON_NAME_CLAIMED),
        ('SYNTHETIC_B', org.OUTCOME_ORGANIZED, ''), ('SYNTHETIC_C', org.OUTCOME_REFUSED, org.REASON_NO_SOP_UID)]
    assert counted == (1, 1) and left == [] and 'SYNTHETIC_C/old.txt' in tree1 and 'SYNTHETIC_B/old.txt' not in tree1


def test_an_unexpected_error_in_the_read_task_aborts_a_two_process_run_with_the_report_and_no_message(tree, capfd):
    import pool_workers
    src, out = tree
    for name in ('a', 'boom', 'c', 'd', 'e', 'f'):
        mk(src / name)
    error = call_guarded(lambda: organize(src, out, workers=2, _read_fn=pool_workers.organize_read))
    assert isinstance(error, org.OrganizeAborted) and str(error) == org.REASON_INTERNAL
    got = [(r['patient'], r['outcome'], r['reason']) for r in rows(out)]
    assert got.count(('boom', org.OUTCOME_REFUSED, org.REASON_INTERNAL)) == 1
    assert len(got) <= 2  # 'a' (before it) at most: a later read is delivered after the abort and adds no row or copy
    err = capfd.readouterr().err
    assert pool_workers.SECRET not in err and 'RuntimeError' in err  # the worker logged the type and site, never the message


def test_an_unexpected_error_in_the_read_task_starts_no_further_patient_with_one_process(tree):
    import pool_workers
    src, out = tree
    for name in ('a', 'boom', 'c', 'd'):
        mk(src / name)
    with pytest.raises(org.OrganizeAborted):
        organize(src, out, _read_fn=pool_workers.organize_read)
    assert [(r['patient'], r['outcome']) for r in rows(out)] == [('a', org.OUTCOME_ORGANIZED), ('boom', org.OUTCOME_REFUSED)]


def test_an_unexpected_error_in_the_copy_task_aborts_the_run_and_leaves_no_partial_folder(tree, capfd):
    import pool_workers
    src, out = tree
    planning_study(src / 'boom' / 'p', pid='SYNTHETIC_A')
    error = call_guarded(lambda: organize(src, out, workers=2, _copy_fn=pool_workers.organize_copy))
    assert isinstance(error, org.OrganizeAborted)
    assert [(r['outcome'], r['reason']) for r in rows(out)] == [(org.OUTCOME_REFUSED, org.REASON_INTERNAL)]
    assert pool_workers.SECRET not in capfd.readouterr().err and hidden(out) == []


def test_an_oserror_in_the_copy_task_or_in_recover_refuses_that_patient_and_the_run_goes_on(tree, monkeypatch):
    src, out = tree
    planning_study(src / 'a' / 'p', pid='SYNTHETIC_A')
    planning_study(src / 'b' / 'p', pid='SYNTHETIC_B')
    planning_study(src / 'c' / 'p', pid='SYNTHETIC_C')
    real_copy, real_recover = org._copy_all, dicom_io.recover  # noqa: SLF001

    def copy(paths, folder):
        if 'SYNTHETIC_A' in str(folder):
            raise OSError('permission denied /secret/path')
        return real_copy(paths, folder)

    def recover(root, name):
        if name == 'SYNTHETIC_B':
            raise OSError('permission denied /secret/path')
        return real_recover(root, name)

    monkeypatch.setattr(org, '_copy_all', copy)
    monkeypatch.setattr(dicom_io, 'recover', recover)
    report = organize(src, out)
    assert [(r.patient, r.outcome, r.reason) for r in report.rows] == [
        ('a', org.OUTCOME_REFUSED, org.REASON_IO_ERROR), ('b', org.OUTCOME_REFUSED, org.REASON_IO_ERROR),
        ('SYNTHETIC_C', org.OUTCOME_ORGANIZED, '')]
    assert hidden(out) == [] and not (out / 'SYNTHETIC_A').exists()


def test_a_target_outside_the_root_still_raises_value_error_not_an_internal_row(tree, monkeypatch):
    src, out = tree
    planning_study(src / 'a' / 'p', pid='SYNTHETIC_A')
    real = org.sanitize_path_component
    monkeypatch.setattr(org, 'sanitize_path_component', lambda text: '..' if text == 'SYNTHETIC_A' else real(text))  # a hostile id
    with pytest.raises(ValueError, match=re.escape(org.REASON_OUTSIDE_ROOT)):
        organize(src, out)
    assert rows(out) == []


@pytest.mark.parametrize('read_fn, expected', [('organize_read_escaping', (ValueError, org.REASON_OUTSIDE_ROOT)),
                                               ('organize_read_a_exits', (SystemExit, '3'))])
def test_a_read_delivered_after_the_handle_raise_is_not_claimed_or_recovered(tree, read_fn, expected):
    # #182/#185: patient b's read was running when a's handle raised; it must not recover (delete) b's leftover partial
    import pool_workers
    src, out = tree
    mk(src / 'a')
    mk(src / 'b')
    (out / '.b.partial').mkdir(parents=True)
    error = call_guarded(lambda: organize(src, out, workers=2, _read_fn=getattr(pool_workers, read_fn)))
    assert type(error) is expected[0] and str(error) == expected[1]
    assert (out / '.b.partial').is_dir() and rows(out) == []


@pytest.mark.parametrize('workers', [1, 2])
def test_a_resumed_run_copies_nothing_and_creates_no_partial_for_a_skipped_patient(tree, tmp_path, workers):
    src, out = tree
    planning_study(src / 'a' / 'p', pid='SYNTHETIC_A')
    planning_study(src / 'b' / 'p', pid='SYNTHETIC_B')
    organize(src, out)
    before = written(out)
    seen = []
    # a partial created and removed again would leave no trace: watch the directory from a thread
    import threading
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            seen.extend(p.name for p in out.iterdir() if p.name.endswith('.partial'))
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        report = organize(src, out, workers=workers)
    finally:
        stop.set()
        thread.join(10)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_SKIPPED, org.REASON_EXISTS)] * 2
    assert seen == [] and written(out) == before
