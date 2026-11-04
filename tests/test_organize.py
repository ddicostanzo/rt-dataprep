"""organize: acceptance tests A1.1-A1.7 plus the guardrails. Every tree is synthetic and built in tmp_path."""
import csv
import hashlib
import logging
import re
import subprocess
import sys
from collections import Counter

import numpy as np
import pydicom
import pytest
import synthetic_dicom as sd

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


def test_unreadable_dicom_is_copied_to_misc_files(tree):
    src, out = tree
    planning_study(src / PID / 'a')
    (src / PID / 'a' / 'cut.dcm').write_bytes(TRUNCATED)
    report = organize(src, out)
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


def test_empty_sop_instance_uids_refuse_the_patient(tree):
    src, out = tree
    planning_study(src / PID / 'p')
    for path in (src / PID / 'p').glob('ct*.dcm'):
        ds = pydicom.dcmread(path)
        ds.SOPInstanceUID = ''
        ds.save_as(path)
    organize(src, out)
    assert (rows(out)[0]['outcome'], rows(out)[0]['reason']) == ('refused', org.REASON_NO_SOP_UID)
