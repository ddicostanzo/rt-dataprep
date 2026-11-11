"""organize zip staging (task-01): classify, cap, allow-list, extract. Every zip is built at test time from synthetic bytes."""
import io
import logging
import lzma
import traceback
import zipfile
import zlib

import pytest

from rt_dataprep import dicom_io
from rt_dataprep import organize as org

DICOM = bytes(128) + b'DICM' + b'SYNTHETIC_payload'
NAME = 'SYNTHETIC_PATIENT_NAME'  # a member name that must never reach a log line, message or traceback


def make_zip(path, members, method=zipfile.ZIP_DEFLATED):
    """members: (name, bytes) pairs, or (ZipInfo, bytes)."""
    with zipfile.ZipFile(path, 'w', method) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return path


def patch_headers(path, *, central=(), local=()):
    """Overwrite bytes in every central (PK\\x01\\x02) and local (PK\\x03\\x04) header: (offset, bytes) pairs."""
    data = bytearray(path.read_bytes())
    for sig, edits in ((b'PK\x01\x02', central), (b'PK\x03\x04', local)):
        at = data.find(sig)
        while at >= 0:
            for offset, value in edits:
                data[at + offset:at + offset + len(value)] = value
            at = data.find(sig, at + 4)
    path.write_bytes(bytes(data))


@pytest.fixture
def folder(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    return src


@pytest.fixture
def stage(tmp_path):
    return tmp_path / 'stage'


def run(folder, stage):
    return org._stage_zips(sorted(org._walk(folder)), stage)


def staged_files(stage):
    return sorted(p for p in stage.rglob('*') if p.is_file()) if stage.exists() else []


# --- AC1: classification ---------------------------------------------------------------------------------------------

def test_has_preamble_classifies_by_the_one_132_byte_read(tmp_path):
    dicom, zipped, empty = tmp_path / 'a', tmp_path / 'b', tmp_path / 'c'
    dicom.write_bytes(DICOM)
    make_zip(zipped, [('x', b'y')])
    make_zip(empty, [])  # PK\x05\x06 only
    short, other = tmp_path / 'd', tmp_path / 'e'
    short.write_bytes(b'PK\x03')
    other.write_bytes(b'not dicom' * 30)
    assert empty.read_bytes()[:4] == b'PK\x05\x06'
    assert [org._has_preamble(p) for p in (dicom, zipped, empty, short, other, tmp_path)] == ['dicom', 'zip', None, None, None, None]


def test_dicom_preamble_wins_over_a_pk_start(tmp_path):
    path = tmp_path / 'a'
    path.write_bytes(b'PK\x03\x04' + bytes(124) + b'DICM')
    assert org._has_preamble(path) == 'dicom'


def test_read_one_returns_none_for_a_zip_and_never_reads_its_header(tmp_path, monkeypatch):
    zipped, dicom = make_zip(tmp_path / 'z', [('x', b'y')]), tmp_path / 'd'
    dicom.write_bytes(DICOM)
    monkeypatch.setattr(dicom_io, 'read_header_core', lambda path: 'HEADER')
    assert org._read_one(zipped) is None
    assert org._read_one(dicom) == (dicom, 'HEADER')


def test_a_monkeypatched_has_preamble_still_drives_read_one(tmp_path, monkeypatch):
    path = tmp_path / 'x'
    path.write_bytes(b'anything')
    monkeypatch.setattr(org, '_has_preamble', lambda p: 'dicom')
    monkeypatch.setattr(dicom_io, 'read_header_core', lambda p: 'HEADER')
    assert org._read_one(path) == (path, 'HEADER')
    monkeypatch.setattr(org, '_has_preamble', lambda p: 'zip')
    assert org._read_one(path) is None


# --- AC2/AC3: layout --------------------------------------------------------------------------------------------------

def test_no_zip_stages_nothing_and_creates_nothing(folder, stage):
    (folder / 'a.dcm').write_bytes(DICOM)
    assert run(folder, stage) == ([], 0, 0, 0)
    assert not stage.exists()


def test_zips_are_staged_as_zip_number_member_number_in_sorted_order(folder, stage):
    make_zip(folder / 'b.zip', [('one', DICOM + b'b1'), ('two', DICOM + b'b2')])
    make_zip(folder / 'a.zip', [('d/', b''), ('d/x.dcm', DICOM + b'a2'), ('y.dcm', DICOM + b'a3')], method=zipfile.ZIP_STORED)
    staged, zips, members, skipped = run(folder, stage)
    assert (zips, members, skipped) == (2, 4, 0)
    # a.zip is zip 1; its directory member is infolist position 1, so its files are members 2 and 3
    assert staged == [stage / '1' / '2', stage / '1' / '3', stage / '2' / '1', stage / '2' / '2']
    assert [p.read_bytes() for p in staged] == [DICOM + b'a2', DICOM + b'a3', DICOM + b'b1', DICOM + b'b2']
    assert staged_files(stage) == staged


@pytest.mark.parametrize('method', [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_every_allowed_method_extracts(folder, stage, method):
    make_zip(folder / 'a.zip', [('x', DICOM)], method=method)
    staged, *_ = run(folder, stage)
    assert [p.read_bytes() for p in staged] == [DICOM]


def test_extraction_reads_the_scanned_member_sizes_not_a_zip_rewritten_after_the_scan(folder, stage, monkeypatch):
    make_zip(folder / 'a.zip', [('x', DICOM)], method=zipfile.ZIP_STORED)
    real = org._zip_scan

    def scan_then_grow(path):
        scan = real(path)
        make_zip(path, [('x', DICOM * 100)], method=zipfile.ZIP_STORED)  # same member, same offset, 100x the bytes
        return scan

    monkeypatch.setattr(org, '_zip_scan', scan_then_grow)
    staged, *_ = run(folder, stage)
    assert [p.read_bytes() for p in staged] == [DICOM]


def test_a_zip_inside_a_zip_is_staged_as_a_plain_file_and_never_opened(folder, stage, monkeypatch):
    inner = io.BytesIO()
    make_zip(inner, [(NAME + 'inner', DICOM)])
    make_zip(folder / 'a.zip', [('in.zip', inner.getvalue())])
    opened = []
    real = zipfile.ZipFile.__init__
    monkeypatch.setattr(zipfile.ZipFile, '__init__', lambda self, file, *a, **k: (opened.append(file), real(self, file, *a, **k))[1])
    staged, zips, members, skipped = run(folder, stage)
    assert (zips, members, skipped) == (1, 1, 0) and len(opened) == 2  # scan + extract of the one outer zip only
    assert staged[0].read_bytes() == inner.getvalue() and org._has_preamble(staged[0]) == 'zip'
    assert staged_files(stage) == staged


def test_a_symlink_entry_is_staged_as_a_regular_file_of_its_bytes(folder, stage):
    link = zipfile.ZipInfo('link')
    link.external_attr = 0o120777 << 16
    make_zip(folder / 'a.zip', [(link, b'/etc/passwd')])
    (staged,), *_ = run(folder, stage)
    assert staged.is_file() and not staged.is_symlink() and staged.read_bytes() == b'/etc/passwd'


@pytest.mark.filterwarnings('ignore:Duplicate name')
def test_traversal_absolute_and_duplicate_member_names_stay_inside_staging(tmp_path, folder, stage):
    make_zip(folder / 'a.zip', [('../x.dcm', DICOM + b'1'), ('/abs/x.dcm', DICOM + b'2'), ('same.dcm', DICOM + b'3'),
                                ('same.dcm', DICOM + b'4')])
    staged, _, members, _ = run(folder, stage)
    assert members == 4 and sorted(p.read_bytes() for p in staged) == [DICOM + bytes([c]) for c in b'1234']
    assert all(p.resolve().is_relative_to(stage.resolve()) for p in staged)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['src', 'stage']  # nothing escaped next to the staging folder
    assert not (tmp_path / 'x.dcm').exists() and sorted(p.name for p in folder.iterdir()) == ['a.zip']


# --- AC4: caps ----------------------------------------------------------------------------------------------------------

def test_the_module_constants_and_reason_texts():
    assert org.MAX_UNZIP_BYTES == 16 * 2**30 and org.MAX_ZIP_MEMBERS == 200_000
    assert org.REASON_ZIP_TOO_LARGE == 'zip archives in the source folder exceed the size or member cap; nothing kept'
    assert org.REASON_ZIP_UNREADABLE == 'a zip archive in the source folder cannot be read safely; nothing kept'
    assert org.REASON_LOCKED == 'another organize run holds the output root lock'


def test_declared_size_over_the_cap_refuses_and_stages_nothing(folder, stage, monkeypatch):
    monkeypatch.setattr(org, 'MAX_UNZIP_BYTES', 10)
    make_zip(folder / 'a.zip', [('x', b'1' * 6)])
    make_zip(folder / 'b.zip', [('y', b'2' * 6)])  # each under the cap, the sum over it
    with pytest.raises(org.ZipRefused) as refused:
        run(folder, stage)
    assert str(refused.value) == refused.value.reason == org.REASON_ZIP_TOO_LARGE
    assert staged_files(stage) == []


def test_declared_size_exactly_at_the_cap_is_allowed(folder, stage, monkeypatch):
    monkeypatch.setattr(org, 'MAX_UNZIP_BYTES', 12)
    make_zip(folder / 'a.zip', [('x', b'1' * 6), ('y', b'2' * 6)])
    assert run(folder, stage)[2] == 2


def test_member_count_over_the_cap_refuses_and_directories_count_toward_none(folder, stage, monkeypatch):
    monkeypatch.setattr(org, 'MAX_ZIP_MEMBERS', 2)
    make_zip(folder / 'a.zip', [('d/', b''), ('e/', b''), ('f/', b''), ('x', b'1'), ('y', b'2')])
    assert run(folder, stage)[2] == 2  # five members, two files: at the cap
    make_zip(folder / 'b.zip', [('z', b'3')])
    with pytest.raises(org.ZipRefused) as refused:
        run(folder, stage / 'again')
    assert str(refused.value) == org.REASON_ZIP_TOO_LARGE and staged_files(stage / 'again') == []


# --- AC5/AC6: allow-list and bad zips -------------------------------------------------------------------------------------

def encrypted(path):
    make_zip(path, [(NAME + 'enc', b'secret' * 10)])
    patch_headers(path, central=[(8, b'\x01\x00')], local=[(6, b'\x01\x00')])


def crc_flipped(path):
    make_zip(path, [(NAME + 'crc', b'payload' * 10)], method=zipfile.ZIP_STORED)
    data = bytearray(path.read_bytes())
    data[data.index(b'payload')] ^= 0xFF  # a stored byte: the declared CRC no longer matches
    path.write_bytes(bytes(data))


def truncated(path):
    make_zip(path, [(NAME + 'cut', DICOM * 20)])
    path.write_bytes(path.read_bytes()[:-30])


def method_9(path):
    make_zip(path, [(NAME + 'm9', DICOM)])
    patch_headers(path, central=[(10, b'\x09\x00')], local=[(8, b'\x09\x00')])


def bad_name(path):
    info = zipfile.ZipInfo(NAME + 'é')  # non-ASCII: written with the UTF-8 flag
    make_zip(path, [(info, DICOM)])
    path.write_bytes(path.read_bytes().replace('é'.encode(), b'\xff\xfe'))


def bzip2(path):
    make_zip(path, [(NAME + 'bz', DICOM)], method=zipfile.ZIP_BZIP2)  # off the allow-list: unbounded output on early 3.12.x (#243)


def lzma_zip(path):
    make_zip(path, [(NAME + 'xz', DICOM)], method=zipfile.ZIP_LZMA)


BAD = [encrypted, crc_flipped, truncated, method_9, bad_name, bzip2, lzma_zip]


@pytest.mark.parametrize('build', BAD)
def test_a_bad_zip_alone_refuses_as_unreadable_and_leaves_nothing(folder, stage, build):
    build(folder / 'a.zip')
    with pytest.raises(org.ZipRefused) as refused:
        run(folder, stage)
    assert str(refused.value) == org.REASON_ZIP_UNREADABLE and staged_files(stage) == []


def test_a_member_that_fails_after_good_ones_removes_everything_staged(folder, stage):
    make_zip(folder / 'a.zip', [('x', DICOM)])
    crc_flipped(folder / 'b.zip')
    with pytest.raises(org.ZipRefused) as refused:
        run(folder, stage)
    assert str(refused.value) == org.REASON_ZIP_UNREADABLE and staged_files(stage) == []


@pytest.mark.parametrize('error', [zipfile.BadZipFile, EOFError, zlib.error, lzma.LZMAError, NotImplementedError, RuntimeError])
def test_every_listed_error_while_reading_makes_the_zip_unreadable(folder, stage, monkeypatch, error):
    make_zip(folder / 'a.zip', [('x', DICOM)])

    def boom(self, *args, **kwargs):
        raise error(NAME)

    monkeypatch.setattr(zipfile.ZipFile, 'open', boom)
    with pytest.raises(org.ZipRefused) as refused:
        run(folder, stage)
    assert str(refused.value) == org.REASON_ZIP_UNREADABLE


def test_a_unicode_decode_error_while_opening_makes_the_zip_unreadable(folder, stage, monkeypatch):
    make_zip(folder / 'a.zip', [('x', DICOM)])

    def boom(self, *args, **kwargs):
        raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, NAME)

    monkeypatch.setattr(zipfile.ZipFile, '__init__', boom)
    with pytest.raises(org.ZipRefused) as refused:
        run(folder, stage)
    assert str(refused.value) == org.REASON_ZIP_UNREADABLE


def test_an_oserror_propagates_unchanged(folder, stage, monkeypatch):
    make_zip(folder / 'a.zip', [('x', DICOM)])

    def boom(self, *args, **kwargs):
        raise PermissionError('denied')

    monkeypatch.setattr(zipfile.ZipFile, 'open', boom)
    with pytest.raises(PermissionError):
        run(folder, stage)


# --- AC7: loose DICOM decides skip or refuse -------------------------------------------------------------------------------

def test_a_bad_zip_beside_loose_dicom_is_skipped_and_its_staged_members_removed(folder, stage):
    (folder / 'loose.dcm').write_bytes(DICOM)
    make_zip(folder / 'a.zip', [('x', DICOM + b'good')])
    crc_flipped(folder / 'b.zip')  # fails while streaming, after its first byte may be staged
    staged, zips, members, skipped = run(folder, stage)
    assert (zips, members, skipped) == (1, 1, 1)
    assert staged == [stage / '1' / '1'] and staged_files(stage) == staged and not (stage / '2').exists()


def test_every_bad_zip_kind_is_skipped_beside_loose_dicom(folder, stage):
    (folder / 'loose.dcm').write_bytes(DICOM)
    for n, build in enumerate(BAD):
        build(folder / f'bad{n}.zip')
    assert run(folder, stage) == ([], 0, 0, len(BAD))


def test_an_over_cap_total_beside_loose_dicom_skips_every_zip(folder, stage, monkeypatch):
    monkeypatch.setattr(org, 'MAX_UNZIP_BYTES', 10)
    (folder / 'loose.dcm').write_bytes(DICOM)
    make_zip(folder / 'a.zip', [('x', b'1' * 6)])
    make_zip(folder / 'b.zip', [('y', b'2' * 6)])
    assert run(folder, stage) == ([], 0, 0, 2) and staged_files(stage) == []


def test_a_loose_file_without_the_preamble_does_not_count_as_loose_dicom(folder, stage):
    (folder / 'notes.txt').write_text('not dicom')
    crc_flipped(folder / 'a.zip')
    with pytest.raises(org.ZipRefused):
        run(folder, stage)


def test_a_loose_preamble_file_counts_even_if_its_header_would_not_read(folder, stage):
    (folder / 'cut.dcm').write_bytes(bytes(128) + b'DICM')  # no elements: pydicom would fail, the preamble is enough
    crc_flipped(folder / 'a.zip')
    assert run(folder, stage) == ([], 0, 0, 1)


def test_over_cap_and_unreadable_zips_together_refuse_as_too_large(folder, stage, monkeypatch):
    monkeypatch.setattr(org, 'MAX_ZIP_MEMBERS', 1)
    make_zip(folder / 'a.zip', [('x', b'1'), ('y', b'2')])
    truncated(folder / 'b.zip')
    with pytest.raises(org.ZipRefused) as refused:
        run(folder, stage)
    assert str(refused.value) == org.REASON_ZIP_TOO_LARGE


# --- AC8: no member name or path anywhere -------------------------------------------------------------------------------------

def test_traversal_names_log_nothing(folder, stage, caplog):
    make_zip(folder / 'a.zip', [(f'../{NAME}.dcm', DICOM), (f'/abs/{NAME}.dcm', DICOM)])
    with caplog.at_level(logging.DEBUG):
        assert run(folder, stage)[2] == 2
    assert NAME not in caplog.text and str(folder) not in caplog.text and str(stage) not in caplog.text


@pytest.mark.parametrize('build', [crc_flipped, encrypted, truncated, method_9, bad_name])
@pytest.mark.parametrize('loose', [False, True])
def test_no_member_name_or_path_reaches_a_log_a_message_or_a_chained_exception(folder, stage, caplog, build, loose):
    if loose:
        (folder / 'loose.dcm').write_bytes(DICOM)
    build(folder / 'a.zip')
    seen = ''
    with caplog.at_level(logging.DEBUG):
        try:
            run(folder, stage)
        except org.ZipRefused as exc:
            assert not loose and exc.__cause__ is None and exc.__suppress_context__ and exc.__context__ is None
            seen = ''.join(traceback.format_exception(exc)) + repr(exc.args)
    assert NAME not in seen + caplog.text and str(folder) not in seen + caplog.text and str(stage) not in seen + caplog.text
    assert (not loose) == bool(seen)


# === task-02: organize() reads the zips ======================================================================================

import fcntl  # noqa: E402 (POSIX-only tests below)
import hashlib  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pydicom  # noqa: E402
import synthetic_dicom as sd  # noqa: E402
from test_organize import LATE, PID, planning_study, snapshot  # noqa: E402

from rt_dataprep.organize import organize  # noqa: E402

SUMMARY_LINE = re.compile(r'organize: (\d+) patient folders, (\d+) organized, (\d+) plan folders, (\d+) misc files, (\d+) unreadable files, '
                          r'(\d+) zips, (\d+) zip members, (\d+) zips skipped; (\d+) replaced, (\d+) kept')


@pytest.fixture
def tree(tmp_path):
    src, out = tmp_path / 'src', tmp_path / 'out'
    src.mkdir()
    return src, out


def zip_up(folder, name='export.zip', only=None):
    """Zip the files directly under `folder` (those named in `only`, default all) into folder/name and remove the originals."""
    members = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix != '.zip' and (only is None or p.name in only)]
    with zipfile.ZipFile(folder / name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in members:
            zf.write(p, p.name)
    for p in members:
        p.unlink()
    return folder / name


def leftovers(out):
    return sorted(p.name for p in out.iterdir() if p.name.startswith('.unzip-'))


def plan_hashes(out, pid=PID):
    """{plan folder: sorted content hashes}: file names differ between loose and zipped sources, the bytes do not."""
    return {d.name: sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()) for d in sorted((out / pid).iterdir())}


def log_line(caplog, text):
    return next(r.getMessage() for r in caplog.records if text in r.getMessage())


# --- AC9/AC10/AC17: the happy path --------------------------------------------------------------------------------------

def test_a_zipped_patient_organizes_like_the_loose_files_and_the_source_is_untouched(tree, tmp_path):
    src, out = tree
    planning_study(src / 'loose' / 'x')
    shutil.copytree(src / 'loose', src / 'zipped')
    zip_up(src / 'zipped' / 'x')
    before = snapshot(src)
    organize(src, out)
    assert not (out / PID / 'plan_2').exists()
    organize(src / 'loose', tmp_path / 'ref')
    assert plan_hashes(out) == plan_hashes(tmp_path / 'ref') and list(plan_hashes(out)) == ['plan_1']
    assert snapshot(src) == before and leftovers(out) == []


def test_the_summary_line_carries_the_zip_counts_in_the_pinned_format(tree, caplog):
    src, out = tree
    planning_study(src / 'a' / 'x')
    zip_up(src / 'a' / 'x')
    planning_study(src / 'b' / 'x', pid='SYNTHETIC_B')
    with caplog.at_level(logging.INFO):
        organize(src, out)
    match = SUMMARY_LINE.fullmatch(log_line(caplog, 'patient folders'))
    assert match.groups() == ('2', '2', '2', '0', '0', '1', '6', '0', '0', '0')
    assert ', 13 files,' in log_line(caplog, ' s total')  # the zip, its 6 members and b's 6 loose files


def test_staged_files_join_the_loose_ones_and_keep_index_names(tree):
    src, out = tree
    planning_study(src / 'a')
    zip_up(src / 'a', only={'plan.dcm', 'struct.dcm'})
    organize(src, out)
    names = {p.name for p in (out / PID / 'plan_1').iterdir()}
    assert len(names) == 6 and {'1', '2'} <= names and not names & {'plan.dcm', 'struct.dcm'}


# --- AC11: zipped converts to the same outputs as loose -----------------------------------------------------------------

def test_a_zipped_patient_converts_to_the_same_outputs_as_the_same_files_loose(tmp_path):
    import SimpleITK as sitk  # noqa: PLC0415
    from test_convert import BOX, dose_spec, make_plan  # noqa: PLC0415
    from test_convert import PID as CPID  # noqa: PLC0415

    from rt_dataprep import convert as conv  # noqa: PLC0415

    folder = tmp_path / 'loose' / CPID / 'p'
    made = make_plan(folder, doses=[dose_spec(10, 'PLAN')], rois=[BOX])
    sd.write_ct_series(folder, np.ones((3, 4, 4)), patient_id=CPID, prefix='mr', modality='MR', study_uid=made['ct']['study_uid'])
    shutil.copytree(tmp_path / 'loose', tmp_path / 'zipped')
    zip_up(tmp_path / 'zipped' / CPID / 'p')
    converted = {}
    for label in ('loose', 'zipped'):
        organize(tmp_path / label, tmp_path / f'org_{label}')
        converted[label] = conv.convert(tmp_path / f'org_{label}', tmp_path / f'out_{label}').rows
    assert converted['zipped'] and all(r.outcome == 'converted' for r in converted['zipped'])
    assert [(r.folder, r.files) for r in converted['loose']] == [(r.folder, r.files) for r in converted['zipped']]
    images = lambda label: sorted(p.relative_to(tmp_path / f'out_{label}') for p in (tmp_path / f'out_{label}').rglob('*.nii.gz'))  # noqa: E731
    assert images('loose') == images('zipped') and len(images('zipped')) >= 3  # CT, ROI mask, dose and the MR series
    for path in images('zipped'):
        a, b = (sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / f'out_{k}' / path))) for k in ('loose', 'zipped'))
        assert np.array_equal(a, b)


# --- AC12: loose plus zipped, deduplicated by SOP -------------------------------------------------------------------------

def test_loose_files_plus_a_zip_with_the_rest_are_one_patient_and_a_duplicate_uses_the_loose_copy(tree):
    src, out = tree
    folder = src / 'a'
    planning_study(folder)
    first = sorted(folder.glob('ct*.dcm'))[0]
    zip_up(folder, only={'struct.dcm', 'plan.dcm', 'dose.dcm'})
    with zipfile.ZipFile(folder / 'export.zip', 'a') as zf:
        zf.write(first, 'duplicate')  # the same instance, loose and zipped
    report = organize(src, out)
    assert [(r.patient, r.outcome, r.plans, r.misc) for r in report.rows] == [(PID, org.OUTCOME_ORGANIZED, 1, 1)]
    plan_names = {p.name for p in (out / PID / 'plan_1').iterdir()}
    assert len(plan_names) == 6 and first.name in plan_names
    assert [p.read_bytes() for p in (out / PID / 'misc_files').iterdir()] == [first.read_bytes()]


# --- AC13: two plans on one planning CT, zipped -------------------------------------------------------------------------

def test_a_zipped_patient_with_two_plans_on_one_ct_gives_both_folders_the_full_ct_series(tree):
    src, out = tree
    folder = src / 'a'
    made = planning_study(folder)
    sd.write_rtplan(folder / 'plan2.dcm', made['struct'], np.zeros((2, 3)), plan_date=LATE, plan_time='120000',
                    patient_id=PID, study_uid=made['study'])
    zip_up(folder)
    organize(src, out)
    for plan in ('plan_1', 'plan_2'):
        modalities = sorted(pydicom.dcmread(p, stop_before_pixels=True).Modality for p in (out / PID / plan).iterdir())
        assert modalities.count('CT') == 3 and modalities.count('RTPLAN') == 1 and modalities.count('RTSTRUCT') == 1
    assert leftovers(out) == []


# --- AC14: refusals and skips -------------------------------------------------------------------------------------------

def test_a_zip_only_folder_over_a_cap_is_refused_with_a_fixed_reason_and_the_run_goes_on(tree, monkeypatch, caplog):
    src, out = tree
    planning_study(src / 'a')
    zip_up(src / 'a')
    planning_study(src / 'b', pid='SYNTHETIC_B')
    monkeypatch.setattr(org, 'MAX_ZIP_MEMBERS', 3)
    with caplog.at_level(logging.DEBUG):
        report = organize(src, out)
    assert [(r.patient, r.outcome, r.reason) for r in report.rows] == [
        ('a', org.OUTCOME_REFUSED, org.REASON_ZIP_TOO_LARGE), ('SYNTHETIC_B', org.OUTCOME_ORGANIZED, '')]
    assert leftovers(out) == [] and str(src) not in caplog.text


@pytest.mark.parametrize('build', [crc_flipped, encrypted, truncated, method_9, bad_name])
def test_a_zip_only_folder_with_an_unreadable_zip_is_refused_and_the_next_patient_organizes(tree, build, caplog):
    src, out = tree
    (src / 'a').mkdir()
    build(src / 'a' / 'a.zip')
    planning_study(src / 'b', pid='SYNTHETIC_B')
    with caplog.at_level(logging.DEBUG):
        report = organize(src, out)
    assert [(r.patient, r.outcome, r.reason, r.unreadable) for r in report.rows] == [
        ('a', org.OUTCOME_REFUSED, org.REASON_ZIP_UNREADABLE, 0), ('SYNTHETIC_B', org.OUTCOME_ORGANIZED, '', 0)]
    assert NAME not in caplog.text and str(src) not in caplog.text and leftovers(out) == []


def test_a_refused_zip_folder_leaves_an_existing_output_untouched_with_overwrite(tree):
    src, out = tree
    (out / 'a').mkdir(parents=True)
    (out / 'a' / 'keep.txt').write_text('x')
    (src / 'a').mkdir()
    crc_flipped(src / 'a' / 'a.zip')
    report = organize(src, out, overwrite=True)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_REFUSED, org.REASON_ZIP_UNREADABLE)]
    assert [p.name for p in (out / 'a').iterdir()] == ['keep.txt']


def test_loose_dicom_plus_a_bad_zip_organizes_from_the_loose_files_and_logs_the_skip(tree, caplog):
    src, out = tree
    planning_study(src / 'a')
    crc_flipped(src / 'a' / 'bad.zip')
    with caplog.at_level(logging.INFO):
        report = organize(src, out)
    assert [(r.outcome, r.plans) for r in report.rows] == [(org.OUTCOME_ORGANIZED, 1)]
    assert SUMMARY_LINE.fullmatch(log_line(caplog, 'patient folders')).group(6, 7, 8) == ('0', '0', '1')
    assert '1 zips skipped' in caplog.text and leftovers(out) == []


# --- AC14/AC15b: _read_patient called directly --------------------------------------------------------------------------

@pytest.fixture
def parent(tmp_path):
    path = tmp_path / 'staging'
    path.mkdir()
    return path


def test_read_patient_returns_the_refusal_not_a_raise_for_a_zip_only_folder(tree, parent):
    src, _ = tree
    (src / 'a').mkdir()
    crc_flipped(src / 'a' / 'a.zip')
    name, refusal, cases, misc, unreadable, walked, _, staging, zips, members, skipped = org._read_patient(src / 'a', ('Acuity',), None, parent)
    row, reads_kept = refusal
    assert (name, reads_kept, row.reason, cases, misc, unreadable, walked, staging, (zips, members, skipped)) == (
        'a', False, org.REASON_ZIP_UNREADABLE, [], [], 0, 1, None, (0, 0, 0))
    assert list(parent.iterdir()) == []


def test_read_patient_removes_its_staging_on_a_header_refusal(tree, parent):
    src, _ = tree
    planning_study(src / 'a', pid='SYNTHETIC_ONE')
    planning_study(src / 'a', pid='SYNTHETIC_TWO', tag='x')
    zip_up(src / 'a')
    result = org._read_patient(src / 'a', ('Acuity',), None, parent)
    assert result[1][0].reason == org.REASON_MIXED_IDS and result[7] is None and result[8:] == (1, 12, 0)
    assert list(parent.iterdir()) == []


def test_read_patient_removes_its_staging_when_the_output_exists(tree, parent):
    src, out = tree
    planning_study(src / 'a')
    zip_up(src / 'a')
    (out / PID).mkdir(parents=True)
    result = org._read_patient(src / 'a', ('Acuity',), out, parent)
    assert result[1] is None and result[2] is None and result[7] is None
    assert list(parent.iterdir()) == []


def test_read_patient_keeps_the_staging_it_returns_with_files_to_copy(tree, parent):
    src, _ = tree
    planning_study(src / 'a')
    zip_up(src / 'a')
    result = org._read_patient(src / 'a', ('Acuity',), None, parent)
    staging = result[7]
    assert staging is not None and staging.parent == parent and len(result[2][0]) == 6
    assert all(p.is_relative_to(staging) for p in result[2][0])


def test_read_patient_removes_its_staging_when_a_staged_read_raises_oserror(tree, parent, monkeypatch):
    src, _ = tree
    planning_study(src / 'a')
    zip_up(src / 'a')
    real = org._has_preamble

    def flaky(path):
        if path.is_relative_to(parent):
            raise OSError(f'denied {path}')
        return real(path)

    monkeypatch.setattr(org, '_has_preamble', flaky)
    with pytest.raises(OSError):
        org._read_patient(src / 'a', ('Acuity',), None, parent)
    assert list(parent.iterdir()) == []


def test_a_loose_file_is_classified_once(tree, parent, monkeypatch):
    src, _ = tree
    planning_study(src / 'a')
    seen, real = [], org._has_preamble
    monkeypatch.setattr(org, '_has_preamble', lambda p: seen.append(p) or real(p))
    org._read_patient(src / 'a', ('Acuity',), None, parent)
    assert len(seen) == len(set(seen)) == 6


def test_a_name_claimed_refusal_removes_that_folders_staging_in_the_handler(tree, monkeypatch):
    src, out = tree
    for name in ('a', 'b'):
        planning_study(src / name)
        zip_up(src / name)
    removed, real = [], dicom_io.remove_tree

    def record(path):
        removed.append(Path(path))
        return real(path)

    monkeypatch.setattr(dicom_io, 'remove_tree', record)
    report = organize(src, out)
    assert [r.reason for r in report.rows] == ['', org.REASON_NAME_CLAIMED]
    assert len([p for p in removed if p.parent.name.startswith('.unzip-')]) == 2  # a's by task B, b's by the handler
    assert removed[-1].parent == out and removed[-1].name.startswith('.unzip-')  # the parent, after both


# --- AC15: nothing left behind ------------------------------------------------------------------------------------------

def test_no_staging_is_left_after_a_run_that_skips_an_existing_output(tree):
    src, out = tree
    planning_study(src / 'a')
    zip_up(src / 'a')
    organize(src, out)
    report = organize(src, out)
    assert [r.outcome for r in report.rows] == [org.OUTCOME_SKIPPED] and leftovers(out) == []


def test_no_staging_is_left_after_a_classification_oserror(tree, monkeypatch):
    src, out = tree
    planning_study(src / 'a')
    zip_up(src / 'a')
    real = org._has_preamble

    def flaky(path):
        if '.unzip-' in str(path):
            raise OSError(f'denied {path}')
        return real(path)

    monkeypatch.setattr(org, '_has_preamble', flaky)
    report = organize(src, out)
    assert [(r.outcome, r.reason) for r in report.rows] == [(org.OUTCOME_REFUSED, org.REASON_IO_ERROR)] and leftovers(out) == []


def test_no_staging_is_left_after_a_reason_internal_abort(tree, monkeypatch):
    src, out = tree
    planning_study(src / 'a')
    zip_up(src / 'a')

    def boom(path):
        raise RuntimeError('SYNTHETIC_secret')

    monkeypatch.setattr(dicom_io, 'read_header_core', boom)
    with pytest.raises(org.OrganizeAborted):
        organize(src, out)
    assert leftovers(out) == []


def test_no_staging_is_left_after_a_keyboard_interrupt_during_extraction(tree, monkeypatch):
    src, out = tree
    planning_study(src / 'a')
    zip_up(src / 'a')
    calls, real = [], zipfile.ZipFile.open

    def interrupt(self, *args, **kwargs):
        if calls:
            raise KeyboardInterrupt
        calls.append(1)
        return real(self, *args, **kwargs)  # one member is staged before the interrupt

    monkeypatch.setattr(zipfile.ZipFile, 'open', interrupt)
    with pytest.raises(KeyboardInterrupt):
        organize(src, out, workers=1)
    assert calls and leftovers(out) == [] and not (out / PID).exists()


def test_no_staging_is_left_after_a_two_process_run(tree):
    src, out = tree
    for n in range(3):
        planning_study(src / f'f{n}', pid=f'SYNTHETIC_P{n}')
        zip_up(src / f'f{n}')
    report = organize(src, out, workers=2)
    assert [r.outcome for r in report.rows] == [org.OUTCOME_ORGANIZED] * 3 and leftovers(out) == []


def test_a_staging_parent_that_cannot_be_removed_logs_one_fixed_warning_with_a_count(tree, monkeypatch, caplog):
    src, out = tree
    planning_study(src / 'a')
    real = shutil.rmtree

    def stubborn(path, *args, **kwargs):
        if Path(path).name.startswith('.unzip-'):
            raise OSError(f'busy {path}')
        return real(path, *args, **kwargs)

    monkeypatch.setattr(shutil, 'rmtree', stubborn)
    with caplog.at_level(logging.WARNING):
        organize(src, out)
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [org.WARN_STAGING_LEFT % 1]
    assert str(out) not in caplog.text and len(leftovers(out)) == 1
    monkeypatch.undo()
    organize(src, out)  # the next run sweeps it
    assert leftovers(out) == []


def test_the_log_of_a_zipped_run_holds_no_member_name_or_path(tree, caplog):
    src, out = tree
    planning_study(src / 'a')
    make_zip(src / 'a' / 'x.zip', [(f'../{NAME}.dcm', DICOM), (f'/abs/{NAME}.dcm', DICOM), (NAME, DICOM)])
    with caplog.at_level(logging.DEBUG):
        organize(src, out)
    assert NAME not in caplog.text and str(src) not in caplog.text and str(out) not in caplog.text


def test_zipped_members_that_make_pydicom_warn_print_and_log_nothing(tree, capfd, caplog):
    src, out = tree
    planning_study(src / 'a')
    sd.write_rtdose(src / 'a' / 'odd.dcm', [[[1]]], patient_id=PID, invalid_values=True)
    zip_up(src / 'a')
    capfd.readouterr()
    with caplog.at_level(logging.DEBUG):
        report = organize(src, out)
    assert [r.outcome for r in report.rows] == [org.OUTCOME_ORGANIZED]
    assert [r for r in caplog.records if r.name.startswith('pydicom')] == [] and tuple(capfd.readouterr()) == ('', '')


# --- AC16: the lock -----------------------------------------------------------------------------------------------------

def hold(out):
    out.mkdir(parents=True, exist_ok=True)
    handle = open(out / org.LOCK_NAME, 'a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def test_a_second_run_on_a_locked_root_refuses_and_writes_no_report(tree):
    src, out = tree
    planning_study(src / 'a')
    handle = hold(out)
    (out / org.REPORT_NAME).write_text('live run')
    try:
        with pytest.raises(ValueError, match='^' + re.escape(org.REASON_LOCKED) + '$'):
            organize(src, out)
    finally:
        handle.close()
    assert (out / org.REPORT_NAME).read_text() == 'live run' and leftovers(out) == []


def test_the_lock_is_released_and_the_file_stays_after_a_run(tree):
    src, out = tree
    planning_study(src / 'a')
    organize(src, out)
    assert (out / org.LOCK_NAME).is_file()
    organize(src, out)  # not locked any more


def test_a_run_sweeps_the_leftover_staging_of_a_killed_run(tree):
    src, out = tree
    planning_study(src / 'a')
    (out / '.unzip-dead' / '1').mkdir(parents=True)
    (out / '.unzip-dead' / '1' / '1').write_bytes(b'x')
    organize(src, out)
    assert leftovers(out) == []


def test_the_sweep_keeps_the_old_and_partial_folders_of_a_patient_named_unzip(tree):
    src, out = tree
    planning_study(src / 'a')
    for hidden in ('.unzip-x.old', '.unzip-x.partial'):  # .old is a killed --overwrite swap's only copy: recover restores it
        (out / hidden).mkdir(parents=True)
        (out / hidden / 'f').write_bytes(b'x')
    organize(src, out)
    assert leftovers(out) == ['.unzip-x.old', '.unzip-x.partial']


def test_a_file_system_without_flock_logs_one_fixed_warning_and_runs_unlocked_without_a_sweep(tree, monkeypatch, caplog):
    src, out = tree
    planning_study(src / 'a')
    (out / '.unzip-live').mkdir(parents=True)

    def unsupported(*args):
        raise OSError(95, 'Operation not supported')

    monkeypatch.setattr(fcntl, 'flock', unsupported)
    with caplog.at_level(logging.WARNING):
        report = organize(src, out)
    assert [r.outcome for r in report.rows] == [org.OUTCOME_ORGANIZED]
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [org.WARN_NO_LOCK]
    assert leftovers(out) == ['.unzip-live']


def test_fcntl_is_not_a_module_level_import():
    assert 'fcntl' not in vars(org)


# --- AC17: convert ignores the lock file --------------------------------------------------------------------------------

def test_convert_reads_no_patient_from_the_lock_file(tmp_path):
    from test_convert import BOX, make_plan  # noqa: PLC0415

    from rt_dataprep import convert as conv  # noqa: PLC0415
    root = tmp_path / 'org'
    make_plan(root / 'SYNTHETIC_CP' / 'plan_1', rois=[BOX], patient='SYNTHETIC_CP')
    (root / org.LOCK_NAME).write_text('')
    assert {r.patient for r in conv.convert(root, tmp_path / 'out').rows} == {'SYNTHETIC_CP'}
