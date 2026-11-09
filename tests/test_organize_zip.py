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


@pytest.mark.parametrize('method', [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
def test_every_allowed_method_extracts(folder, stage, method):
    make_zip(folder / 'a.zip', [('x', DICOM)], method=method)
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


BAD = [encrypted, crc_flipped, truncated, method_9, bad_name]


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
