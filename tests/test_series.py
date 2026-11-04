"""convert error paths (task-01, #33): the narrowed tag catch and the REASON_INTERNAL crash path. Every tree is synthetic."""
import csv
import inspect
import logging
import re

import pytest
from test_convert import PID, dose_spec, edit, make_plan

from rt_dataprep import convert as conv
from rt_dataprep import masks
from rt_dataprep.dicom_io import CaseRefused

PID_B = 'SYNTHETICPTB'
SECRET = '1.2.3.4.5'  # UID-like, in the message of the exception the tests raise


@pytest.fixture
def tree(tmp_path):
    org, out = tmp_path / 'org', tmp_path / 'out'
    org.mkdir()
    return org, out


def report(out):
    with open(out / conv.REPORT_NAME, newline='') as handle:
        return list(csv.DictReader(handle))


def crash_in_masks_for(monkeypatch, patient, exc):
    """build_masks as is, except that it raises `exc` for the plan folders of `patient`."""
    real = masks.build_masks

    def build(kept, ct_paths, struct_path, *args):
        if patient in str(struct_path):
            raise exc
        return real(kept, ct_paths, struct_path, *args)
    monkeypatch.setattr(masks, 'build_masks', build)


def raise_in(monkeypatch, name, exc):
    def boom(*args, **kwargs):
        raise exc
    monkeypatch.setattr(conv, name, boom)


# ---------------------------------------------------------------- the tag catch is per header-only call

@pytest.mark.parametrize('what', ['plan_struct_reference', 'ct_spacing', 'dose_units'])
def test_a_missing_header_tag_still_refuses_the_plan_with_missing_tag(tree, what):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1', doses=[dose_spec(10)] if what == 'dose_units' else [])
    if what == 'plan_struct_reference':
        edit(plan['folder'] / 'plan.dcm', lambda ds: delattr(ds, 'ReferencedStructureSetSequence'))
    elif what == 'ct_spacing':
        edit(plan['ct']['paths'][1], lambda ds: delattr(ds, 'PixelSpacing'))
    else:
        edit(plan['folder'] / 'dose_0.dcm', lambda ds: delattr(ds, 'DoseUnits'))
    with pytest.raises(CaseRefused) as caught:
        conv.convert_plan(plan['folder'], out / 'case')
    assert str(caught.value) == conv.REASON_MISSING_TAG and not (out / 'case').exists()
    assert getattr(caught.value, '__cause__') is None and getattr(caught.value, '__context__') is None  # noqa: B009


def test_a_tag_error_while_building_the_series_rows_refuses_the_plan_and_removes_the_folder(tree, monkeypatch):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    raise_in(monkeypatch, '_series_rows', KeyError(SECRET))
    with pytest.raises(CaseRefused) as caught:
        conv.convert_plan(plan['folder'], out / 'case')
    assert str(caught.value) == conv.REASON_MISSING_TAG and not (out / 'case').exists()


@pytest.mark.parametrize('name', ['_read_ct', 'write_atomic'])
def test_calls_outside_the_tag_catch_do_not_turn_a_type_error_into_a_refusal(tree, monkeypatch, name):
    org, out = tree
    plan = make_plan(org / PID / 'plan_1')
    raise_in(monkeypatch, name, TypeError(SECRET))
    with pytest.raises(TypeError):
        conv.convert_plan(plan['folder'], out / 'case')


# ---------------------------------------------------------------- the crash path

def test_convert_aborted_is_a_plain_exception_with_the_fixed_text():
    assert issubclass(conv.ConvertAborted, Exception)
    assert not issubclass(conv.ConvertAborted, (ValueError, OSError, CaseRefused))
    assert str(conv.ConvertAborted(conv.REASON_INTERNAL)) == conv.REASON_INTERNAL


def two_patients(org):
    make_plan(org / PID / 'plan_1', patient=PID)
    make_plan(org / PID_B / 'plan_1', patient=PID_B)
    make_plan(org / PID_B / 'plan_2', patient=PID_B)


def test_an_unexpected_error_on_the_second_patient_keeps_the_first_and_flushes_the_report(tree, tmp_path, monkeypatch, capfd, caplog):
    org, out = tree
    two_patients(org)
    crash_in_masks_for(monkeypatch, PID_B, TypeError(SECRET))
    caplog.set_level(logging.DEBUG)
    capfd.readouterr()
    with pytest.raises(conv.ConvertAborted) as caught:
        conv.convert(org, out)
    err = caught.value
    assert str(err) == conv.REASON_INTERNAL and err.args == (conv.REASON_INTERNAL,)
    assert getattr(err, '__cause__') is None and getattr(err, '__context__') is None  # noqa: B009
    assert [(r['patient'], r['folder'], r['outcome'], r['reason']) for r in report(out)] == [
        (PID, 'plan_1', conv.OUTCOME_CONVERTED, ''), (PID_B, '', conv.OUTCOME_REFUSED, conv.REASON_INTERNAL)]
    assert sorted(p.name for p in out.iterdir()) == sorted([conv.REPORT_NAME, PID])  # no output and no partial folder of the second

    captured = capfd.readouterr()
    texts = [captured.out, captured.err, caplog.text, f'{err!r} {err}']
    assert [s for s in (SECRET, str(tmp_path)) for t in texts if s in t] == []

    records = [r.getMessage() for r in caplog.records if 'TypeError' in r.getMessage()]
    assert len(records) == 1
    module, line = re.fullmatch(r'.*TypeError.* (\S+):(\d+)', records[0]).groups()
    source = inspect.getsource(conv).splitlines()
    call = next(i for i, text in enumerate(source, 1) if 'masks.build_masks(' in text)
    assert module == 'rt_dataprep.convert' and call <= int(line) <= call + 2  # the innermost frame inside rt_dataprep
    assert '/' not in records[0] and '.py' not in records[0] and SECRET not in records[0]


@pytest.mark.parametrize('exc', [KeyboardInterrupt, SystemExit])
def test_ctrl_c_and_exit_add_no_row_and_still_remove_the_partial(tree, monkeypatch, exc):
    org, out = tree
    two_patients(org)
    crash_in_masks_for(monkeypatch, PID_B, exc())
    with pytest.raises(exc):
        conv.convert(org, out)
    assert [p.name for p in out.iterdir()] == [PID]  # the first patient is done; no partial, no report


def test_an_output_side_oserror_is_still_an_io_error_row_and_the_run_goes_on(tree, monkeypatch):
    org, out = tree
    two_patients(org)
    real = conv.write_atomic

    def write(image, path):
        if PID_B in str(path):
            raise OSError(conv.REASON_WRITE)
        return real(image, path)
    monkeypatch.setattr(conv, 'write_atomic', write)
    rows = conv.convert(org, out).rows
    assert [(r.patient, r.outcome, r.reason) for r in rows] == [
        (PID, conv.OUTCOME_CONVERTED, ''), (PID_B, conv.OUTCOME_REFUSED, conv.REASON_IO_ERROR)]
