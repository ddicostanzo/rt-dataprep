"""CLI (task-06, task-03 of F2b): `python -m rt_dataprep convert` and `preprocess`, always run as a subprocess. Every tree is synthetic and built in tmp_path."""
import csv
import dataclasses
import json
import re
import subprocess
import sys

import pytest
import SimpleITK as sitk
from test_convert import PID, make_plan
from test_preprocess import FOUR_STEPS, SMALL, report_rows, write_plan

from rt_dataprep import convert as conv
from rt_dataprep import preprocess as pre
from rt_dataprep import __main__ as cli
from rt_dataprep import split as sp
from rt_dataprep.manifest import MANIFEST_NAME


@pytest.fixture
def org(tmp_path):
    root = tmp_path / 'org'
    make_plan(root / PID / 'plan_1')
    return root


def run_convert(org, out):
    return subprocess.run([sys.executable, '-m', 'rt_dataprep', 'convert', str(org), str(out)], capture_output=True, text=True)


def test_a2_12_convert_runs_end_to_end_and_writes_the_report(tmp_path, org):
    out = tmp_path / 'out'
    result = run_convert(org, out)
    assert result.returncode == 0
    with open(out / conv.REPORT_NAME, newline='') as handle:
        assert [(r['patient'], r['folder'], r['outcome']) for r in csv.DictReader(handle)] == [(PID, 'plan_1', 'converted')]
    assert sorted(p.name for p in (out / PID / 'plan_1').iterdir()) == ['planning_ct.nii.gz', 'series.csv']
    assert PID not in result.stdout + result.stderr and str(tmp_path) not in result.stdout + result.stderr


@pytest.mark.parametrize('where', ['same', 'inside'])
def test_out_inside_org_exits_1_with_the_reason_and_no_path(tmp_path, org, where):
    out = org if where == 'same' else org / 'deeper' / 'out'
    result = run_convert(org, out)
    assert result.returncode == 1
    assert result.stderr == f'rt_dataprep: {conv.REASON_OUTSIDE_ORG}\n' and result.stdout == ''
    assert not (org / 'deeper').exists() and not (org / conv.REPORT_NAME).exists()


def test_oserror_exits_1_with_a_path_free_message(tmp_path, org):
    blocker = tmp_path / 'blocker'
    blocker.write_text('a file, not a folder')
    result = run_convert(org, blocker / 'out')  # the output root cannot be made
    assert result.returncode == 1
    assert result.stderr == 'rt_dataprep: file system error on the source or output root\n' and result.stdout == ''


# Runs the entry module as __main__ with build_masks patched to raise; argv comes after -c, so no path is embedded.
CRASH = """
import runpy, sys
from rt_dataprep import masks
def boom(*args, **kwargs):
    raise TypeError('1.2.3.4.5')
masks.build_masks = boom
sys.argv = ['rt_dataprep', 'convert', sys.argv[1], sys.argv[2]]
runpy.run_module('rt_dataprep', run_name='__main__')
"""


def test_an_unexpected_error_exits_1_and_the_last_stderr_line_is_the_fixed_text(tmp_path, org):
    out = tmp_path / 'out'
    result = subprocess.run([sys.executable, '-c', CRASH, str(org), str(out)], capture_output=True, text=True)
    assert result.returncode == 1
    lines = result.stderr.splitlines()
    assert lines[-1] == f'rt_dataprep: {conv.REASON_INTERNAL}' and result.stdout == ''
    assert [line for line in lines if '1.2.3.4.5' in line or str(tmp_path) in line] == []  # log lines stay, minus the secrets
    with open(out / conv.REPORT_NAME, newline='') as handle:
        assert [(r['outcome'], r['reason']) for r in csv.DictReader(handle)] == [(conv.OUTCOME_REFUSED, conv.REASON_INTERNAL)]


# F2b: the preprocess subcommand. The reshape step pads to 701 x 701 x 433, so these runs shrink new_shape inside the child process
# by wrapping preprocess() before main imports it (as CRASH patches build_masks); argv comes after -c, so no path is embedded.
# %s is extra patch code, run before main.
PRE_RUN = """
import dataclasses, runpy, sys
from rt_dataprep import preprocess
real = preprocess.preprocess
preprocess.preprocess = lambda converted, out, steps, **kw: real(converted, out, dataclasses.replace(steps, new_shape=%r), **kw)
%s
sys.argv = ['rt_dataprep', 'preprocess', *sys.argv[1:]]
runpy.run_module('rt_dataprep', run_name='__main__')
"""
FOUR_FLAGS = ('--initial-resample', '--rescale', '--resample', '--reshape')
PRE_CRASH = "def boom(*args, **kwargs):\n    raise TypeError('1.2.3.4.5')\npreprocess.process_ct = boom"


def run_pre(converted, out, *flags, patch=''):
    code = PRE_RUN % (SMALL.new_shape, patch)
    return subprocess.run([sys.executable, '-c', code, str(converted), str(out), *flags], capture_output=True, text=True)


def mask_pixel_type(out, steps):
    return sitk.ReadImage(str(out / 'SYNTHETICPTA' / 'plan_1' / pre.output_name('ptv.nii.gz', steps))).GetPixelID()


@pytest.fixture
def converted(tmp_path):
    root = tmp_path / 'converted'
    write_plan(root)
    return root


def test_preprocess_four_step_flags_run_end_to_end(tmp_path, converted):
    out = tmp_path / 'out'
    result = run_pre(converted, out, *FOUR_FLAGS)
    assert result.returncode == 0 and result.stdout == ''
    assert [(r['patient'], r['folder'], r['outcome']) for r in report_rows(out)] == [('SYNTHETICPTA', 'plan_1', pre.OUTCOME_PREPROCESSED)]
    names = {p.name for p in (out / 'SYNTHETICPTA' / 'plan_1').iterdir()}
    assert {'reshape_resample_rescale_initialresample_planning_ct.nii.gz', 'reshape_resample_initialresample_dose.nii.gz'} <= names
    assert mask_pixel_type(out, SMALL) == sitk.sitkUInt8
    # the PRE_RUN seam must shrink the output; a broken seam only shows up as a slow 701 x 701 x 433 run
    assert sitk.ReadImage(str(out / 'SYNTHETICPTA' / 'plan_1' / 'reshape_resample_rescale_initialresample_planning_ct.nii.gz')).GetSize() == SMALL.new_shape
    assert str(tmp_path) not in result.stderr


def test_legacy_linear_masks_flag_sets_the_field(tmp_path, converted):
    out = tmp_path / 'out'
    result = run_pre(converted, out, *FOUR_FLAGS, '--legacy-linear-masks')
    assert result.returncode == 0
    assert mask_pixel_type(out, dataclasses.replace(SMALL, legacy_linear_masks=True)) == sitk.sitkFloat32


def test_preprocess_without_a_step_flag_exits_1_with_no_steps(tmp_path, converted):
    out = tmp_path / 'out'
    for flags in ([], ['--legacy-linear-masks'], ['--series-fov-mask'], ['--legacy-linear-masks', '--series-fov-mask']):
        result = subprocess.run([sys.executable, '-m', 'rt_dataprep', 'preprocess', str(converted), str(out), *flags], capture_output=True, text=True)
        assert result.returncode == 1
        assert result.stderr == f'rt_dataprep: {pre.REASON_NO_STEPS}\n' and result.stdout == ''
        assert not out.exists()


@pytest.mark.parametrize('where', ['same', 'inside'])
def test_preprocess_out_inside_the_input_exits_1_with_the_reason(tmp_path, converted, where):
    out = converted if where == 'same' else converted / 'deeper' / 'out'
    result = run_pre(converted, out, *FOUR_FLAGS)
    assert result.returncode == 1
    assert result.stderr == f'rt_dataprep: {pre.REASON_OUTSIDE_INPUT}\n' and result.stdout == ''
    assert not (converted / 'deeper').exists() and not (converted / pre.REPORT_NAME).exists()


def test_preprocess_oserror_exits_1_with_a_path_free_message(tmp_path, converted):
    blocker = tmp_path / 'blocker'
    blocker.write_text('a file, not a folder')
    result = run_pre(converted, blocker / 'out', *FOUR_FLAGS)
    assert result.returncode == 1
    assert result.stderr == 'rt_dataprep: file system error on the source or output root\n' and result.stdout == ''


def test_preprocess_unexpected_error_exits_1_with_its_own_fixed_text(tmp_path, converted):
    out = tmp_path / 'out'
    result = run_pre(converted, out, *FOUR_FLAGS, patch=PRE_CRASH)
    assert result.returncode == 1 and result.stdout == ''
    lines = result.stderr.splitlines()
    assert lines[-1] == f'rt_dataprep: {pre.REASON_INTERNAL}' and pre.REASON_INTERNAL != conv.REASON_INTERNAL
    assert [line for line in lines if '1.2.3.4.5' in line or str(tmp_path) in line] == []
    assert [(r['outcome'], r['reason']) for r in report_rows(out)] == [(conv.OUTCOME_REFUSED, pre.REASON_INTERNAL)]


def test_preprocess_help_lists_the_five_step_flags_with_their_values_and_prefixes():
    result = subprocess.run([sys.executable, '-m', 'rt_dataprep', 'preprocess', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    text = ' '.join(result.stdout.split())
    assert 'at least one step flag' in text and '--legacy-rnt' not in text
    for flag, value, prefix in [('--initial-resample', '1 mm', 'initialresample_'), ('--rescale', '(HU + 1024) / 4096', 'rescale_'),
                                ('--ahe', 'CLAHE', 'ahe_'), ('--resample', '1.25 mm', 'resample_'), ('--reshape', '701 x 701 x 433', 'reshape_')]:
        entry = text[text.index(f'{flag} ') + len(flag):]
        entry = entry[:entry.index(' --')] if ' --' in entry else entry  # this flag's help, up to the next option
        assert value in entry and prefix in entry, flag


def test_preprocess_ahe_alone_equalizes_the_ct_and_copies_dose_and_masks(tmp_path, converted):
    out = tmp_path / 'out'
    result = run_pre(converted, out, '--ahe')
    assert result.returncode == 0 and result.stdout == '' and str(tmp_path) not in result.stderr
    src, dst = converted / 'SYNTHETICPTA' / 'plan_1', out / 'SYNTHETICPTA' / 'plan_1'
    assert [(r['outcome']) for r in report_rows(out)] == [pre.OUTCOME_PREPROCESSED]
    assert (dst / f'ahe_{conv.CT_NAME}').is_file() and not (dst / conv.CT_NAME).exists()
    for name in (conv.DOSE_NAME, 'ptv.nii.gz'):  # geometry-free run: dose and masks are copied as they are
        assert (dst / name).read_bytes() == (src / name).read_bytes()


def test_preprocess_ahe_with_the_four_step_flags_writes_the_full_ct_name(tmp_path, converted):
    out = tmp_path / 'out'
    result = run_pre(converted, out, '--ahe', *FOUR_FLAGS)
    assert result.returncode == 0 and result.stdout == ''
    names = {p.name for p in (out / 'SYNTHETICPTA' / 'plan_1').iterdir()}
    assert f'reshape_resample_ahe_rescale_initialresample_{conv.CT_NAME}' in names
    assert f'reshape_resample_rescale_initialresample_{conv.CT_NAME}' not in names


@pytest.mark.parametrize('flags, want', [
    (['--ahe'], pre.PreprocessSteps(ahe=True)),
    (['--ahe', '--series-fov-mask'], pre.PreprocessSteps(ahe=True, series_fov_mask=True)),
    ([*FOUR_FLAGS, '--ahe'], dataclasses.replace(FOUR_STEPS, ahe=True)),
    (list(FOUR_FLAGS), FOUR_STEPS),
])
def test_ahe_flag_reaches_preprocess_steps(flags, want, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, 'preprocess', lambda *args, **kwargs: seen.append(args))
    cli.main(['preprocess', 'a', 'b', *flags])
    assert seen == [('a', 'b', want)]
    assert seen[0][2].ahe is ('--ahe' in flags)


# Flags are named after the output prefix, not the field: --resample is harmonize_spacing (1.25 mm), --initial-resample is resample (1 mm).
STEP_FIELDS = {'--initial-resample': 'resample', '--rescale': 'rescale', '--ahe': 'ahe', '--resample': 'harmonize_spacing', '--reshape': 'reshape'}


@pytest.mark.parametrize('flags', [[flag] for flag in STEP_FIELDS] + [list(STEP_FIELDS)])
def test_each_step_flag_sets_exactly_its_field(flags, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, 'preprocess', lambda *args, **kwargs: seen.append(args))
    cli.main(['preprocess', 'a', 'b', *flags])
    assert seen == [('a', 'b', pre.PreprocessSteps(**{STEP_FIELDS[flag]: True for flag in flags}))]


def test_legacy_rnt_is_an_argparse_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['preprocess', 'a', 'b', '--legacy-rnt'])
    assert exc.value.code == 2 and 'unrecognized arguments: --legacy-rnt' in capsys.readouterr().err


# convert itself raises outside the patient loop, so the CLI's own handler reports it; argv comes after -c.
BUG = """
import runpy, sys
from rt_dataprep import convert
def bug(*args, **kwargs):
    raise %s('SYNTHETIC_ID in ' + sys.argv[1])
convert.convert = bug
sys.argv = ['rt_dataprep', 'convert', sys.argv[1], sys.argv[2]]
runpy.run_module('rt_dataprep', run_name='__main__')
"""


@pytest.mark.parametrize('kind', ['ValueError', 'TypeError'])
def test_an_unexpected_exception_prints_its_type_and_site_never_its_message(tmp_path, org, kind):
    result = subprocess.run([sys.executable, '-c', BUG % kind, str(org), str(tmp_path / 'out')], capture_output=True, text=True)
    assert result.returncode == 1 and result.stdout == ''
    assert re.fullmatch(r'rt_dataprep: unexpected error \(' + kind + r' at [\w.]+:\d+\)\n', result.stderr)
    assert 'SYNTHETIC_ID' not in result.stderr and str(tmp_path) not in result.stderr


# F3: the manifest and split subcommands, called as cli.main([...]); capfd reads the file descriptors.
NAMES = [f'SYNTHETIC_P{i}' for i in range(5)]


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / 'converted'
    for name in NAMES:
        write_plan(root, patient=name)
    return root


def split_args(root, *extra, test_size='0.2', val_size='0.2', seed='0'):
    return ['split', str(root), '--test-size', test_size, '--val-size', val_size, '--seed', seed, *extra]


def refusal(capsys_result, reason):
    assert capsys_result.out == '' and capsys_result.err.splitlines()[-1] == f'rt_dataprep: {reason}'


def test_manifest_then_split_run_end_to_end_and_print_counts_only(tree, capfd):
    cli.main(['manifest', str(tree)])
    first = capfd.readouterr()
    assert first.out == 'manifest: 5 written, 0 refused\n'
    cli.main(split_args(tree))
    second = capfd.readouterr()
    assert second.out == 'train: 3 patients, 3 cases\nval: 1 patients, 1 cases\ntest: 1 patients, 1 cases\n'
    with open(tree / MANIFEST_NAME, newline='') as handle:
        assert sorted(r['patient'] for r in csv.DictReader(handle)) == NAMES
    splits = json.loads((tree / sp.SPLITS_NAME).read_text())
    assert sorted(p for name in sp.SPLIT_NAMES for p in splits[name]) == NAMES
    for out in (first, second):
        assert 'SYNTHETIC' not in out.out + out.err and str(tree) not in out.out + out.err and '/plan_' not in out.out + out.err


def test_seed_test_size_and_val_size_are_required(tree, capfd):
    for missing in ('--seed', '--test-size', '--val-size'):
        argv = split_args(tree)
        del argv[argv.index(missing):argv.index(missing) + 2]
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2
        capfd.readouterr()
    assert not (tree / sp.SPLITS_NAME).exists()


def test_overlapping_external_exits_1_with_the_fixed_overlap_text(tree, tmp_path, capfd):
    cli.main(['manifest', str(tree)])
    other = tmp_path / 'other'
    for name in NAMES:
        write_plan(other, patient=name)
    cli.main(['manifest', str(other)])
    capfd.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(split_args(tree, '--external', str(other / MANIFEST_NAME)))
    assert exc.value.code == 1
    out = capfd.readouterr()
    refusal(out, sp.REASON_OVERLAP)
    assert 'SYNTHETIC' not in out.out + out.err and not (tree / sp.SPLITS_NAME).exists()


def test_a_disjoint_external_manifest_splits(tree, tmp_path, capfd):
    other = tmp_path / 'other'
    write_plan(other, patient='SYNTHETIC_OTHER')
    cli.main(['manifest', str(tree)])
    cli.main(['manifest', str(other)])
    capfd.readouterr()
    cli.main(split_args(tree, '--external', str(other / MANIFEST_NAME)))
    assert (tree / sp.SPLITS_NAME).exists()


@pytest.mark.parametrize(('kwargs', 'reason'), [
    ({'test_size': '1'}, sp.REASON_SIZE), ({'val_size': '-0.1'}, sp.REASON_SIZE),
    ({'test_size': '0.5', 'val_size': '0.5'}, sp.REASON_NO_TRAIN)])
def test_split_refusals_exit_1_with_their_fixed_text(tree, capfd, kwargs, reason):
    cli.main(['manifest', str(tree)])
    capfd.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(split_args(tree, **kwargs))
    assert exc.value.code == 1
    out = capfd.readouterr()
    refusal(out, reason)
    assert 'SYNTHETIC' not in out.out + out.err and not (tree / sp.SPLITS_NAME).exists()


def test_a_different_existing_splits_file_is_refused(tree, capfd):
    cli.main(['manifest', str(tree)])
    cli.main(split_args(tree))
    capfd.readouterr()
    cli.main(split_args(tree))  # the same arguments rewrite the same bytes: fine
    capfd.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(split_args(tree, seed='1'))
    assert exc.value.code == 1
    result = capfd.readouterr()
    refusal(result, sp.REASON_EXISTS)
    assert 'SYNTHETIC_' not in result.out + result.err


def test_a_manifest_without_the_patient_column_prints_the_refusal(tree, capfd):
    (tree / 'manifest.csv').write_text('case_dir\nSYNTHETIC_P0/plan_1\n')
    with pytest.raises(SystemExit) as exc:
        cli.main(split_args(tree, seed='-1'))
    assert exc.value.code == 1
    result = capfd.readouterr()
    refusal(result, sp.REASON_COLUMNS)
    assert 'SYNTHETIC_' not in result.out + result.err


def test_split_without_a_manifest_exits_1_with_the_path_free_message(tree, capfd):
    with pytest.raises(SystemExit) as exc:
        cli.main(split_args(tree))
    assert exc.value.code == 1
    out = capfd.readouterr()
    assert out.out == '' and out.err.splitlines()[-1] == 'rt_dataprep: file system error on the source or output root'
    assert str(tree) not in out.err


# --workers on organize and convert: the names __main__ imports are patched to record their kwargs.
@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, 'organize', lambda *args, **kwargs: seen.append(('organize', args, kwargs)))
    monkeypatch.setattr(cli, 'convert', lambda *args, **kwargs: seen.append(('convert', args, kwargs)))
    return seen


@pytest.mark.parametrize('flags, want', [
    (list(FOUR_FLAGS), FOUR_STEPS),
    ([*FOUR_FLAGS, '--series-fov-mask'], dataclasses.replace(FOUR_STEPS, series_fov_mask=True)),
    ([*FOUR_FLAGS, '--legacy-linear-masks', '--series-fov-mask'],
     dataclasses.replace(FOUR_STEPS, legacy_linear_masks=True, series_fov_mask=True)),
])
def test_series_fov_mask_flag_reaches_preprocess_steps(flags, want, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, 'preprocess', lambda *args, **kwargs: seen.append(args))
    cli.main(['preprocess', 'a', 'b', *flags])
    assert seen == [('a', 'b', want)]
    assert seen[0][2].series_fov_mask is ('--series-fov-mask' in flags)


@pytest.mark.parametrize('command', ['organize', 'convert'])
@pytest.mark.parametrize('flag, workers', [([], 1), (['--workers', '3'], 3)])
def test_workers_reaches_organize_and_convert_and_defaults_to_one(command, flag, workers, calls):
    cli.main([command, 'a', 'b', *flag])
    assert calls == [(command, ('a', 'b'), {'workers': workers, 'overwrite': False})]


@pytest.mark.parametrize('command', ['organize', 'convert'])
def test_workers_accepts_the_cap(command, calls):
    cli.main([command, 'a', 'b', '--workers', '64'])
    assert calls == [(command, ('a', 'b'), {'workers': 64, 'overwrite': False})]


@pytest.mark.parametrize('command', ['organize', 'convert'])
@pytest.mark.parametrize('value', ['0', '-1', 'x', '1.5', '1_0', '+3', '٣', ' 3', '65', '1000000'])
def test_workers_below_one_or_not_an_integer_is_a_usage_error(command, value, calls, capfd):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, 'a', 'b', '--workers', value])
    assert exc.value.code == 2 and calls == []
    assert '--workers' in capfd.readouterr().err


@pytest.mark.parametrize('argv', [['preprocess', 'a', 'b'], ['manifest', 'a'], ['split', 'a', '--test-size', '0', '--val-size', '0', '--seed', '0']])
def test_no_other_subcommand_takes_workers(argv, calls, capfd):
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, '--workers', '2'])
    assert exc.value.code == 2 and calls == []
    capfd.readouterr()


@pytest.mark.parametrize('command', ['organize', 'convert'])
def test_workers_help_says_threads_and_same_output(command, capfd):
    with pytest.raises(SystemExit):
        cli.main([command, '--help'])
    text = ' '.join(capfd.readouterr().out.split())
    assert 'number of threads reading headers; output is the same for any value' in text


# --overwrite on organize, convert and preprocess: passed through as a keyword; split and manifest reject it.
@pytest.mark.parametrize('command', ['organize', 'convert', 'preprocess'])
@pytest.mark.parametrize('flag, want', [([], False), (['--overwrite'], True)])
def test_overwrite_reaches_each_stage_and_defaults_to_false(command, flag, want, calls, monkeypatch):
    monkeypatch.setattr(cli, 'preprocess', lambda *args, **kwargs: calls.append(('preprocess', args[:2], kwargs)))
    cli.main([command, 'a', 'b', *flag])
    assert len(calls) == 1 and calls[0][0] == command and calls[0][1] == ('a', 'b') and calls[0][2]['overwrite'] is want


@pytest.mark.parametrize('argv', [['manifest', 'a'], ['split', 'a', '--test-size', '0', '--val-size', '0', '--seed', '0']])
def test_manifest_and_split_reject_overwrite(argv, calls, capfd):
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, '--overwrite'])
    assert exc.value.code == 2 and calls == []
    assert 'unrecognized arguments: --overwrite' in capfd.readouterr().err


@pytest.mark.parametrize('command', ['organize', 'convert', 'preprocess'])
def test_overwrite_help_lists_the_flag(command, capfd):
    with pytest.raises(SystemExit):
        cli.main([command, '--help'])
    text = ' '.join(capfd.readouterr().out.split())
    assert '--overwrite' in text and 'replace a patient\'s existing output folder once its new output holds at least one successful case' in text
