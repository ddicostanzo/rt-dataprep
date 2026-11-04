import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from rt_dataprep.convert import REASON_INTERNAL, REASON_OUTSIDE_ORG, ConvertAborted, convert, crash_site
from rt_dataprep.manifest import MANIFEST_NAME, manifest
from rt_dataprep.organize import REASON_OUTSIDE_SRC, organize
from rt_dataprep.preprocess import (
    LEGACY_RNT,
    REASON_NO_STEPS,
    REASON_OUTSIDE_INPUT,
    PreprocessAborted,
    PreprocessSteps,
    preprocess,
)
from rt_dataprep.preprocess import REASON_INTERNAL as PREPROCESS_REASON_INTERNAL
from rt_dataprep.split import REASON_COLUMNS, REASON_EXISTS, REASON_NO_TRAIN, REASON_OVERLAP, REASON_SIZE, SPLIT_NAMES, split


PREPROCESS_HELP = 'resample, rescale and crop/pad a converted tree into a separate output tree; --legacy-rnt is required for any output'


def main(argv=None):
    parser = argparse.ArgumentParser(prog='rt_dataprep', description='Turn radiotherapy DICOM exports into ML-ready cases.')
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('organize', help='sort an export tree into <patient>/plan_<n>/ folders, one per RTPLAN')
    sub.add_argument('src', help='export root, one folder per patient')
    sub.add_argument('out', help='output root')
    sub = commands.add_parser('convert', help='turn an organized tree into NIfTI files on the native CT grid, one folder per plan')
    sub.add_argument('org', help='organized root, from the organize command')
    sub.add_argument('out', help='output root, outside the organized root')
    sub = commands.add_parser('preprocess', help=PREPROCESS_HELP, description=PREPROCESS_HELP)
    sub.add_argument('converted', help='converted root, from the convert command')
    sub.add_argument('out', help='output root, outside the converted root')
    sub.add_argument('--legacy-rnt', action='store_true', help='the legacy RNT steps; required for any output')
    sub.add_argument('--legacy-linear-masks', action='store_true', help='resample masks linearly to float32, not nearest to uint8')
    sub = commands.add_parser('manifest', help='write manifest.csv in a converted or preprocessed root, one row per plan folder')
    sub.add_argument('root', help='converted or preprocessed root; manifest.csv is written here')
    sub = commands.add_parser('split', help='write splits.json next to manifest.csv, whole patients per split')
    sub.add_argument('root', help='root holding manifest.csv; splits.json is written here')
    sub.add_argument('--test-size', type=float, required=True, help='fraction of patients in test, at least 0 and below 1')
    sub.add_argument('--val-size', type=float, required=True, help='fraction of patients in val, at least 0 and below 1')
    sub.add_argument('--seed', type=int, required=True, help='integer seed; the same seed gives the same split')
    sub.add_argument('--external', help='manifest.csv of another dataset; refused if it shares a patient with a split')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        if args.command == 'convert':
            convert(args.org, args.out)
        elif args.command == 'preprocess':
            steps = LEGACY_RNT if args.legacy_rnt else PreprocessSteps()
            preprocess(args.converted, args.out, dataclasses.replace(steps, legacy_linear_masks=args.legacy_linear_masks))
        elif args.command == 'manifest':
            written, refused = manifest(args.root)
            print(f'manifest: {written} written, {refused} refused')
        elif args.command == 'split':
            splits = split(Path(args.root) / MANIFEST_NAME, test_size=args.test_size, val_size=args.val_size, seed=args.seed, external=args.external)
            for name in SPLIT_NAMES:  # counts only, never the ids
                print(f'{name}: {len(splits[name])} patients, {sum(map(len, splits[name].values()))} cases')
        else:
            organize(args.src, args.out)
    except ValueError as exc:  # a fixed refusal text; any other ValueError is a bug, reported as below
        if str(exc) not in (REASON_OUTSIDE_ORG, REASON_OUTSIDE_INPUT, REASON_NO_STEPS, REASON_OUTSIDE_SRC,
                            REASON_SIZE, REASON_NO_TRAIN, REASON_OVERLAP, REASON_EXISTS, REASON_COLUMNS):
            parser.exit(1, unexpected(exc))
        parser.exit(1, f'rt_dataprep: {exc}\n')
    except ConvertAborted:  # the report is written; the text is fixed, nothing of the original reaches the terminal
        parser.exit(1, f'rt_dataprep: {REASON_INTERNAL}\n')
    except PreprocessAborted:  # same as above, with preprocess's own fixed text
        parser.exit(1, f'rt_dataprep: {PREPROCESS_REASON_INTERNAL}\n')
    except OSError:  # the message names a path, which can hold a patient id
        parser.exit(1, 'rt_dataprep: file system error on the source or output root\n')
    except Exception as exc:  # a bug: its type and site only, never its message, which may name a path (#26)
        parser.exit(1, unexpected(exc))


def unexpected(exc):
    """Fixed text for an unexpected exception being handled: its type and innermost rt_dataprep site, no message."""
    module, line = crash_site(sys.exc_info()[2])
    return f'rt_dataprep: unexpected error ({getattr(type(exc), "__name__")} at {module}:{line})\n'  # noqa: B009 (no dunder access)


if __name__ == '__main__':
    main()
