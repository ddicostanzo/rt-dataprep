import argparse
import logging
import math
import sys
from pathlib import Path

from rt_dataprep.dicom_io import REASON_INPUT_INSIDE_OUTPUT
from rt_dataprep.convert import REASON_INTERNAL, REASON_OUTSIDE_ORG, ConvertAborted, convert, crash_site
from rt_dataprep.manifest import MANIFEST_NAME, manifest
from rt_dataprep.organize import REASON_INTERNAL as ORGANIZE_REASON_INTERNAL
from rt_dataprep.organize import REASON_LOCKED, REASON_OUTSIDE_SRC, OrganizeAborted, organize
from rt_dataprep.pool import MAX_WORKERS
from rt_dataprep.preprocess import (
    REASON_AHE_OPTIONS,
    REASON_MASK_RETENTION_LIMIT,
    REASON_NO_STEPS,
    REASON_OUTSIDE_INPUT,
    PreprocessAborted,
    PreprocessSteps,
    preprocess,
)
from rt_dataprep.preprocess import REASON_INTERNAL as PREPROCESS_REASON_INTERNAL
from rt_dataprep.split import REASON_COLUMNS, REASON_EXISTS, REASON_NO_TRAIN, REASON_OVERLAP, REASON_ROWS, REASON_SIZE, SPLIT_NAMES, split


PREPROCESS_HELP = 'resample, rescale and crop/pad a converted tree into a separate output tree; give at least one step flag to write anything'
WORKERS_HELP = ("number of patients processed in parallel, each in its own process (about 7 GB RAM each at 701x701x433); "
                "default 'auto' sizes it from the first processed patient's peak RAM (0.8 of available memory, capped by CPUs), "
                "an integer sets it; output is the same for any value")
OVERWRITE_HELP = "replace a patient's existing output folder once its new output holds at least one successful case"


def workers_arg(text):
    if text == 'auto':
        return text
    value = int(text) if text.isascii() and text.isdigit() else 0  # no sign, space, underscore or non-ASCII digit
    if not 1 <= value <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(f"must be 'auto' or an integer from 1 to {MAX_WORKERS}")
    return value


def positive_ints(text):
    value = int(text) if text.isascii() and text.isdigit() else 0
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def spacing_mm(text):
    value = float(text)  # a ValueError is argparse's "invalid value"
    if not 0.1 <= value < math.inf:  # also rejects nan; a finer grid only runs out of memory (#229)
        raise argparse.ArgumentTypeError('must be a finite spacing of at least 0.1 mm')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog='rt_dataprep', description='Turn radiotherapy DICOM exports into ML-ready cases.')
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('organize', help='sort an export tree into <patient>/plan_<n>/ folders, one per RTPLAN')
    sub.add_argument('src', help='export root, one folder per patient')
    sub.add_argument('out', help='output root')
    sub.add_argument('--workers', type=workers_arg, default='auto', help=WORKERS_HELP)
    sub.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
    sub = commands.add_parser('convert', help='turn an organized tree into NIfTI files on the native CT grid, one folder per plan and one per study')
    sub.add_argument('org', help='organized root, from the organize command')
    sub.add_argument('out', help='output root, outside the organized root')
    sub.add_argument('--workers', type=workers_arg, default='auto', help=WORKERS_HELP)
    sub.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
    sub = commands.add_parser('preprocess', help=PREPROCESS_HELP, description=PREPROCESS_HELP)
    sub.add_argument('converted', help='converted root, from the convert command')
    sub.add_argument('out', help='output root, outside the converted root')
    sub.add_argument('--workers', type=workers_arg, default='auto', help=WORKERS_HELP)
    sub.add_argument('--overwrite', action='store_true', help=OVERWRITE_HELP)
    sub.add_argument('--initial-resample', action='store_true', help='step: resample to 1 mm spacing (set it with --new-spacing); writes the initialresample_ prefix')
    sub.add_argument('--rescale', action='store_true', help='step: rescale a CT to (HU + 1024) / 4096; writes the rescale_ prefix; as the only step, MR/PT study series are refused')
    sub.add_argument('--ahe', action='store_true', help='step: CLAHE adaptive histogram equalization of every image volume; writes the ahe_ prefix')
    sub.add_argument('--resample', action='store_true', help='step: resample to 1.25 mm spacing (set it with --final-spacing); writes the resample_ prefix')
    sub.add_argument('--reshape', action='store_true', help='step: centred crop/pad to 701 x 701 x 433 voxels (set it with --new-shape); writes the reshape_ prefix')
    sub.add_argument('--new-shape', type=positive_ints, nargs=3, metavar=('X', 'Y', 'Z'), help='--reshape size in voxels (default 701 701 433); requires --reshape')
    sub.add_argument('--new-spacing', type=spacing_mm, nargs=3, metavar=('X', 'Y', 'Z'), help='--initial-resample spacing in mm (default 1 1 1); requires --initial-resample')
    sub.add_argument('--final-spacing', type=spacing_mm, nargs=3, metavar=('X', 'Y', 'Z'), help='--resample spacing in mm (default 1.25 1.25 1.25); requires --resample')
    sub.add_argument('--allow-crop', action='store_true',
                     help='crop a dose or mask that extends past the final grid, not refuse the plan; needs --initial-resample, --resample or --reshape')
    sub.add_argument('--ahe-kernel-mm', type=float, nargs=3, metavar=('X', 'Y', 'Z'), help='AHE window in physical mm; requires --ahe')
    sub.add_argument('--ahe-bins', type=int, default=None, help='AHE histogram bins (default 128, at least 2); requires --ahe')
    sub.add_argument('--ahe-clip-limit', type=float, default=None, help='AHE clip limit (default 0.05, from 0 to 1); requires --ahe')
    sub.add_argument('--legacy-linear-masks', action='store_true', help='resample masks linearly to float32, not nearest to uint8')
    sub.add_argument('--min-mask-retention', type=float, help='optional minimum output/input mask volume fraction, greater than 0 and at most 1')
    sub.add_argument('--series-fov-mask', action='store_true',
                     help='with a step flag, write uint8 FOV/padding masks for planning CT, study images and aligned plan series')
    sub = commands.add_parser('manifest', help='write manifest.jsonl in a converted or preprocessed root, one line per plan folder or study series')
    sub.add_argument('root', help='converted or preprocessed root; manifest.jsonl is written here')
    sub = commands.add_parser('split', help='write splits.json next to manifest.jsonl, whole patients per split')
    sub.add_argument('root', help='root holding manifest.jsonl; splits.json is written here')
    sub.add_argument('--test-size', type=float, required=True, help='fraction of patients in test, at least 0 and below 1')
    sub.add_argument('--val-size', type=float, required=True, help='fraction of patients in val, at least 0 and below 1')
    sub.add_argument('--seed', type=int, required=True, help='integer seed; the same seed gives the same split')
    sub.add_argument('--external', help='manifest.jsonl of another dataset; refused if it shares a patient with a split')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        if args.command == 'convert':
            convert(args.org, args.out, workers=args.workers, overwrite=args.overwrite)
        elif args.command == 'preprocess':
            if not args.ahe and any(v is not None for v in (args.ahe_kernel_mm, args.ahe_bins, args.ahe_clip_limit)):
                parser.error('AHE options require --ahe')
            for option, value, step in (('--new-shape', args.new_shape, args.reshape), ('--new-spacing', args.new_spacing, args.initial_resample),
                                        ('--final-spacing', args.final_spacing, args.resample)):
                if value is not None and not step:
                    parser.error(f'{option} requires its step flag')
            if args.allow_crop and not (args.initial_resample or args.resample or args.reshape):
                parser.error('--allow-crop requires --initial-resample, --resample or --reshape')
            grids = {name: tuple(value) for name, value in (('new_shape', args.new_shape), ('new_spacing', args.new_spacing),
                                                           ('final_spacing', args.final_spacing)) if value is not None}
            # flags are named after the output prefix: --initial-resample is the resample field, --resample is harmonize_spacing
            preprocess(args.converted, args.out, PreprocessSteps(
                resample=args.initial_resample, rescale=args.rescale, ahe=args.ahe, harmonize_spacing=args.resample, reshape=args.reshape,
                legacy_linear_masks=args.legacy_linear_masks, series_fov_mask=args.series_fov_mask,
                min_mask_retention=args.min_mask_retention, allow_crop=args.allow_crop, **grids,
                ahe_kernel_mm=None if args.ahe_kernel_mm is None else tuple(args.ahe_kernel_mm),
                ahe_bins=128 if args.ahe_bins is None else args.ahe_bins,
                ahe_clip_limit=0.05 if args.ahe_clip_limit is None else args.ahe_clip_limit), workers=args.workers, overwrite=args.overwrite)
        elif args.command == 'manifest':
            manifest(args.root)  # logs its own counts line
        elif args.command == 'split':
            splits = split(Path(args.root) / MANIFEST_NAME, test_size=args.test_size, val_size=args.val_size, seed=args.seed, external=args.external)
            for name in SPLIT_NAMES:  # counts only, never the ids
                print(f'{name}: {len(splits[name])} patients, {sum(map(len, splits[name].values()))} cases')
        else:
            organize(args.src, args.out, workers=args.workers, overwrite=args.overwrite)
    except ValueError as exc:  # a fixed refusal text; any other ValueError is a bug, reported as below
        if str(exc) not in (REASON_OUTSIDE_ORG, REASON_OUTSIDE_INPUT, REASON_NO_STEPS, REASON_AHE_OPTIONS,
                            REASON_MASK_RETENTION_LIMIT, REASON_OUTSIDE_SRC, REASON_INPUT_INSIDE_OUTPUT,
                            REASON_SIZE, REASON_NO_TRAIN, REASON_OVERLAP, REASON_EXISTS, REASON_COLUMNS, REASON_ROWS,
                            REASON_LOCKED):
            parser.exit(1, unexpected(exc))
        parser.exit(1, f'rt_dataprep: {exc}\n')
    except ConvertAborted:  # the report is written; the text is fixed, nothing of the original reaches the terminal
        parser.exit(1, f'rt_dataprep: {REASON_INTERNAL}\n')
    except OrganizeAborted:  # same as above, with organize's own fixed text
        parser.exit(1, f'rt_dataprep: {ORGANIZE_REASON_INTERNAL}\n')
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
