import argparse
import dataclasses
import logging
import sys

from rt_dataprep.convert import REASON_INTERNAL, REASON_OUTSIDE_ORG, ConvertAborted, convert, crash_site
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        if args.command == 'convert':
            convert(args.org, args.out)
        elif args.command == 'preprocess':
            steps = LEGACY_RNT if args.legacy_rnt else PreprocessSteps()
            preprocess(args.converted, args.out, dataclasses.replace(steps, legacy_linear_masks=args.legacy_linear_masks))
        else:
            organize(args.src, args.out)
    except ValueError as exc:  # a fixed refusal text; any other ValueError is a bug, reported as below
        if str(exc) not in (REASON_OUTSIDE_ORG, REASON_OUTSIDE_INPUT, REASON_NO_STEPS, REASON_OUTSIDE_SRC):
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
