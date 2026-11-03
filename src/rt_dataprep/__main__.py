import argparse
import logging

from rt_dataprep.convert import REASON_OUTSIDE_ORG, convert
from rt_dataprep.organize import organize


def main(argv=None):
    parser = argparse.ArgumentParser(prog='rt_dataprep', description='Turn radiotherapy DICOM exports into ML-ready cases.')
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('organize', help='sort an export tree into <patient>/plan_<n>/ folders, one per RTPLAN')
    sub.add_argument('src', help='export root, one folder per patient')
    sub.add_argument('out', help='output root')
    sub = commands.add_parser('convert', help='turn an organized tree into NIfTI files on the native CT grid, one folder per plan')
    sub.add_argument('org', help='organized root, from the organize command')
    sub.add_argument('out', help='output root, outside the organized root')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        if args.command == 'convert':
            convert(args.org, args.out)
        else:
            organize(args.src, args.out)
    except ValueError as exc:  # the output root is inside the organized root; any other ValueError is a bug and propagates
        if str(exc) != REASON_OUTSIDE_ORG:
            raise
        parser.exit(1, f'rt_dataprep: {REASON_OUTSIDE_ORG}\n')
    except OSError:  # the message names a path, which can hold a patient id
        parser.exit(1, 'rt_dataprep: file system error on the source or output root\n')


if __name__ == '__main__':
    main()
