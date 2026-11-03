import argparse
import logging

from rt_dataprep.organize import organize


def main(argv=None):
    parser = argparse.ArgumentParser(prog='rt_dataprep', description='Turn radiotherapy DICOM exports into ML-ready cases.')
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('organize', help='sort an export tree into <patient>/fx_<n>/ folders, one per RTPLAN')
    sub.add_argument('src', help='export root, one folder per patient')
    sub.add_argument('out', help='output root')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    organize(args.src, args.out)


if __name__ == '__main__':
    main()
