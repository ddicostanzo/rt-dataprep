import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog='rt_dataprep', description='Turn radiotherapy DICOM exports into ML-ready cases.')
    commands = parser.add_subparsers(dest='command', required=True)
    organize = commands.add_parser('organize', help='sort an export tree into <patient>/fx_<n>/ folders, one per RTPLAN')
    organize.add_argument('src', help='export root, one folder per patient')
    organize.add_argument('out', help='output root')
    parser.parse_args(argv)
    parser.error('organize is not wired up yet')  # task-02 imports and calls organize here


if __name__ == '__main__':
    main()
