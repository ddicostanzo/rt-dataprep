"""Preprocess an existing converted tree with an explicit, configurable grid, then write its manifest.

SPDX-License-Identifier: AGPL-3.0-only
"""
import argparse
import logging
from pathlib import Path

from rt_dataprep.manifest import manifest
from rt_dataprep.preprocess import PreprocessSteps, preprocess


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('dimensions must be positive integers')
    return number


def positive_float(value):
    number = float(value)
    if not 0 < number < float('inf'):
        raise argparse.ArgumentTypeError('spacing must be finite and positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('converted', type=Path, help='existing converted input tree')
    parser.add_argument('out', type=Path, help='new output root outside the input tree')
    parser.add_argument('--shape', type=positive_int, nargs=3, default=(128, 128, 96), metavar=('X', 'Y', 'Z'))
    parser.add_argument('--spacing', type=positive_float, nargs=3, default=(1.25, 1.25, 1.25), metavar=('X', 'Y', 'Z'))
    parser.add_argument('--series-fov-mask', action='store_true', help='add field-of-view masks for aligned plan series')
    parser.add_argument('--ahe', action='store_true', help='also apply adaptive histogram equalization to image volumes')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('choose a new output root; this example never reuses an existing destination')
    if args.out.resolve().is_relative_to(args.converted.resolve()):
        parser.error('output must be outside the converted input tree')
    if not args.converted.is_dir():
        parser.error('the converted input must be an existing directory')
    steps = PreprocessSteps(resample=True, rescale=True, harmonize_spacing=True, reshape=True,
                            new_spacing=(1, 1, 1), final_spacing=tuple(args.spacing), new_shape=tuple(args.shape),
                            series_fov_mask=args.series_fov_mask, ahe=args.ahe)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        result = preprocess(args.converted, args.out, steps)
        written, refused = manifest(args.out)
    except Exception as exc:
        parser.exit(1, f'example failed ({type(exc).__name__}); inspect reports in approved storage\n')
    if not result.rows or any(row.outcome != 'preprocessed' or row.skipped_series for row in result.rows) or refused:
        parser.exit(1, 'no cases were processed, or cases/registered series were refused or skipped; inspect the reports\n')
    print(f'Custom preprocessing complete: {written} manifest cases, {refused} manifest refusals.')


if __name__ == '__main__':
    main()
