"""Run the full imaging-only pipeline on small, generated CT/MR/PT phantoms. No clinical data or test imports.

From the checkout root: uv run python examples/synthetic_workflow.py /tmp/rt-dataprep-demo
The output directory must be new. See examples/README.md for expected results.
SPDX-License-Identifier: AGPL-3.0-only
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    MRImageStorage,
    PositronEmissionTomographyImageStorage,
    generate_uid,
)

from rt_dataprep.convert import convert
from rt_dataprep.manifest import manifest
from rt_dataprep.organize import organize
from rt_dataprep.preprocess import PreprocessSteps, preprocess
from rt_dataprep.split import SPLIT_NAMES, split

PATIENTS = 6
MODALITIES = {'CT': CTImageStorage, 'MR': MRImageStorage, 'PT': PositronEmissionTomographyImageStorage}


def write_series(folder, patient, modality, study_uid, frame_uid):
    """One synthetic 8 x 12 x 12 single-frame series, in patient coordinates with 1 x 1 x 2 mm spacing."""
    folder.mkdir(parents=True)
    series_uid = generate_uid()
    z, y, x = np.indices((8, 12, 12))
    values = (100 + 500 * (((x - 5.5) / 4) ** 2 + ((y - 5.5) / 4) ** 2 + ((z - 3.5) / 3) ** 2 < 1)).astype('<i2')
    for index, plane in enumerate(values):
        sop_uid = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MODALITIES[modality]
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = folder / f'slice_{index:02d}.dcm'
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = MODALITIES[modality], sop_uid
        ds.PatientID, ds.PatientName = patient, 'SYNTHETIC^Phantom'
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
        ds.Modality = modality
        ds.ManufacturerModelName = 'SYNTHETIC_PHANTOM'
        ds.SeriesDescription = f'SYNTHETIC_{modality}_PHANTOM'
        ds.StudyDate = '20240201' if modality == 'PT' else '20240101'
        ds.StudyTime = '090000'
        ds.SeriesDate, ds.SeriesTime = ds.StudyDate, ds.StudyTime
        ds.SeriesNumber, ds.InstanceNumber = list(MODALITIES).index(modality) + 1, index + 1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, index * 2]
        ds.PixelSpacing, ds.SliceThickness = [1, 1], 2
        ds.Rows, ds.Columns = plane.shape
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024 if modality == 'CT' else 0
        ds.PixelData = plane.tobytes()
        ds.save_as(path, enforce_file_format=True)


def run(workspace):
    """Create a new workspace, run each stage, and validate aggregate counts without printing metadata."""
    workspace.mkdir(parents=True, mode=0o700)  # no exist_ok: never overwrite or silently reuse a prior run
    source, organized = workspace / 'export', workspace / 'organized'
    converted, processed = workspace / 'converted', workspace / 'preprocessed'
    for number in range(1, PATIENTS + 1):
        patient = f'SYNTHETIC_P{number:03d}'
        study_uid, other_study_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
        for modality in MODALITIES:
            write_series(source / patient / modality, patient, modality,
                         other_study_uid if modality == 'PT' else study_uid, frame_uid)

    organization = organize(source, organized, workers=2)
    conversion = convert(organized, converted, workers=2)
    if any(row.outcome != 'organized' for row in organization.rows):
        raise RuntimeError('synthetic organization had an unexpected outcome')
    if any(row.outcome != 'converted' or row.skipped_series for row in conversion.rows):
        raise RuntimeError('synthetic conversion had an unexpected outcome')

    # All four geometric steps (the CLI's --initial-resample --rescale --resample --reshape), with a deliberately compact grid for a quick local example.
    steps = PreprocessSteps(resample=True, rescale=True, harmonize_spacing=True, reshape=True,
                            new_spacing=(1, 1, 2), final_spacing=(1.25, 1.25, 2), new_shape=(16, 16, 8))
    preprocessing = preprocess(converted, processed, steps)
    if any(row.outcome != 'preprocessed' for row in preprocessing.rows):
        raise RuntimeError('synthetic preprocessing had an unexpected outcome')
    for root in (converted, processed):
        written, refused = manifest(root)
        if (written, refused) != (PATIENTS * len(MODALITIES), 0):
            raise RuntimeError('synthetic manifest had unexpected case counts')
        cohorts = split(root / 'manifest.csv', test_size=0.2, val_size=0.2, seed=42)
        if [len(cohorts[name]) for name in SPLIT_NAMES] != [2, 2, 2]:
            raise RuntimeError('synthetic split had unexpected patient counts')
    print('Synthetic workflow complete: 6 patients, 18 imaging cases; train=2, val=2, test=2; no refusals.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path, help='new directory for synthetic inputs and all pipeline outputs')
    args = parser.parse_args()
    if args.workspace.exists():
        parser.error('the example needs a new workspace directory; existing output is never overwritten')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        run(args.workspace)
    except Exception as exc:
        parser.exit(1, f'example failed ({type(exc).__name__}); inspect reports in the selected workspace\n')


if __name__ == '__main__':
    main()
