"""Read and write a plan's or study's series.json in tests (one place for the file's format)."""
import json

from rt_dataprep import convert as conv


def read(folder):
    """The objects of folder/series.json, as written."""
    with open(folder / conv.SERIES_NAME, encoding='utf-8') as handle:
        return json.load(handle)


def columns(row):
    """The SERIES_COLUMNS part of a convert object, in order: `dicom` (a dict on every object convert writes) is checked apart."""
    assert list(row)[-1] == 'dicom' and isinstance(row['dicom'], dict)
    return {column: row[column] for column in conv.SERIES_COLUMNS}


def write(rows, path):
    conv.write_series_json(rows, path)
