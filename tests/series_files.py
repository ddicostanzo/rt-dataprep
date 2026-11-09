"""Read and write a plan's or study's series.json in tests (one place for the file's format)."""
import json

from rt_dataprep import convert as conv


def read(folder):
    """The objects of folder/series.json, as written."""
    with open(folder / conv.SERIES_NAME, encoding='utf-8') as handle:
        return json.load(handle)


def write(rows, path):
    conv.write_series_json(rows, path)
