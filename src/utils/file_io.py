import gzip
import json
import logging
from typing import List

from jsonlines import jsonlines

log = logging.getLogger(__name__)


def _auto_read_text(ext, f):
    if ext == "json":
        data = json.load(f)
    elif ext == "jsonl":
        data = [json.loads(line) for line in f]
    elif ext == "txt":
        data = f.read().splitlines()
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return data


def auto_read_text(path_to_file):
    ext = path_to_file.split(".")[-1]
    if ext == "gz":
        return auto_read_gzipped_text(path_to_file)
    with open(path_to_file, "r") as f:
        data = _auto_read_text(ext, f)
    log.debug(f"Loaded text from {path_to_file}")
    return data

def auto_read_gzipped_text(path_to_file):
    assert path_to_file.endswith(".gz"), f"File must be gzipped, i.e. end with .gz, but got {path_to_file}"
    ext = path_to_file.split(".")[-2]
    with gzip.open(path_to_file, "r") as f:
        data = _auto_read_text(ext, f)
    log.debug(f"Loaded text from {path_to_file}")
    return data



def _auto_write_text(ext, data, f):
    if ext == "json":
        json.dump(data, f)
    elif ext == "jsonl":
        json_writer = jsonlines.Writer(f)
        json_writer.write_all(data)
    elif ext == "txt":
        f.write("\n".join(data))
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def auto_write_text(path_to_file, data, mode="w"):
    ext = path_to_file.split(".")[-1]

    if ext == "gz":
        return auto_write_gzipped_text(path_to_file, data, mode=mode)

    with open(path_to_file, mode) as f:
        _auto_write_text(ext, data, f)
    log.debug(f"Wrote text to {path_to_file}")


def auto_write_gzipped_text(path_to_file, data, mode="wb"):

    assert path_to_file.endswith(".gz"), f"File must be gzipped, i.e. end with .gz, but got {path_to_file}"
    ext = path_to_file.split(".")[-2]
    with gzip.open(path_to_file, mode) as f:
        _auto_write_text(ext, data, f)
    log.debug(f"Wrote text to {path_to_file}")


def read_json(path_to_file):
    with open(path_to_file, "r") as f:
        data = json.load(f)
    log.debug(f"Loaded json from {path_to_file}")
    return data


def write_json(path_to_file, data, mode="w"):
    with open(path_to_file, mode) as f:
        json.dump(data, f)
    log.debug(f"Wrote json to {path_to_file}")


def read_jsonlines(path_to_file):
    with open(path_to_file, "r") as f:
        data = [json.loads(line) for line in f]
    log.debug(f"Loaded {len(data)} lines from {path_to_file}")
    return data


def write_jsonlines(path_to_file, data, mode="w"):
    with open(path_to_file, mode) as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")
    log.debug(f"Wrote {len(data)} lines to {path_to_file}")



def write_gzipped_jsonlines(path_to_file, data:List, mode="w"):
    """
    In case where the content has a lot of repeated strings, it is better to use gzip compression.
    The compression ratio can be as high as 90%.
    """
    with gzip.open(path_to_file, mode) as fp:
        json_writer = jsonlines.Writer(fp)
        json_writer.write_all(data)


def read_gzipped_jsonlines(path_to_file) -> List:
    with gzip.open(path_to_file, "r") as fp:
        json_reader = jsonlines.Reader(fp)
        return list(json_reader)
