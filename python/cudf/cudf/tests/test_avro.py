# Copyright (c) 2019, NVIDIA CORPORATION.

from io import BytesIO

import fastavro as fa
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.fixture(scope="module")
def datadir(datadir):
    return datadir / "avro"


@pytest.fixture
def path_or_buf(datadir):
    fname = datadir / "example.avro"
    try:
        with open(fname, "rb") as f:
            buffer = BytesIO(f.read())
    except Exception as excpr:
        if type(excpr).__name__ == "FileNotFoundError":
            pytest.skip(".parquet file is not found")
        else:
            print(type(excpr).__name__)

    def _make_path_or_buf(src):
        if src == "filepath":
            return str(fname)
        if src == "pathobj":
            return fname
        if src == "bytes_io":
            return buffer
        if src == "bytes":
            return buffer.getvalue()
        if src == "url":
            return fname.as_uri()

        raise ValueError("Invalid source type")

    yield _make_path_or_buf


@pytest.mark.filterwarnings("ignore:Using CPU")
@pytest.mark.parametrize("engine", ["cudf"])
@pytest.mark.parametrize("inputfile, columns", [("example.avro", None)])
def test_avro_reader_basic(datadir, inputfile, columns, engine):
    path = datadir / inputfile
    try:
        reader = fa.reader(open(path, "rb"))
    except Exception as excpr:
        if type(excpr).__name__ == "FileNotFoundError":
            pytest.skip(".avro file is not found")
        else:
            print(type(excpr).__name__)

    expect = pd.DataFrame.from_records(reader)
    got = cudf.read_avro(path, engine=engine, columns=columns)

    # PANDAS uses NaN to represent invalid data, which forces float dtype
    # For comparison, we can replace NaN with 0 and cast to the cuDF dtype
    # FASTAVRO produces int64 columns from avro int32 dtype, so convert
    # it back to int32 here
    for col in expect.columns:
        expect[col] = expect[col].astype(got[col].dtype)

    # fastavro appears to return columns in reverse order
    # (actual order may depend on pandas/python version)
    assert_eq(expect, got[expect.columns], check_categorical=False)
