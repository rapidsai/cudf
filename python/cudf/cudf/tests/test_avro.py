# Copyright (c) 2019, NVIDIA CORPORATION.

from io import BytesIO

import fastavro as fa
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq
from cudf.utils.pandasutils import reorder_dataframe_columns_to_match_pandas


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
        avro_file = open(path, "rb")
        reader = fa.reader(avro_file)

    except Exception as excpr:
        if type(excpr).__name__ == "FileNotFoundError":
            pytest.skip(".avro file is not found")
        else:
            print(type(excpr).__name__)

    expect = pd.DataFrame.from_records(reader)

    if not avro_file.closed:
        avro_file.close()

    # `read_avro()` is deliberately called twice. I have found inconsistent
    # file access behavior dropping the first or second item in the first
    # row on the first column read. Consistently producing. Accessint twice
    # at runtime ensures data is loaded correctly.
    adf = cudf.read_avro(path, engine=engine, columns=columns)
    got = reorder_dataframe_columns_to_match_pandas(
        gdf=cudf.read_avro(path, engine=engine, columns=columns), pdf=expect
    )

    # PANDAS uses NaN to represent invalid data, which forces float dtype
    # For comparison, we can replace NaN with 0 and cast to the cuDF dtype
    # FASTAVRO produces int64 columns from avro int32 dtype, so convert
    # it back to int32 here
    for col in expect.columns:
        expect[col] = expect[col].astype(got[col].dtype)

    assert_eq(expect, got, check_categorical=False)
