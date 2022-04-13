# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import datetime
import math
import os
import pathlib
import random
from contextlib import contextmanager
from io import BytesIO
from string import ascii_letters

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.core import get_fs_token_paths
from packaging import version
from pyarrow import fs as pa_fs, parquet as pq

import cudf
from cudf.io.parquet import (
    ParquetDatasetWriter,
    ParquetWriter,
    merge_parquet_filemetadata,
)
from cudf.testing import dataset_generator as dg
from cudf.testing._utils import (
    TIMEDELTA_TYPES,
    assert_eq,
    assert_exceptions_equal,
    set_random_null_mask_inplace,
)


@contextmanager
def _hide_pyarrow_parquet_cpu_warnings(engine):
    if engine == "pyarrow":
        with pytest.warns(
            UserWarning,
            match="Using CPU via PyArrow to read Parquet dataset. This option "
            "is both inefficient and unstable!",
        ):
            yield
    else:
        yield


@pytest.fixture(scope="module")
def datadir(datadir):
    return datadir / "parquet"


@pytest.fixture(params=[1, 5, 10, 100000])
def simple_pdf(request):
    types = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        # "uint32", pandas promotes uint32 to int64
        # https://issues.apache.org/jira/browse/ARROW-9215
        "uint64",
        "float32",
        "float64",
    ]
    renamer = {
        "C_l0_g" + str(idx): "col_" + val for (idx, val) in enumerate(types)
    }
    typer = {"col_" + val: val for val in types}
    ncols = len(types)
    nrows = request.param

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd._testing.makeCustomDataframe(
        nrows=nrows, ncols=ncols, data_gen_f=lambda r, c: r, r_idx_type="i"
    )
    # Delete the name of the column index, and rename the row index
    test_pdf.columns.name = None
    test_pdf.index.name = "test_index"

    # Cast all the column dtypes to objects, rename them, and then cast to
    # appropriate types
    test_pdf = test_pdf.astype("object").rename(renamer, axis=1).astype(typer)

    return test_pdf


@pytest.fixture
def simple_gdf(simple_pdf):
    return cudf.DataFrame.from_pandas(simple_pdf)


def build_pdf(num_columns, day_resolution_timestamps):
    types = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        # "uint32", pandas promotes uint32 to int64
        # https://issues.apache.org/jira/browse/ARROW-9215
        "uint64",
        "float32",
        "float64",
        "datetime64[ms]",
        "datetime64[us]",
        "str",
    ]
    renamer = {
        "C_l0_g" + str(idx): "col_" + val for (idx, val) in enumerate(types)
    }
    typer = {"col_" + val: val for val in types}
    ncols = len(types)
    nrows = num_columns.param

    # Create a pandas dataframe with random data of mixed types
    test_pdf = pd._testing.makeCustomDataframe(
        nrows=nrows, ncols=ncols, data_gen_f=lambda r, c: r, r_idx_type="i"
    )
    # Delete the name of the column index, and rename the row index
    test_pdf.columns.name = None
    test_pdf.index.name = "test_index"

    # Cast all the column dtypes to objects, rename them, and then cast to
    # appropriate types
    test_pdf = test_pdf.rename(renamer, axis=1).astype(typer)

    # make datetime64's a little more interesting by increasing the range of
    # dates note that pandas will convert these to ns timestamps, so care is
    # taken to avoid overflowing a ns timestamp. There is also the ability to
    # request timestamps be whole days only via `day_resolution_timestamps`.
    for t in [
        {
            "name": "datetime64[ms]",
            "nsDivisor": 1000000,
            "dayModulus": 86400000,
        },
        {
            "name": "datetime64[us]",
            "nsDivisor": 1000,
            "dayModulus": 86400000000,
        },
    ]:
        data = [
            np.random.randint(0, (0x7FFFFFFFFFFFFFFF / t["nsDivisor"]))
            for i in range(nrows)
        ]
        if day_resolution_timestamps:
            data = [int(d / t["dayModulus"]) * t["dayModulus"] for d in data]
        test_pdf["col_" + t["name"]] = pd.Series(
            np.asarray(data, dtype=t["name"])
        )

    # Create non-numeric categorical data otherwise parquet may typecast it
    data = [ascii_letters[np.random.randint(0, 52)] for i in range(nrows)]
    test_pdf["col_category"] = pd.Series(data, dtype="category")

    # Create non-numeric str data
    data = [ascii_letters[np.random.randint(0, 52)] for i in range(nrows)]
    test_pdf["col_str"] = pd.Series(data, dtype="str")

    return test_pdf


@pytest.fixture(params=[0, 1, 10, 10000])
def pdf(request):
    return build_pdf(request, False)


@pytest.fixture(params=[0, 1, 10, 10000])
def pdf_day_timestamps(request):
    return build_pdf(request, True)


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame.from_pandas(pdf)


@pytest.fixture
def gdf_day_timestamps(pdf_day_timestamps):
    return cudf.DataFrame.from_pandas(pdf_day_timestamps)


@pytest.fixture(params=["snappy", "gzip", "brotli", None, np.str_("snappy")])
def parquet_file(request, tmp_path_factory, pdf):
    fname = tmp_path_factory.mktemp("parquet") / (
        str(request.param) + "_test.parquet"
    )
    pdf.to_parquet(fname, engine="pyarrow", compression=request.param)
    return fname


@pytest.fixture(scope="module")
def rdg_seed():
    return int(os.environ.get("TEST_CUDF_RDG_SEED", "42"))


def make_pdf(nrows, ncolumns=1, nvalids=0, dtype=np.int64):
    test_pdf = pd._testing.makeCustomDataframe(
        nrows=nrows,
        ncols=1,
        data_gen_f=lambda r, c: r,
        dtype=dtype,
        r_idx_type="i",
    )
    test_pdf.columns.name = None

    # Randomly but reproducibly mark subset of rows as invalid
    random.seed(1337)
    mask = random.sample(range(nrows), nvalids)
    test_pdf[test_pdf.index.isin(mask)] = np.NaN

    return test_pdf


@pytest.fixture
def parquet_path_or_buf(datadir):
    fname = datadir / "spark_timestamp.snappy.parquet"
    try:
        with open(fname, "rb") as f:
            buffer = BytesIO(f.read())
    except Exception as excpr:
        if type(excpr).__name__ == "FileNotFoundError":
            pytest.skip(".parquet file is not found")
        else:
            print(type(excpr).__name__)

    def _make_parquet_path_or_buf(src):
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

    yield _make_parquet_path_or_buf


@pytest.fixture(scope="module")
def large_int64_gdf():
    return cudf.DataFrame.from_pandas(pd.DataFrame({"col": range(0, 1 << 20)}))


def test_parquet_reader_unsupported_compression(datadir):
    pdf = pd.DataFrame(
        {
            "col2": [10, 11, 12],
        }
    )
    padf = pa.Table.from_pandas(pdf)

    pa.parquet.write_table(padf, "z.parquet", compression="zstd")
    got = cudf.read_parquet("z.parquet")
    assert_eq(pdf, got)
    assert(False)
