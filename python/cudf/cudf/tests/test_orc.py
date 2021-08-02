# Copyright (c) 2019-2021, NVIDIA CORPORATION.

import datetime
import decimal
import os
import random
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc
import pyorc as po
import pytest

import cudf
from cudf.core.dtypes import Decimal64Dtype
from cudf.io.orc import ORCWriter
from cudf.testing._utils import (
    assert_eq,
    gen_rand_series,
    supported_numpy_dtypes,
)


@pytest.fixture(scope="module")
def datadir(datadir):
    return datadir / "orc"


@pytest.fixture
def path_or_buf(datadir):
    fname = datadir / "TestOrcFile.test1.orc"
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


def test_writer_struct():
    data = {
        "X": [
            {"Name": "A", "Age": 27},
            {"Name": "B", "Age": 25},
            None,
        ],
        "Y": [
            {"W": "C", "H": 12},
            None,
            {"W": "D", "H": 13},
        ]
    }
    pdf_in = pd.DataFrame(data)
    buffer = BytesIO()
    cudf_in = cudf.from_pandas(pdf_in)
    print(cudf_in)
    
    cudf_in.to_orc(buffer)

    pdf_out = pa.orc.ORCFile(buffer).read().to_pandas()
    assert_eq(pdf_out, pdf_in)


def test_chunked_orc_writer_lists():
    num_rows = 12345
    pdf_in = pd.DataFrame(
        {
            "ls": [[str(i), str(2 * i)] for i in range(num_rows)],
            "ld": [[dec(i / 2)] * 5 for i in range(num_rows)],
        }
    )

    gdf = cudf.from_pandas(pdf_in)
    expect = pd.concat([pdf_in, pdf_in]).reset_index(drop=True)

    buffer = BytesIO()
    writer = ORCWriter(buffer)
    writer.write_table(gdf)
    writer.write_table(gdf)
    writer.close()

    got = pa.orc.ORCFile(buffer).read().to_pandas()
    assert_eq(expect, got)


def test_writer_timestamp_stream_size(datadir, tmpdir):
    pdf_fname = datadir / "TestOrcFile.largeTimestamps.orc"
    gdf_fname = tmpdir.join("gdf.orc")

    try:
        orcfile = pa.orc.ORCFile(pdf_fname)
    except Exception as excpr:
        if type(excpr).__name__ == "ArrowIOError":
            pytest.skip(".orc file is not found")
        else:
            print(type(excpr).__name__)

    expect = orcfile.read().to_pandas()
    cudf.from_pandas(expect).to_orc(gdf_fname.strpath)
    got = pa.orc.ORCFile(gdf_fname).read().to_pandas()

    assert_eq(expect, got)
