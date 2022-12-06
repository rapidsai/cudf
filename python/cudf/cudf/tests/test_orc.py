# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import datetime
import decimal
import os
import random
from io import BytesIO
from string import ascii_lowercase

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc
import pyorc
import pytest

import cudf
from cudf.io.orc import ORCWriter
from cudf.testing import assert_frame_equal
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


def test_reader_unsupported_offsets():
    # needs enough data for more than one row group
    expected = cudf.DataFrame({"str": ["*"] * 10001}, dtype="string")

    buffer = BytesIO()
    expected.to_pandas().to_orc(buffer)

    # Reading this file should not lead to data corruption, even if it fails
    try:
        got = cudf.read_orc(buffer)
    except RuntimeError:
        pytest.mark.xfail(
            reason="Unsupported file, " 
            "see https://github.com/rapidsai/cudf/issues/11890"
        )
    else:
        assert_eq(expected, got)
