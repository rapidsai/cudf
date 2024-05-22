# Copyright (c) 2024, NVIDIA CORPORATION.

import datetime

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc


@pytest.fixture(params=[False, True])
def nullable(request):
    return request.param


@pytest.fixture
def column(nullable):
    values = [
        datetime.date(1999, 1, 1),
        datetime.date(2024, 10, 12),
        datetime.date(1, 1, 1),
        datetime.date(9999, 1, 1),
    ]
    if nullable:
        values[2] = None
    return plc.interop.from_arrow(pa.array(values, type=pa.date32()))


def test_extract_year(column):
    got = plc.datetime.extract_year(column)
    # libcudf produces an in16, arrow produces an int64
    expect = pa.compute.year(plc.interop.to_arrow(column)).cast(pa.int16())

    assert_column_eq(expect, got)
