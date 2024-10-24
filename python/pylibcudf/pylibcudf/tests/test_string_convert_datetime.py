# Copyright (c) 2024, NVIDIA CORPORATION.
import datetime

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture
def fmt():
    return "%Y-%m-%dT%H:%M:%S"


def test_to_timestamp(fmt):
    arr = pa.array(["2020-01-01T01:01:01", None])
    result = plc.strings.convert.convert_datetime.to_timestamps(
        plc.interop.from_arrow(arr),
        plc.DataType(plc.TypeId.TIMESTAMP_SECONDS),
        fmt,
    )
    expected = pc.strptime(arr, fmt, "s")
    assert_column_eq(result, expected)


def test_from_timestamp(fmt):
    arr = pa.array([datetime.datetime(2020, 1, 1, 1, 1, 1), None])
    result = plc.strings.convert.convert_datetime.from_timestamps(
        plc.interop.from_arrow(arr),
        fmt,
        plc.interop.from_arrow(pa.array([], type=pa.string())),
    )
    # pc.strftime will add the extra %f
    expected = pa.array(["2020-01-01T01:01:01", None])
    assert_column_eq(result, expected)


def test_is_timestamp(fmt):
    arr = pa.array(["2020-01-01T01:01:01", None, "2020-01-01"])
    result = plc.strings.convert.convert_datetime.is_timestamp(
        plc.interop.from_arrow(arr),
        fmt,
    )
    expected = pa.array([True, None, False])
    assert_column_eq(result, expected)
