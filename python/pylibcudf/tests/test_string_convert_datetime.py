# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
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
    got = plc.strings.convert.convert_datetime.to_timestamps(
        plc.Column.from_arrow(arr),
        plc.DataType(plc.TypeId.TIMESTAMP_SECONDS),
        fmt,
    )
    expect = pc.strptime(arr, fmt, "s")
    assert_column_eq(expect, got)


def test_from_timestamp(fmt):
    arr = pa.array([datetime.datetime(2020, 1, 1, 1, 1, 1), None])
    got = plc.strings.convert.convert_datetime.from_timestamps(
        plc.Column.from_arrow(arr),
        fmt,
        plc.Column.from_arrow(pa.array([], type=pa.string())),
    )
    # pc.strftime will add the extra %f
    expect = pa.array(["2020-01-01T01:01:01", None])
    assert_column_eq(expect, got)


def test_is_timestamp(fmt):
    arr = pa.array(["2020-01-01T01:01:01", None, "2020-01-01"])
    got = plc.strings.convert.convert_datetime.is_timestamp(
        plc.Column.from_arrow(arr),
        fmt,
    )
    expect = pa.array([True, None, False])
    assert_column_eq(expect, got)
