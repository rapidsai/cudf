# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(
    scope="module",
    params=[
        pa.timestamp("ns"),
        pa.timestamp("us"),
        pa.timestamp("ms"),
        pa.timestamp("s"),
    ],
)
def timestamp_type(request):
    return request.param


@pytest.fixture(scope="module")
def pa_timestamp_col():
    return pa.array(["2011-01-01", "2011-01-02", "2011-01-03"])


@pytest.fixture(scope="module")
def plc_timestamp_col(pa_timestamp_col):
    return plc.interop.from_arrow(pa_timestamp_col)


@pytest.mark.parametrize("format", ["%Y-%m-%d"])
def test_to_datetime(
    pa_timestamp_col, plc_timestamp_col, timestamp_type, format
):
    expect = pa.compute.strptime(pa_timestamp_col, format, timestamp_type.unit)
    got = plc.strings.convert.convert_datetime.to_timestamps(
        plc_timestamp_col,
        plc.interop.from_arrow(timestamp_type),
        format,
    )
    assert_column_eq(expect, got)
