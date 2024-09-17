# Copyright (c) 2024, NVIDIA CORPORATION.

from datetime import datetime

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc


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


@pytest.fixture(
    scope="module",
    params=[
        pa.duration("ns"),
        pa.duration("us"),
        pa.duration("ms"),
        pa.duration("s"),
    ],
)
def duration_type(request):
    return request.param


@pytest.fixture(scope="module")
def pa_timestamp_col():
    return pa.array(["2011-01-01", "2011-01-02", "2011-01-03"])


@pytest.fixture(scope="module")
def pa_duration_col():
    return pa.array(["05:20:25"])


@pytest.fixture(scope="module")
def plc_timestamp_col(pa_timestamp_col):
    return plc.interop.from_arrow(pa_timestamp_col)


@pytest.fixture(scope="module")
def plc_duration_col(pa_duration_col):
    return plc.interop.from_arrow(pa_duration_col)


@pytest.mark.parametrize("format", ["%Y-%m-%d"])
def test_to_datetime(
    pa_timestamp_col, plc_timestamp_col, timestamp_type, format
):
    expect = pa.compute.strptime(pa_timestamp_col, format, timestamp_type.unit)
    got = plc.strings.convert.convert_datetime.to_timestamps(
        plc_timestamp_col,
        plc.interop.from_arrow(timestamp_type),
        format.encode(),
    )
    assert_column_eq(expect, got)


@pytest.mark.parametrize("format", ["%H:%M:%S"])
def test_to_duration(pa_duration_col, plc_duration_col, duration_type, format):
    def to_timedelta(duration_str):
        date = datetime.strptime(duration_str, format)
        return date - datetime(1900, 1, 1)  # "%H:%M:%S" zero date

    expect = pa.array([to_timedelta(d.as_py()) for d in pa_duration_col]).cast(
        duration_type
    )

    got = plc.strings.convert.convert_durations.to_durations(
        plc_duration_col,
        plc.interop.from_arrow(duration_type),
        format.encode(),
    )
    assert_column_eq(expect, got)
