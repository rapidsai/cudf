# Copyright (c) 2024, NVIDIA CORPORATION.

from datetime import datetime, timedelta

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(
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
def pa_duration_col():
    return pa.array(["05:20:25"])


@pytest.fixture(scope="module")
def plc_duration_col(pa_duration_col):
    return plc.interop.from_arrow(pa_duration_col)


def test_to_duration(pa_duration_col, plc_duration_col, duration_type):
    format = "%H:%M:%S"

    def to_timedelta(duration_str):
        date = datetime.strptime(duration_str, format)
        return date - datetime(1900, 1, 1)  # "%H:%M:%S" zero date

    expect = pa.array([to_timedelta(d.as_py()) for d in pa_duration_col]).cast(
        duration_type
    )

    got = plc.strings.convert.convert_durations.to_durations(
        plc_duration_col,
        plc.interop.from_arrow(duration_type),
        format,
    )
    assert_column_eq(expect, got)


@pytest.mark.parametrize("format", [None, "%D days %H:%M:%S"])
def test_from_durations(format):
    pa_array = pa.array(
        [timedelta(days=1, hours=1, minutes=1, seconds=1), None]
    )
    result = plc.strings.convert.convert_durations.from_durations(
        plc.interop.from_arrow(pa_array), format
    )
    expected = pa.array(["1 days 01:01:01", None])
    assert_column_eq(result, expected)
