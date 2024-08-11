# Copyright (c) 2024, NVIDIA CORPORATION.

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
def timedelta_type(request):
    return request.param


@pytest.fixture(scope="module")
def pa_col():
    return pa.array(["2011-01-01", "2011-01-02", "2011-01-03"])


@pytest.fixture(scope="module")
def plc_col(pa_col):
    return plc.interop.from_arrow(pa_col)


@pytest.mark.parametrize("format", ["%Y-%m-%d"])
def test_to_datetime(pa_col, plc_col, timestamp_type, format):
    expect = pa.compute.strptime(pa_col, format, timestamp_type.unit)
    got = plc.strings.convert.convert_datetime.to_timestamps(
        plc_col, plc.interop.from_arrow(timestamp_type), format.encode()
    )
    assert_column_eq(expect, got)
