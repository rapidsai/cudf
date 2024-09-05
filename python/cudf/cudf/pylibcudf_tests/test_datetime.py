# Copyright (c) 2024, NVIDIA CORPORATION.

import datetime
import functools

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc


@pytest.fixture
def date_column(has_nulls):
    values = [
        datetime.date(1999, 1, 1),
        datetime.date(2024, 10, 12),
        datetime.date(1, 1, 1),
        datetime.date(9999, 1, 1),
    ]
    if has_nulls:
        values[2] = None
    return plc.interop.from_arrow(pa.array(values, type=pa.date32()))


@pytest.fixture(scope="module", params=["s", "ms", "us", "ns"])
def datetime_column(has_nulls, request):
    values = [
        datetime.datetime(1999, 1, 1),
        datetime.datetime(2024, 10, 12),
        datetime.datetime(1970, 1, 1),
        datetime.datetime(2260, 1, 1),
        datetime.datetime(2024, 2, 29, 3, 14, 15),
        datetime.datetime(2024, 2, 29, 3, 14, 15, 999),
    ]
    if has_nulls:
        values[2] = None
    return plc.interop.from_arrow(
        pa.array(values, type=pa.timestamp(request.param))
    )


@pytest.mark.parametrize(
    "component, pc_fun",
    [
        ("year", pc.year),
        ("month", pc.month),
        ("day", pc.day),
        ("weekday", functools.partial(pc.day_of_week, count_from_zero=False)),
        ("hour", pc.hour),
        ("minute", pc.minute),
        ("second", pc.second),
        ("millisecond", pc.millisecond),
        ("microsecond", pc.microsecond),
        ("nanosecond", pc.nanosecond),
    ],
)
def test_extraction(datetime_column, component, pc_fun):
    got = plc.datetime.extract_datetime_component(datetime_column, component)
    # libcudf produces an int16, arrow produces an int64
    expect = pc_fun(plc.interop.to_arrow(datetime_column)).cast(pa.int16())

    assert_column_eq(expect, got)
