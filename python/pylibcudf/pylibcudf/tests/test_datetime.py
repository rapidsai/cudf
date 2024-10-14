# Copyright (c) 2024, NVIDIA CORPORATION.

import datetime

import pyarrow as pa
import pyarrow.compute as pc
import pylibcudf as plc
import pytest
from utils import assert_column_eq


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


@pytest.fixture(
    params=[
        ("year", plc.datetime.DatetimeComponent.YEAR),
        ("month", plc.datetime.DatetimeComponent.MONTH),
        ("day", plc.datetime.DatetimeComponent.DAY),
        ("day_of_week", plc.datetime.DatetimeComponent.WEEKDAY),
        ("hour", plc.datetime.DatetimeComponent.HOUR),
        ("minute", plc.datetime.DatetimeComponent.MINUTE),
        ("second", plc.datetime.DatetimeComponent.SECOND),
        ("millisecond", plc.datetime.DatetimeComponent.MILLISECOND),
        ("microsecond", plc.datetime.DatetimeComponent.MICROSECOND),
        ("nanosecond", plc.datetime.DatetimeComponent.NANOSECOND),
    ],
    ids=lambda x: x[0],
)
def component(request):
    return request.param


def test_extract_datetime_component(datetime_column, component):
    attr, component = component
    kwargs = {}
    if attr == "day_of_week":
        kwargs = {"count_from_zero": False}
    got = plc.datetime.extract_datetime_component(datetime_column, component)
    # libcudf produces an int16, arrow produces an int64

    expect = getattr(pc, attr)(
        plc.interop.to_arrow(datetime_column), **kwargs
    ).cast(pa.int16())

    assert_column_eq(expect, got)
