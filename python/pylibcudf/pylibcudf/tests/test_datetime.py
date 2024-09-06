# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pylibcudf as plc
import pytest
from utils import assert_column_eq


@pytest.fixture
def column(has_nulls):
    values = [
        pa.scalar(1694004645123456789, pa.timestamp("ns")),
        pa.scalar(1544024645123456789, pa.timestamp("ns")),
        pa.scalar(1682342345346235434, pa.timestamp("ns")),
        pa.scalar(1445624625623452452, pa.timestamp("ns")),
    ]
    if has_nulls:
        values[2] = None
    return plc.interop.from_arrow(pa.array(values))


def test_extract_year(column):
    got = plc.datetime.extract_year(column)
    expect = pc.year(plc.interop.to_arrow(column)).cast(pa.int16())

    assert_column_eq(expect, got)


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


def test_extract_datetime_component(column, component):
    attr, component = component
    kwargs = {}
    if attr == "day_of_week":
        kwargs = {"count_from_zero": False}
    got = plc.datetime.extract_datetime_component(column, component)
    # libcudf produces an int16, arrow produces an int64

    expect = getattr(pc, attr)(plc.interop.to_arrow(column), **kwargs).cast(
        pa.int16()
    )

    assert_column_eq(expect, got)
