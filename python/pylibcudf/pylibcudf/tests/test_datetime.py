# Copyright (c) 2024, NVIDIA CORPORATION.

import calendar
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


@pytest.fixture(
    params=[
        pa.scalar(-3, pa.int32()),
        pa.scalar(1, pa.int16()),
        pa.array([1, -3, 2, 4, -1, 5], pa.int32()),
    ]
)
def months(request):
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


@pytest.mark.parametrize(
    "datetime_func",
    [
        "extract_year",
        "extract_month",
        "extract_day",
        "extract_weekday",
        "extract_hour",
        "extract_minute",
        "extract_second",
        "extract_millisecond_fraction",
        "extract_microsecond_fraction",
        "extract_nanosecond_fraction",
    ],
)
def test_datetime_extracting_functions(datetime_column, datetime_func):
    pa_col = plc.interop.to_arrow(datetime_column)
    got = getattr(plc.datetime, datetime_func)(datetime_column)
    kwargs = {}
    attr = datetime_func.split("_")[1]
    if attr == "weekday":
        kwargs = {"count_from_zero": False}
        attr = "day_of_week"
    expect = getattr(pc, attr)(pa_col, **kwargs).cast(pa.int16())
    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "op",
    [
        ("ceil_temporal", "ceil_datetimes"),
        ("floor_temporal", "floor_datetimes"),
        ("round_temporal", "round_datetimes"),
    ],
)
@pytest.mark.parametrize(
    "freq",
    [
        ("day", plc.datetime.RoundingFrequency.DAY),
        ("hour", plc.datetime.RoundingFrequency.HOUR),
        ("minute", plc.datetime.RoundingFrequency.MINUTE),
        ("second", plc.datetime.RoundingFrequency.SECOND),
        ("millisecond", plc.datetime.RoundingFrequency.MILLISECOND),
        ("microsecond", plc.datetime.RoundingFrequency.MICROSECOND),
        ("nanosecond", plc.datetime.RoundingFrequency.NANOSECOND),
    ],
)
def test_rounding_operations(datetime_column, op, freq):
    got = getattr(plc.datetime, op[1])(datetime_column, freq[1])
    pa_col = plc.interop.to_arrow(datetime_column)
    pa_got = plc.interop.to_arrow(got)
    expect = getattr(pc, op[0])(
        pa_col,
        unit=freq[0],
    ).cast(pa_got.type)
    assert_column_eq(expect, got)


def test_calendrical_months(datetime_column, months):
    pa_col = plc.interop.to_arrow(datetime_column)
    got = plc.datetime.add_calendrical_months(
        datetime_column, plc.interop.from_arrow(months)
    )
    pa_got = plc.interop.to_arrow(got)

    def add_calendrical_months(timestamps, months):
        result = []
        if isinstance(months, (pa.Int32Scalar, pa.Int16Scalar)):
            months = months.as_py()
            months = [months] * len(timestamps)
        else:
            months = months.to_pylist()
        for i, d in enumerate(timestamps):
            dt = d.as_py()
            if dt is not None:
                year, month = dt.year, dt.month
                new_month = month + months[i]
                new_year = year + (new_month - 1) // 12
                new_month = (new_month - 1) % 12 + 1
                dt = dt.replace(year=new_year, month=new_month)
                result.append(dt)
            else:
                result.append(dt)
        return pa.array(result)

    expect = add_calendrical_months(pa_col, months).cast(pa_got.type)
    assert_column_eq(expect, got)


def test_day_of_year(datetime_column):
    got = plc.datetime.day_of_year(datetime_column)
    pa_got = plc.interop.to_arrow(got)
    pa_col = plc.interop.to_arrow(datetime_column)
    expect = pa.array(
        [
            d.as_py().timetuple().tm_yday if d.as_py() is not None else None
            for d in pa_col
        ],
        pa_got.type,
    )
    assert_column_eq(expect, got)


def test_is_leap_year(datetime_column):
    got = plc.datetime.is_leap_year(datetime_column)
    pa_col = plc.interop.to_arrow(datetime_column)
    expect = pc.is_leap_year(pa_col)
    assert_column_eq(expect, got)


def test_last_day_of_month(datetime_column):
    def last_day_of_month(dates):
        last_days = []
        for d in dates:
            py_d = d.as_py()
            if py_d is not None:
                last_day = calendar.monthrange(py_d.year, py_d.month)[1]
                py_d = py_d.replace(day=last_day)
                last_days.append(py_d)
            else:
                last_days.append(py_d)
        return last_days

    got = plc.datetime.last_day_of_month(datetime_column)
    pa_got = plc.interop.to_arrow(got)
    pa_col = plc.interop.to_arrow(datetime_column)
    expect = pa.array(last_day_of_month(pa_col), pa_got.type)
    assert_column_eq(expect, got)


def test_extract_quarter(datetime_column):
    got = plc.datetime.extract_quarter(datetime_column)
    pa_col = plc.interop.to_arrow(datetime_column)
    pa_got = plc.interop.to_arrow(got)
    expect = pc.quarter(pa_col).cast(pa_got.type)
    assert_column_eq(expect, got)


def test_days_in_month(datetime_column):
    def first_day(dates):
        days = []
        for d in dates:
            py_d = d.as_py()
            if py_d is not None:
                py_d = py_d.replace(day=1)
                days.append(py_d)
            else:
                days.append(None)
        return days

    got = plc.datetime.days_in_month(datetime_column)
    pa_col = plc.interop.to_arrow(datetime_column)
    pa_got = plc.interop.to_arrow(got)
    expect = pa.array(
        [
            calendar.monthrange(d.as_py().year, d.as_py().month)[1]
            if d.as_py() is not None
            else None
            for d in pa_col
        ],
        pa_got.type,
    )
    assert_column_eq(expect, got)
