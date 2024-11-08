# Copyright (c) 2024, NVIDIA CORPORATION.

from datetime import datetime

import pyarrow as pa
import pytest
from utils import assert_column_eq, assert_table_eq

import pylibcudf as plc


@pytest.fixture
def pa_col():
    return pa.array([2, 3, 5, 7, 11])


@pytest.fixture
def pa_table():
    pa_col = pa.array([1, 2, 3])
    return pa.table([pa_col], names=["a"])


def test_fill(pa_col):
    result = plc.filling.fill(
        plc.interop.from_arrow(pa_col),
        1,
        3,
        plc.interop.from_arrow(pa.scalar(5)),
    )
    expect = pa.array([2, 5, 5, 7, 11])
    assert_column_eq(result, expect)


def test_fill_in_place(pa_col):
    result = plc.interop.from_arrow(pa_col)
    plc.filling.fill_in_place(
        result,
        1,
        3,
        plc.interop.from_arrow(pa.scalar(5)),
    )
    expect = pa.array([2, 5, 5, 7, 11])
    assert_column_eq(result, expect)


def test_sequence():
    size = 5
    init_scalar = plc.interop.from_arrow(pa.scalar(10))
    step_scalar = plc.interop.from_arrow(pa.scalar(2))
    result = plc.filling.sequence(
        size,
        init_scalar,
        step_scalar,
    )
    expect = pa.array([10, 12, 14, 16, 18])
    assert_column_eq(result, expect)


def test_repeat_with_count_int(pa_table):
    input_table = plc.interop.from_arrow(pa_table)
    count = 2
    result = plc.filling.repeat(input_table, count)
    expect = pa.table([[1, 1, 2, 2, 3, 3]], names=["a"])
    assert_table_eq(expect, result)


def test_repeat_with_count_column(pa_table):
    input_table = plc.interop.from_arrow(pa_table)
    count = plc.interop.from_arrow(pa.array([1, 2, 3]))
    result = plc.filling.repeat(input_table, count)
    expect = pa.table([[1] + [2] * 2 + [3] * 3], names=["a"])
    assert_table_eq(expect, result)


def test_calendrical_month_sequence():
    n = 5
    init_date = datetime(2020, 1, 31)
    init = plc.interop.from_arrow(
        pa.scalar(init_date, type=pa.timestamp("ms"))
    )
    months = 1
    result = plc.filling.calendrical_month_sequence(n, init, months)
    expected_dates = [
        datetime(2020, 1, 31),
        datetime(2020, 2, 29),
        datetime(2020, 3, 31),
        datetime(2020, 4, 30),
        datetime(2020, 5, 31),
    ]
    expect = pa.array(expected_dates, type=pa.timestamp("ms"))
    assert_column_eq(result, expect)
