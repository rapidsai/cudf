# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import (
    assert_array_eq,
    assert_table_eq,
    column_from_arrow,
    column_to_arrow,
)

from cudf._lib import pylibcudf as plc


@pytest.fixture(scope="module")
def input_column():
    return column_from_arrow(pa.array([1, 2, 3]))


@pytest.fixture(scope="module")
def target_column():
    return column_from_arrow(pa.array([4, 5, 6, 7, 8, 9]))


@pytest.fixture
def mutable_target_column(target_column):
    return target_column.copy()


@pytest.fixture(scope="module")
def source_table(target_column):
    return plc.Table([target_column] * 3)


def test_gather(source_table, input_column):
    result = plc.copying.gather(
        source_table,
        input_column,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
    )
    pa_table = source_table.to_arrow(
        [plc.interop.ColumnMetadata("")] * source_table.num_columns()
    )
    expected = pa_table.take(column_to_arrow(input_column))
    assert_table_eq(result, expected)


def test_copy_range(input_column, target_column):
    result = plc.copying.copy_range(
        input_column,
        target_column,
        0,
        input_column.size(),
        0,
    )
    expected = pa.array([1, 2, 3, 7, 8, 9])
    assert_array_eq(result, expected)


def test_copy_range_in_place(input_column, mutable_target_column):
    plc.copying.copy_range_in_place(
        input_column,
        mutable_target_column,
        0,
        input_column.size(),
        0,
    )
    expected = pa.array([1, 2, 3, 7, 8, 9])
    assert_array_eq(mutable_target_column, expected)
