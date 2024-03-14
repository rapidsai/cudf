# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_array_eq, column_from_arrow

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
