# Copyright (c) 2025 NVIDIA CORPORATION.
import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3],
        [1, None, 3],
        [[1, 2], [3]],
        [[1, 2], [3], None],
        [{"a": 1}, {"a": 2}],
        [{"a": 1}, {"a": 2}, None],
    ],
)
def test_assert_column_eq_ok(values: list) -> None:
    array = pa.array(values)
    column = plc.Column.from_arrow(array)
    assert_column_eq(column, array)  # no error


@pytest.mark.parametrize(
    "left, right",
    [
        ([1, 2, 3], [1, 2, 4]),
        ([1, 2, 3], [1, 2, None]),
        ([[1, 2], [3]], [[1, 2], [3, 4]]),
        ([[1, 2], [3]], [[1, 2], None]),
        ([{"a": 1}, {"a": 2}], [{"a": 1}, {"a": 3}]),
        ([{"a": 1}, {"a": 2}], [{"a": 1}, None]),
    ],
)
def test_assert_column_eq_ok_raises(left: list, right: list) -> None:
    array = pa.array(left)
    column = plc.Column.from_arrow(array)
    with pytest.raises(AssertionError):
        assert_column_eq(column, pa.array(right))
