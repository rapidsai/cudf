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
    "left, right, match",
    [
        ([1, 2, 3], [1, 2, 4], "Arrays are not equal"),
        ([1, 2, 3], [1, 2, None], "assert"),
        ([[1, 2], [3]], [[1, 2], [3, 4]], "assert"),
        ([[1, 2], [3]], [[1, 2], None], "assert"),
        ([{"a": 1}, {"a": 2}], [{"a": 1}, {"a": 3}], "assert"),
        ([{"a": 1}, {"a": 2}], [{"a": 1}, None], "assert"),
    ],
)
def test_assert_column_eq_ok_raises(
    left: list, right: list, match: str
) -> None:
    array = pa.array(left)
    column = plc.Column.from_arrow(array)
    with pytest.raises(AssertionError, match=match):
        assert_column_eq(column, pa.array(right))
