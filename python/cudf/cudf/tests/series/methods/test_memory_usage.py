# Copyright (c) 2025, NVIDIA CORPORATION.

import cudf
from cudf.core.column.column import column_empty
from cudf.testing import assert_eq


def test_memory_usage_list():
    s1 = cudf.Series([[1, 2], [3, 4]])
    assert s1.memory_usage() == 44
    s2 = cudf.Series([[[[1, 2]]], [[[3, 4]]]])
    assert s2.memory_usage() == 68
    s3 = cudf.Series([[{"b": 1, "a": 10}, {"b": 2, "a": 100}]])
    assert s3.memory_usage() == 40


def test_empty_nested_list_uninitialized_offsets_memory_usage():
    col = column_empty(0, cudf.ListDtype(cudf.ListDtype("int64")))
    nested_col = col.children[1]
    empty_inner = type(nested_col)(
        data=None,
        size=nested_col.size,
        dtype=nested_col.dtype,
        mask=nested_col.mask,
        offset=nested_col.offset,
        null_count=nested_col.null_count,
        children=(
            column_empty(0, nested_col.children[0].dtype),
            nested_col.children[1],
        ),
    )
    col_empty_offset = type(col)(
        data=None,
        size=col.size,
        dtype=col.dtype,
        mask=col.mask,
        offset=col.offset,
        null_count=col.null_count,
        children=(column_empty(0, col.children[0].dtype), empty_inner),
    )
    ser = cudf.Series._from_column(col_empty_offset)
    assert ser.memory_usage() == 8


def test_series_memory_usage():
    sr = cudf.Series([1, 2, 3, 4], dtype="int64")
    assert sr.memory_usage() == 32

    sliced_sr = sr[2:]
    assert sliced_sr.memory_usage() == 16

    sliced_sr[3] = None
    assert sliced_sr.memory_usage() == 80

    sr = cudf.Series(["hello world", "rapids ai", "abc", "z"])
    assert sr.memory_usage() == 44

    assert sr[3:].memory_usage() == 9  # z
    assert sr[:1].memory_usage() == 19  # hello world


def test_struct_with_null_memory_usage():
    df = cudf.DataFrame(
        {
            "a": cudf.Series([1, 2, -1, -1, 3], dtype="int64"),
            "b": cudf.Series([10, 20, -1, -1, 30], dtype="int64"),
        }
    )
    s = df.to_struct()
    assert s.memory_usage() == 80

    s[2:4] = None
    assert s.memory_usage() == 272


def test_struct_memory_usage():
    s = cudf.Series([{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}])
    df = s.struct.explode()

    assert_eq(s.memory_usage(), df.memory_usage().sum())
