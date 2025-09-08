# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def test_series_duplicate_index_reindex():
    gs = cudf.Series([0, 1, 2, 3], index=[0, 0, 1, 1])
    ps = gs.to_pandas()

    assert_exceptions_equal(
        gs.reindex,
        ps.reindex,
        lfunc_args_and_kwargs=([10, 11, 12, 13], {}),
        rfunc_args_and_kwargs=([10, 11, 12, 13], {}),
    )


@pytest.mark.parametrize("fill_value", [3, 99, -1, 0])
def test_series_reindex_fill_value_preserves_existing_nas(
    fill_value, nan_as_null
):
    """
    Test that reindex with fill_value only fills newly introduced positions,
    not existing NA values.

    This test reproduces and validates the fix for issue #19854:
    https://github.com/rapidsai/cudf/issues/19854

    Before the fix: fill_value was applied to ALL NAs (including existing ones)
    After the fix: fill_value is only applied to newly introduced positions
    """
    gs = cudf.Series([1.0, 2.0, None], nan_as_null=nan_as_null)
    ps = gs.to_pandas()

    result = gs.reindex([0, 1, 2, 3], fill_value=fill_value)
    expected = ps.reindex([0, 1, 2, 3], fill_value=fill_value)

    assert_eq(result, expected)


@pytest.mark.parametrize("fill_value", [99, -5, 0])
def test_series_reindex_fill_value_multiple_new_positions(fill_value):
    """Test reindex with multiple new positions."""
    gs = cudf.Series([1.0, 2.0, None])
    ps = gs.to_pandas()

    result = gs.reindex([0, 1, 2, 3, 4, 5], fill_value=fill_value)
    expected = ps.reindex([0, 1, 2, 3, 4, 5], fill_value=fill_value)

    assert_eq(result, expected)


@pytest.mark.parametrize("fill_value", [99, -1, 0])
def test_series_reindex_fill_value_no_existing_nas(fill_value):
    """Test reindex when original series has no NA values."""
    gs = cudf.Series([1.0, 2.0, 3.0])
    ps = gs.to_pandas()

    result = gs.reindex([0, 1, 2, 3, 4], fill_value=fill_value)
    expected = ps.reindex([0, 1, 2, 3, 4], fill_value=fill_value)

    assert_eq(result, expected)


@pytest.mark.parametrize("fill_value", [99, -1, 0])
def test_series_reindex_fill_value_all_existing_nas(fill_value):
    """Test reindex when original series has all NA values."""
    gs = cudf.Series([None, None, None])
    ps = gs.to_pandas()

    result = gs.reindex([0, 1, 2, 3], fill_value=fill_value)
    expected = ps.reindex([0, 1, 2, 3], fill_value=fill_value)

    assert_eq(result, expected)


@pytest.mark.parametrize("fill_value", [99, -1, 0])
def test_series_reindex_fill_value_mixed_dtypes(fill_value):
    """Test reindex with different data types."""
    gs = cudf.Series([1, 2, None], dtype="int64")
    ps = gs.to_pandas()

    result = gs.reindex([0, 1, 2, 3], fill_value=fill_value)
    expected = ps.reindex([0, 1, 2, 3], fill_value=fill_value)

    assert_eq(result, expected)


@pytest.mark.parametrize("fill_value", ["new", "test", "x"])
def test_series_reindex_fill_value_string_dtype(fill_value):
    """Test reindex with string data type."""
    gs = cudf.Series(["a", "b", None])
    ps = gs.to_pandas()

    result = gs.reindex([0, 1, 2, 3], fill_value=fill_value)
    expected = ps.reindex([0, 1, 2, 3], fill_value=fill_value)

    assert_eq(result, expected)


def test_series_reindex_fill_value_none():
    """Test reindex with fill_value=None (should use default NA)."""
    gs = cudf.Series([1.0, 2.0, None])
    ps = gs.to_pandas()

    result = gs.reindex([0, 1, 2, 3], fill_value=None)
    expected = ps.reindex([0, 1, 2, 3], fill_value=None)

    assert_eq(result, expected)


def test_series_reindex_fill_value_na():
    """Test reindex with fill_value=cudf.NA."""
    gs = cudf.Series([1.0, 2.0, None])
    ps = gs.to_pandas()

    result = gs.reindex([0, 1, 2, 3], fill_value=cudf.NA)
    expected = ps.reindex([0, 1, 2, 3], fill_value=cudf.NA)

    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data", [[1.0, 2.0, None], [None, 1.0, 2.0], [1.0, None, 2.0]]
)
@pytest.mark.parametrize("fill_value", [0, 99, -1])
def test_series_reindex_fill_value_various_na_positions(data, fill_value):
    """Test reindex with NA values at different positions."""
    gs = cudf.Series(data)
    ps = gs.to_pandas()

    result = gs.reindex([0, 1, 2, 3, 4], fill_value=fill_value)
    expected = ps.reindex([0, 1, 2, 3, 4], fill_value=fill_value)

    assert_eq(result, expected)
