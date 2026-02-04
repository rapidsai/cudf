# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for _pylibcudf_helpers optimization functions."""

import numpy as np
import pytest

import cudf
from cudf.core.column._pylibcudf_helpers import (
    all_strings_match_type,
    count_false,
    count_true,
    reduce_boolean_column,
)


class TestAllStringsMatchType:
    """Tests for all_strings_match_type() helper function."""

    def test_all_integers(self):
        """Test with all valid integers."""
        col = cudf.core.column.as_column(["1", "2", "3", "100"])
        assert all_strings_match_type(col, "integer") is True

    def test_mixed_integer_float(self):
        """Test with mix of integers and floats - should fail integer check."""
        col = cudf.core.column.as_column(["1", "2.5", "3"])
        assert all_strings_match_type(col, "integer") is False

    def test_all_floats(self):
        """Test with all valid floats."""
        col = cudf.core.column.as_column(["1.0", "2.5", "3.14"])
        assert all_strings_match_type(col, "float") is True

    def test_integers_are_floats(self):
        """Test that integers are also valid floats."""
        col = cudf.core.column.as_column(["1", "2", "3"])
        assert all_strings_match_type(col, "float") is True

    def test_mixed_float_nonnumeric(self):
        """Test with mix of floats and non-numeric strings."""
        col = cudf.core.column.as_column(["1.5", "abc", "3.14"])
        assert all_strings_match_type(col, "float") is False

    def test_with_nulls_integers(self):
        """Test that nulls are ignored in integer validation."""
        col = cudf.core.column.as_column(["1", "2", None, "3"])
        assert all_strings_match_type(col, "integer") is True

    def test_with_nulls_non_integers(self):
        """Test with nulls and non-integers."""
        col = cudf.core.column.as_column(["1", "abc", None, "3"])
        assert all_strings_match_type(col, "integer") is False

    def test_empty_column(self):
        """Test with empty column."""
        col = cudf.core.column.as_column([], dtype="object")
        # Empty column should return True (vacuous truth)
        assert all_strings_match_type(col, "integer") is True

    def test_all_nulls(self):
        """Test with all null values."""
        col = cudf.core.column.as_column([None, None, None], dtype="object")
        # All nulls should return True (vacuous truth)
        assert all_strings_match_type(col, "integer") is True

    def test_invalid_type_check(self):
        """Test that invalid type_check raises ValueError."""
        col = cudf.core.column.as_column(["1", "2", "3"])
        with pytest.raises(ValueError, match="Unknown type_check"):
            all_strings_match_type(col, "invalid")

    def test_equivalence_to_old_pattern_integer(self):
        """Verify equivalence to old .is_integer().all() pattern."""
        test_cases = [
            ["1", "2", "3"],
            ["1", "2.5", "3"],
            ["1", "abc", "3"],
            ["1", None, "3"],
        ]
        for data in test_cases:
            col = cudf.core.column.as_column(data)
            old_result = col.is_integer().all()
            new_result = all_strings_match_type(col, "integer")
            assert old_result == new_result, (
                f"Mismatch for {data}: old={old_result}, new={new_result}"
            )

    def test_equivalence_to_old_pattern_float(self):
        """Verify equivalence to old .is_float().all() pattern."""
        test_cases = [
            ["1.0", "2.5", "3.14"],
            ["1", "2", "3"],  # integers are valid floats
            ["1.5", "abc", "3.14"],
            ["1.0", None, "3.14"],
        ]
        for data in test_cases:
            col = cudf.core.column.as_column(data)
            old_result = col.is_float().all()
            new_result = all_strings_match_type(col, "float")
            assert old_result == new_result, (
                f"Mismatch for {data}: old={old_result}, new={new_result}"
            )


class TestReduceBooleanColumn:
    """Tests for reduce_boolean_column() helper function."""

    def test_all_true(self):
        """Test all() with all True values."""
        col = cudf.core.column.as_column([True, True, True])
        assert reduce_boolean_column(col, "all") is True

    def test_all_false(self):
        """Test all() with all False values."""
        col = cudf.core.column.as_column([False, False, False])
        assert reduce_boolean_column(col, "all") is False

    def test_all_mixed(self):
        """Test all() with mixed values."""
        col = cudf.core.column.as_column([True, False, True])
        assert reduce_boolean_column(col, "all") is False

    def test_any_true(self):
        """Test any() with all True values."""
        col = cudf.core.column.as_column([True, True, True])
        assert reduce_boolean_column(col, "any") is True

    def test_any_false(self):
        """Test any() with all False values."""
        col = cudf.core.column.as_column([False, False, False])
        assert reduce_boolean_column(col, "any") is False

    def test_any_mixed(self):
        """Test any() with mixed values."""
        col = cudf.core.column.as_column([False, True, False])
        assert reduce_boolean_column(col, "any") is True

    def test_with_nulls_all(self):
        """Test all() with null values."""
        col = cudf.core.column.as_column([True, None, True])
        # pylibcudf reduction should skip nulls
        assert reduce_boolean_column(col, "all") is True

    def test_with_nulls_any(self):
        """Test any() with null values."""
        col = cudf.core.column.as_column([False, None, False])
        # pylibcudf reduction should skip nulls
        assert reduce_boolean_column(col, "any") is False

    def test_empty_column_all(self):
        """Test all() with empty column."""
        col = cudf.core.column.as_column([], dtype="bool")
        # Empty column should return True for all() (vacuous truth)
        assert reduce_boolean_column(col, "all") is True

    def test_empty_column_any(self):
        """Test any() with empty column."""
        col = cudf.core.column.as_column([], dtype="bool")
        # Empty column should return False for any()
        assert reduce_boolean_column(col, "any") is False

    def test_invalid_operation(self):
        """Test that invalid operation raises ValueError."""
        col = cudf.core.column.as_column([True, False])
        with pytest.raises(ValueError, match="Unknown operation"):
            reduce_boolean_column(col, "invalid")


class TestIntegration:
    """Integration tests verifying optimizations work in real scenarios."""

    def test_string_to_int_cast_valid(self):
        """Test that string to int casting uses the optimization."""
        s = cudf.Series(["1", "2", "3", "100"])
        result = s.astype("int64")
        expected = cudf.Series([1, 2, 3, 100], dtype="int64")
        cudf.testing.assert_series_equal(result, expected)

    def test_string_to_int_cast_invalid(self):
        """Test that invalid string to int casting raises error."""
        s = cudf.Series(["1", "2.5", "3"])
        with pytest.raises(ValueError, match="non-integer values"):
            s.astype("int64")

    def test_string_to_float_cast_valid(self):
        """Test that string to float casting uses the optimization."""
        s = cudf.Series(["1.5", "2.7", "3.14"])
        result = s.astype("float64")
        expected = cudf.Series([1.5, 2.7, 3.14], dtype="float64")
        cudf.testing.assert_series_equal(result, expected)

    def test_string_to_float_cast_invalid(self):
        """Test that invalid string to float casting raises error."""
        s = cudf.Series(["1.5", "abc", "3.14"])
        with pytest.raises(ValueError, match="non-floating values"):
            s.astype("float64")

    def test_can_cast_safely_integer(self):
        """Test can_cast_safely with integer strings."""
        s = cudf.Series(["1", "2", "3"])
        assert s._column.can_cast_safely(np.dtype("int64")) is True

    def test_can_cast_safely_float(self):
        """Test can_cast_safely with float strings."""
        s = cudf.Series(["1.5", "2.7", "3.14"])
        assert s._column.can_cast_safely(np.dtype("float64")) is True

    def test_can_cast_safely_invalid(self):
        """Test can_cast_safely with non-numeric strings."""
        s = cudf.Series(["abc", "def", "ghi"])
        assert s._column.can_cast_safely(np.dtype("int64")) is False
        assert s._column.can_cast_safely(np.dtype("float64")) is False

    def test_minmax_optimization(self):
        """Test that minmax() optimization works correctly."""
        s = cudf.Series([1, 5, 3, 9, 2], dtype="int64")
        min_val, max_val = s._column.minmax()
        assert min_val == 1
        assert max_val == 9

    def test_numeric_conversion_integers(self):
        """Test to_numeric with integer strings."""
        result = cudf.to_numeric(cudf.Series(["1", "2", "3"]))
        expected = cudf.Series([1, 2, 3], dtype="int64")
        cudf.testing.assert_series_equal(result, expected)

    def test_numeric_conversion_floats(self):
        """Test to_numeric with float strings."""
        result = cudf.to_numeric(cudf.Series(["1.5", "2.7", "3.14"]))
        expected = cudf.Series([1.5, 2.7, 3.14], dtype="float64")
        cudf.testing.assert_series_equal(result, expected)


class TestCountTrueCountFalse:
    """Tests for count_true() and count_false() helper functions."""

    def test_count_true_all_true(self):
        """Test counting True values when all are True."""
        col = cudf.core.column.as_column([True, True, True, True])
        assert count_true(col) == 4

    def test_count_true_all_false(self):
        """Test counting True values when all are False."""
        col = cudf.core.column.as_column([False, False, False])
        assert count_true(col) == 0

    def test_count_true_mixed(self):
        """Test counting True values in mixed boolean column."""
        col = cudf.core.column.as_column([True, False, True, False, True])
        assert count_true(col) == 3

    def test_count_true_with_nulls(self):
        """Test that nulls are not counted as True."""
        col = cudf.core.column.as_column([True, None, True, None, False])
        assert count_true(col) == 2

    def test_count_true_empty(self):
        """Test counting True values in empty column."""
        col = cudf.core.column.as_column([], dtype="bool")
        assert count_true(col) == 0

    def test_count_false_all_false(self):
        """Test counting False values when all are False."""
        col = cudf.core.column.as_column([False, False, False, False])
        assert count_false(col) == 4

    def test_count_false_all_true(self):
        """Test counting False values when all are True."""
        col = cudf.core.column.as_column([True, True, True])
        assert count_false(col) == 0

    def test_count_false_mixed(self):
        """Test counting False values in mixed boolean column."""
        col = cudf.core.column.as_column([True, False, True, False, True])
        assert count_false(col) == 2

    def test_count_false_with_nulls(self):
        """Test that nulls are not counted as False."""
        col = cudf.core.column.as_column([False, None, False, None, True])
        assert count_false(col) == 2

    def test_count_false_empty(self):
        """Test counting False values in empty column."""
        col = cudf.core.column.as_column([], dtype="bool")
        assert count_false(col) == 0

    def test_count_true_false_consistency(self):
        """Test that count_true + count_false + null_count == len."""
        col = cudf.core.column.as_column(
            [True, False, None, True, False, None, True]
        )
        true_count = count_true(col)
        false_count = count_false(col)
        null_count = col.null_count
        total = true_count + false_count + null_count
        assert total == len(col)
        assert true_count == 3
        assert false_count == 2
        assert null_count == 2
