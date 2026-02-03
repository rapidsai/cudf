# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Tests for _pylibcudf_helpers module."""

import pytest

from cudf.core.column._pylibcudf_helpers import (
    all_strings_match_type,
    create_non_null_mask,
    reduce_boolean_column,
)
from cudf.core.column.column import as_column


class TestAllStringsMatchType:
    """Tests for all_strings_match_type() helper function."""

    def test_all_strings_are_integers(self):
        """Test that all_strings_match_type detects all integer strings."""
        col = as_column(["1", "2", "3", "42", "-5"])
        assert all_strings_match_type(col, "integer") is True

    def test_not_all_strings_are_integers(self):
        """Test that all_strings_match_type detects non-integer strings."""
        col = as_column(["1", "2.5", "3"])
        assert all_strings_match_type(col, "integer") is False

    def test_all_strings_are_floats(self):
        """Test that all_strings_match_type detects all float strings."""
        col = as_column(["1.0", "2.5", "3.14", "-5.2"])
        assert all_strings_match_type(col, "float") is True

    def test_integers_are_also_floats(self):
        """Test that integer strings are also valid float strings."""
        col = as_column(["1", "2", "3"])
        assert all_strings_match_type(col, "float") is True

    def test_not_all_strings_are_floats(self):
        """Test that all_strings_match_type detects non-float strings."""
        col = as_column(["1.0", "abc", "3.14"])
        assert all_strings_match_type(col, "float") is False

    def test_with_nulls_integer(self):
        """Test that nulls are ignored when checking integer strings."""
        col = as_column(["1", None, "3", "42"])
        # Nulls should be ignored - only non-null values are checked
        assert all_strings_match_type(col, "integer") is True

    def test_empty_column(self):
        """Test with empty column."""
        col = as_column([], dtype="object")
        # Empty column should return True (vacuous truth)
        assert all_strings_match_type(col, "integer") is True

    def test_invalid_type_check(self):
        """Test that invalid type_check raises ValueError."""
        col = as_column(["1", "2", "3"])
        with pytest.raises(ValueError, match="Unknown type_check"):
            all_strings_match_type(col, "invalid")


class TestCreateNonNullMask:
    """Tests for create_non_null_mask() helper function."""

    def test_no_nulls(self):
        """Test mask creation with no nulls."""
        col = as_column([1, 2, 3, 4, 5])
        mask = create_non_null_mask(col)
        # All elements should be valid
        assert mask.size > 0

    def test_with_nulls(self):
        """Test mask creation with nulls."""
        col = as_column([1, None, 3, None, 5])
        mask = create_non_null_mask(col)
        # Mask should be created
        assert mask.size > 0

    def test_all_nulls(self):
        """Test mask creation with all nulls."""
        col = as_column([None, None, None], dtype="int64")
        mask = create_non_null_mask(col)
        # Mask should be created even if all nulls
        assert mask.size > 0

    def test_string_column_with_nulls(self):
        """Test mask creation with string column containing nulls."""
        col = as_column(["a", None, "c", None, "e"])
        mask = create_non_null_mask(col)
        assert mask.size > 0

    def test_equivalence_to_original_pattern(self):
        """Test that result is equivalent to notnull().fillna(False).as_mask()."""
        col = as_column([1, None, 3, None, 5])

        # Our optimized version
        mask_optimized = create_non_null_mask(col)

        # Original pattern
        mask_original = col.notnull().fillna(False).as_mask()

        # Compare the buffers
        assert mask_optimized.size == mask_original.size
        # The actual bit patterns should be identical
        assert bytes(mask_optimized.memoryview()) == bytes(
            mask_original.memoryview()
        )


class TestReduceBooleanColumn:
    """Tests for reduce_boolean_column() helper function."""

    def test_all_true(self):
        """Test reduction of all True values."""
        col = as_column([True, True, True, True])
        assert reduce_boolean_column(col, "all") is True
        assert reduce_boolean_column(col, "any") is True

    def test_all_false(self):
        """Test reduction of all False values."""
        col = as_column([False, False, False, False])
        assert reduce_boolean_column(col, "all") is False
        assert reduce_boolean_column(col, "any") is False

    def test_mixed_values_all(self):
        """Test 'all' reduction with mixed values."""
        col = as_column([True, False, True, True])
        assert reduce_boolean_column(col, "all") is False

    def test_mixed_values_any(self):
        """Test 'any' reduction with mixed values."""
        col = as_column([False, False, True, False])
        assert reduce_boolean_column(col, "any") is True

    def test_with_nulls(self):
        """Test reduction with null values."""
        col = as_column([True, None, True, True])
        # Behavior with nulls - typically nulls are skipped in reductions
        # all() should return True if all non-null values are True
        assert reduce_boolean_column(col, "all") is True

    def test_empty_column(self):
        """Test reduction of empty column."""
        col = as_column([], dtype="bool")
        # Empty all() should be True, empty any() should be False
        assert reduce_boolean_column(col, "all") is True
        assert reduce_boolean_column(col, "any") is False

    def test_invalid_operation(self):
        """Test that invalid operation raises ValueError."""
        col = as_column([True, False, True])
        with pytest.raises(ValueError, match="Unknown operation"):
            reduce_boolean_column(col, "invalid")


class TestIntegration:
    """Integration tests comparing helper functions to original patterns."""

    def test_string_validation_equivalence(self):
        """Test that all_strings_match_type gives same result as is_integer().all()."""
        test_cases = [
            (["1", "2", "3"], "integer", True),
            (["1", "2.5", "3"], "integer", False),
            (["1.0", "2.5", "3.14"], "float", True),
            (["1", "abc", "3"], "float", False),
        ]

        for data, type_check, expected in test_cases:
            col = as_column(data)

            # Our optimized version
            result_optimized = all_strings_match_type(col, type_check)

            # Original pattern
            if type_check == "integer":
                result_original = col.is_integer().all()
            elif type_check == "float":
                result_original = col.is_float().all()
            elif type_check == "hex":
                result_original = col.is_hex().all()

            assert result_optimized == result_original == expected

    def test_performance_benefit(self):
        """Demonstrate that helpers avoid creating intermediate columns."""
        # This is more of a documentation test than a functional test
        # The actual performance benefit would be measured in benchmarks

        col = as_column(["1", "2", "3"] * 1000)

        # Using helper - no intermediate column created
        result = all_strings_match_type(col, "integer")
        assert result is True

        # Original pattern - creates intermediate NumericalColumn
        result_original = col.is_integer().all()
        assert result_original

        # Both should give same result
        assert result == result_original
