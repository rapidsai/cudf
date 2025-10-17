# Copyright (c) 2025, NVIDIA CORPORATION.
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


class TestIndexConstructorProperty:
    """Tests for the _constructor property of Index classes."""

    def test_index_constructor(self):
        """Test that _constructor returns Index class."""
        gidx = cudf.Index([1, 2, 3])

        assert gidx._constructor is cudf.Index

    def test_rangeindex_constructor(self):
        """Test that _constructor returns RangeIndex class."""
        gidx = cudf.RangeIndex(10)

        assert gidx._constructor is cudf.RangeIndex

    def test_datetimeindex_constructor(self):
        """Test that _constructor returns DatetimeIndex class."""
        gidx = cudf.DatetimeIndex(["2020-01-01", "2020-01-02"])

        assert gidx._constructor is cudf.DatetimeIndex

    def test_timedeltaindex_constructor(self):
        """Test that _constructor returns TimedeltaIndex class."""
        gidx = cudf.TimedeltaIndex([1, 2, 3])

        assert gidx._constructor is cudf.TimedeltaIndex

    def test_categoricalindex_constructor(self):
        """Test that _constructor returns CategoricalIndex class."""
        gidx = cudf.CategoricalIndex(["a", "b", "c"])

        assert gidx._constructor is cudf.CategoricalIndex

    def test_intervalindex_constructor(self):
        """Test that _constructor returns IntervalIndex class."""
        gidx = cudf.IntervalIndex.from_breaks([0, 1, 2, 3])

        assert gidx._constructor is cudf.IntervalIndex


class TestIndexInferredType:
    """Tests for the inferred_type property of Index classes."""

    @pytest.mark.parametrize(
        "data,expected_type",
        [
            ([1, 2, 3], "int64"),
            ([1.0, 2.0, 3.0], "float64"),
            (["a", "b", "c"], "string"),
            ([True, False, True], "boolean"),
        ],
    )
    def test_index_inferred_type(self, data, expected_type):
        """Test inferred_type for generic Index."""
        gidx = cudf.Index(data)
        pidx = pd.Index(data)

        assert_eq(gidx.inferred_type, pidx.inferred_type)

    def test_rangeindex_inferred_type(self):
        """Test inferred_type for RangeIndex."""
        gidx = cudf.RangeIndex(10)
        pidx = pd.RangeIndex(10)
        assert_eq(gidx.inferred_type, pidx.inferred_type)

    @pytest.mark.parametrize(
        "dates",
        [
            pd.date_range("2020-01-01", periods=5),
            pd.date_range("2020-01-01", periods=5, freq="h"),
        ],
    )
    def test_datetimeindex_inferred_type(self, dates):
        """Test inferred_type for DatetimeIndex."""
        gidx = cudf.DatetimeIndex(dates)
        pidx = pd.DatetimeIndex(dates)
        assert_eq(gidx.inferred_type, pidx.inferred_type)

    @pytest.mark.parametrize(
        "timedeltas",
        [
            pd.timedelta_range("1 day", periods=5),
            [pd.Timedelta(days=i) for i in range(5)],
        ],
    )
    def test_timedeltaindex_inferred_type(self, timedeltas):
        """Test inferred_type for TimedeltaIndex."""
        gidx = cudf.TimedeltaIndex(timedeltas)
        pidx = pd.TimedeltaIndex(timedeltas)
        assert_eq(gidx.inferred_type, pidx.inferred_type)

    @pytest.mark.parametrize(
        "data",
        [
            ["a", "b", "c"],
            [1, 2, 3],
            pd.Categorical(["a", "b", "c"]),
        ],
    )
    def test_categoricalindex_inferred_type(self, data):
        """Test inferred_type for CategoricalIndex."""
        gidx = cudf.CategoricalIndex(data)
        pidx = pd.CategoricalIndex(data)
        assert_eq(gidx.inferred_type, pidx.inferred_type)

    @pytest.mark.parametrize(
        "closed",
        ["left", "right", "both", "neither"],
    )
    def test_intervalindex_inferred_type(self, closed):
        """Test inferred_type for IntervalIndex."""
        gidx = cudf.IntervalIndex.from_breaks([0, 1, 2, 3], closed=closed)
        pidx = pd.IntervalIndex.from_breaks([0, 1, 2, 3], closed=closed)
        assert_eq(gidx.inferred_type, pidx.inferred_type)

    def test_empty_index_inferred_type(self):
        """Test inferred_type for empty Index."""
        gidx = cudf.Index([])
        pidx = pd.Index([])
        assert_eq(gidx.inferred_type, pidx.inferred_type)


class TestIntervalIndexClosedProperties:
    """Tests for closed_left and closed_right properties of IntervalIndex."""

    @pytest.mark.parametrize(
        "closed,expected_left,expected_right",
        [
            ("left", True, False),
            ("right", False, True),
            ("both", True, True),
            ("neither", False, False),
        ],
    )
    def test_intervalindex_closed_left_right(
        self, closed, expected_left, expected_right
    ):
        """Test closed_left and closed_right properties."""
        gidx = cudf.IntervalIndex.from_breaks([0, 1, 2, 3], closed=closed)
        pidx = pd.IntervalIndex.from_breaks([0, 1, 2, 3], closed=closed)

        assert gidx.closed_left == expected_left
        assert gidx.closed_right == expected_right
        assert pidx.closed_left == expected_left
        assert pidx.closed_right == expected_right

    @pytest.mark.parametrize(
        "closed,expected_left,expected_right",
        [
            ("left", True, False),
            ("right", False, True),
            ("both", True, True),
            ("neither", False, False),
        ],
    )
    def test_intervalindex_closed_properties_from_tuples(
        self, closed, expected_left, expected_right
    ):
        """Test closed_left and closed_right with from_tuples constructor."""
        gidx = cudf.IntervalIndex.from_tuples(
            [(0, 1), (1, 2), (2, 3)], closed=closed
        )
        pidx = pd.IntervalIndex.from_tuples(
            [(0, 1), (1, 2), (2, 3)], closed=closed
        )

        assert gidx.closed_left == expected_left
        assert gidx.closed_right == expected_right
        assert pidx.closed_left == expected_left
        assert pidx.closed_right == expected_right

    def test_intervalindex_closed_properties_numeric(self):
        """Test closed properties with numeric intervals."""
        gidx = cudf.IntervalIndex.from_breaks(
            [0.0, 1.5, 3.0, 4.5], closed="left"
        )
        pidx = pd.IntervalIndex.from_breaks(
            [0.0, 1.5, 3.0, 4.5], closed="left"
        )

        assert gidx.closed_left is True
        assert gidx.closed_right is False
        assert pidx.closed_left is True
        assert pidx.closed_right is False

    def test_intervalindex_closed_properties_datetime(self):
        """Test closed properties with datetime intervals."""
        dates = pd.date_range("2020-01-01", periods=4)
        gidx = cudf.IntervalIndex.from_breaks(dates, closed="right")
        pidx = pd.IntervalIndex.from_breaks(dates, closed="right")

        assert gidx.closed_left is False
        assert gidx.closed_right is True
        assert pidx.closed_left is False
        assert pidx.closed_right is True

    def test_intervalindex_closed_properties_timedelta(self):
        """Test closed properties with timedelta intervals."""
        timedeltas = pd.timedelta_range("1 day", periods=4)
        gidx = cudf.IntervalIndex.from_breaks(timedeltas, closed="both")
        pidx = pd.IntervalIndex.from_breaks(timedeltas, closed="both")

        assert gidx.closed_left is True
        assert gidx.closed_right is True
        assert pidx.closed_left is True
        assert pidx.closed_right is True

    def test_intervalindex_closed_properties_consistency(self):
        """Test that closed properties are consistent with closed attribute."""
        for closed in ["left", "right", "both", "neither"]:
            gidx = cudf.IntervalIndex.from_breaks([0, 1, 2], closed=closed)
            pidx = pd.IntervalIndex.from_breaks([0, 1, 2], closed=closed)

            # Verify consistency with the closed attribute
            assert gidx.closed == closed
            assert pidx.closed == closed

            # Verify the boolean properties match expectations for both cudf and pandas
            if closed == "left":
                assert gidx.closed_left and not gidx.closed_right
                assert pidx.closed_left and not pidx.closed_right
            elif closed == "right":
                assert not gidx.closed_left and gidx.closed_right
                assert not pidx.closed_left and pidx.closed_right
            elif closed == "both":
                assert gidx.closed_left and gidx.closed_right
                assert pidx.closed_left and pidx.closed_right
            elif closed == "neither":
                assert not gidx.closed_left and not gidx.closed_right
                assert not pidx.closed_left and not pidx.closed_right
