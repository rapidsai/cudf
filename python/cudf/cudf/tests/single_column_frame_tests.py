# Copyright (c) 2021, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

# from cudf.tests.utils import (
#     assert_eq,
# )


class SingleColumnFrameTests:
    """Common set of tests for use with subclasses of SingleColumnFrame.

    Subclasses must define the _cls class variable, which is used to construct
    suitable objects in the tests.
    """

    _cls = None

    @pytest.mark.parametrize(
        "binop",
        [
            # Arithmetic operations.
            "__add__",
            "__sub__",
            "__mul__",
            "__floordiv__",
            "__truediv__",
            # Logical operations.
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
        ],
    )
    @given(
        x=arrays(np.float64, (100,), elements=floats(-10, 10, width=64)),
        y=arrays(np.float64, (100,), elements=floats(-10, 10, width=64)),
    )
    @settings(deadline=None)
    def test_binops(self, binop, x, y):
        """Test binary operations."""
        x = self._cls(x)
        y = self._cls(y)
        got = (getattr(x, binop)(y)).values
        expected = getattr(x.values, binop)(y.values)
        assert cp.all(got == expected) or (
            cp.all(cp.isnan(got)) and cp.all(cp.isnan(expected))
        )


# def test_binop_add_nullable_fill():
#     """Test a binary op with a fill value on two nullable series."""
#     x = cudf.Series([1, 2, None])
#     y = cudf.Series([1, None, 3])
#     assert_eq(x.add(y), cudf.Series([2, None, None]))
#     assert_eq(x.add(y, fill_value=0), cudf.Series([2, 2, 3]))
#
#
# def test_binop_indexed_series_withindex():
#     """Test a binary op (addition) with a non-integral index and an Index."""
#     psr = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
#     pidx = pd.Index([1, None, 3])
#
#     gsr = cudf.Series(psr)
#     gidx = cudf.Index(pidx)
#     assert_eq(psr + pidx, gsr + gidx)
#     assert_eq(psr + pd.Series(pidx), gsr + cudf.Series(gidx))
#     assert_eq(pidx + psr, gidx + gsr)
#     assert_eq(pd.Series(pidx) + psr, cudf.Series(gidx) + gsr)
#
#     # TODO: Also test with arbitrary sequence objects (like lists).
