# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "idx1, idx2",
    [
        (pd.RangeIndex(0, 10), pd.RangeIndex(3, 7)),
        (pd.RangeIndex(0, 10), pd.RangeIndex(10, 20)),
        (pd.RangeIndex(0, 10, 2), pd.RangeIndex(1, 5, 3)),
        (pd.RangeIndex(1, 5, 3), pd.RangeIndex(0, 10, 2)),
        (pd.RangeIndex(1, 10, 3), pd.RangeIndex(1, 5, 2)),
        (pd.RangeIndex(1, 5, 2), pd.RangeIndex(1, 10, 3)),
        (pd.RangeIndex(1, 100, 3), pd.RangeIndex(1, 50, 3)),
        (pd.RangeIndex(1, 100, 3), pd.RangeIndex(1, 50, 6)),
        (pd.RangeIndex(1, 100, 6), pd.RangeIndex(1, 50, 3)),
        (pd.RangeIndex(0, 10, name="a"), pd.RangeIndex(90, 100, name="b")),
        (pd.Index([0, 1, 2, 30], name="a"), pd.Index([90, 100])),
        (pd.Index([0, 1, 2, 30], name="a"), [90, 100]),
        (pd.Index([0, 1, 2, 30]), pd.Index([0, 10, 1.0, 11])),
        (pd.Index(["a", "b", "c", "d", "c"]), pd.Index(["a", "c", "z"])),
        (
            pd.IntervalIndex.from_tuples([(0, 2), (0, 2), (2, 4)]),
            pd.IntervalIndex.from_tuples([(0, 2), (2, 4)]),
        ),
        (pd.RangeIndex(0, 10), pd.Index([8, 1, 2, 4])),
        (pd.Index([8, 1, 2, 4], name="a"), pd.Index([8, 1, 2, 4], name="b")),
        (
            pd.Index([8, 1, 2, 4], name="a"),
            pd.Index([], name="b", dtype="int64"),
        ),
        (pd.Index([], dtype="int64", name="a"), pd.Index([10, 12], name="b")),
        (pd.Index([True, True, True], name="a"), pd.Index([], dtype="bool")),
        (
            pd.Index([True, True, True]),
            pd.Index([False, True], dtype="bool", name="b"),
        ),
    ],
)
@pytest.mark.parametrize("sort", [None, False, True])
def test_union_index(idx1, idx2, sort):
    expected = idx1.union(idx2, sort=sort)

    idx1 = cudf.from_pandas(idx1) if isinstance(idx1, pd.Index) else idx1
    idx2 = cudf.from_pandas(idx2) if isinstance(idx2, pd.Index) else idx2

    actual = idx1.union(idx2, sort=sort)

    assert_eq(expected, actual)


def test_union_bool_with_other():
    idx1 = cudf.Index([True, True, True])
    idx2 = cudf.Index([0, 1], name="b")
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(cudf.errors.MixedTypeError):
            idx1.union(idx2)


def test_union_unsigned_vs_signed(
    signed_integer_types_as_str, unsigned_integer_types_as_str
):
    idx1 = cudf.Index([10, 20, 30], dtype=signed_integer_types_as_str)
    idx2 = cudf.Index([0, 1], dtype=unsigned_integer_types_as_str)
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(cudf.errors.MixedTypeError):
            idx1.union(idx2)
