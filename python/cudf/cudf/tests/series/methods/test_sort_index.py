# Copyright (c) 2020-2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(0, 3, 1),
        [3.0, 1.0, np.nan],
        ["a", "z", None],
        pd.RangeIndex(4, -1, -2),
    ],
)
@pytest.mark.parametrize("axis", [0, "index"])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_series_sort_index(
    index, axis, ascending, inplace, ignore_index, na_position
):
    ps = pd.Series([10, 3, 12], index=index)
    gs = cudf.from_pandas(ps)

    expected = ps.sort_index(
        axis=axis,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )
    got = gs.sort_index(
        axis=axis,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )

    if inplace is True:
        assert_eq(ps, gs, check_index_type=True)
    else:
        assert_eq(expected, got, check_index_type=True)
