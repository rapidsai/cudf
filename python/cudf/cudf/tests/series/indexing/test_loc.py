# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/13031")
@pytest.mark.parametrize("other_index", [["1", "3", "2"], [1, 2, 3]])
def test_loc_setitem_series_index_alignment_13031(other_index):
    s = pd.Series([1, 2, 3], index=["1", "2", "3"])
    other = pd.Series([5, 6, 7], index=other_index)

    cs = cudf.from_pandas(s)
    cother = cudf.from_pandas(other)

    s.loc[["1", "3"]] = other

    cs.loc[["1", "3"]] = cother

    assert_eq(s, cs)


def test_series_set_item_index_reference():
    gs1 = cudf.Series([1], index=[7])
    gs2 = cudf.Series([2], index=gs1.index)

    gs1.loc[11] = 2
    ps1 = pd.Series([1], index=[7])
    ps2 = pd.Series([2], index=ps1.index)
    ps1.loc[11] = 2

    assert_eq(ps1, gs1)
    assert_eq(ps2, gs2)
