# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


def test_dataframe_midx_columns_loc():
    idx_1 = ["Hi", "Lo"]
    idx_2 = ["I", "II", "III"]
    idx = cudf.MultiIndex.from_product([idx_1, idx_2])

    data_rand = (
        np.random.default_rng(seed=0)
        .uniform(0, 1, 3 * len(idx))
        .reshape(3, -1)
    )
    df = cudf.DataFrame(data_rand, index=["A", "B", "C"], columns=idx)
    pdf = df.to_pandas()

    assert_eq(df.shape, pdf.shape)

    expected = pdf.loc[["A", "B"]]
    actual = df.loc[["A", "B"]]

    assert_eq(expected, actual)
    assert_eq(df, pdf)


@pytest.mark.parametrize("dtype1", ["int16", "float32"])
@pytest.mark.parametrize("dtype2", ["int16", "float32"])
def test_dataframe_loc_int_float(dtype1, dtype2):
    df = cudf.DataFrame(
        {"a": [10, 11, 12, 13, 14]},
        index=cudf.Index([1, 2, 3, 4, 5], dtype=dtype1),
    )
    pdf = df.to_pandas()

    gidx = cudf.Index([2, 3, 4], dtype=dtype2)
    pidx = gidx.to_pandas()

    actual = df.loc[gidx]
    expected = pdf.loc[pidx]

    assert_eq(actual, expected, check_index_type=True, check_dtype=True)


@pytest.mark.xfail(reason="Not yet properly supported.")
def test_multiindex_wildcard_selection_three_level_all():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2, c3) for c1 in "abcd" for c2 in "abc" for c3 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(24)})
    df.columns = midx

    expect = df.to_pandas().loc[:, (slice("a", "c"), slice("a", "b"), "b")]
    got = df.loc[:, (slice(None), "b")]
    assert_eq(expect, got)
