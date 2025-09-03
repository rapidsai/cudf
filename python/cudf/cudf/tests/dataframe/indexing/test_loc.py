# Copyright (c) 2025, NVIDIA CORPORATION.
import re

import numpy as np
import pandas as pd
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


def test_multiindex_wildcard_selection_all():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2) for c1 in "abc" for c2 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(6)})
    df.columns = midx
    expect = df.to_pandas().loc[:, (slice(None), "b")]
    got = df.loc[:, (slice(None), "b")]
    assert_eq(expect, got)


@pytest.mark.xfail(reason="Not yet properly supported.")
def test_multiindex_wildcard_selection_partial():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2) for c1 in "abc" for c2 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(6)})
    df.columns = midx
    expect = df.to_pandas().loc[:, (slice("a", "b"), "b")]
    got = df.loc[:, (slice("a", "b"), "b")]
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "value",
    [
        "7",
        pytest.param(
            ["7", "8"],
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/11298"
            ),
        ),
    ],
)
def test_loc_setitem_string_11298(value):
    df = pd.DataFrame({"a": ["a", "b", "c"]})
    cdf = cudf.from_pandas(df)

    df.loc[:1, "a"] = value

    cdf.loc[:1, "a"] = value

    assert_eq(df, cdf)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/11944")
def test_loc_setitem_list_11944():
    df = pd.DataFrame(
        data={"a": ["yes", "no"], "b": [["l1", "l2"], ["c", "d"]]}
    )
    cdf = cudf.from_pandas(df)
    df.loc[df.a == "yes", "b"] = [["hello"]]
    cdf.loc[cdf.a == "yes", "b"] = [["hello"]]
    assert_eq(df, cdf)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/12504")
def test_loc_setitem_extend_empty_12504():
    df = pd.DataFrame(columns=["a"])
    cdf = cudf.from_pandas(df)

    df.loc[0] = [1]

    cdf.loc[0] = [1]

    assert_eq(df, cdf)


def test_loc_setitem_extend_existing_12505():
    df = pd.DataFrame({"a": [0]})
    cdf = cudf.from_pandas(df)

    df.loc[1] = 1

    cdf.loc[1] = 1

    assert_eq(df, cdf)


def test_loc_setitem_list_arg_missing_raises():
    data = {"a": [0]}
    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    cudf_msg = re.escape("[1] not in the index.")
    with pytest.raises(KeyError, match=cudf_msg):
        gdf.loc[[1]] = 1

    with pytest.raises(KeyError, match=cudf_msg):
        gdf.loc[[1], "a"] = 1

    with pytest.raises(KeyError):
        pdf.loc[[1]] = 1

    with pytest.raises(KeyError):
        pdf.loc[[1], "a"] = 1


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/12801")
def test_loc_setitem_add_column_partial_12801():
    df = pd.DataFrame({"a": [0, 1, 2]})
    cdf = cudf.from_pandas(df)

    df.loc[df.a < 2, "b"] = 1

    cdf.loc[cdf.a < 2, "b"] = 1

    assert_eq(df, cdf)
