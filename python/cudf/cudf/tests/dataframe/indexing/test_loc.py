# Copyright (c) 2025, NVIDIA CORPORATION.
import re

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


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
