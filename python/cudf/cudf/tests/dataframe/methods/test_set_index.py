# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "index",
    [
        "a",
        ["a", "b"],
        pd.CategoricalIndex(["I", "II", "III", "IV", "V"]),
        pd.Series(["h", "i", "k", "l", "m"]),
        ["b", pd.Index(["I", "II", "III", "IV", "V"])],
        ["c", [11, 12, 13, 14, 15]],
        pd.MultiIndex(
            levels=[
                ["I", "II", "III", "IV", "V"],
                ["one", "two", "three", "four", "five"],
            ],
            codes=[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]],
            names=["col1", "col2"],
        ),
        pd.RangeIndex(0, 5),  # corner case
        [pd.Series(["h", "i", "k", "l", "m"]), pd.RangeIndex(0, 5)],
        [
            pd.MultiIndex(
                levels=[
                    ["I", "II", "III", "IV", "V"],
                    ["one", "two", "three", "four", "five"],
                ],
                codes=[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]],
                names=["col1", "col2"],
            ),
            pd.RangeIndex(0, 5),
        ],
    ],
)
@pytest.mark.parametrize("append", [True, False])
def test_set_index(index, drop, append, inplace):
    gdf = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "b", "c", "d", "e"],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    pdf = gdf.to_pandas()

    expected = pdf.set_index(index, inplace=inplace, drop=drop, append=append)
    actual = gdf.set_index(index, inplace=inplace, drop=drop, append=append)

    if inplace:
        expected = pdf
        actual = gdf
    assert_eq(expected, actual)


@pytest.mark.parametrize("index", ["a", pd.Index([1, 1, 2, 2, 3])])
def test_set_index_verify_integrity(index):
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 2, 2, 5],
            "b": ["a", "b", "c", "d", "e"],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    with pytest.raises(ValueError):
        gdf.set_index(index, verify_integrity=True)


def test_set_index_multi(drop):
    nelem = 10
    rng = np.random.default_rng(seed=0)
    a = np.arange(nelem)
    rng.shuffle(a)
    df = pd.DataFrame(
        {
            "a": a,
            "b": rng.integers(0, 4, size=nelem),
            "c": rng.uniform(low=0, high=4, size=nelem),
            "d": rng.choice(["green", "black", "white"], nelem),
        }
    )
    df["e"] = df["d"].astype("category")
    gdf = cudf.DataFrame.from_pandas(df)

    assert_eq(gdf.set_index("a", drop=drop), gdf.set_index(["a"], drop=drop))
    assert_eq(
        df.set_index(["b", "c"], drop=drop),
        gdf.set_index(["b", "c"], drop=drop),
    )
    assert_eq(
        df.set_index(["d", "b"], drop=drop),
        gdf.set_index(["d", "b"], drop=drop),
    )
    assert_eq(
        df.set_index(["b", "d", "e"], drop=drop),
        gdf.set_index(["b", "d", "e"], drop=drop),
    )


def test_df_cat_set_index():
    df = cudf.DataFrame(
        {
            "a": pd.Categorical(list("aababcabbc"), categories=list("abc")),
            "b": np.arange(10),
        }
    )
    got = df.set_index("a")

    pddf = df.to_pandas()
    expect = pddf.set_index("a")

    assert_eq(got, expect)
