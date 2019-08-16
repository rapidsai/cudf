# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf as gd
from cudf.tests.utils import assert_eq


def make_frames(index=None, nulls="none"):
    df = pd.DataFrame(
        {
            "x": range(10),
            "y": list(map(float, range(10))),
            "z": list("abcde") * 2,
        }
    )
    df.z = df.z.astype("category")
    df2 = pd.DataFrame(
        {
            "x": range(10, 20),
            "y": list(map(float, range(10, 20))),
            "z": list("edcba") * 2,
        }
    )
    df2.z = df2.z.astype("category")
    if nulls == "all":
        df.y = np.full_like(df.y, np.nan)
        df2.y = np.full_like(df2.y, np.nan)
    if nulls == "some":
        mask = np.arange(10)
        np.random.shuffle(mask)
        mask = mask[:5]
        df.y.loc[mask] = np.nan
        df2.y.loc[mask] = np.nan
    gdf = gd.DataFrame.from_pandas(df)
    gdf2 = gd.DataFrame.from_pandas(df2)
    if index:
        df = df.set_index(index)
        df2 = df2.set_index(index)
        gdf = gdf.set_index(index)
        gdf2 = gdf2.set_index(index)
    return df, df2, gdf, gdf2


@pytest.mark.parametrize("nulls", ["none", "some", "all"])
@pytest.mark.parametrize("index", [False, "z", "y"])
def test_concat(index, nulls):
    if index == "y" and nulls in ("some", "all"):
        pytest.skip("nulls in columns, dont index")
    df, df2, gdf, gdf2 = make_frames(index, nulls=nulls)
    # Make empty frame
    gdf_empty1 = gdf2[:0]
    assert len(gdf_empty1) == 0
    df_empty1 = gdf_empty1.to_pandas()

    # DataFrame
    res = gd.concat([gdf, gdf2, gdf, gdf_empty1]).to_pandas()
    sol = pd.concat([df, df2, df, df_empty1])
    pd.util.testing.assert_frame_equal(
        res, sol, check_names=False, check_categorical=False
    )

    # Series
    for c in [i for i in ("x", "y", "z") if i != index]:
        res = gd.concat([gdf[c], gdf2[c], gdf[c]]).to_pandas()
        sol = pd.concat([df[c], df2[c], df[c]])
        pd.util.testing.assert_series_equal(
            res, sol, check_names=False, check_categorical=False
        )

    # Index
    res = gd.concat([gdf.index, gdf2.index]).to_pandas()
    sol = df.index.append(df2.index)
    pd.util.testing.assert_index_equal(
        res, sol, check_names=False, check_categorical=False
    )


@pytest.mark.parametrize(
    "values",
    [["foo", "bar"], [1.0, 2.0], pd.Series(["one", "two"], dtype="category")],
)
def test_concat_all_nulls(values):
    pa = pd.Series(values)
    pb = pd.Series([None])
    ps = pd.concat([pa, pb])

    ga = gd.Series(values)
    gb = gd.Series([None])
    gs = gd.concat([ga, gb])

    assert_eq(ps, gs, check_dtype=False, check_categorical=False)


def test_concat_errors():
    df, df2, gdf, gdf2 = make_frames()

    # No objs
    with pytest.raises(ValueError):
        gd.concat([])

    # Mismatched types
    with pytest.raises(ValueError):
        gd.concat([gdf, gdf.x])

    # Unknown type
    with pytest.raises(ValueError):
        gd.concat(["bar", "foo"])

    # Mismatched index dtypes
    gdf3 = gdf2.set_index("z")
    del gdf2["z"]
    with pytest.raises(ValueError):
        gd.concat([gdf2, gdf3])

    # Mismatched columns
    with pytest.raises(ValueError):
        gd.concat([gdf, gdf2])


def test_concat_misordered_columns():
    df, df2, gdf, gdf2 = make_frames(False)
    gdf2 = gdf2[["z", "x", "y"]]
    df2 = df2[["z", "x", "y"]]

    res = gd.concat([gdf, gdf2]).to_pandas()
    sol = pd.concat([df, df2], sort=False)

    pd.util.testing.assert_frame_equal(
        res, sol, check_names=False, check_categorical=False
    )
