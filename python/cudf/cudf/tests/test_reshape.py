import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import melt as cudf_melt
from cudf.core import DataFrame
from cudf.tests.utils import (
    ALL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    assert_eq,
)


@pytest.mark.parametrize("num_id_vars", [0, 1, 2, 10])
@pytest.mark.parametrize("num_value_vars", [0, 1, 2, 10])
@pytest.mark.parametrize("num_rows", [1, 2, 1000])
@pytest.mark.parametrize("dtype", NUMERIC_TYPES + DATETIME_TYPES)
@pytest.mark.parametrize("nulls", ["none", "some", "all"])
def test_melt(nulls, num_id_vars, num_value_vars, num_rows, dtype):
    if dtype not in ["float32", "float64"] and nulls in ["some", "all"]:
        pytest.skip(msg="nulls not supported in dtype: " + dtype)

    pdf = pd.DataFrame()
    id_vars = []
    for i in range(num_id_vars):
        colname = "id" + str(i)
        data = np.random.randint(0, 26, num_rows).astype(dtype)
        if nulls == "some":
            idx = np.random.choice(
                num_rows, size=int(num_rows / 2), replace=False
            )
            data[idx] = np.nan
        elif nulls == "all":
            data[:] = np.nan
        pdf[colname] = data
        id_vars.append(colname)

    value_vars = []
    for i in range(num_value_vars):
        colname = "val" + str(i)
        data = np.random.randint(0, 26, num_rows).astype(dtype)
        if nulls == "some":
            idx = np.random.choice(
                num_rows, size=int(num_rows / 2), replace=False
            )
            data[idx] = np.nan
        elif nulls == "all":
            data[:] = np.nan
        pdf[colname] = data
        value_vars.append(colname)

    gdf = DataFrame.from_pandas(pdf)

    got = cudf_melt(frame=gdf, id_vars=id_vars, value_vars=value_vars)
    got_from_melt_method = gdf.melt(id_vars=id_vars, value_vars=value_vars)

    expect = pd.melt(frame=pdf, id_vars=id_vars, value_vars=value_vars)
    # pandas' melt makes the 'variable' column of 'object' type (string)
    # cuDF's melt makes it Categorical because it doesn't support strings
    expect["variable"] = expect["variable"].astype("category")

    pd.testing.assert_frame_equal(expect, got.to_pandas())

    pd.testing.assert_frame_equal(expect, got_from_melt_method.to_pandas())


@pytest.mark.parametrize("num_cols", [1, 2, 10])
@pytest.mark.parametrize("num_rows", [1, 2, 1000])
@pytest.mark.parametrize(
    "dtype",
    list(NUMERIC_TYPES + DATETIME_TYPES)
    + [pytest.param("str", marks=pytest.mark.xfail())],
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_df_stack(nulls, num_cols, num_rows, dtype):
    if dtype not in ["float32", "float64"] and nulls in ["some"]:
        pytest.skip(msg="nulls not supported in dtype: " + dtype)

    pdf = pd.DataFrame()
    for i in range(num_cols):
        colname = str(i)
        data = np.random.randint(0, 26, num_rows).astype(dtype)
        if nulls == "some":
            idx = np.random.choice(
                num_rows, size=int(num_rows / 2), replace=False
            )
            data[idx] = np.nan
        pdf[colname] = data

    gdf = DataFrame.from_pandas(pdf)

    got = gdf.stack()

    expect = pdf.stack()
    if {None} == set(expect.index.names):
        expect.rename_axis(
            list(range(0, len(expect.index.names))), inplace=True
        )

    assert_eq(expect, got)
    pass


@pytest.mark.parametrize("num_rows", [1, 2, 10, 1000])
@pytest.mark.parametrize("num_cols", [1, 2, 10])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + ["category"]
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_interleave_columns(nulls, num_cols, num_rows, dtype):

    if dtype not in ["float32", "float64"] and nulls in ["some"]:
        pytest.skip(msg="nulls not supported in dtype: " + dtype)

    pdf = pd.DataFrame(dtype=dtype)
    for i in range(num_cols):
        colname = str(i)
        data = pd.Series(np.random.randint(0, 26, num_rows)).astype(dtype)

        if nulls == "some":
            idx = np.random.choice(
                num_rows, size=int(num_rows / 2), replace=False
            )
            data[idx] = np.nan
        pdf[colname] = data

    gdf = DataFrame.from_pandas(pdf)

    if dtype == "category":
        with pytest.raises(ValueError):
            assert gdf.interleave_columns()
    else:
        got = gdf.interleave_columns()

        expect = pd.Series(np.vstack(pdf.to_numpy()).reshape((-1,))).astype(
            dtype
        )

        assert_eq(expect, got)


@pytest.mark.parametrize("num_cols", [1, 2, 10])
@pytest.mark.parametrize("num_rows", [1, 2, 1000])
@pytest.mark.parametrize("count", [1, 2, 10])
@pytest.mark.parametrize("dtype", ALL_TYPES)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_tile(nulls, num_cols, num_rows, dtype, count):

    if dtype not in ["float32", "float64"] and nulls in ["some"]:
        pytest.skip(msg="nulls not supported in dtype: " + dtype)

    pdf = pd.DataFrame(dtype=dtype)
    for i in range(num_cols):
        colname = str(i)
        data = pd.Series(np.random.randint(num_cols, 26, num_rows)).astype(
            dtype
        )

        if nulls == "some":
            idx = np.random.choice(
                num_rows, size=int(num_rows / 2), replace=False
            )
            data[idx] = np.nan
        pdf[colname] = data

    gdf = DataFrame.from_pandas(pdf)

    got = gdf.tile(count)
    expect = pd.DataFrame(pd.concat([pdf] * count))

    assert_eq(expect, got)


def _prepare_merge_sorted_test(
    size,
    nparts,
    keys,
    add_null=False,
    na_position="last",
    ascending=True,
    series=False,
    index=False,
):
    if index:
        df = (
            cudf.datasets.timeseries()[:size]
            .reset_index(drop=False)
            .set_index(keys, drop=True)
        )
    else:
        df = cudf.datasets.timeseries()[:size].reset_index(drop=False)
        if add_null:
            df.iloc[1, df.columns.get_loc(keys[0])] = None
    chunk = int(size / nparts)
    indices = [i * chunk for i in range(0, nparts)] + [size]
    if index:
        dfs = [
            df.iloc[indices[i] : indices[i + 1]]
            .copy()
            .sort_index(ascending=ascending)
            for i in range(nparts)
        ]
    elif series:
        df = df[keys[0]]
        dfs = [
            df.iloc[indices[i] : indices[i + 1]]
            .copy()
            .sort_values(na_position=na_position, ascending=ascending)
            for i in range(nparts)
        ]
    else:
        dfs = [
            df.iloc[indices[i] : indices[i + 1]]
            .copy()
            .sort_values(keys, na_position=na_position, ascending=ascending)
            for i in range(nparts)
        ]
    return df, dfs


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
@pytest.mark.parametrize("keys", [None, ["id"], ["name", "timestamp"]])
@pytest.mark.parametrize("nparts", [2, 10])
def test_df_merge_sorted(nparts, keys, na_position, ascending):
    size = 100
    keys_1 = keys or ["timestamp"]
    # Null values NOT currently supported with Categorical data
    # or when `ascending=False`
    add_null = keys_1[0] not in ("name")
    df, dfs = _prepare_merge_sorted_test(
        size,
        nparts,
        keys_1,
        add_null=add_null,
        na_position=na_position,
        ascending=ascending,
    )

    expect = df.sort_values(
        keys_1, na_position=na_position, ascending=ascending
    )
    result = cudf.merge_sorted(
        dfs, keys=keys, na_position=na_position, ascending=ascending
    )
    if keys:
        expect = expect[keys]
        result = result[keys]

    assert expect.index.dtype == result.index.dtype
    assert_eq(expect.reset_index(drop=True), result.reset_index(drop=True))


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("index", ["id", "x"])
@pytest.mark.parametrize("nparts", [2, 10])
def test_df_merge_sorted_index(nparts, index, ascending):
    size = 100
    df, dfs = _prepare_merge_sorted_test(
        size, nparts, index, ascending=ascending, index=True
    )

    expect = df.sort_index(ascending=ascending)
    result = cudf.merge_sorted(dfs, by_index=True, ascending=ascending)

    assert_eq(expect.index, result.index)


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
@pytest.mark.parametrize("keys", [None, ["name", "timestamp"]])
def test_df_merge_sorted_ignore_index(keys, na_position, ascending):
    size = 100
    nparts = 3
    keys_1 = keys or ["timestamp"]
    # Null values NOT currently supported with Categorical data
    # or when `ascending=False`
    add_null = keys_1[0] not in ("name")
    df, dfs = _prepare_merge_sorted_test(
        size,
        nparts,
        keys_1,
        add_null=add_null,
        na_position=na_position,
        ascending=ascending,
    )

    expect = df.sort_values(
        keys_1, na_position=na_position, ascending=ascending
    )
    result = cudf.merge_sorted(
        dfs,
        keys=keys,
        na_position=na_position,
        ascending=ascending,
        ignore_index=True,
    )
    if keys:
        expect = expect[keys]
        result = result[keys]

    assert_eq(expect.reset_index(drop=True), result)


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
@pytest.mark.parametrize("key", ["id", "name", "timestamp"])
@pytest.mark.parametrize("nparts", [2, 10])
def test_series_merge_sorted(nparts, key, na_position, ascending):
    size = 100
    df, dfs = _prepare_merge_sorted_test(
        size,
        nparts,
        [key],
        na_position=na_position,
        ascending=ascending,
        series=True,
    )

    expect = df.sort_values(na_position=na_position, ascending=ascending)
    result = cudf.merge_sorted(
        dfs, na_position=na_position, ascending=ascending
    )

    assert_eq(expect.reset_index(drop=True), result.reset_index(drop=True))
