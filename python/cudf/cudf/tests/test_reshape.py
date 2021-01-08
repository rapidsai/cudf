import re

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

    assert_eq(expect, got)

    assert_eq(expect, got_from_melt_method)


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


@pytest.mark.parametrize(
    "index, column, data",
    [
        ([], [], []),
        ([0], [0], [0]),
        ([0, 0], [0, 1], [1, 2.0]),
        ([0, 1], [0, 0], [1, 2.0]),
        ([0, 1], [0, 1], [1, 2.0]),
        (["a", "a", "b", "b"], ["c", "d", "c", "d"], [1, 2, 3, 4]),
        (
            ["a", "a", "b", "b", "a"],
            ["c", "d", "c", "d", "e"],
            [1, 2, 3, 4, 5],
        ),
    ],
)
def test_pivot_simple(index, column, data):
    pdf = pd.DataFrame({"index": index, "column": column, "data": data})
    gdf = cudf.DataFrame.from_pandas(pdf)

    expect = pdf.pivot("index", "column")
    got = gdf.pivot("index", "column")

    check_index_and_columns = expect.shape != (0, 0)
    assert_eq(
        expect,
        got,
        check_dtype=False,
        check_index_type=check_index_and_columns,
        check_column_type=check_index_and_columns,
    )


def test_pivot_multi_values():
    # from Pandas docs:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot.html
    pdf = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.pivot(index="foo", columns="bar", values=["baz", "zoo"]),
        gdf.pivot(index="foo", columns="bar", values=["baz", "zoo"]),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "level",
    [
        0,
        1,
        2,
        "foo",
        "bar",
        "baz",
        [],
        [0, 1],
        ["foo"],
        ["foo", "bar"],
        pytest.param(
            [0, 1, 2],
            marks=pytest.mark.xfail(reason="Pandas behaviour unclear"),
        ),
        pytest.param(
            ["foo", "bar", "baz"],
            marks=pytest.mark.xfail(reason="Pandas behaviour unclear"),
        ),
    ],
)
def test_unstack_multiindex(level):
    pdf = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    ).set_index(["foo", "bar", "baz"])
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.unstack(level=level), gdf.unstack(level=level), check_dtype=False,
    )


@pytest.mark.parametrize(
    "data",
    [{"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [11.0, 12.0, 13.0, 14.0, 15.0]}],
)
@pytest.mark.parametrize(
    "index",
    [
        pd.Index(range(0, 5), name=None),
        pd.Index(range(0, 5), name="row_index"),
    ],
)
@pytest.mark.parametrize(
    "col_idx",
    [
        pd.Index(["a", "b"], name=None),
        pd.Index(["a", "b"], name="col_index"),
        pd.MultiIndex.from_tuples([("c", 1), ("c", 2)], names=[None, None]),
        pd.MultiIndex.from_tuples(
            [("c", 1), ("c", 2)], names=["col_index1", "col_index2"]
        ),
    ],
)
def test_unstack_index(data, index, col_idx):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    pdf.index = index
    pdf.columns = col_idx

    gdf.index = cudf.from_pandas(index)
    gdf.columns = cudf.from_pandas(col_idx)

    assert_eq(pdf.unstack(), gdf.unstack())


def test_unstack_index_invalid():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Calling unstack() on single index dataframe with "
            "different column datatype is not supported."
        ),
    ):
        gdf.unstack()


def test_pivot_duplicate_error():
    gdf = cudf.DataFrame(
        {"a": [0, 1, 2, 2], "b": [1, 2, 3, 3], "d": [1, 2, 3, 4]}
    )
    with pytest.raises(ValueError):
        gdf.pivot(index="a", columns="b")
    with pytest.raises(ValueError):
        gdf.pivot(index="b", columns="a")
