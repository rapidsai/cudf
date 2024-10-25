# Copyright (c) 2021-2024, NVIDIA CORPORATION.

import re
from itertools import chain

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_GE_220,
    PANDAS_VERSION,
)
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.testing import assert_eq
from cudf.testing._utils import (
    ALL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    expect_warning_if,
)

pytest_xfail = pytest.mark.xfail
pytestmark = pytest.mark.spilling

# If spilling is enabled globally, we skip many test permutations
# to reduce running time.
if get_global_manager() is not None:
    ALL_TYPES = ["float32"]  # noqa: F811
    DATETIME_TYPES = ["datetime64[ms]"]  # noqa: F811
    NUMERIC_TYPES = ["float32"]  # noqa: F811
    # To save time, we skip tests marked "pytest.mark.xfail"
    pytest_xfail = pytest.mark.skipif


@pytest.mark.parametrize("num_id_vars", [0, 1, 2])
@pytest.mark.parametrize("num_value_vars", [0, 1, 2])
@pytest.mark.parametrize("num_rows", [1, 2, 100])
@pytest.mark.parametrize("dtype", NUMERIC_TYPES + DATETIME_TYPES)
@pytest.mark.parametrize("nulls", ["none", "some", "all"])
def test_melt(nulls, num_id_vars, num_value_vars, num_rows, dtype):
    if dtype not in ["float32", "float64"] and nulls in ["some", "all"]:
        pytest.skip(reason="nulls not supported in dtype: " + dtype)

    pdf = pd.DataFrame()
    id_vars = []
    rng = np.random.default_rng(seed=0)
    for i in range(num_id_vars):
        colname = "id" + str(i)
        data = rng.integers(0, 26, num_rows).astype(dtype)
        if nulls == "some":
            idx = rng.choice(num_rows, size=int(num_rows / 2), replace=False)
            data[idx] = np.nan
        elif nulls == "all":
            data[:] = np.nan
        pdf[colname] = data
        id_vars.append(colname)

    value_vars = []
    for i in range(num_value_vars):
        colname = "val" + str(i)
        data = rng.integers(0, 26, num_rows).astype(dtype)
        if nulls == "some":
            idx = rng.choice(num_rows, size=int(num_rows / 2), replace=False)
            data[idx] = np.nan
        elif nulls == "all":
            data[:] = np.nan
        pdf[colname] = data
        value_vars.append(colname)

    gdf = cudf.from_pandas(pdf)

    got = cudf.melt(frame=gdf, id_vars=id_vars, value_vars=value_vars)
    got_from_melt_method = gdf.melt(id_vars=id_vars, value_vars=value_vars)

    expect = pd.melt(frame=pdf, id_vars=id_vars, value_vars=value_vars)

    assert_eq(expect, got)

    assert_eq(expect, got_from_melt_method)


def test_melt_many_columns():
    mydict = {"id": ["foobar"]}
    for i in range(1, 1942):
        mydict[f"d_{i}"] = i

    df = pd.DataFrame(mydict)
    grid_df = pd.melt(df, id_vars=["id"], var_name="d", value_name="sales")

    df_d = cudf.DataFrame(mydict)
    grid_df_d = cudf.melt(
        df_d, id_vars=["id"], var_name="d", value_name="sales"
    )
    grid_df_d["d"] = grid_df_d["d"]

    assert_eq(grid_df, grid_df_d)


def test_melt_str_scalar_id_var():
    data = {"index": [1, 2], "id": [1, 2], "d0": [10, 20], "d1": [30, 40]}
    result = cudf.melt(
        cudf.DataFrame(data),
        id_vars="index",
        var_name="column",
        value_name="value",
    )
    expected = pd.melt(
        pd.DataFrame(data),
        id_vars="index",
        var_name="column",
        value_name="value",
    )
    assert_eq(result, expected)


def test_melt_falsy_var_name():
    df = cudf.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})
    result = cudf.melt(df, id_vars=["A"], value_vars=["B"], var_name="")
    expected = pd.melt(
        df.to_pandas(), id_vars=["A"], value_vars=["B"], var_name=""
    )
    assert_eq(result, expected)


@pytest.mark.parametrize("num_cols", [1, 2, 10])
@pytest.mark.parametrize("num_rows", [1, 2, 1000])
@pytest.mark.parametrize(
    "dtype", list(chain(NUMERIC_TYPES, DATETIME_TYPES, ["str"]))
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_df_stack(nulls, num_cols, num_rows, dtype):
    if dtype not in ["float32", "float64"] and nulls in ["some"]:
        pytest.skip(reason="nulls not supported in dtype: " + dtype)

    pdf = pd.DataFrame()
    rng = np.random.default_rng(seed=0)
    for i in range(num_cols):
        colname = str(i)
        data = rng.integers(0, 26, num_rows).astype(dtype)
        if nulls == "some":
            idx = rng.choice(num_rows, size=int(num_rows / 2), replace=False)
            data[idx] = np.nan
        pdf[colname] = data

    gdf = cudf.from_pandas(pdf)

    got = gdf.stack()
    expect = pdf.stack()

    assert_eq(expect, got)


def test_df_stack_reset_index():
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [10, 11, 12, 13],
            "c": ["ab", "cd", None, "gh"],
        }
    )
    df = df.set_index(["a", "b"])
    pdf = df.to_pandas()

    expected = pdf.stack()
    actual = df.stack()

    assert_eq(expected, actual)

    expected = expected.reset_index()
    actual = actual.reset_index()

    assert_eq(expected, actual)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Need pandas-2.1.0+ to match `stack` api",
)
@pytest.mark.parametrize(
    "columns",
    [
        pd.MultiIndex.from_tuples(
            [("A", "cat"), ("A", "dog"), ("B", "cat"), ("B", "dog")],
            names=["letter", "animal"],
        ),
        pd.MultiIndex.from_tuples(
            [("A", "cat"), ("B", "bird"), ("A", "dog"), ("B", "dog")],
            names=["letter", "animal"],
        ),
    ],
)
@pytest.mark.parametrize(
    "level",
    [
        -1,
        0,
        1,
        "letter",
        "animal",
        [0, 1],
        [1, 0],
        ["letter", "animal"],
        ["animal", "letter"],
    ],
)
@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(2, name="range"),
        pd.Index([9, 8], name="myindex"),
        pd.MultiIndex.from_arrays(
            [
                ["A", "B"],
                [101, 102],
            ],
            names=["first", "second"],
        ),
    ],
)
@pytest.mark.parametrize("dropna", [True, False])
def test_df_stack_multiindex_column_axis(columns, index, level, dropna):
    if isinstance(level, list) and len(level) > 1 and not dropna:
        pytest.skip(
            "Stacking multiple levels with dropna==False is unsupported."
        )

    pdf = pd.DataFrame(
        data=[[1, 2, 3, 4], [2, 4, 6, 8]], columns=columns, index=index
    )
    gdf = cudf.from_pandas(pdf)

    with pytest.warns(FutureWarning):
        got = gdf.stack(level=level, dropna=dropna, future_stack=False)
    with expect_warning_if(PANDAS_GE_220, FutureWarning):
        expect = pdf.stack(level=level, dropna=dropna, future_stack=False)

    assert_eq(expect, got, check_dtype=False)

    got = gdf.stack(level=level, future_stack=True)
    expect = pdf.stack(level=level, future_stack=True)

    assert_eq(expect, got, check_dtype=False)


def test_df_stack_mixed_dtypes():
    pdf = pd.DataFrame(
        {
            "A": pd.Series([1, 2, 3], dtype="f4"),
            "B": pd.Series([4, 5, 6], dtype="f8"),
        }
    )

    gdf = cudf.from_pandas(pdf)

    got = gdf.stack()
    expect = pdf.stack()

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Need pandas-2.1.0+ to match `stack` api",
)
@pytest.mark.parametrize("level", [["animal", "hair_length"], [1, 2]])
def test_df_stack_multiindex_column_axis_pd_example(level):
    columns = pd.MultiIndex.from_tuples(
        [
            ("A", "cat", "long"),
            ("B", "cat", "long"),
            ("A", "dog", "short"),
            ("B", "dog", "short"),
        ],
        names=["exp", "animal", "hair_length"],
    )
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(rng.standard_normal(size=(4, 4)), columns=columns)

    with expect_warning_if(PANDAS_GE_220, FutureWarning):
        expect = df.stack(level=level, future_stack=False)
    gdf = cudf.from_pandas(df)
    with pytest.warns(FutureWarning):
        got = gdf.stack(level=level, future_stack=False)

    assert_eq(expect, got)

    expect = df.stack(level=level, future_stack=True)
    got = gdf.stack(level=level, future_stack=True)

    assert_eq(expect, got)


@pytest.mark.parametrize("num_rows", [1, 2, 10, 1000])
@pytest.mark.parametrize("num_cols", [1, 2, 10])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + ["category"]
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_interleave_columns(nulls, num_cols, num_rows, dtype):
    if dtype not in ["float32", "float64"] and nulls in ["some"]:
        pytest.skip(reason="nulls not supported in dtype: " + dtype)

    pdf = pd.DataFrame(dtype=dtype)
    rng = np.random.default_rng(seed=0)
    for i in range(num_cols):
        colname = str(i)
        data = pd.Series(rng.integers(0, 26, num_rows)).astype(dtype)

        if nulls == "some":
            idx = rng.choice(num_rows, size=int(num_rows / 2), replace=False)
            data[idx] = np.nan
        pdf[colname] = data

    gdf = cudf.from_pandas(pdf)

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
        pytest.skip(reason="nulls not supported in dtype: " + dtype)

    pdf = pd.DataFrame(dtype=dtype)
    rng = np.random.default_rng(seed=0)
    for i in range(num_cols):
        colname = str(i)
        data = pd.Series(rng.integers(num_cols, 26, num_rows)).astype(dtype)

        if nulls == "some":
            idx = rng.choice(num_rows, size=int(num_rows / 2), replace=False)
            data[idx] = np.nan
        pdf[colname] = data

    gdf = cudf.from_pandas(pdf)

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
    result = cudf.core.reshape._merge_sorted(
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
    result = cudf.core.reshape._merge_sorted(
        dfs, by_index=True, ascending=ascending
    )

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
    result = cudf.core.reshape._merge_sorted(
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
    result = cudf.core.reshape._merge_sorted(
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
    gdf = cudf.from_pandas(pdf)

    expect = pdf.pivot(columns="column", index="index")
    got = gdf.pivot(columns="column", index="index")

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
    "values", ["z", "z123", ["z123"], ["z", "z123", "123z"]]
)
def test_pivot_values(values):
    data = [
        ["A", "a", 0, 0, 0],
        ["A", "b", 1, 1, 1],
        ["A", "c", 2, 2, 2],
        ["B", "a", 0, 0, 0],
        ["B", "b", 1, 1, 1],
        ["B", "c", 2, 2, 2],
        ["C", "a", 0, 0, 0],
        ["C", "b", 1, 1, 1],
        ["C", "c", 2, 2, 2],
    ]
    columns = ["x", "y", "z", "z123", "123z"]
    pdf = pd.DataFrame(data, columns=columns)
    cdf = cudf.DataFrame(data, columns=columns)
    expected = pd.pivot(pdf, index="x", columns="y", values=values)
    actual = cudf.pivot(cdf, index="x", columns="y", values=values)
    assert_eq(
        expected,
        actual,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "level",
    [
        0,
        pytest.param(
            1,
            marks=pytest_xfail(
                reason="Categorical column indexes not supported"
            ),
        ),
        2,
        "foo",
        pytest.param(
            "bar",
            marks=pytest_xfail(
                reason="Categorical column indexes not supported"
            ),
        ),
        "baz",
        [],
        pytest.param(
            [0, 1],
            marks=pytest_xfail(
                reason="Categorical column indexes not supported"
            ),
        ),
        ["foo"],
        pytest.param(
            ["foo", "bar"],
            marks=pytest_xfail(
                reason="Categorical column indexes not supported"
            ),
        ),
        pytest.param(
            [0, 1, 2],
            marks=pytest_xfail(reason="Pandas behaviour unclear"),
        ),
        pytest.param(
            ["foo", "bar", "baz"],
            marks=pytest_xfail(reason="Pandas behaviour unclear"),
        ),
    ],
)
def test_unstack_multiindex(level):
    pdf = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": pd.Categorical(["A", "B", "C", "A", "B", "C"]),
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    ).set_index(["foo", "bar", "baz"])
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.unstack(level=level),
        gdf.unstack(level=level),
        check_dtype=False,
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
        pytest.param(
            pd.CategoricalIndex(["d", "e", "f", "g", "h"]),
            marks=pytest_xfail(
                reason="Categorical column indexes not supported"
            ),
        ),
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


@pytest.mark.parametrize(
    "aggfunc", ["mean", "count", {"D": "sum", "E": "count"}]
)
@pytest.mark.parametrize("fill_value", [0])
def test_pivot_table_simple(aggfunc, fill_value):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "A": ["one", "one", "two", "three"] * 6,
            "B": ["A", "B", "C"] * 8,
            "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
            "D": rng.standard_normal(size=24),
            "E": rng.standard_normal(size=24),
        }
    )
    expected = pd.pivot_table(
        pdf,
        values=["D", "E"],
        index=["A", "B"],
        columns=["C"],
        aggfunc=aggfunc,
        fill_value=fill_value,
    )
    cdf = cudf.DataFrame.from_pandas(pdf)
    actual = cudf.pivot_table(
        cdf,
        values=["D", "E"],
        index=["A", "B"],
        columns=["C"],
        aggfunc=aggfunc,
        fill_value=fill_value,
    )
    assert_eq(expected, actual, check_dtype=False)


@pytest.mark.parametrize(
    "aggfunc", ["mean", "count", {"D": "sum", "E": "count"}]
)
@pytest.mark.parametrize("fill_value", [0])
def test_dataframe_pivot_table_simple(aggfunc, fill_value):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "A": ["one", "one", "two", "three"] * 6,
            "B": ["A", "B", "C"] * 8,
            "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
            "D": rng.standard_normal(size=24),
            "E": rng.standard_normal(size=24),
        }
    )
    expected = pdf.pivot_table(
        values=["D", "E"],
        index=["A", "B"],
        columns=["C"],
        aggfunc=aggfunc,
        fill_value=fill_value,
    )
    cdf = cudf.DataFrame.from_pandas(pdf)
    actual = cdf.pivot_table(
        values=["D", "E"],
        index=["A", "B"],
        columns=["C"],
        aggfunc=aggfunc,
        fill_value=fill_value,
    )
    assert_eq(expected, actual, check_dtype=False)


def test_crosstab_simple():
    a = np.array(
        [
            "foo",
            "foo",
            "foo",
            "foo",
            "bar",
            "bar",
            "bar",
            "bar",
            "foo",
            "foo",
            "foo",
        ],
        dtype=object,
    )
    b = np.array(
        [
            "one",
            "one",
            "one",
            "two",
            "one",
            "one",
            "one",
            "two",
            "two",
            "two",
            "one",
        ],
        dtype=object,
    )
    c = np.array(
        [
            "dull",
            "dull",
            "shiny",
            "dull",
            "dull",
            "shiny",
            "shiny",
            "dull",
            "shiny",
            "shiny",
            "shiny",
        ],
        dtype=object,
    )
    expected = pd.crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"])
    actual = cudf.crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"])
    assert_eq(expected, actual, check_dtype=False)
