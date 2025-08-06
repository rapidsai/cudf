# Copyright (c) 2021-2025, NVIDIA CORPORATION.

import re

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.testing import assert_eq
from cudf.testing._utils import (
    ALL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
)

pytest_xfail = pytest.mark.xfail
pytestmark = pytest.mark.spilling

# If spilling is enabled globally, we skip many test permutations
# to reduce running time.
if get_global_manager() is not None:
    ALL_TYPES = ["float32"]
    DATETIME_TYPES = ["datetime64[ms]"]
    NUMERIC_TYPES = ["float32"]
    # To save time, we skip tests marked "pytest.mark.xfail"
    pytest_xfail = pytest.mark.skipif


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
def test_unstack_index(index, col_idx):
    data = {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [11.0, 12.0, 13.0, 14.0, 15.0],
    }
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
def test_pivot_table_simple(aggfunc):
    rng = np.random.default_rng(seed=0)
    fill_value = 0
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
def test_dataframe_pivot_table_simple(aggfunc):
    rng = np.random.default_rng(seed=0)
    fill_value = 0
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


@pytest.mark.parametrize("index", ["A", ["A"]])
@pytest.mark.parametrize("columns", ["C", ["C"]])
def test_pivot_table_scalar_index_columns(index, columns):
    data = {
        "A": ["one", "one", "two", "three"] * 6,
        "B": ["A", "B", "C"] * 8,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
        "D": range(24),
        "E": range(24),
    }
    result = cudf.DataFrame(data).pivot_table(
        values="D", index=index, columns=columns, aggfunc="sum"
    )
    expected = pd.DataFrame(data).pivot_table(
        values="D", index=index, columns=columns, aggfunc="sum"
    )
    assert_eq(result, expected)


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


@pytest.mark.parametrize("index", [["ix"], ["ix", "foo"]])
@pytest.mark.parametrize("columns", [["col"], ["col", "baz"]])
def test_pivot_list_like_index_columns(index, columns):
    data = {
        "bar": ["x", "y", "z", "w"],
        "col": ["a", "b", "a", "b"],
        "foo": [1, 2, 3, 4],
        "ix": [1, 1, 2, 2],
        "baz": [0, 0, 0, 0],
    }
    pd_df = pd.DataFrame(data)
    cudf_df = cudf.DataFrame(data)
    result = cudf_df.pivot(columns=columns, index=index)
    expected = pd_df.pivot(columns=columns, index=index)
    assert_eq(result, expected)
