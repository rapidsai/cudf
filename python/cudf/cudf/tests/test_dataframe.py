# Copyright (c) 2018-2025, NVIDIA CORPORATION.

import array as arr
import operator
import textwrap
from copy import copy

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from packaging import version

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.core.column.column import as_column
from cudf.testing import assert_eq
from cudf.testing._utils import (
    ALL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    assert_exceptions_equal,
    expect_warning_if,
)
from cudf.utils.dtypes import SIZE_TYPE_DTYPE

pytest_xfail = pytest.mark.xfail
pytestmark = pytest.mark.spilling

# Use this to "unmark" the module level spilling mark
pytest_unmark_spilling = pytest.mark.skipif(
    get_global_manager() is not None, reason="unmarked spilling"
)

# If spilling is enabled globally, we skip many test permutations
# to reduce running time.
if get_global_manager() is not None:
    ALL_TYPES = ["float32"]
    DATETIME_TYPES = ["datetime64[ms]"]
    NUMERIC_TYPES = ["float32"]
    # To save time, we skip tests marked "xfail"
    pytest_xfail = pytest.mark.skipif


def test_dataframe_basic():
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame()

    # Populate with cuda memory
    df["keys"] = np.arange(10, dtype=np.float64)
    np.testing.assert_equal(df["keys"].to_numpy(), np.arange(10))
    assert len(df) == 10

    # Populate with numpy array
    rnd_vals = rng.random(10)
    df["vals"] = rnd_vals
    np.testing.assert_equal(df["vals"].to_numpy(), rnd_vals)
    assert len(df) == 10
    assert tuple(df.columns) == ("keys", "vals")

    # Make another dataframe
    df2 = cudf.DataFrame()
    df2["keys"] = np.array([123], dtype=np.float64)
    df2["vals"] = np.array([321], dtype=np.float64)

    # Concat
    df = cudf.concat([df, df2])
    assert len(df) == 11

    hkeys = np.asarray([*np.arange(10, dtype=np.float64).tolist(), 123])
    hvals = np.asarray([*rnd_vals.tolist(), 321])

    np.testing.assert_equal(df["keys"].to_numpy(), hkeys)
    np.testing.assert_equal(df["vals"].to_numpy(), hvals)

    # As matrix
    mat = df.values_host

    expect = np.vstack([hkeys, hvals]).T

    np.testing.assert_equal(mat, expect)

    # test dataframe with tuple name
    df_tup = cudf.DataFrame()
    data = np.arange(10)
    df_tup[(1, "foobar")] = data
    np.testing.assert_equal(data, df_tup[(1, "foobar")].to_numpy())

    df = cudf.DataFrame(pd.DataFrame({"a": [1, 2, 3], "c": ["a", "b", "c"]}))
    pdf = pd.DataFrame(pd.DataFrame({"a": [1, 2, 3], "c": ["a", "b", "c"]}))
    assert_eq(df, pdf)

    gdf = cudf.DataFrame({"id": [0, 1], "val": [None, None]})
    gdf["val"] = gdf["val"].astype("int")

    assert gdf["val"].isnull().all()


def test_dataframe_column_add_drop_via_setitem():
    df = cudf.DataFrame()
    data = np.asarray(range(10))
    df["a"] = data
    df["b"] = data
    assert tuple(df.columns) == ("a", "b")
    del df["a"]
    assert tuple(df.columns) == ("b",)
    df["c"] = data
    assert tuple(df.columns) == ("b", "c")
    df["a"] = data
    assert tuple(df.columns) == ("b", "c", "a")


def test_dataframe_column_set_via_attr():
    data_0 = np.asarray([0, 2, 4, 5])
    data_1 = np.asarray([1, 4, 2, 3])
    data_2 = np.asarray([2, 0, 3, 0])
    df = cudf.DataFrame({"a": data_0, "b": data_1, "c": data_2})

    for i in range(10):
        df.c = df.a
        assert assert_eq(df.c, df.a, check_names=False)
        assert tuple(df.columns) == ("a", "b", "c")

        df.c = df.b
        assert assert_eq(df.c, df.b, check_names=False)
        assert tuple(df.columns) == ("a", "b", "c")


def test_dataframe_column_drop_via_attr():
    df = cudf.DataFrame({"a": []})

    with pytest.raises(AttributeError):
        del df.a

    assert tuple(df.columns) == tuple("a")


@pytest.mark.parametrize("nelem", [0, 10])
def test_dataframe_astype(nelem):
    df = cudf.DataFrame()
    data = np.asarray(range(nelem), dtype=np.int32)
    df["a"] = data
    assert df["a"].dtype is np.dtype(np.int32)
    df["b"] = df["a"].astype(np.float32)
    assert df["b"].dtype is np.dtype(np.float32)
    np.testing.assert_equal(df["a"].to_numpy(), df["b"].to_numpy())


def test_astype_dict():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": ["1", "2", "3"]})
    pdf = gdf.to_pandas()

    assert_eq(pdf.astype({"a": "str"}), gdf.astype({"a": "str"}))
    assert_eq(
        pdf.astype({"a": "str", "b": np.int64}),
        gdf.astype({"a": "str", "b": np.int64}),
    )


@pytest.mark.parametrize("nelem", [0, 100])
def test_index_astype(nelem):
    df = cudf.DataFrame()
    data = np.asarray(range(nelem), dtype=np.int32)
    df["a"] = data
    assert df.index.dtype is np.dtype(np.int64)
    df.index = df.index.astype(np.float32)
    assert df.index.dtype is np.dtype(np.float32)
    df["a"] = df["a"].astype(np.float32)
    np.testing.assert_equal(df.index.to_numpy(), df["a"].to_numpy())
    df["b"] = df["a"]
    df = df.set_index("b")
    df["a"] = df["a"].astype(np.int16)
    df.index = df.index.astype(np.int16)
    np.testing.assert_equal(df.index.to_numpy(), df["a"].to_numpy())


def test_dataframe_to_string_with_skipped_rows():
    # Test skipped rows
    df = cudf.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [11, 12, 13, 14, 15, 16]}
    )

    with pd.option_context("display.max_rows", 5):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
            a   b
        0   1  11
        1   2  12
        .. ..  ..
        4   5  15
        5   6  16

        [6 rows x 2 columns]"""
    )
    assert got == expect


def test_dataframe_to_string_with_skipped_rows_and_columns():
    # Test skipped rows and skipped columns
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [11, 12, 13, 14, 15, 16],
            "c": [11, 12, 13, 14, 15, 16],
            "d": [11, 12, 13, 14, 15, 16],
        }
    )

    with pd.option_context("display.max_rows", 5, "display.max_columns", 3):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
            a  ...   d
        0   1  ...  11
        1   2  ...  12
        .. ..  ...  ..
        4   5  ...  15
        5   6  ...  16

        [6 rows x 4 columns]"""
    )
    assert got == expect


def test_dataframe_to_string_with_masked_data():
    # Test masked data
    df = cudf.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [11, 12, 13, 14, 15, 16]}
    )

    data = np.arange(6)
    mask = np.zeros(1, dtype=SIZE_TYPE_DTYPE)
    mask[0] = 0b00101101

    masked = cudf.Series._from_column(as_column(data).set_mask(mask))
    assert masked.null_count == 2
    df["c"] = masked

    # Check data
    values = masked.copy()
    validids = [0, 2, 3, 5]
    densearray = masked.dropna().to_numpy()
    np.testing.assert_equal(data[validids], densearray)
    # Valid position is correct
    for i in validids:
        assert data[i] == values[i]
    # Null position is correct
    for i in range(len(values)):
        if i not in validids:
            assert values[i] is cudf.NA

    with pd.option_context("display.max_rows", 10):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
           a   b     c
        0  1  11     0
        1  2  12  <NA>
        2  3  13     2
        3  4  14     3
        4  5  15  <NA>
        5  6  16     5"""
    )
    assert got == expect


def test_dataframe_to_string_wide():
    # Test basic
    df = cudf.DataFrame({f"a{i}": [0, 1, 2] for i in range(100)})
    with pd.option_context("display.max_columns", 16):
        got = df.to_string()

    expect = textwrap.dedent(
        """\
           a0  a1  a2  a3  a4  a5  a6  a7  ...  a92  a93  a94  a95  a96  a97  a98  a99
        0   0   0   0   0   0   0   0   0  ...    0    0    0    0    0    0    0    0
        1   1   1   1   1   1   1   1   1  ...    1    1    1    1    1    1    1    1
        2   2   2   2   2   2   2   2   2  ...    2    2    2    2    2    2    2    2

        [3 rows x 100 columns]"""
    )
    assert got == expect


def test_dataframe_empty_to_string():
    # Test for printing empty dataframe
    df = cudf.DataFrame()
    got = df.to_string()

    expect = "Empty DataFrame\nColumns: []\nIndex: []"
    assert got == expect


def test_dataframe_emptycolumns_to_string():
    # Test for printing dataframe having empty columns
    df = cudf.DataFrame()
    df["a"] = []
    df["b"] = []
    got = df.to_string()

    expect = "Empty DataFrame\nColumns: [a, b]\nIndex: []"
    assert got == expect


def test_dataframe_copy():
    # Test for copying the dataframe using python copy pkg
    df = cudf.DataFrame()
    df["a"] = [1, 2, 3]
    df2 = copy(df)
    df2["b"] = [4, 5, 6]
    got = df.to_string()

    expect = textwrap.dedent(
        """\
           a
        0  1
        1  2
        2  3"""
    )
    assert got == expect


def test_dataframe_copy_shallow():
    # Test for copy dataframe using class method
    df = cudf.DataFrame()
    df["a"] = [1, 2, 3]
    df2 = df.copy()
    df2["b"] = [4, 2, 3]
    got = df.to_string()

    expect = textwrap.dedent(
        """\
           a
        0  1
        1  2
        2  3"""
    )
    assert got == expect


def test_dataframe_dtypes():
    dtypes = pd.Series(
        [np.int32, np.float32, np.float64], index=["c", "a", "b"]
    )
    df = cudf.DataFrame({k: np.ones(10, dtype=v) for k, v in dtypes.items()})
    assert df.dtypes.equals(dtypes)


def test_dataframe_add_col_to_object_dataframe():
    # Test for adding column to an empty object dataframe
    cols = ["a", "b", "c"]
    df = pd.DataFrame(columns=cols, dtype="str")

    data = {k: ["a"] for k in cols}

    gdf = cudf.DataFrame(data)
    gdf = gdf[:0]

    assert gdf.dtypes.equals(df.dtypes)
    gdf["a"] = [1]
    df["a"] = [10]
    assert gdf.dtypes.equals(df.dtypes)
    gdf["b"] = [1.0]
    df["b"] = [10.0]
    assert gdf.dtypes.equals(df.dtypes)


def test_dataframe_dir_and_getattr():
    df = cudf.DataFrame(
        {
            "a": np.ones(10),
            "b": np.ones(10),
            "not an id": np.ones(10),
            "oop$": np.ones(10),
        }
    )
    o = dir(df)
    assert {"a", "b"}.issubset(o)
    assert "not an id" not in o
    assert "oop$" not in o

    # Getattr works
    assert df.a.equals(df["a"])
    assert df.b.equals(df["b"])
    with pytest.raises(AttributeError):
        df.not_a_column


def test_dataframe_append_empty():
    pdf = pd.DataFrame(
        {
            "key": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

    gdf["newcol"] = 100
    pdf["newcol"] = 100

    assert len(gdf["newcol"]) == len(pdf)
    assert len(pdf["newcol"]) == len(pdf)
    assert_eq(gdf, pdf)


def test_dataframe_setitem_from_masked_object():
    rng = np.random.default_rng(seed=0)
    ary = rng.standard_normal(100)
    mask = np.zeros(100, dtype=bool)
    mask[:20] = True
    rng.shuffle(mask)
    ary[mask] = np.nan

    test1_null = cudf.Series(ary, nan_as_null=True)
    assert test1_null.null_count == 20
    test1_nan = cudf.Series(ary, nan_as_null=False)
    assert test1_nan.null_count == 0

    test2_null = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ary}), nan_as_null=True
    )
    assert test2_null["a"].null_count == 20
    test2_nan = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ary}), nan_as_null=False
    )
    assert test2_nan["a"].null_count == 0

    gpu_ary = cupy.asarray(ary)
    test3_null = cudf.Series(gpu_ary, nan_as_null=True)
    assert test3_null.null_count == 20
    test3_nan = cudf.Series(gpu_ary, nan_as_null=False)
    assert test3_nan.null_count == 0

    test4 = cudf.DataFrame()
    lst = [1, 2, None, 4, 5, 6, None, 8, 9]
    test4["lst"] = lst
    assert test4["lst"].null_count == 2


def test_dataframe_append_to_empty():
    pdf = pd.DataFrame()
    pdf["a"] = []
    pdf["a"] = pdf["a"].astype("str")
    pdf["b"] = [1, 2, 3]

    gdf = cudf.DataFrame()
    gdf["a"] = []
    gdf["b"] = [1, 2, 3]

    assert_eq(gdf, pdf)


def test_dataframe_setitem_index_len1():
    gdf = cudf.DataFrame()
    gdf["a"] = [1]
    gdf["b"] = gdf.index._column

    np.testing.assert_equal(gdf.b.to_numpy(), [0])


def test_empty_dataframe_setitem_df():
    gdf1 = cudf.DataFrame()
    gdf2 = cudf.DataFrame({"a": [1, 2, 3, 4, 5]})
    gdf1["a"] = gdf2["a"]
    assert_eq(gdf1, gdf2)


@pytest.mark.parametrize("nrows", [0, 3])
def test_nonmatching_index_setitem(nrows):
    rng = np.random.default_rng(seed=0)

    gdf = cudf.DataFrame()
    gdf["a"] = rng.integers(2147483647, size=nrows)
    gdf["b"] = rng.integers(2147483647, size=nrows)
    gdf = gdf.set_index("b")

    test_values = rng.integers(2147483647, size=nrows)
    gdf["c"] = test_values
    assert len(test_values) == len(gdf["c"])
    gdf_series = cudf.Series(test_values, index=gdf.index, name="c")
    assert_eq(gdf["c"].to_pandas(), gdf_series.to_pandas())


@pytest.mark.parametrize("dtype", ["int", "int64[pyarrow]"])
def test_from_pandas(dtype):
    df = pd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0], dtype=dtype)
    df.columns.name = "custom_column_name"
    gdf = cudf.DataFrame.from_pandas(df)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf, check_dtype="pyarrow" not in dtype)

    s = df.x
    gs = cudf.Series.from_pandas(s)
    assert isinstance(gs, cudf.Series)

    assert_eq(s, gs, check_dtype="pyarrow" not in dtype)


def test_from_arrow_missing_categorical():
    pd_cat = pd.Categorical(["a", "b", "c"], categories=["a", "b"])
    pa_cat = pa.array(pd_cat, from_pandas=True)
    gd_cat = cudf.Series(pa_cat)

    assert isinstance(gd_cat, cudf.Series)
    assert_eq(
        pd.Series(pa_cat.to_pandas()),  # PyArrow returns a pd.Categorical
        gd_cat.to_pandas(),
    )


def test_to_arrow_missing_categorical():
    pd_cat = pd.Categorical(["a", "b", "c"], categories=["a", "b"])
    pa_cat = pa.array(pd_cat, from_pandas=True)
    gd_cat = cudf.Series(pa_cat)

    assert isinstance(gd_cat, cudf.Series)
    assert pa.Array.equals(pa_cat, gd_cat.to_arrow())


@pytest.mark.parametrize("data_type", NUMERIC_TYPES)
def test_from_python_array(data_type):
    rng = np.random.default_rng(seed=0)
    np_arr = rng.integers(0, 100, 10).astype(data_type)
    data = memoryview(np_arr)
    data = arr.array(data.format, data)

    gs = cudf.Series(data)

    np.testing.assert_equal(gs.to_numpy(), np_arr)


def test_series_shape():
    ps = pd.Series([1, 2, 3, 4])
    cs = cudf.Series([1, 2, 3, 4])

    assert ps.shape == cs.shape


def test_series_shape_empty():
    ps = pd.Series([], dtype="float64")
    cs = cudf.Series([], dtype="float64")

    assert ps.shape == cs.shape


def test_dataframe_shape():
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
    gdf = cudf.DataFrame.from_pandas(pdf)

    assert pdf.shape == gdf.shape


def test_dataframe_shape_empty():
    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()

    assert pdf.shape == gdf.shape


@pytest.fixture
def pdf():
    return pd.DataFrame({"x": range(10), "y": range(10)})


@pytest.fixture
def gdf(pdf):
    return cudf.DataFrame.from_pandas(pdf)


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "data",
    [
        {
            "x": [np.nan, 2, 3, 4, 100, np.nan],
            "y": [4, 5, 6, 88, 99, np.nan],
            "z": [7, 8, 9, 66, np.nan, 77],
        },
        {"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]},
        {
            "x": [np.nan, np.nan, np.nan],
            "y": [np.nan, np.nan, np.nan],
            "z": [np.nan, np.nan, np.nan],
        },
        pytest.param(
            {"x": [], "y": [], "z": []},
            marks=pytest_xfail(
                condition=version.parse("11")
                <= version.parse(cupy.__version__)
                < version.parse("11.1"),
                reason="Zero-sized array passed to cupy reduction, "
                "https://github.com/cupy/cupy/issues/6937",
            ),
        ),
        pytest.param(
            {"x": []},
            marks=pytest_xfail(
                condition=version.parse("11")
                <= version.parse(cupy.__version__)
                < version.parse("11.1"),
                reason="Zero-sized array passed to cupy reduction, "
                "https://github.com/cupy/cupy/issues/6937",
            ),
        ),
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "func",
    [
        "min",
        "max",
        "sum",
        "prod",
        "product",
        "cummin",
        "cummax",
        "cumsum",
        "cumprod",
        "mean",
        "median",
        "sum",
        "std",
        "var",
        "kurt",
        "skew",
        "all",
        "any",
    ],
)
@pytest.mark.parametrize("skipna", [True, False])
def test_dataframe_reductions(data, axis, func, skipna):
    pdf = pd.DataFrame(data=data)
    gdf = cudf.DataFrame.from_pandas(pdf)

    # Reductions can fail in numerous possible ways when attempting row-wise
    # reductions, which are only partially supported. Catching the appropriate
    # exception here allows us to detect API breakage in the form of changing
    # exceptions.
    expected_exception = None
    if axis == 1:
        if func in ("kurt", "skew"):
            expected_exception = NotImplementedError
        elif func not in cudf.core.dataframe._cupy_nan_methods_map:
            if skipna is False:
                expected_exception = NotImplementedError
            elif any(col._column.nullable for name, col in gdf.items()):
                expected_exception = ValueError
            elif func in ("cummin", "cummax"):
                expected_exception = AttributeError

    # Test different degrees of freedom for var and std.
    all_kwargs = [{"ddof": 1}, {"ddof": 2}] if func in ("var", "std") else [{}]
    for kwargs in all_kwargs:
        if expected_exception is not None:
            with pytest.raises(expected_exception):
                (getattr(gdf, func)(axis=axis, skipna=skipna, **kwargs),)
        else:
            expect = getattr(pdf, func)(axis=axis, skipna=skipna, **kwargs)
            with expect_warning_if(
                skipna
                and func in {"min", "max"}
                and axis == 1
                and any(gdf.T[col].isna().all() for col in gdf.T),
                RuntimeWarning,
            ):
                got = getattr(gdf, func)(axis=axis, skipna=skipna, **kwargs)
            assert_eq(got, expect, check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        {"x": [np.nan, 2, 3, 4, 100, np.nan], "y": [4, 5, 6, 88, 99, np.nan]},
        {"x": [1, 2, 3], "y": [4, 5, 6]},
        {"x": [np.nan, np.nan, np.nan], "y": [np.nan, np.nan, np.nan]},
        {"x": [], "y": []},
        {"x": []},
    ],
)
@pytest.mark.parametrize("func", [lambda df: df.count()])
def test_dataframe_count_reduction(data, func):
    pdf = pd.DataFrame(data=data)
    gdf = cudf.DataFrame.from_pandas(pdf)

    assert_eq(func(pdf), func(gdf))


@pytest.mark.parametrize(
    "data",
    [
        {"x": [np.nan, 2, 3, 4, 100, np.nan], "y": [4, 5, 6, 88, 99, np.nan]},
        {"x": [1, 2, 3], "y": [4, 5, 6]},
        {"x": [np.nan, np.nan, np.nan], "y": [np.nan, np.nan, np.nan]},
        {"x": pd.Series([], dtype="float"), "y": pd.Series([], dtype="float")},
        {"x": pd.Series([], dtype="int")},
    ],
)
@pytest.mark.parametrize("ops", ["sum", "product", "prod"])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("min_count", [-10, -1, 0, 1, 2, 3, 10])
def test_dataframe_min_count_ops(data, ops, skipna, min_count):
    psr = pd.DataFrame(data)
    gsr = cudf.from_pandas(psr)

    assert_eq(
        getattr(psr, ops)(skipna=skipna, min_count=min_count),
        getattr(gsr, ops)(skipna=skipna, min_count=min_count),
        check_dtype=False,
    )


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "binop",
    [
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.pow,
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        1.0,
        pd.Series([1.0]),
        pd.Series([1.0, 2.0]),
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([1.0], index=["x"]),
        pd.Series([1.0, 2.0], index=["x", "y"]),
        pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"]),
        pd.DataFrame({"x": [1.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}),
    ],
)
def test_arithmetic_binops_df(pdf, gdf, binop, other):
    # Avoid 1**NA cases: https://github.com/pandas-dev/pandas/issues/29997
    pdf[pdf == 1.0] = 2
    gdf[gdf == 1.0] = 2
    try:
        d = binop(pdf, other)
    except Exception:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            cudf_other = cudf.from_pandas(other)

        # that returns before we enter this try-except.
        assert_exceptions_equal(
            lfunc=binop,
            rfunc=binop,
            lfunc_args_and_kwargs=([pdf, other], {}),
            rfunc_args_and_kwargs=([gdf, cudf_other], {}),
        )
    else:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            other = cudf.from_pandas(other)
        g = binop(gdf, other)
        assert_eq(d, g)


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "binop",
    [
        operator.eq,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.ne,
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        1.0,
        pd.Series([1.0, 2.0], index=["x", "y"]),
        pd.DataFrame({"x": [1.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0]}),
        pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}),
    ],
)
def test_comparison_binops_df(pdf, gdf, binop, other):
    # Avoid 1**NA cases: https://github.com/pandas-dev/pandas/issues/29997
    pdf[pdf == 1.0] = 2
    gdf[gdf == 1.0] = 2
    try:
        d = binop(pdf, other)
    except Exception:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            cudf_other = cudf.from_pandas(other)

        # that returns before we enter this try-except.
        assert_exceptions_equal(
            lfunc=binop,
            rfunc=binop,
            lfunc_args_and_kwargs=([pdf, other], {}),
            rfunc_args_and_kwargs=([gdf, cudf_other], {}),
        )
    else:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            other = cudf.from_pandas(other)
        g = binop(gdf, other)
        assert_eq(d, g)


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "binop",
    [
        operator.eq,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.ne,
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        pd.Series([1.0]),
        pd.Series([1.0, 2.0]),
        pd.Series([1.0, 2.0, 3.0]),
        pd.Series([1.0], index=["x"]),
        pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"]),
    ],
)
def test_comparison_binops_df_reindexing(request, pdf, gdf, binop, other):
    # Avoid 1**NA cases: https://github.com/pandas-dev/pandas/issues/29997
    pdf[pdf == 1.0] = 2
    gdf[gdf == 1.0] = 2
    try:
        d = binop(pdf, other)
    except Exception:
        if isinstance(other, (pd.Series, pd.DataFrame)):
            cudf_other = cudf.from_pandas(other)

        # that returns before we enter this try-except.
        assert_exceptions_equal(
            lfunc=binop,
            rfunc=binop,
            lfunc_args_and_kwargs=([pdf, other], {}),
            rfunc_args_and_kwargs=([gdf, cudf_other], {}),
        )
    else:
        request.applymarker(
            pytest.mark.xfail(
                condition=pdf.columns.difference(other.index).size > 0,
                reason="""
                Currently we will not match pandas for equality/inequality
                operators when there are columns that exist in a Series but not
                the DataFrame because pandas returns True/False values whereas
                we return NA. However, this reindexing is deprecated in pandas
                so we opt not to add support. This test should start passing
                once pandas removes the deprecated behavior in 2.0.  When that
                happens, this test can be merged with the two tests above into
                a single test with common parameters.
                """,
            )
        )

        if isinstance(other, (pd.Series, pd.DataFrame)):
            other = cudf.from_pandas(other)
        g = binop(gdf, other)
        assert_eq(d, g)


def test_binops_df_invalid(gdf):
    with pytest.raises(TypeError):
        gdf + np.array([1, 2])


@pytest.mark.parametrize("binop", [operator.and_, operator.or_, operator.xor])
def test_bitwise_binops_df(pdf, gdf, binop):
    d = binop(pdf, pdf + 1)
    g = binop(gdf, gdf + 1)
    assert_eq(d, g)


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "binop",
    [
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.pow,
        operator.eq,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.ne,
    ],
)
def test_binops_series(pdf, gdf, binop):
    pdf = pdf + 1.0
    gdf = gdf + 1.0
    d = binop(pdf.x, pdf.y)
    g = binop(gdf.x, gdf.y)
    assert_eq(d, g)


@pytest.mark.parametrize("binop", [operator.and_, operator.or_, operator.xor])
def test_bitwise_binops_series(pdf, gdf, binop):
    d = binop(pdf.x, pdf.y + 1)
    g = binop(gdf.x, gdf.y + 1)
    assert_eq(d, g)


@pytest.mark.parametrize("unaryop", [operator.neg, operator.inv, operator.abs])
@pytest.mark.parametrize(
    "col_name,assign_col_name", [(None, False), (None, True), ("abc", True)]
)
def test_unaryops_df(pdf, unaryop, col_name, assign_col_name):
    pd_df = pdf.copy()
    if assign_col_name:
        pd_df.columns.name = col_name
    gdf = cudf.from_pandas(pd_df)
    d = unaryop(pd_df - 5)
    g = unaryop(gdf - 5)
    assert_eq(d, g)


def test_df_abs(pdf):
    rng = np.random.default_rng(seed=0)
    disturbance = pd.Series(rng.random(10))
    pdf = pdf - 5 + disturbance
    d = pdf.apply(np.abs)
    g = cudf.from_pandas(pdf).abs()
    assert_eq(d, g)


def test_scale_df(gdf):
    got = (gdf - 5).scale()
    expect = cudf.DataFrame(
        {"x": np.linspace(0.0, 1.0, 10), "y": np.linspace(0.0, 1.0, 10)}
    )
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.empty,
        lambda df: df.x.empty,
        lambda df: df.x.fillna(123, limit=None, method=None, axis=None),
        lambda df: df.drop("x", axis=1, errors="raise"),
    ],
)
def test_unary_operators(func, pdf, gdf):
    p = func(pdf)
    g = func(gdf)
    assert_eq(p, g)


def test_is_monotonic(gdf):
    pdf = pd.DataFrame({"x": [1, 2, 3]}, index=[3, 1, 2])
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert not gdf.index.is_monotonic_increasing
    assert not gdf.index.is_monotonic_decreasing


@pytest.mark.parametrize("q", [0.5, 1, 0.001, [0.5], [], [0.005, 0.5, 1]])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_quantile(q, numeric_only):
    ts = pd.date_range("2018-08-24", periods=5, freq="D")
    td = pd.to_timedelta(np.arange(5), unit="h")
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {"date": ts, "delta": td, "val": rng.standard_normal(len(ts))}
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

    assert_eq(pdf["date"].quantile(q), gdf["date"].quantile(q))
    assert_eq(pdf["delta"].quantile(q), gdf["delta"].quantile(q))
    assert_eq(pdf["val"].quantile(q), gdf["val"].quantile(q))

    q = q if isinstance(q, list) else [q]
    assert_eq(
        pdf.quantile(q, numeric_only=numeric_only),
        gdf.quantile(q, numeric_only=numeric_only),
    )


@pytest.mark.parametrize("q", [0.2, 1, 0.001, [0.5], [], [0.005, 0.8, 0.03]])
@pytest.mark.parametrize("interpolation", ["higher", "lower", "nearest"])
@pytest.mark.parametrize(
    "decimal_type",
    [cudf.Decimal32Dtype, cudf.Decimal64Dtype, cudf.Decimal128Dtype],
)
def test_decimal_quantile(q, interpolation, decimal_type):
    rng = np.random.default_rng(seed=0)
    data = ["244.8", "32.24", "2.22", "98.14", "453.23", "5.45"]
    gdf = cudf.DataFrame(
        {"id": rng.integers(0, 10, size=len(data)), "val": data}
    )
    gdf["id"] = gdf["id"].astype("float64")
    gdf["val"] = gdf["val"].astype(decimal_type(7, 2))
    pdf = gdf.to_pandas()

    got = gdf.quantile(q, numeric_only=False, interpolation=interpolation)
    expected = pdf.quantile(
        q if isinstance(q, list) else [q],
        numeric_only=False,
        interpolation=interpolation,
    )

    assert_eq(got, expected)


def test_empty_quantile():
    pdf = pd.DataFrame({"x": []}, dtype="float64")
    df = cudf.DataFrame({"x": []}, dtype="float64")

    actual = df.quantile()
    expected = pdf.quantile()

    assert_eq(actual, expected)


def test_boolmask(pdf, gdf):
    rng = np.random.default_rng(seed=0)
    boolmask = rng.integers(0, 2, len(pdf)) > 0
    gdf = gdf[boolmask]
    pdf = pdf[boolmask]
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "mask_shape",
    [
        (2, "ab"),
        (2, "abc"),
        (3, "ab"),
        (3, "abc"),
        (3, "abcd"),
        (4, "abc"),
        (4, "abcd"),
    ],
)
def test_dataframe_boolmask(mask_shape):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame({col: rng.integers(0, 10, 3) for col in "abc"})
    pdf_mask = pd.DataFrame(
        {col: rng.integers(0, 2, mask_shape[0]) > 0 for col in mask_shape[1]}
    )
    gdf = cudf.DataFrame.from_pandas(pdf)
    gdf_mask = cudf.DataFrame.from_pandas(pdf_mask)
    gdf = gdf[gdf_mask]
    pdf = pdf[pdf_mask]

    assert np.array_equal(gdf.columns, pdf.columns)
    for col in gdf.columns:
        assert np.array_equal(
            gdf[col].fillna(-1).to_pandas().values, pdf[col].fillna(-1).values
        )


@pytest.mark.parametrize(
    "box",
    [
        list,
        pytest.param(
            cudf.Series,
            marks=pytest_xfail(
                reason="Pandas can't index a multiindex with a Series"
            ),
        ),
    ],
)
def test_dataframe_multiindex_boolmask(box):
    mask = box([True, False, True])
    gdf = cudf.DataFrame(
        {"w": [3, 2, 1], "x": [1, 2, 3], "y": [0, 1, 0], "z": [1, 1, 1]}
    )
    gdg = gdf.groupby(["w", "x"]).count()
    pdg = gdg.to_pandas()
    assert_eq(gdg[mask], pdg[mask])


def test_dataframe_assignment():
    pdf = pd.DataFrame()
    for col in "abc":
        pdf[col] = np.array([0, 1, 1, -2, 10])
    gdf = cudf.DataFrame.from_pandas(pdf)
    gdf[gdf < 0] = 999
    pdf[pdf < 0] = 999
    assert_eq(gdf, pdf)


@pytest.mark.parametrize(
    "data",
    [
        [0, 1, 2, 3],
        [-2, -1, 2, 3, 5],
        [-2, -1, 0, 3, 5],
        [True, False, False],
        [True],
        [False],
        [],
        [True, None, False],
        [True, True, None],
        [None, None],
        [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]],
        [[1, True], [2, False], [3, False]],
        [["a", True], ["b", False], ["c", False]],
    ],
)
def test_all(data):
    # Provide a dtype when data is empty to avoid future pandas changes.
    dtype = None if data else float
    if np.array(data).ndim <= 1:
        pdata = pd.Series(data=data, dtype=dtype)
        gdata = cudf.Series.from_pandas(pdata)
        got = gdata.all()
        expected = pdata.all()
        assert_eq(got, expected)
    else:
        pdata = pd.DataFrame(data, columns=["a", "b"], dtype=dtype).replace(
            [None], False
        )
        gdata = cudf.DataFrame.from_pandas(pdata)

        # test bool_only
        if pdata["b"].dtype == "bool":
            got = gdata.all(bool_only=True)
            expected = pdata.all(bool_only=True)
            assert_eq(got, expected)
        else:
            got = gdata.all()
            expected = pdata.all()
            assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [0, 1, 2, 3],
        [-2, -1, 2, 3, 5],
        [-2, -1, 0, 3, 5],
        [0, 0, 0, 0, 0],
        [0, 0, None, 0],
        [True, False, False],
        [True],
        [False],
        [],
        [True, None, False],
        [True, True, None],
        [None, None],
        [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]],
        [[1, True], [2, False], [3, False]],
        [["a", True], ["b", False], ["c", False]],
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
def test_any(data, axis):
    # Provide a dtype when data is empty to avoid future pandas changes.
    dtype = float if all(x is None for x in data) or len(data) < 1 else None
    if np.array(data).ndim <= 1:
        pdata = pd.Series(data=data, dtype=dtype)
        gdata = cudf.Series(data=data, dtype=dtype)

        if axis == 1:
            with pytest.raises(NotImplementedError):
                gdata.any(axis=axis)
        else:
            got = gdata.any(axis=axis)
            expected = pdata.any(axis=axis)
            assert_eq(got, expected)
    else:
        pdata = pd.DataFrame(data, columns=["a", "b"])
        gdata = cudf.DataFrame.from_pandas(pdata)

        # test bool_only
        if pdata["b"].dtype == "bool":
            got = gdata.any(bool_only=True)
            expected = pdata.any(bool_only=True)
            assert_eq(got, expected)
        else:
            got = gdata.any(axis=axis)
            expected = pdata.any(axis=axis)
            assert_eq(got, expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_empty_dataframe_any(axis):
    pdf = pd.DataFrame({}, columns=["a", "b"], dtype=float)
    gdf = cudf.DataFrame.from_pandas(pdf)
    got = gdf.any(axis=axis)
    expected = pdf.any(axis=axis)
    assert_eq(got, expected, check_index_type=False)


@pytest_unmark_spilling
@pytest.mark.parametrize("a", [[], ["123"]])
@pytest.mark.parametrize("b", ["123", ["123"]])
@pytest.mark.parametrize(
    "misc_data",
    ["123", ["123"] * 20, 123, [1, 2, 0.8, 0.9] * 50, 0.9, 0.00001],
)
@pytest.mark.parametrize("non_list_data", [123, "abc", "zyx", "rapids", 0.8])
def test_create_dataframe_cols_empty_data(a, b, misc_data, non_list_data):
    expected = pd.DataFrame({"a": a})
    actual = cudf.DataFrame.from_pandas(expected)
    expected["b"] = b
    actual["b"] = b
    assert_eq(actual, expected)

    expected = pd.DataFrame({"a": []})
    actual = cudf.DataFrame.from_pandas(expected)
    expected["b"] = misc_data
    actual["b"] = misc_data
    assert_eq(actual, expected)

    expected = pd.DataFrame({"a": a})
    actual = cudf.DataFrame.from_pandas(expected)
    expected["b"] = non_list_data
    actual["b"] = non_list_data
    assert_eq(actual, expected)


def test_as_column_types():
    col = as_column(cudf.Series([], dtype="float64"))
    assert_eq(col.dtype, np.dtype("float64"))
    gds = cudf.Series._from_column(col)
    pds = pd.Series(pd.Series([], dtype="float64"))

    assert_eq(pds, gds)

    col = as_column(
        cudf.Series([], dtype="float64"), dtype=np.dtype(np.float32)
    )
    assert_eq(col.dtype, np.dtype("float32"))
    gds = cudf.Series._from_column(col)
    pds = pd.Series(pd.Series([], dtype="float32"))

    assert_eq(pds, gds)

    col = as_column(cudf.Series([], dtype="float64"), dtype=cudf.dtype("str"))
    assert_eq(col.dtype, np.dtype("object"))
    gds = cudf.Series._from_column(col)
    pds = pd.Series(pd.Series([], dtype="str"))

    assert_eq(pds, gds)

    col = as_column(cudf.Series([], dtype="float64"), dtype=cudf.dtype("str"))
    assert_eq(col.dtype, np.dtype("object"))
    gds = cudf.Series._from_column(col)
    pds = pd.Series(pd.Series([], dtype="object"))

    assert_eq(pds, gds)

    pds = pd.Series(np.array([1, 2, 3]), dtype="float32")
    gds = cudf.Series._from_column(
        as_column(np.array([1, 2, 3]), dtype=np.dtype(np.float32))
    )

    assert_eq(pds, gds)

    pds = pd.Series([1, 2, 3], dtype="float32")
    gds = cudf.Series([1, 2, 3], dtype="float32")

    assert_eq(pds, gds)

    pds = pd.Series([], dtype="float64")
    gds = cudf.Series._from_column(as_column(pds))
    assert_eq(pds, gds)

    pds = pd.Series([1, 2, 4], dtype="int64")
    gds = cudf.Series._from_column(
        as_column(cudf.Series([1, 2, 4]), dtype="int64")
    )

    assert_eq(pds, gds)

    pds = pd.Series([1.2, 18.0, 9.0], dtype="float32")
    gds = cudf.Series._from_column(
        as_column(cudf.Series([1.2, 18.0, 9.0]), dtype=np.dtype(np.float32))
    )

    assert_eq(pds, gds)

    pds = pd.Series([1.2, 18.0, 9.0], dtype="str")
    gds = cudf.Series._from_column(
        as_column(cudf.Series([1.2, 18.0, 9.0]), dtype=cudf.dtype("str"))
    )

    assert_eq(pds, gds)

    pds = pd.Series(pd.Index(["1", "18", "9"]), dtype="int")
    gds = cudf.Series(cudf.Index(["1", "18", "9"]), dtype="int")

    assert_eq(pds, gds)


@pytest.mark.parametrize("dtype", ["int64", "datetime64[ns]", "int8"])
def test_dataframe_astype_preserves_column_dtype(dtype):
    result = cudf.DataFrame([1], columns=cudf.Index([1], dtype=dtype))
    result = result.astype(np.int32).columns
    expected = pd.Index([1], dtype=dtype)
    assert_eq(result, expected)


def test_dataframe_astype_preserves_column_rangeindex():
    result = cudf.DataFrame([1], columns=range(1))
    result = result.astype(np.int32).columns
    expected = pd.RangeIndex(1)
    assert_eq(result, expected)


@pytest.mark.parametrize("dtype", ["int64", "datetime64[ns]", "int8"])
def test_dataframe_fillna_preserves_column_dtype(dtype):
    result = cudf.DataFrame([1, None], columns=cudf.Index([1], dtype=dtype))
    result = result.fillna(2).columns
    expected = pd.Index([1], dtype=dtype)
    assert_eq(result, expected)


def test_dataframe_fillna_preserves_column_rangeindex():
    result = cudf.DataFrame([1, None], columns=range(1))
    result = result.fillna(2).columns
    expected = pd.RangeIndex(1)
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 4],
        [],
        [5.0, 7.0, 8.0],
        pd.Categorical(["a", "b", "c"]),
        ["m", "a", "d", "v"],
    ],
)
def test_series_values_host_property(data):
    pds = pd.Series(data=data, dtype=None if data else float)
    gds = cudf.Series(data=data, dtype=None if data else float)

    np.testing.assert_array_equal(pds.values, gds.values_host)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 4],
        [],
        [5.0, 7.0, 8.0],
        pytest.param(
            pd.Categorical(["a", "b", "c"]),
            marks=pytest_xfail(raises=NotImplementedError),
        ),
        pytest.param(
            ["m", "a", "d", "v"],
            marks=pytest_xfail(raises=TypeError),
        ),
    ],
)
def test_series_values_property(data):
    pds = pd.Series(data=data, dtype=None if data else float)
    gds = cudf.from_pandas(pds)
    gds_vals = gds.values
    assert isinstance(gds_vals, cupy.ndarray)
    np.testing.assert_array_equal(gds_vals.get(), pds.values)


@pytest.mark.parametrize(
    "data",
    [
        {"A": [1, 2, 3], "B": [4, 5, 6]},
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]},
        {"A": [1, 2, 3], "B": [1.0, 2.0, 3.0]},
        {"A": np.float32(np.arange(3)), "B": np.float64(np.arange(3))},
        pytest.param(
            {"A": [1, None, 3], "B": [1, 2, None]},
            marks=pytest_xfail(
                reason="Nulls not supported by values accessor"
            ),
        ),
        pytest.param(
            {"A": [None, None, None], "B": [None, None, None]},
            marks=pytest_xfail(
                reason="Nulls not supported by values accessor"
            ),
        ),
        {"A": [], "B": []},
        pytest.param(
            {"A": [1, 2, 3], "B": ["a", "b", "c"]},
            marks=pytest_xfail(
                reason="str or categorical not supported by values accessor"
            ),
        ),
        pytest.param(
            {"A": pd.Categorical(["a", "b", "c"]), "B": ["d", "e", "f"]},
            marks=pytest_xfail(
                reason="str or categorical not supported by values accessor"
            ),
        ),
    ],
)
def test_df_values_property(data):
    pdf = pd.DataFrame.from_dict(data)
    gdf = cudf.DataFrame.from_pandas(pdf)

    pmtr = pdf.values
    gmtr = gdf.values.get()

    np.testing.assert_array_equal(pmtr, gmtr)


def test_numeric_alpha_value_counts():
    pdf = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5, 6, 1, 2, 4] * 10,
            "alpha": ["u", "h", "d", "a", "m", "u", "h", "d", "a"] * 10,
        }
    )

    gdf = cudf.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5, 6, 1, 2, 4] * 10,
            "alpha": ["u", "h", "d", "a", "m", "u", "h", "d", "a"] * 10,
        }
    )

    assert_eq(
        pdf.numeric.value_counts().sort_index(),
        gdf.numeric.value_counts().sort_index(),
        check_dtype=False,
    )
    assert_eq(
        pdf.alpha.value_counts().sort_index(),
        gdf.alpha.value_counts().sort_index(),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame(
            {
                "num_legs": [2, 4],
                "num_wings": [2, 0],
                "bird_cats": pd.Series(
                    ["sparrow", "pigeon"],
                    dtype="category",
                    index=["falcon", "dog"],
                ),
            },
            index=["falcon", "dog"],
        ),
        pd.DataFrame(
            {"num_legs": [8, 2], "num_wings": [0, 2]},
            index=["spider", "falcon"],
        ),
        pd.DataFrame(
            {
                "num_legs": [8, 2, 1, 0, 2, 4, 5],
                "num_wings": [2, 0, 2, 1, 2, 4, -1],
            }
        ),
        pd.DataFrame({"a": ["a", "b", "c"]}, dtype="category"),
        pd.DataFrame({"a": ["a", "b", "c"]}),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [0, 2],
        {"num_wings": [0, 3]},
        pd.DataFrame(
            {"num_legs": [8, 2], "num_wings": [0, 2]},
            index=["spider", "falcon"],
        ),
        pd.DataFrame(
            {
                "num_legs": [2, 4],
                "num_wings": [2, 0],
                "bird_cats": pd.Series(
                    ["sparrow", "pigeon"],
                    dtype="category",
                    index=["falcon", "dog"],
                ),
            },
            index=["falcon", "dog"],
        ),
        ["sparrow", "pigeon"],
        pd.Series(["sparrow", "pigeon"], dtype="category"),
        pd.Series([1, 2, 3, 4, 5]),
        "abc",
        123,
        pd.Series(["a", "b", "c"]),
        pd.Series(["a", "b", "c"], dtype="category"),
        pd.DataFrame({"a": ["a", "b", "c"]}, dtype="category"),
    ],
)
def test_isin_dataframe(data, values):
    pdf = data
    gdf = cudf.from_pandas(pdf)

    if cudf.api.types.is_scalar(values):
        assert_exceptions_equal(
            lfunc=pdf.isin,
            rfunc=gdf.isin,
            lfunc_args_and_kwargs=([values],),
            rfunc_args_and_kwargs=([values],),
        )
    else:
        try:
            expected = pdf.isin(values)
        except TypeError as e:
            # Can't do isin with different categories
            if str(e) == (
                "Categoricals can only be compared if 'categories' "
                "are the same."
            ):
                return

        if isinstance(values, (pd.DataFrame, pd.Series)):
            values = cudf.from_pandas(values)

        got = gdf.isin(values)
        assert_eq(got, expected)


def test_isin_axis_duplicated_error():
    df = cudf.DataFrame(range(2))
    with pytest.raises(ValueError):
        df.isin(cudf.Series(range(2), index=[1, 1]))

    with pytest.raises(ValueError):
        df.isin(cudf.DataFrame(range(2), index=[1, 1]))

    with pytest.raises(ValueError):
        df.isin(cudf.DataFrame([[1, 2]], columns=[1, 1]))


def test_constructor_properties():
    df = cudf.DataFrame()
    key1 = "a"
    key2 = "b"
    val1 = np.array([123], dtype=np.float64)
    val2 = np.array([321], dtype=np.float64)
    df[key1] = val1
    df[key2] = val2

    # Correct use of _constructor_sliced (for DataFrame)
    assert_eq(df[key1], df._constructor_sliced(val1, name=key1))

    # Correct use of _constructor_expanddim (for cudf.Series)
    assert_eq(df, df[key2]._constructor_expanddim({key1: val1, key2: val2}))

    # Incorrect use of _constructor_sliced (Raises for cudf.Series)
    with pytest.raises(NotImplementedError):
        df[key1]._constructor_sliced

    # Incorrect use of _constructor_expanddim (Raises for DataFrame)
    with pytest.raises(NotImplementedError):
        df._constructor_expanddim


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("as_dtype", ALL_TYPES)
def test_df_astype_numeric_to_all(dtype, as_dtype):
    if "uint" in dtype:
        data = [1, 2, None, 4, 7]
    elif "int" in dtype or "longlong" in dtype:
        data = [1, 2, None, 4, -7]
    elif "float" in dtype:
        data = [1.0, 2.0, None, 4.0, np.nan, -7.0]

    gdf = cudf.DataFrame()

    gdf["foo"] = cudf.Series(data, dtype=dtype)
    gdf["bar"] = cudf.Series(data, dtype=dtype)

    insert_data = cudf.Series(data, dtype=dtype)

    expect = cudf.DataFrame()
    expect["foo"] = insert_data.astype(as_dtype)
    expect["bar"] = insert_data.astype(as_dtype)

    got = gdf.astype(as_dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_df_astype_string_to_other(as_dtype):
    if "datetime64" in as_dtype:
        # change None to "NaT" after this issue is fixed:
        # https://github.com/rapidsai/cudf/issues/5117
        data = ["2001-01-01", "2002-02-02", "2000-01-05", None]
    elif as_dtype == "int32":
        data = [1, 2, 3]
    elif as_dtype == "category":
        data = ["1", "2", "3", None]
    elif "float" in as_dtype:
        data = [1.0, 2.0, 3.0, np.nan]

    insert_data = cudf.Series.from_pandas(pd.Series(data, dtype="str"))
    expect_data = cudf.Series(data, dtype=as_dtype)

    gdf = cudf.DataFrame()
    expect = cudf.DataFrame()

    gdf["foo"] = insert_data
    gdf["bar"] = insert_data

    expect["foo"] = expect_data
    expect["bar"] = expect_data

    got = gdf.astype(as_dtype)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int64",
        "datetime64[s]",
        "datetime64[us]",
        "datetime64[ns]",
        "str",
        "category",
    ],
)
def test_df_astype_datetime_to_other(as_dtype):
    data = [
        "1991-11-20 00:00:00.000",
        "2004-12-04 00:00:00.000",
        "2016-09-13 00:00:00.000",
        None,
    ]

    gdf = cudf.DataFrame()
    expect = cudf.DataFrame()

    gdf["foo"] = cudf.Series(data, dtype="datetime64[ms]")
    gdf["bar"] = cudf.Series(data, dtype="datetime64[ms]")

    if as_dtype == "int64":
        expect["foo"] = cudf.Series(
            [690595200000, 1102118400000, 1473724800000, None], dtype="int64"
        )
        expect["bar"] = cudf.Series(
            [690595200000, 1102118400000, 1473724800000, None], dtype="int64"
        )
    elif as_dtype == "str":
        expect["foo"] = cudf.Series(data, dtype="str")
        expect["bar"] = cudf.Series(data, dtype="str")
    elif as_dtype == "category":
        expect["foo"] = cudf.Series(gdf["foo"], dtype="category")
        expect["bar"] = cudf.Series(gdf["bar"], dtype="category")
    else:
        expect["foo"] = cudf.Series(data, dtype=as_dtype)
        expect["bar"] = cudf.Series(data, dtype=as_dtype)

    got = gdf.astype(as_dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
        "str",
    ],
)
def test_df_astype_categorical_to_other(as_dtype):
    if "datetime64" in as_dtype:
        data = ["2001-01-01", "2002-02-02", "2000-01-05", "2001-01-01"]
    else:
        data = [1, 2, 3, 1]
    psr = pd.Series(data, dtype="category")
    pdf = pd.DataFrame()
    pdf["foo"] = psr
    pdf["bar"] = psr
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert_eq(pdf.astype(as_dtype), gdf.astype(as_dtype))


@pytest.mark.parametrize("ordered", [True, False])
def test_df_astype_to_categorical_ordered(ordered):
    psr = pd.Series([1, 2, 3, 1], dtype="category")
    pdf = pd.DataFrame()
    pdf["foo"] = psr
    pdf["bar"] = psr
    gdf = cudf.DataFrame.from_pandas(pdf)

    ordered_dtype_pd = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=ordered
    )
    ordered_dtype_gd = cudf.CategoricalDtype.from_pandas(ordered_dtype_pd)

    assert_eq(
        pdf.astype(ordered_dtype_pd).astype("int32"),
        gdf.astype(ordered_dtype_gd).astype("int32"),
    )


@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in ALL_TYPES]
    + [
        cudf.CategoricalDtype(ordered=True),
        cudf.CategoricalDtype(ordered=False),
    ],
)
def test_empty_df_astype(dtype):
    df = cudf.DataFrame()
    result = df.astype(dtype=dtype)
    assert_eq(df, result)
    assert_eq(df.to_pandas().astype(dtype=dtype), result)


@pytest.mark.parametrize(
    "errors",
    [
        pytest.param(
            "raise", marks=pytest_xfail(reason="should raise error here")
        ),
        pytest.param("other", marks=pytest_xfail(raises=ValueError)),
        "ignore",
    ],
)
def test_series_astype_error_handling(errors):
    sr = cudf.Series(["random", "words"])
    got = sr.astype("datetime64[ns]", errors=errors)
    assert_eq(sr, got)


@pytest.mark.parametrize("dtype", ALL_TYPES)
def test_df_constructor_dtype(dtype):
    if "datetime" in dtype:
        data = ["1991-11-20", "2004-12-04", "2016-09-13", None]
    elif dtype == "str":
        data = ["a", "b", "c", None]
    elif "float" in dtype:
        data = [1.0, 0.5, -1.1, np.nan, None]
    elif "bool" in dtype:
        data = [True, False, None]
    else:
        data = [1, 2, 3, None]

    sr = cudf.Series(data, dtype=dtype)

    expect = cudf.DataFrame()
    expect["foo"] = sr
    expect["bar"] = sr
    got = cudf.DataFrame({"foo": data, "bar": data}, dtype=dtype)

    assert_eq(expect, got)


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "data",
    [
        lambda: cudf.datasets.randomdata(
            nrows=10, dtypes={"a": "category", "b": int, "c": float, "d": int}
        ),
        lambda: cudf.datasets.randomdata(
            nrows=10, dtypes={"a": "category", "b": int, "c": float, "d": str}
        ),
        lambda: cudf.datasets.randomdata(
            nrows=10, dtypes={"a": bool, "b": int, "c": float, "d": str}
        ),
        lambda: cudf.DataFrame(),
        lambda: cudf.DataFrame({"a": [0, 1, 2], "b": [1, None, 3]}),
        lambda: cudf.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [7, np.nan, 9, 10],
                "c": cudf.Series(
                    [np.nan, np.nan, np.nan, np.nan], nan_as_null=False
                ),
                "d": cudf.Series([None, None, None, None], dtype="int64"),
                "e": [100, None, 200, None],
                "f": cudf.Series([10, None, np.nan, 11], nan_as_null=False),
            }
        ),
        lambda: cudf.DataFrame(
            {
                "a": [10, 11, 12, 13, 14, 15],
                "b": cudf.Series(
                    [10, None, np.nan, 2234, None, np.nan], nan_as_null=False
                ),
            }
        ),
    ],
)
@pytest.mark.parametrize(
    "op", ["max", "min", "sum", "product", "mean", "var", "std"]
)
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_rowwise_ops(data, op, skipna, numeric_only):
    gdf = data()
    pdf = gdf.to_pandas()

    kwargs = {"axis": 1, "skipna": skipna, "numeric_only": numeric_only}
    if op in ("var", "std"):
        kwargs["ddof"] = 0

    if not numeric_only and not all(
        (
            (pdf[column].count() == 0)
            if skipna
            else (pdf[column].notna().count() == 0)
        )
        or cudf.api.types.is_numeric_dtype(pdf[column].dtype)
        or pdf[column].dtype.kind == "b"
        for column in pdf
    ):
        with pytest.raises(TypeError):
            expected = getattr(pdf, op)(**kwargs)
        with pytest.raises(TypeError):
            got = getattr(gdf, op)(**kwargs)
    else:
        expected = getattr(pdf, op)(**kwargs)
        got = getattr(gdf, op)(**kwargs)

        assert_eq(
            expected,
            got,
            check_dtype=False,
            check_index_type=False if len(got.index) == 0 else True,
        )


@pytest.mark.parametrize(
    "op", ["max", "min", "sum", "product", "mean", "var", "std"]
)
def test_rowwise_ops_nullable_dtypes_all_null(op):
    gdf = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [7, np.nan, 9, 10],
            "c": cudf.Series([np.nan, np.nan, np.nan, np.nan], dtype=float),
            "d": cudf.Series([None, None, None, None], dtype="int64"),
            "e": [100, None, 200, None],
            "f": cudf.Series([10, None, np.nan, 11], nan_as_null=False),
        }
    )

    expected = cudf.Series([None, None, None, None], dtype="float64")

    if op in ("var", "std"):
        got = getattr(gdf, op)(axis=1, ddof=0, skipna=False)
    else:
        got = getattr(gdf, op)(axis=1, skipna=False)

    assert_eq(got.null_count, expected.null_count)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "op",
    [
        "max",
        "min",
        "sum",
        "product",
        "mean",
        "var",
        "std",
    ],
)
def test_rowwise_ops_nullable_dtypes_partial_null(op):
    gdf = cudf.DataFrame(
        {
            "a": [10, 11, 12, 13, 14, 15],
            "b": cudf.Series(
                [10, None, np.nan, 2234, None, np.nan],
                nan_as_null=False,
            ),
        }
    )

    if op in ("var", "std"):
        got = getattr(gdf, op)(axis=1, ddof=0, skipna=False)
        expected = getattr(gdf.to_pandas(), op)(axis=1, ddof=0, skipna=False)
    else:
        got = getattr(gdf, op)(axis=1, skipna=False)
        expected = getattr(gdf.to_pandas(), op)(axis=1, skipna=False)

    assert_eq(got.null_count, 2)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "op,expected",
    [
        (
            "max",
            lambda: cudf.Series(
                [10, None, None, 2234, None, 453],
                dtype="int64",
            ),
        ),
        (
            "min",
            lambda: cudf.Series(
                [10, None, None, 13, None, 15],
                dtype="int64",
            ),
        ),
        (
            "sum",
            lambda: cudf.Series(
                [20, None, None, 2247, None, 468],
                dtype="int64",
            ),
        ),
        (
            "product",
            lambda: cudf.Series(
                [100, None, None, 29042, None, 6795],
                dtype="int64",
            ),
        ),
        (
            "mean",
            lambda: cudf.Series(
                [10.0, None, None, 1123.5, None, 234.0],
                dtype="float32",
            ),
        ),
        (
            "var",
            lambda: cudf.Series(
                [0.0, None, None, 1233210.25, None, 47961.0],
                dtype="float32",
            ),
        ),
        (
            "std",
            lambda: cudf.Series(
                [0.0, None, None, 1110.5, None, 219.0],
                dtype="float32",
            ),
        ),
    ],
)
def test_rowwise_ops_nullable_int_dtypes(op, expected):
    gdf = cudf.DataFrame(
        {
            "a": [10, 11, None, 13, None, 15],
            "b": cudf.Series(
                [10, None, 323, 2234, None, 453],
                nan_as_null=False,
            ),
        }
    )

    if op in ("var", "std"):
        got = getattr(gdf, op)(axis=1, ddof=0, skipna=False)
    else:
        got = getattr(gdf, op)(axis=1, skipna=False)

    expected = expected()
    assert_eq(got.null_count, expected.null_count)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": pd.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[ms]"
            ),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": pd.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[ns]"
            ),
            "t3": pd.Series(
                ["1960-08-31 06:00:00", "2030-08-02 10:00:00"], dtype="<M8[s]"
            ),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": pd.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[us]"
            ),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": pd.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[ms]"
            ),
            "i1": pd.Series([1001, 2002], dtype="int64"),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": pd.Series(["1940-08-31 06:00:00", None], dtype="<M8[ms]"),
            "i1": pd.Series([1001, 2002], dtype="int64"),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "i1": pd.Series([1001, 2002], dtype="int64"),
            "f1": pd.Series([-100.001, 123.456], dtype="float64"),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "i1": pd.Series([1001, 2002], dtype="int64"),
            "f1": pd.Series([-100.001, 123.456], dtype="float64"),
            "b1": pd.Series([True, False], dtype="bool"),
        },
    ],
)
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_rowwise_ops_datetime_dtypes(data, op, skipna, numeric_only):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    if not numeric_only and not all(dt.kind == "M" for dt in gdf.dtypes):
        with pytest.raises(TypeError):
            got = getattr(gdf, op)(
                axis=1, skipna=skipna, numeric_only=numeric_only
            )
        with pytest.raises(TypeError):
            expected = getattr(pdf, op)(
                axis=1, skipna=skipna, numeric_only=numeric_only
            )
    else:
        got = getattr(gdf, op)(
            axis=1, skipna=skipna, numeric_only=numeric_only
        )
        expected = getattr(pdf, op)(
            axis=1, skipna=skipna, numeric_only=numeric_only
        )
        if got.dtype == cudf.dtype(
            "datetime64[us]"
        ) and expected.dtype == np.dtype("datetime64[ns]"):
            # Workaround for a PANDAS-BUG:
            # https://github.com/pandas-dev/pandas/issues/52524
            assert_eq(got.astype("datetime64[ns]"), expected)
        else:
            assert_eq(got, expected, check_dtype=False)


@pytest.mark.parametrize(
    "data,op,skipna",
    [
        (
            {
                "t1": pd.Series(
                    ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
                    dtype="<M8[ms]",
                ),
                "t2": pd.Series(
                    ["1940-08-31 06:00:00", None], dtype="<M8[ms]"
                ),
            },
            "max",
            True,
        ),
        (
            {
                "t1": pd.Series(
                    ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
                    dtype="<M8[ms]",
                ),
                "t2": pd.Series(
                    ["1940-08-31 06:00:00", None], dtype="<M8[ms]"
                ),
            },
            "min",
            False,
        ),
        (
            {
                "t1": pd.Series(
                    ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
                    dtype="<M8[ms]",
                ),
                "t2": pd.Series(
                    ["1940-08-31 06:00:00", None], dtype="<M8[ms]"
                ),
            },
            "min",
            True,
        ),
    ],
)
def test_rowwise_ops_datetime_dtypes_2(data, op, skipna):
    gdf = cudf.DataFrame(data)

    pdf = gdf.to_pandas()

    got = getattr(gdf, op)(axis=1, skipna=skipna)
    expected = getattr(pdf, op)(axis=1, skipna=skipna)

    assert_eq(got, expected)


def test_rowwise_ops_datetime_dtypes_pdbug():
    data = {
        "t1": pd.Series(
            ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
            dtype="<M8[ns]",
        ),
        "t2": pd.Series(["1940-08-31 06:00:00", pd.NaT], dtype="<M8[ns]"),
    }
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    expected = pdf.max(axis=1, skipna=False)
    got = gdf.max(axis=1, skipna=False)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [5.0, 6.0, 7.0],
        "single value",
        np.array(1, dtype="int64"),
        np.array(0.6273643, dtype="float64"),
    ],
)
def test_insert(data):
    pdf = pd.DataFrame.from_dict({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    gdf = cudf.DataFrame.from_pandas(pdf)

    # insertion by index

    pdf.insert(0, "foo", data)
    gdf.insert(0, "foo", data)

    assert_eq(pdf, gdf)

    pdf.insert(3, "bar", data)
    gdf.insert(3, "bar", data)

    assert_eq(pdf, gdf)

    pdf.insert(1, "baz", data)
    gdf.insert(1, "baz", data)

    assert_eq(pdf, gdf)

    # pandas insert doesn't support negative indexing
    pdf.insert(len(pdf.columns), "qux", data)
    gdf.insert(-1, "qux", data)

    assert_eq(pdf, gdf)


def test_insert_NA():
    pdf = pd.DataFrame.from_dict({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    gdf = cudf.DataFrame.from_pandas(pdf)

    pdf["C"] = pd.NA
    gdf["C"] = cudf.NA
    assert_eq(pdf, gdf)


def test_cov():
    gdf = cudf.datasets.randomdata(10)
    pdf = gdf.to_pandas()

    assert_eq(pdf.cov(), gdf.cov())


@pytest_xfail(reason="cupy-based cov does not support nulls")
def test_cov_nans():
    pdf = pd.DataFrame()
    pdf["a"] = [None, None, None, 2.00758632, None]
    pdf["b"] = [0.36403686, None, None, None, None]
    pdf["c"] = [None, None, None, 0.64882227, None]
    pdf["d"] = [None, -1.46863125, None, 1.22477948, -0.06031689]
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.cov(), gdf.cov())


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "psr",
    [
        pd.Series([4, 2, 3]),
        pd.Series([4, 2, 3], index=["a", "b", "c"]),
        pd.Series([4, 2, 3], index=["a", "b", "d"]),
        pd.Series([4, 2], index=["a", "b"]),
        pd.Series([4, 2, 3]),
        pd.Series([4, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"]),
    ],
)
@pytest.mark.parametrize("colnames", [["a", "b", "c"], [0, 1, 2]])
@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.pow,
        operator.eq,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.ne,
    ],
)
def test_df_sr_binop(psr, colnames, op):
    data = [[3.0, 2.0, 5.0], [3.0, None, 5.0], [6.0, 7.0, np.nan]]
    data = dict(zip(colnames, data, strict=True))

    gsr = cudf.Series.from_pandas(psr).astype("float64")

    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas(nullable=True)

    psr = gsr.to_pandas(nullable=True)

    try:
        expect = op(pdf, psr)
    except ValueError:
        with pytest.raises(ValueError):
            op(gdf, gsr)
        with pytest.raises(ValueError):
            op(psr, pdf)
        with pytest.raises(ValueError):
            op(gsr, gdf)
    else:
        got = op(gdf, gsr).to_pandas(nullable=True)
        assert_eq(expect, got, check_dtype=False, check_like=True)

        expect = op(psr, pdf)
        got = op(gsr, gdf).to_pandas(nullable=True)
        assert_eq(expect, got, check_dtype=False, check_like=True)


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.pow,
        # comparison ops will temporarily XFAIL
        # see PR  https://github.com/rapidsai/cudf/pull/7491
        pytest.param(operator.eq, marks=pytest_xfail()),
        pytest.param(operator.lt, marks=pytest_xfail()),
        pytest.param(operator.le, marks=pytest_xfail()),
        pytest.param(operator.gt, marks=pytest_xfail()),
        pytest.param(operator.ge, marks=pytest_xfail()),
        pytest.param(operator.ne, marks=pytest_xfail()),
    ],
)
def test_df_sr_binop_col_order(op):
    colnames = [0, 1, 2]
    data = [[0, 2, 5], [3, None, 5], [6, 7, np.nan]]
    data = dict(zip(colnames, data, strict=True))

    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame.from_dict(data)

    gsr = cudf.Series([1, 2, 3, 4, 5], index=["a", "b", "d", "0", "12"])
    psr = gsr.to_pandas()

    with expect_warning_if(
        op
        in {
            operator.eq,
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            operator.ne,
        },
        FutureWarning,
    ):
        expect = op(pdf, psr).astype("float")
    out = op(gdf, gsr).astype("float")
    got = out[expect.columns]

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "list_input",
    [
        pytest.param([1, 2, 3, 4], id="smaller"),
        pytest.param([1, 2, 3, 4, 5, 6], id="larger"),
    ],
)
@pytest.mark.parametrize(
    "key",
    [
        pytest.param("list_test", id="new_column"),
        pytest.param("id", id="existing_column"),
    ],
)
def test_setitem_diff_size_list(list_input, key):
    gdf = cudf.datasets.randomdata(5)
    with pytest.raises(
        ValueError, match=("All columns must be of equal length")
    ):
        gdf[key] = list_input


@pytest.mark.parametrize(
    "data, index",
    [
        [[1, 2, 3, 4], None],
        [[1, 2, 3, 4, 5, 6], None],
        [[1, 2, 3], [4, 5, 6]],
    ],
)
@pytest.mark.parametrize("klass", [pd.Series, cudf.Series])
@pytest.mark.parametrize(
    "key",
    [
        pytest.param("list_test", id="new_column"),
        pytest.param("id", id="existing_column"),
    ],
)
def test_setitem_diff_size_series(klass, data, index, key):
    gdf = cudf.datasets.randomdata(5)
    pdf = gdf.to_pandas()

    series_input = klass(data, index=index)
    pandas_input = series_input
    if isinstance(pandas_input, cudf.Series):
        pandas_input = pandas_input.to_pandas()

    expect = pdf
    expect[key] = pandas_input

    got = gdf
    got[key] = series_input

    # Pandas uses NaN and typecasts to float64 if there's missing values on
    # alignment, so need to typecast to float64 for equality comparison
    expect = expect.astype("float64")
    got = got.astype("float64")

    assert_eq(expect, got)


def test_tupleize_cols_False_set():
    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()
    pdf[("a", "b")] = [1]
    gdf[("a", "b")] = [1]
    assert_eq(pdf, gdf)
    assert_eq(pdf.columns, gdf.columns)


@pytest.mark.parametrize(
    "arg", [slice(2, 8, 3), slice(1, 20, 4), slice(-2, -6, -2)]
)
def test_dataframe_strided_slice(arg):
    mul = pd.DataFrame(
        {
            "Index": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "AlphaIndex": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        }
    )
    pdf = pd.DataFrame(
        {"Val": [10, 9, 8, 7, 6, 5, 4, 3, 2]},
        index=pd.MultiIndex.from_frame(mul),
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

    expect = pdf[arg]
    got = gdf[arg]

    assert_eq(expect, got)


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "col_data",
    [
        range(5),
        ["a", "b", "x", "y", "z"],
        [1.0, 0.213, 0.34332],
        ["a"],
        [1],
        [0.2323],
        [],
    ],
)
@pytest.mark.parametrize(
    "assign_val",
    [
        1,
        2,
        np.array(2),
        cupy.array(2),
        0.32324,
        np.array(0.34248),
        cupy.array(0.34248),
        "abc",
        np.array("abc", dtype="object"),
        np.array("abc", dtype="str"),
        np.array("abc"),
        None,
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_dataframe_assign_scalar(request, col_data, assign_val):
    request.applymarker(
        pytest.mark.xfail(
            condition=PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION
            and len(col_data) == 0,
            reason="https://github.com/pandas-dev/pandas/issues/56679",
        )
    )
    pdf = pd.DataFrame({"a": col_data})
    gdf = cudf.DataFrame({"a": col_data})

    pdf["b"] = (
        cupy.asnumpy(assign_val)
        if isinstance(assign_val, cupy.ndarray)
        else assign_val
    )
    gdf["b"] = assign_val

    assert_eq(pdf, gdf)


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "col_data",
    [
        1,
        2,
        np.array(2),
        cupy.array(2),
        0.32324,
        np.array(0.34248),
        cupy.array(0.34248),
        "abc",
        np.array("abc", dtype="object"),
        np.array("abc", dtype="str"),
        np.array("abc"),
        None,
    ],
)
@pytest.mark.parametrize(
    "assign_val",
    [
        1,
        2,
        np.array(2),
        cupy.array(2),
        0.32324,
        np.array(0.34248),
        cupy.array(0.34248),
        "abc",
        np.array("abc", dtype="object"),
        np.array("abc", dtype="str"),
        np.array("abc"),
        None,
    ],
)
def test_dataframe_assign_scalar_with_scalar_cols(col_data, assign_val):
    pdf = pd.DataFrame(
        {
            "a": cupy.asnumpy(col_data)
            if isinstance(col_data, cupy.ndarray)
            else col_data
        },
        index=["dummy_mandatory_index"],
    )
    gdf = cudf.DataFrame({"a": col_data}, index=["dummy_mandatory_index"])

    pdf["b"] = (
        cupy.asnumpy(assign_val)
        if isinstance(assign_val, cupy.ndarray)
        else assign_val
    )
    gdf["b"] = assign_val

    assert_eq(pdf, gdf)
