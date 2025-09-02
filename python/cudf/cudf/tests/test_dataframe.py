# Copyright (c) 2018-2025, NVIDIA CORPORATION.

import array as arr
import datetime
import decimal
import io
import operator
import re
import textwrap
import warnings
from contextlib import contextmanager
from copy import copy

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from packaging import version

import cudf
from cudf.api.extensions import no_default
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.core.column.column import as_column
from cudf.testing import _utils as utils, assert_eq
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


@contextmanager
def _hide_ufunc_warnings(eval_str):
    # pandas raises warnings for some inputs to the following ufuncs:
    if any(
        x in eval_str
        for x in {
            "arctanh",
            "log",
        }
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "invalid value encountered in",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                "divide by zero encountered in",
                category=RuntimeWarning,
            )
            yield
    else:
        yield


@contextmanager
def _hide_concat_empty_dtype_warning():
    with warnings.catch_warnings():
        # Ignoring warnings in this test as warnings are
        # being caught and validated in other tests.
        warnings.filterwarnings(
            "ignore",
            "The behavior of array concatenation with empty "
            "entries is deprecated.",
            category=FutureWarning,
        )
        yield


@pytest.fixture(
    params=[
        pd.DataFrame(
            {
                "a": [0, 1, 2, np.nan, 4, None, 6],
                "b": [np.nan, None, "u", "h", "d", "a", "m"],
            },
            index=["q", "w", "e", "r", "t", "y", "u"],
        ),
        pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": ["a", "b", "u", "h", "d"]}),
        pd.DataFrame(
            {
                "a": [None, None, np.nan, None],
                "b": [np.nan, None, np.nan, None],
            }
        ),
        pd.DataFrame({"a": []}),
        pd.DataFrame({"a": [np.nan], "b": [None]}),
        pd.DataFrame({"a": ["a", "b", "c", None, "e"]}),
        pd.DataFrame({"a": ["a", "b", "c", "d", "e"]}),
    ]
)
def na_data(request):
    return request.param


@pytest.mark.parametrize("a", [[1, 2, 3], [1, 10, 30]])
@pytest.mark.parametrize("b", [[4, 5, 6], [-11, -100, 30]])
def test_concat_index(a, b):
    df = pd.DataFrame()
    df["a"] = a
    df["b"] = b

    gdf = cudf.DataFrame()
    gdf["a"] = a
    gdf["b"] = b

    expected = pd.concat([df.a, df.b])
    actual = cudf.concat([gdf.a, gdf.b])

    assert len(expected) == len(actual)
    assert_eq(expected.index, actual.index)

    expected = pd.concat([df.a, df.b], ignore_index=True)
    actual = cudf.concat([gdf.a, gdf.b], ignore_index=True)

    assert len(expected) == len(actual)
    assert_eq(expected.index, actual.index)


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


@pytest.mark.parametrize("dtype1", utils.supported_numpy_dtypes)
@pytest.mark.parametrize("dtype2", utils.supported_numpy_dtypes)
def test_dataframe_concat_different_numerical_columns(dtype1, dtype2):
    df1 = pd.DataFrame(dict(x=pd.Series(np.arange(5)).astype(dtype1)))
    df2 = pd.DataFrame(dict(x=pd.Series(np.arange(5)).astype(dtype2)))
    if dtype1 != dtype2 and "datetime" in dtype1 or "datetime" in dtype2:
        with pytest.raises(TypeError):
            cudf.concat([df1, df2])
    else:
        pres = pd.concat([df1, df2])
        gres = cudf.concat([cudf.from_pandas(df1), cudf.from_pandas(df2)])
        assert_eq(pres, gres, check_dtype=False, check_index_type=True)


def test_dataframe_concat_different_column_types():
    df1 = cudf.Series([42], dtype=np.float64)
    df2 = cudf.Series(["a"], dtype="category")
    with pytest.raises(ValueError):
        cudf.concat([df1, df2])

    df2 = cudf.Series(["a string"])
    with pytest.raises(TypeError):
        cudf.concat([df1, df2])


@pytest.mark.parametrize("df_1_data", [{"a": [1, 2], "b": [1, 3]}, {}])
@pytest.mark.parametrize("df_2_data", [{"a": [], "b": []}, {}])
def test_concat_empty_dataframe(df_1_data, df_2_data):
    df_1 = cudf.DataFrame(df_1_data)
    df_2 = cudf.DataFrame(df_2_data)
    with _hide_concat_empty_dtype_warning():
        got = cudf.concat([df_1, df_2])
        expect = pd.concat([df_1.to_pandas(), df_2.to_pandas()], sort=False)

    # ignoring dtypes as pandas upcasts int to float
    # on concatenation with empty dataframes

    assert_eq(got, expect, check_dtype=False, check_index_type=True)


@pytest.mark.parametrize(
    "df1_d",
    [
        {"a": [1, 2], "b": [1, 2], "c": ["s1", "s2"], "d": [1.0, 2.0]},
        {"b": [1.9, 10.9], "c": ["s1", "s2"]},
        {"c": ["s1"], "b": pd.Series([None], dtype="float"), "a": [False]},
    ],
)
@pytest.mark.parametrize(
    "df2_d",
    [
        {"a": [1, 2, 3]},
        {"a": [1, None, 3], "b": [True, True, False], "c": ["s3", None, "s4"]},
        {"a": [], "b": []},
        {},
    ],
)
def test_concat_different_column_dataframe(df1_d, df2_d):
    with _hide_concat_empty_dtype_warning():
        got = cudf.concat(
            [
                cudf.DataFrame(df1_d),
                cudf.DataFrame(df2_d),
                cudf.DataFrame(df1_d),
            ],
            sort=False,
        )

    pdf1 = pd.DataFrame(df1_d)
    pdf2 = pd.DataFrame(df2_d)

    expect = pd.concat([pdf1, pdf2, pdf1], sort=False)

    # numerical columns are upcasted to float in cudf.DataFrame.to_pandas()
    # casts nan to 0 in non-float numerical columns

    numeric_cols = got.dtypes[got.dtypes != "object"].index
    for col in numeric_cols:
        got[col] = got[col].astype(np.float64).fillna(np.nan)

    assert_eq(got, expect, check_dtype=False, check_index_type=True)


@pytest.mark.parametrize(
    "ser_1", [pd.Series([1, 2, 3]), pd.Series([], dtype="float64")]
)
def test_concat_empty_series(ser_1):
    ser_2 = pd.Series([], dtype="float64")
    with _hide_concat_empty_dtype_warning():
        got = cudf.concat([cudf.Series(ser_1), cudf.Series(ser_2)])
        expect = pd.concat([ser_1, ser_2])

    assert_eq(got, expect, check_index_type=True)


def test_concat_with_axis():
    df1 = pd.DataFrame(dict(x=np.arange(5), y=np.arange(5)))
    df2 = pd.DataFrame(dict(a=np.arange(5), b=np.arange(5)))

    concat_df = pd.concat([df1, df2], axis=1)
    cdf1 = cudf.from_pandas(df1)
    cdf2 = cudf.from_pandas(df2)

    # concat only dataframes
    concat_cdf = cudf.concat([cdf1, cdf2], axis=1)
    assert_eq(concat_cdf, concat_df, check_index_type=True)

    # concat only series
    concat_s = pd.concat([df1.x, df1.y], axis=1)
    cs1 = cudf.Series.from_pandas(df1.x)
    cs2 = cudf.Series.from_pandas(df1.y)
    concat_cdf_s = cudf.concat([cs1, cs2], axis=1)

    assert_eq(concat_cdf_s, concat_s, check_index_type=True)

    rng = np.random.default_rng(seed=0)
    # concat series and dataframes
    s3 = pd.Series(rng.random(5))
    cs3 = cudf.Series.from_pandas(s3)

    concat_cdf_all = cudf.concat([cdf1, cs3, cdf2], axis=1)
    concat_df_all = pd.concat([df1, s3, df2], axis=1)
    assert_eq(concat_cdf_all, concat_df_all, check_index_type=True)

    # concat manual multi index
    midf1 = cudf.from_pandas(df1)
    midf1.index = cudf.MultiIndex(
        levels=[[0, 1, 2, 3], [0, 1]], codes=[[0, 1, 2, 3, 2], [0, 1, 0, 1, 0]]
    )
    midf2 = midf1[2:]
    midf2.index = cudf.MultiIndex(
        levels=[[3, 4, 5], [2, 0]], codes=[[0, 1, 2], [1, 0, 1]]
    )
    mipdf1 = midf1.to_pandas()
    mipdf2 = midf2.to_pandas()

    assert_eq(
        cudf.concat([midf1, midf2]),
        pd.concat([mipdf1, mipdf2]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([midf2, midf1]),
        pd.concat([mipdf2, mipdf1]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([midf1, midf2, midf1]),
        pd.concat([mipdf1, mipdf2, mipdf1]),
        check_index_type=True,
    )

    rng = np.random.default_rng(seed=0)
    # concat groupby multi index
    gdf1 = cudf.DataFrame(
        {
            "x": rng.integers(0, 10, 10),
            "y": rng.integers(0, 10, 10),
            "z": rng.integers(0, 10, 10),
            "v": rng.integers(0, 10, 10),
        }
    )
    gdf2 = gdf1[5:]
    gdg1 = gdf1.groupby(["x", "y"]).min()
    gdg2 = gdf2.groupby(["x", "y"]).min()
    pdg1 = gdg1.to_pandas()
    pdg2 = gdg2.to_pandas()

    assert_eq(
        cudf.concat([gdg1, gdg2]),
        pd.concat([pdg1, pdg2]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([gdg2, gdg1]),
        pd.concat([pdg2, pdg1]),
        check_index_type=True,
    )

    # series multi index concat
    gdgz1 = gdg1.z
    gdgz2 = gdg2.z
    pdgz1 = gdgz1.to_pandas()
    pdgz2 = gdgz2.to_pandas()

    assert_eq(
        cudf.concat([gdgz1, gdgz2]),
        pd.concat([pdgz1, pdgz2]),
        check_index_type=True,
    )
    assert_eq(
        cudf.concat([gdgz2, gdgz1]),
        pd.concat([pdgz2, pdgz1]),
        check_index_type=True,
    )


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


@pytest.mark.parametrize("set_index", [None, "A", "C", "D"])
@pytest.mark.parametrize("index", [True, False])
@pytest.mark.parametrize("deep", [True, False])
def test_memory_usage(deep, index, set_index):
    # Testing numerical/datetime by comparing with pandas
    # (string and categorical columns will be different)
    rows = 100
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int64"),
            "B": np.arange(rows, dtype="int32"),
            "C": np.arange(rows, dtype="float64"),
        }
    )
    df["D"] = pd.to_datetime(df.A)
    if set_index:
        df = df.set_index(set_index)

    gdf = cudf.from_pandas(df)

    if index and set_index is None:
        # Special Case: Assume RangeIndex size == 0
        with expect_warning_if(deep, UserWarning):
            assert gdf.index.memory_usage(deep=deep) == 0

    else:
        # Check for Series only
        assert df["B"].memory_usage(index=index, deep=deep) == gdf[
            "B"
        ].memory_usage(index=index, deep=deep)

        # Check for entire DataFrame
        assert_eq(
            df.memory_usage(index=index, deep=deep).sort_index(),
            gdf.memory_usage(index=index, deep=deep).sort_index(),
        )


@pytest_xfail
def test_memory_usage_string():
    rows = 100
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int32"),
            "B": rng.choice(["apple", "banana", "orange"], rows),
        }
    )
    gdf = cudf.from_pandas(df)

    # Check deep=False (should match pandas)
    assert gdf.B.memory_usage(deep=False, index=False) == df.B.memory_usage(
        deep=False, index=False
    )

    # Check string column
    assert gdf.B.memory_usage(deep=True, index=False) == df.B.memory_usage(
        deep=True, index=False
    )

    # Check string index
    assert gdf.set_index("B").index.memory_usage(
        deep=True
    ) == df.B.memory_usage(deep=True, index=False)


def test_memory_usage_cat():
    rows = 100
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int32"),
            "B": rng.choice(["apple", "banana", "orange"], rows),
        }
    )
    df["B"] = df.B.astype("category")
    gdf = cudf.from_pandas(df)

    expected = (
        gdf.B._column.categories.memory_usage
        + gdf.B._column.codes.memory_usage
    )

    # Check cat column
    assert gdf.B.memory_usage(deep=True, index=False) == expected

    # Check cat index
    assert gdf.set_index("B").index.memory_usage(deep=True) == expected


def test_memory_usage_list():
    df = cudf.DataFrame({"A": [[0, 1, 2, 3], [4, 5, 6], [7, 8], [9]]})
    expected = (
        df.A._column.offsets.memory_usage + df.A._column.elements.memory_usage
    )
    assert expected == df.A.memory_usage()


def test_memory_usage_multi():
    # We need to sample without replacement to guarantee that the size of the
    # levels are always the same.
    rng = np.random.default_rng(seed=0)
    rows = 10
    df = pd.DataFrame(
        {
            "A": np.arange(rows, dtype="int32"),
            "B": rng.choice(
                np.arange(rows, dtype="int64"), rows, replace=False
            ),
            "C": rng.choice(
                np.arange(rows, dtype="float64"), rows, replace=False
            ),
        }
    ).set_index(["B", "C"])
    gdf = cudf.from_pandas(df)
    # Assume MultiIndex memory footprint is just that
    # of the underlying columns, levels, and codes
    expect = rows * 16  # Source Columns
    expect += rows * 16  # Codes
    expect += rows * 8  # Level 0
    expect += rows * 8  # Level 1

    assert expect == gdf.index.memory_usage(deep=True)


@pytest.mark.parametrize("index", [False, True])
def test_memory_usage_index_preserve_types(index):
    data = [[1, 2, 3]]
    columns = pd.Index(np.array([1, 2, 3], dtype=np.int8), name="a")
    result = (
        cudf.DataFrame(data, columns=columns).memory_usage(index=index).index
    )
    expected = (
        pd.DataFrame(data, columns=columns).memory_usage(index=index).index
    )
    if index:
        # pandas returns an Index[object] with int and string elements
        expected = expected.astype(str)
    assert_eq(result, expected)


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


def test_dataframe_info_basic():
    buffer = io.StringIO()
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    Index: 10 entries, a to 1111
    Data columns (total 10 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   0       10 non-null     float64
     1   1       10 non-null     float64
     2   2       10 non-null     float64
     3   3       10 non-null     float64
     4   4       10 non-null     float64
     5   5       10 non-null     float64
     6   6       10 non-null     float64
     7   7       10 non-null     float64
     8   8       10 non-null     float64
     9   9       10 non-null     float64
    dtypes: float64(10)
    memory usage: 859.0+ bytes
    """
    )
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        rng.standard_normal(size=(10, 10)),
        index=["a", "2", "3", "4", "5", "6", "7", "8", "100", "1111"],
    )
    cudf.from_pandas(df).info(buf=buffer, verbose=True)
    s = buffer.getvalue()
    assert str_cmp == s


def test_dataframe_info_verbose_mem_usage():
    buffer = io.StringIO()
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["safdas", "assa", "asdasd"]})
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       3 non-null      int64
     1   b       3 non-null      object
    dtypes: int64(1), object(1)
    memory usage: 56.0+ bytes
    """
    )
    cudf.from_pandas(df).info(buf=buffer, verbose=True)
    s = buffer.getvalue()
    assert str_cmp == s

    buffer.truncate(0)
    buffer.seek(0)

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Columns: 2 entries, a to b
    dtypes: int64(1), object(1)
    memory usage: 56.0+ bytes
    """
    )
    cudf.from_pandas(df).info(buf=buffer, verbose=False)
    s = buffer.getvalue()
    assert str_cmp == s

    buffer.truncate(0)
    buffer.seek(0)

    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["safdas", "assa", "asdasd"]},
        index=["sdfdsf", "sdfsdfds", "dsfdf"],
    )
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    Index: 3 entries, sdfdsf to dsfdf
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       3 non-null      int64
     1   b       3 non-null      object
    dtypes: int64(1), object(1)
    memory usage: 91.0 bytes
    """
    )
    cudf.from_pandas(df).info(buf=buffer, verbose=True, memory_usage="deep")
    s = buffer.getvalue()
    assert str_cmp == s

    buffer.truncate(0)
    buffer.seek(0)

    int_values = [1, 2, 3, 4, 5]
    text_values = ["alpha", "beta", "gamma", "delta", "epsilon"]
    float_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    df = cudf.DataFrame(
        {
            "int_col": int_values,
            "text_col": text_values,
            "float_col": float_values,
        }
    )
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
     0   int_col    5 non-null      int64
     1   text_col   5 non-null      object
     2   float_col  5 non-null      float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 130.0 bytes
    """
    )
    df.info(buf=buffer, verbose=True, memory_usage="deep")
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)


def test_dataframe_info_null_counts():
    int_values = [1, 2, 3, 4, 5]
    text_values = ["alpha", "beta", "gamma", "delta", "epsilon"]
    float_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    df = cudf.DataFrame(
        {
            "int_col": int_values,
            "text_col": text_values,
            "float_col": float_values,
        }
    )
    buffer = io.StringIO()
    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
     #   Column     Dtype
    ---  ------     -----
     0   int_col    int64
     1   text_col   object
     2   float_col  float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 130.0+ bytes
    """
    )
    df.info(buf=buffer, verbose=True, null_counts=False)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df.info(buf=buffer, verbose=True, max_cols=0)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df = cudf.DataFrame()

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 0 entries
    Empty DataFrame"""
    )
    df.info(buf=buffer, verbose=True)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, None, 10, 11, 12, None],
            "b": ["a", "b", "c", "sd", "sdf", "sd", None, None],
        }
    )

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 8 entries, 0 to 7
    Data columns (total 2 columns):
     #   Column  Dtype
    ---  ------  -----
     0   a       int64
     1   b       object
    dtypes: int64(1), object(1)
    memory usage: 238.0+ bytes
    """
    )
    pd.options.display.max_info_rows = 2
    df.info(buf=buffer, max_cols=2, null_counts=None)
    pd.reset_option("display.max_info_rows")
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    str_cmp = textwrap.dedent(
        """\
    <class 'cudf.core.dataframe.DataFrame'>
    RangeIndex: 8 entries, 0 to 7
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   a       6 non-null      int64
     1   b       6 non-null      object
    dtypes: int64(1), object(1)
    memory usage: 238.0+ bytes
    """
    )

    df.info(buf=buffer, max_cols=2, null_counts=None)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string

    buffer.truncate(0)
    buffer.seek(0)

    df.info(buf=buffer, null_counts=True)
    actual_string = buffer.getvalue()
    assert str_cmp == actual_string


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": [1, 2, 3, 4, 5, 10, 11, 12, 33, 55, 19]}),
        pd.DataFrame(
            {
                "one": [1, 2, 3, 4, 5, 10],
                "two": ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            }
        ),
        pd.DataFrame(
            {
                "one": [1, 2, 3, 4, 5, 10],
                "two": ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            },
            index=[10, 20, 30, 40, 50, 60],
        ),
        pd.DataFrame(
            {
                "one": [1, 2, 3, 4, 5, 10],
                "two": ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            },
            index=["a", "b", "c", "d", "e", "f"],
        ),
        pd.DataFrame(index=["a", "b", "c", "d", "e", "f"]),
        pd.DataFrame(columns=["a", "b", "c", "d", "e", "f"]),
        pd.DataFrame(index=[10, 11, 12]),
        pd.DataFrame(columns=[10, 11, 12]),
        pd.DataFrame(),
        pd.DataFrame({"one": [], "two": []}),
        pd.DataFrame({2: [], 1: []}),
        pd.DataFrame(
            {
                0: [1, 2, 3, 4, 5, 10],
                1: ["abc", "def", "ghi", "xyz", "pqr", "abc"],
                100: ["a", "b", "b", "x", "z", "a"],
            },
            index=[10, 20, 30, 40, 50, 60],
        ),
    ],
)
def test_dataframe_keys(df):
    gdf = cudf.from_pandas(df)

    assert_eq(
        df.keys(),
        gdf.keys(),
    )


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series([1, 2, 3, 4, 5, 10, 11, 12, 33, 55, 19]),
        pd.Series(["abc", "def", "ghi", "xyz", "pqr", "abc"]),
        pd.Series(
            [1, 2, 3, 4, 5, 10],
            index=["abc", "def", "ghi", "xyz", "pqr", "abc"],
        ),
        pd.Series(
            ["abc", "def", "ghi", "xyz", "pqr", "abc"],
            index=[1, 2, 3, 4, 5, 10],
        ),
        pd.Series(index=["a", "b", "c", "d", "e", "f"], dtype="float64"),
        pd.Series(index=[10, 11, 12], dtype="float64"),
        pd.Series(dtype="float64"),
        pd.Series([], dtype="float64"),
    ],
)
def test_series_keys(ps):
    gds = cudf.from_pandas(ps)

    assert_eq(ps.keys(), gds.keys())


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB")),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[7, 8]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[100]),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame(
            {"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]},
            index=[100, 200, 300, 400, 500, 0],
        ),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        pd.DataFrame([[5, 6], [7, 8]], columns=list("BD")),
        pd.DataFrame([[5, 6], [7, 8]], columns=list("DE")),
        pd.DataFrame(),
        pd.DataFrame(
            {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
        ),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[200]),
        pd.DataFrame([]),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        pd.DataFrame([], index=[100]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[0, 100, 200, 300],
        ),
    ],
)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_dataframe_concat_dataframe(df, other, sort, ignore_index):
    pdf = df
    other_pd = other

    gdf = cudf.from_pandas(df)
    other_gd = cudf.from_pandas(other)

    with _hide_concat_empty_dtype_warning():
        expected = pd.concat(
            [pdf, other_pd], sort=sort, ignore_index=ignore_index
        )
        actual = cudf.concat(
            [gdf, other_gd], sort=sort, ignore_index=ignore_index
        )

    # In empty dataframe cases, Pandas & cudf differ in columns
    # creation, pandas creates RangeIndex(0, 0)
    # whereas cudf creates an empty Index([], dtype="object").
    check_column_type = (
        False if len(expected.columns) == len(df.columns) == 0 else True
    )

    if expected.shape != df.shape:
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_column_type=check_column_type,
        )
    else:
        assert_eq(
            expected,
            actual,
            check_index_type=not gdf.empty,
            check_column_type=check_column_type,
        )


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame({12: [], 22: []}),
        pd.DataFrame([[1, 2], [3, 4]], columns=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=[0, 1], index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=[1, 0], index=[7, 8]),
        pd.DataFrame(
            {
                23: [315.3324, 3243.32432, 3232.332, -100.32],
                33: [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                0: [315.3324, 3243.32432, 3232.332, -100.32],
                1: [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        pd.Series([10, 11, 23, 234, 13]),
        pd.Series([10, 11, 23, 234, 13], index=[11, 12, 13, 44, 33]),
        {1: 1},
        {0: 10, 1: 100, 2: 102},
    ],
)
@pytest.mark.parametrize("sort", [False, True])
def test_dataframe_concat_series(df, other, sort):
    pdf = df
    gdf = cudf.from_pandas(df)

    if isinstance(other, dict):
        other_pd = pd.Series(other)
    else:
        other_pd = other
    other_gd = cudf.from_pandas(other_pd)

    expected = pd.concat([pdf, other_pd], ignore_index=True, sort=sort)
    actual = cudf.concat([gdf, other_gd], ignore_index=True, sort=sort)

    if expected.shape != df.shape:
        # Ignore the column type comparison because pandas incorrectly
        # returns pd.Index([1, 2, 3], dtype="object") instead
        # of pd.Index([1, 2, 3], dtype="int64")
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_column_type=False,
            check_index_type=True,
        )
    else:
        assert_eq(expected, actual, check_index_type=not gdf.empty)


def test_dataframe_concat_series_mixed_index():
    df = cudf.DataFrame({"first": [], "d": []})
    pdf = df.to_pandas()

    sr = cudf.Series([1, 2, 3, 4])
    psr = sr.to_pandas()

    assert_eq(
        cudf.concat([df, sr], ignore_index=True),
        pd.concat([pdf, psr], ignore_index=True),
        check_dtype=False,
    )


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[10, 20, 30]),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB")),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[7, 8]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[100]),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame(
            {"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]},
            index=[100, 200, 300, 400, 500, 0],
        ),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [pd.DataFrame([[5, 6], [7, 8]], columns=list("AB"))],
        [
            pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("BD")),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("DE")),
        ],
        [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()],
        [
            pd.DataFrame(
                {"c": [10, 11, 22, 33, 44, 100]}, index=[7, 8, 9, 10, 11, 20]
            ),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame([[5, 6], [7, 8]], columns=list("AB")),
        ],
        [
            pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
            pd.DataFrame({"l": [10]}),
            pd.DataFrame({"l": [10]}, index=[200]),
        ],
        [pd.DataFrame([]), pd.DataFrame([], index=[100])],
        [
            pd.DataFrame([]),
            pd.DataFrame([], index=[100]),
            pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                }
            ),
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
        ],
        [
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
            pd.DataFrame(
                {
                    "a": [315.3324, 3243.32432, 3232.332, -100.32],
                    "z": [0.3223, 0.32, 0.0000232, 0.32224],
                },
                index=[0, 100, 200, 300],
            ),
            pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
        ],
    ],
)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_dataframe_concat_dataframe_lists(df, other, sort, ignore_index):
    pdf = df
    other_pd = other

    gdf = cudf.from_pandas(df)
    other_gd = [cudf.from_pandas(o) for o in other]

    with _hide_concat_empty_dtype_warning():
        expected = pd.concat(
            [pdf, *other_pd], sort=sort, ignore_index=ignore_index
        )
        actual = cudf.concat(
            [gdf, *other_gd], sort=sort, ignore_index=ignore_index
        )

    # In some cases, Pandas creates an empty Index([], dtype="object") for
    # columns whereas cudf creates a RangeIndex(0, 0).
    check_column_type = (
        False if len(expected.columns) == len(df.columns) == 0 else True
    )

    if expected.shape != df.shape:
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_column_type=check_column_type,
        )
    else:
        assert_eq(
            expected,
            actual,
            check_index_type=not gdf.empty,
            check_column_type=check_column_type,
        )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"A": [1, 2, 3, np.nan, None, 6]}),
        pd.Series([1, 2, 3, None, np.nan, 5, 6, np.nan]),
    ],
)
@pytest.mark.parametrize("alias", ["bfill", "backfill"])
def test_dataframe_bfill(df, alias):
    gdf = cudf.from_pandas(df)

    with expect_warning_if(alias == "backfill"):
        actual = getattr(df, alias)()
    with expect_warning_if(alias == "backfill"):
        expected = getattr(gdf, alias)()
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"A": [1, 2, 3, np.nan, None, 6]}),
        pd.Series([1, 2, 3, None, np.nan, 5, 6, np.nan]),
    ],
)
@pytest.mark.parametrize("alias", ["ffill", "pad"])
def test_dataframe_ffill(df, alias):
    gdf = cudf.from_pandas(df)

    with expect_warning_if(alias == "pad"):
        actual = getattr(df, alias)()
    with expect_warning_if(alias == "pad"):
        expected = getattr(gdf, alias)()
    assert_eq(expected, actual)


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB")),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[10, 20]),
        pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[7, 8]),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            }
        ),
        pd.DataFrame(
            {
                "a": [315.3324, 3243.32432, 3232.332, -100.32],
                "z": [0.3223, 0.32, 0.0000232, 0.32224],
            },
            index=[7, 20, 11, 9],
        ),
        pd.DataFrame({"l": [10]}),
        pd.DataFrame({"l": [10]}, index=[100]),
        pd.DataFrame({"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]}),
        pd.DataFrame(
            {"f": [10.2, 11.2332, 0.22, 3.3, 44.23, 10.0]},
            index=[100, 200, 300, 400, 500, 0],
        ),
        pd.DataFrame({"first_col": [], "second_col": [], "third_col": []}),
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [[1, 2], [10, 100]],
        [[1, 2, 10, 100, 0.1, 0.2, 0.0021]],
        [[]],
        [[], [], [], []],
        [[0.23, 0.00023, -10.00, 100, 200, 1000232, 1232.32323]],
    ],
)
@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_dataframe_concat_lists(df, other, sort, ignore_index):
    pdf = df
    other_pd = [pd.DataFrame(o) for o in other]

    gdf = cudf.from_pandas(df)
    other_gd = [cudf.from_pandas(o) for o in other_pd]

    with _hide_concat_empty_dtype_warning():
        expected = pd.concat(
            [pdf, *other_pd], sort=sort, ignore_index=ignore_index
        )
        actual = cudf.concat(
            [gdf, *other_gd], sort=sort, ignore_index=ignore_index
        )

    if expected.shape != df.shape:
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_column_type=not gdf.empty,
        )
    else:
        assert_eq(
            expected,
            actual,
            check_index_type=not gdf.empty,
            check_column_type=len(gdf.columns) != 0,
        )


def test_dataframe_concat_series_without_name():
    df = cudf.DataFrame({"a": [1, 2, 3]})
    pdf = df.to_pandas()
    gs = cudf.Series([1, 2, 3])
    ps = gs.to_pandas()

    assert_eq(pd.concat([pdf, ps]), cudf.concat([df, gs]))


def test_cudf_arrow_array_error():
    df = cudf.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        df.__arrow_array__()

    sr = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        sr.__arrow_array__()

    sr = cudf.Series(["a", "b", "c"])
    with pytest.raises(
        TypeError,
        match="Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        sr.__arrow_array__()


@pytest.mark.parametrize(
    "make_weights_axis_1",
    [lambda _: None, lambda s: [1] * s, lambda s: np.ones(s)],
)
def test_sample_axis_1(
    sample_n_frac, random_state_tuple_axis_1, make_weights_axis_1
):
    n, frac = sample_n_frac
    pd_random_state, gd_random_state, checker = random_state_tuple_axis_1

    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "float": [0.05, 0.2, 0.3, 0.2, 0.25],
            "int": [1, 3, 5, 4, 2],
        },
    )
    df = cudf.DataFrame.from_pandas(pdf)

    weights = make_weights_axis_1(len(pdf.columns))

    expected = pdf.sample(
        n=n,
        frac=frac,
        replace=False,
        random_state=pd_random_state,
        weights=weights,
        axis=1,
    )
    got = df.sample(
        n=n,
        frac=frac,
        replace=False,
        random_state=gd_random_state,
        weights=weights,
        axis=1,
    )
    checker(expected, got)


@pytest.mark.parametrize(
    "pdf",
    [
        pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "float": [0.05, 0.2, 0.3, 0.2, 0.25],
                "int": [1, 3, 5, 4, 2],
            },
        ),
        pd.Series([1, 2, 3, 4, 5]),
    ],
)
@pytest.mark.parametrize("replace", [True, False])
def test_sample_axis_0(
    pdf, sample_n_frac, replace, random_state_tuple_axis_0, make_weights_axis_0
):
    n, frac = sample_n_frac
    pd_random_state, gd_random_state, checker = random_state_tuple_axis_0

    df = cudf.from_pandas(pdf)

    pd_weights, gd_weights = make_weights_axis_0(
        len(pdf), isinstance(gd_random_state, np.random.RandomState)
    )
    if (
        not replace
        and not isinstance(gd_random_state, np.random.RandomState)
        and gd_weights is not None
    ):
        pytest.skip(
            "`cupy.random.RandomState` doesn't support weighted sampling "
            "without replacement."
        )

    expected = pdf.sample(
        n=n,
        frac=frac,
        replace=replace,
        random_state=pd_random_state,
        weights=pd_weights,
        axis=0,
    )

    got = df.sample(
        n=n,
        frac=frac,
        replace=replace,
        random_state=gd_random_state,
        weights=gd_weights,
        axis=0,
    )
    checker(expected, got)


@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize(
    "random_state_lib", [cupy.random.RandomState, np.random.RandomState]
)
def test_sample_reproducibility(replace, random_state_lib):
    df = cudf.DataFrame({"a": cupy.arange(0, 1024)})

    n = 1024
    expected = df.sample(n, replace=replace, random_state=random_state_lib(10))
    out = df.sample(n, replace=replace, random_state=random_state_lib(10))

    assert_eq(expected, out)


@pytest.mark.parametrize("axis", [0, 1])
def test_sample_invalid_n_frac_combo(axis):
    n, frac = 2, 0.5
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "float": [0.05, 0.2, 0.3, 0.2, 0.25],
            "int": [1, 3, 5, 4, 2],
        },
    )
    df = cudf.DataFrame.from_pandas(pdf)

    assert_exceptions_equal(
        lfunc=pdf.sample,
        rfunc=df.sample,
        lfunc_args_and_kwargs=([], {"n": n, "frac": frac, "axis": axis}),
        rfunc_args_and_kwargs=([], {"n": n, "frac": frac, "axis": axis}),
    )


@pytest.mark.parametrize("n, frac", [(100, None), (None, 3)])
@pytest.mark.parametrize("axis", [0, 1])
def test_oversample_without_replace(n, frac, axis):
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    df = cudf.DataFrame.from_pandas(pdf)

    assert_exceptions_equal(
        lfunc=pdf.sample,
        rfunc=df.sample,
        lfunc_args_and_kwargs=(
            [],
            {"n": n, "frac": frac, "axis": axis, "replace": False},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"n": n, "frac": frac, "axis": axis, "replace": False},
        ),
    )


@pytest.mark.parametrize("random_state", [None, cupy.random.RandomState(42)])
def test_sample_unsupported_arguments(random_state):
    df = cudf.DataFrame({"float": [0.05, 0.2, 0.3, 0.2, 0.25]})
    with pytest.raises(
        NotImplementedError,
        match="Random sampling with cupy does not support these inputs.",
    ):
        df.sample(
            n=2, replace=False, random_state=random_state, weights=[1] * 5
        )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[100, 10, 1, 0]),
        pd.DataFrame(columns=["a", "b", "c", "d"]),
        pd.DataFrame(columns=["a", "b", "c", "d"], index=[100]),
        pd.DataFrame(
            columns=["a", "b", "c", "d"], index=[100, 10000, 2131, 133]
        ),
        pd.DataFrame({"a": [1, 2, 3], "b": ["abc", "xyz", "klm"]}),
    ],
)
def test_dataframe_empty(df):
    pdf = df
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.empty, gdf.empty)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[100, 10, 1, 0]),
        pd.DataFrame(columns=["a", "b", "c", "d"]),
        pd.DataFrame(columns=["a", "b", "c", "d"], index=[100]),
        pd.DataFrame(
            columns=["a", "b", "c", "d"], index=[100, 10000, 2131, 133]
        ),
        pd.DataFrame({"a": [1, 2, 3], "b": ["abc", "xyz", "klm"]}),
    ],
)
def test_dataframe_size(df):
    pdf = df
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.size, gdf.size)


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series(dtype="float64"),
        pd.Series(index=[100, 10, 1, 0], dtype="float64"),
        pd.Series([], dtype="float64"),
        pd.Series(["a", "b", "c", "d"]),
        pd.Series(["a", "b", "c", "d"], index=[0, 1, 10, 11]),
    ],
)
def test_series_empty(ps):
    ps = ps
    gs = cudf.from_pandas(ps)

    assert_eq(ps.empty, gs.empty)


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        [1],
        {"a": [10, 11, 12]},
        {
            "a": [10, 11, 12],
            "another column name": [12, 22, 34],
            "xyz": [0, 10, 11],
        },
    ],
)
@pytest.mark.parametrize(
    "columns",
    [["a"], ["another column name"], None, pd.Index(["a"], name="index name")],
)
def test_dataframe_init_with_columns(data, columns):
    pdf = pd.DataFrame(data, columns=columns)
    gdf = cudf.DataFrame(data, columns=columns)

    assert_eq(
        pdf,
        gdf,
        check_index_type=len(pdf.index) != 0,
        check_dtype=not (pdf.empty and len(pdf.columns)),
        check_column_type=False,
    )


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "data, ignore_dtype",
    [
        ([pd.Series([1, 2, 3])], False),
        ([pd.Series(index=[1, 2, 3], dtype="float64")], False),
        ([pd.Series(name="empty series name", dtype="float64")], False),
        (
            [pd.Series([1]), pd.Series([], dtype="float64"), pd.Series([3])],
            False,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
                pd.Series([3], name="series that is named"),
            ],
            False,
        ),
        ([pd.Series([1, 2, 3], name="hi")] * 10, False),
        ([pd.Series([1, 2, 3], name=None, index=[10, 11, 12])] * 10, False),
        (
            [
                pd.Series([1, 2, 3], name=None, index=[10, 11, 12]),
                pd.Series([1, 2, 30], name=None, index=[13, 144, 15]),
            ],
            True,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
            ],
            False,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], name="abc", dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
            ],
            False,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([1, -100, 200, -399, 400], name="abc"),
                pd.Series([111, 222, 333], index=[10, 11, 12]),
            ],
            False,
        ),
    ],
)
@pytest.mark.parametrize(
    "columns",
    [
        None,
        ["0"],
        [0],
        ["abc"],
        [144, 13],
        [2, 1, 0],
        pd.Index(["abc"], name="custom_name"),
    ],
)
def test_dataframe_init_from_series_list(data, ignore_dtype, columns):
    gd_data = [cudf.from_pandas(obj) for obj in data]

    expected = pd.DataFrame(data, columns=columns)
    actual = cudf.DataFrame(gd_data, columns=columns)

    if ignore_dtype:
        # When a union is performed to generate columns,
        # the order is never guaranteed. Hence sort by
        # columns before comparison.
        if not expected.columns.equals(actual.columns):
            expected = expected.sort_index(axis=1)
            actual = actual.sort_index(axis=1)
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_index_type=True,
        )
    else:
        assert_eq(
            expected,
            actual,
            check_index_type=True,
            check_column_type=False,
        )


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "data, ignore_dtype, index",
    [
        ([pd.Series([1, 2, 3])], False, ["a", "b", "c"]),
        ([pd.Series(index=[1, 2, 3], dtype="float64")], False, ["a", "b"]),
        (
            [pd.Series(name="empty series name", dtype="float64")],
            False,
            ["index1"],
        ),
        (
            [pd.Series([1]), pd.Series([], dtype="float64"), pd.Series([3])],
            False,
            ["0", "2", "1"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
                pd.Series([3], name="series that is named"),
            ],
            False,
            ["_", "+", "*"],
        ),
        ([pd.Series([1, 2, 3], name="hi")] * 10, False, ["mean"] * 10),
        (
            [pd.Series([1, 2, 3], name=None, index=[10, 11, 12])] * 10,
            False,
            ["abc"] * 10,
        ),
        (
            [
                pd.Series([1, 2, 3], name=None, index=[10, 11, 12]),
                pd.Series([1, 2, 30], name=None, index=[13, 144, 15]),
            ],
            True,
            ["set_index_a", "set_index_b"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
            ],
            False,
            ["a", "b", "c"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], name="abc", dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
            ],
            False,
            ["a", "v", "z"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([1, -100, 200, -399, 400], name="abc"),
                pd.Series([111, 222, 333], index=[10, 11, 12]),
            ],
            False,
            ["a", "v", "z"],
        ),
    ],
)
@pytest.mark.parametrize(
    "columns", [None, ["0"], [0], ["abc"], [144, 13], [2, 1, 0]]
)
def test_dataframe_init_from_series_list_with_index(
    data,
    ignore_dtype,
    index,
    columns,
):
    gd_data = [cudf.from_pandas(obj) for obj in data]

    expected = pd.DataFrame(data, columns=columns, index=index)
    actual = cudf.DataFrame(gd_data, columns=columns, index=index)

    if ignore_dtype:
        # When a union is performed to generate columns,
        # the order is never guaranteed. Hence sort by
        # columns before comparison.
        if not expected.columns.equals(actual.columns):
            expected = expected.sort_index(axis=1)
            actual = actual.sort_index(axis=1)
        assert_eq(expected.fillna(-1), actual.fillna(-1), check_dtype=False)
    else:
        assert_eq(expected, actual, check_column_type=False)


@pytest.mark.parametrize(
    "data, index",
    [
        ([pd.Series([1, 2]), pd.Series([1, 2])], ["a", "b", "c"]),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
                pd.Series([3], name="series that is named"),
            ],
            ["_", "+"],
        ),
        ([pd.Series([1, 2, 3], name="hi")] * 10, ["mean"] * 9),
    ],
)
def test_dataframe_init_from_series_list_with_index_error(data, index):
    gd_data = [cudf.from_pandas(obj) for obj in data]

    assert_exceptions_equal(
        pd.DataFrame,
        cudf.DataFrame,
        ([data], {"index": index}),
        ([gd_data], {"index": index}),
    )


@pytest.mark.parametrize(
    "data",
    [
        [pd.Series([1, 2, 3], index=["a", "a", "a"])],
        [pd.Series([1, 2, 3], index=["a", "a", "a"])] * 4,
        [
            pd.Series([1, 2, 3], index=["a", "b", "a"]),
            pd.Series([1, 2, 3], index=["b", "b", "a"]),
        ],
        [
            pd.Series([1, 2, 3], index=["a", "b", "z"]),
            pd.Series([1, 2, 3], index=["u", "b", "a"]),
            pd.Series([1, 2, 3], index=["u", "b", "u"]),
        ],
    ],
)
def test_dataframe_init_from_series_list_duplicate_index_error(data):
    gd_data = [cudf.from_pandas(obj) for obj in data]

    assert_exceptions_equal(
        lfunc=pd.DataFrame,
        rfunc=cudf.DataFrame,
        lfunc_args_and_kwargs=([], {"data": data}),
        rfunc_args_and_kwargs=([], {"data": gd_data}),
        check_exception_type=False,
    )


def test_dataframe_iterrows_itertuples():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with pytest.raises(
        TypeError,
        match=re.escape(
            "cuDF does not support iteration of DataFrame "
            "via itertuples. Consider using "
            "`.to_pandas().itertuples()` "
            "if you wish to iterate over namedtuples."
        ),
    ):
        df.itertuples()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "cuDF does not support iteration of DataFrame "
            "via iterrows. Consider using "
            "`.to_pandas().iterrows()` "
            "if you wish to iterate over each row."
        ),
    ):
        df.iterrows()


@pytest.mark.parametrize(
    "df",
    [
        lambda: cudf.DataFrame({"a": [1, 2, 3]}),
        lambda: cudf.DataFrame(
            {"a": [1, 2, 3], "b": ["a", "z", "c"]}, index=["a", "z", "x"]
        ),
        lambda: cudf.DataFrame(
            {
                "a": [1, 2, 3, None, 2, 1, None],
                "b": ["a", "z", "c", "a", "v", "z", "z"],
            }
        ),
        lambda: cudf.DataFrame({"a": [], "b": []}),
        lambda: cudf.DataFrame({"a": [None, None], "b": [None, None]}),
        lambda: cudf.DataFrame(
            {
                "a": ["hello", "world", "rapids", "ai", "nvidia"],
                "b": cudf.Series(
                    [1, 21, 21, 11, 11],
                    dtype="timedelta64[s]",
                    index=["a", "b", "c", "d", " e"],
                ),
            },
            index=["a", "b", "c", "d", " e"],
        ),
        lambda: cudf.DataFrame(
            {
                "a": ["hello", None, "world", "rapids", None, "ai", "nvidia"],
                "b": cudf.Series(
                    [1, 21, None, 11, None, 11, None], dtype="datetime64[s]"
                ),
            }
        ),
    ],
)
@pytest.mark.parametrize("numeric_only", [True, False])
@pytest.mark.parametrize("dropna", [True, False])
def test_dataframe_mode(df, numeric_only, dropna):
    df = df()
    pdf = df.to_pandas()

    expected = pdf.mode(numeric_only=numeric_only, dropna=dropna)
    actual = df.mode(numeric_only=numeric_only, dropna=dropna)
    if len(actual.columns) == 0:
        # pandas < 3.0 returns an Index[object] instead of RangeIndex
        actual.columns = expected.columns
    assert_eq(expected, actual, check_dtype=False)


@pytest.mark.parametrize(
    "lhs, rhs", [("a", "a"), ("a", "b"), (1, 1.0), (None, None), (None, "a")]
)
def test_equals_names(lhs, rhs):
    lhs = cudf.DataFrame({lhs: [1, 2]})
    rhs = cudf.DataFrame({rhs: [1, 2]})

    got = lhs.equals(rhs)
    expect = lhs.to_pandas().equals(rhs.to_pandas())

    assert_eq(expect, got)


def test_equals_dtypes():
    lhs = cudf.DataFrame({"a": [1, 2.0]})
    rhs = cudf.DataFrame({"a": [1, 2]})

    got = lhs.equals(rhs)
    expect = lhs.to_pandas().equals(rhs.to_pandas())

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "df1",
    [
        pd.DataFrame({"a": [10, 11, 12]}, index=["a", "b", "z"]),
        pd.DataFrame({"z": ["a"]}),
        pd.DataFrame({"a": [], "b": []}),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        pd.DataFrame(),
        pd.DataFrame({"a": ["a", "a", "c", "z", "A"], "z": [1, 2, 3, 4, 5]}),
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ],
)
def test_dataframe_error_equality(df1, df2, op):
    gdf1 = cudf.from_pandas(df1)
    gdf2 = cudf.from_pandas(df2)

    assert_exceptions_equal(op, op, ([df1, df2],), ([gdf1, gdf2],))


@pytest.mark.parametrize(
    "df,expected_pdf",
    [
        (
            lambda: cudf.DataFrame(
                {
                    "a": cudf.Series([1, 2, None, 3], dtype="uint8"),
                    "b": cudf.Series([23, None, None, 32], dtype="uint16"),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series([1, 2, None, 3], dtype=pd.UInt8Dtype()),
                    "b": pd.Series(
                        [23, None, None, 32], dtype=pd.UInt16Dtype()
                    ),
                }
            ),
        ),
        (
            lambda: cudf.DataFrame(
                {
                    "a": cudf.Series([None, 123, None, 1], dtype="uint32"),
                    "b": cudf.Series(
                        [234, 2323, 23432, None, None, 224], dtype="uint64"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [None, 123, None, 1], dtype=pd.UInt32Dtype()
                    ),
                    "b": pd.Series(
                        [234, 2323, 23432, None, None, 224],
                        dtype=pd.UInt64Dtype(),
                    ),
                }
            ),
        ),
        (
            lambda: cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [-10, 1, None, -1, None, 3], dtype="int8"
                    ),
                    "b": cudf.Series(
                        [111, None, 222, None, 13], dtype="int16"
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [-10, 1, None, -1, None, 3], dtype=pd.Int8Dtype()
                    ),
                    "b": pd.Series(
                        [111, None, 222, None, 13], dtype=pd.Int16Dtype()
                    ),
                }
            ),
        ),
        (
            lambda: cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [11, None, 22, 33, None, 2, None, 3], dtype="int32"
                    ),
                    "b": cudf.Series(
                        [32431, None, None, 32322, 0, 10, -32324, None],
                        dtype="int64",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [11, None, 22, 33, None, 2, None, 3],
                        dtype=pd.Int32Dtype(),
                    ),
                    "b": pd.Series(
                        [32431, None, None, 32322, 0, 10, -32324, None],
                        dtype=pd.Int64Dtype(),
                    ),
                }
            ),
        ),
        (
            lambda: cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [True, None, False, None, False, True, True, False],
                        dtype="bool_",
                    ),
                    "b": cudf.Series(
                        [
                            "abc",
                            "a",
                            None,
                            "hello world",
                            "foo buzz",
                            "",
                            None,
                            "rapids ai",
                        ],
                        dtype="object",
                    ),
                    "c": cudf.Series(
                        [0.1, None, 0.2, None, 3, 4, 1000, None],
                        dtype="float64",
                    ),
                }
            ),
            pd.DataFrame(
                {
                    "a": pd.Series(
                        [True, None, False, None, False, True, True, False],
                        dtype=pd.BooleanDtype(),
                    ),
                    "b": pd.Series(
                        [
                            "abc",
                            "a",
                            None,
                            "hello world",
                            "foo buzz",
                            "",
                            None,
                            "rapids ai",
                        ],
                        dtype=pd.StringDtype(),
                    ),
                    "c": pd.Series(
                        [0.1, None, 0.2, None, 3, 4, 1000, None],
                        dtype=pd.Float64Dtype(),
                    ),
                }
            ),
        ),
    ],
)
def test_dataframe_to_pandas_nullable_dtypes(df, expected_pdf):
    actual_pdf = df().to_pandas(nullable=True)

    assert_eq(actual_pdf, expected_pdf)


@pytest.mark.parametrize(
    "data",
    [
        [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}],
        [{"a": 1, "b": 2, "c": None}, {"a": None, "b": 5, "c": 6}],
        [{"a": 1, "b": 2}, {"a": 1, "b": 5, "c": 6}],
        [{"a": 1, "b": 2}, {"b": 5, "c": 6}],
        [{}, {"a": 1, "b": 5, "c": 6}],
        [{"a": 1, "b": 2, "c": 3}, {"a": 4.5, "b": 5.5, "c": 6.5}],
    ],
)
def test_dataframe_init_from_list_of_dicts(data):
    expect = pd.DataFrame(data)
    got = cudf.DataFrame(data)

    assert_eq(expect, got)


def test_dataframe_pipe():
    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()

    def add_int_col(df, column):
        df[column] = df._constructor_sliced([10, 20, 30, 40])
        return df

    def add_str_col(df, column):
        df[column] = df._constructor_sliced(["a", "b", "xyz", "ai"])
        return df

    expected = (
        pdf.pipe(add_int_col, "one")
        .pipe(add_int_col, column="two")
        .pipe(add_str_col, "three")
    )
    actual = (
        gdf.pipe(add_int_col, "one")
        .pipe(add_int_col, column="two")
        .pipe(add_str_col, "three")
    )

    assert_eq(expected, actual)

    expected = (
        pdf.pipe((add_str_col, "df"), column="one")
        .pipe(add_str_col, column="two")
        .pipe(add_int_col, "three")
    )
    actual = (
        gdf.pipe((add_str_col, "df"), column="one")
        .pipe(add_str_col, column="two")
        .pipe(add_int_col, "three")
    )

    assert_eq(expected, actual)


def test_dataframe_pipe_error():
    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()

    def custom_func(df, column):
        df[column] = df._constructor_sliced([10, 20, 30, 40])
        return df

    assert_exceptions_equal(
        lfunc=pdf.pipe,
        rfunc=gdf.pipe,
        lfunc_args_and_kwargs=([(custom_func, "columns")], {"columns": "d"}),
        rfunc_args_and_kwargs=([(custom_func, "columns")], {"columns": "d"}),
    )


@pytest.mark.parametrize(
    "op",
    ["count", "kurt", "kurtosis", "skew"],
)
def test_dataframe_axis1_unsupported_ops(op):
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [8, 9, 10]})

    with pytest.raises(
        NotImplementedError, match="Only axis=0 is currently supported."
    ):
        getattr(df, op)(axis=1)


def test_dataframe_from_pandas_duplicate_columns():
    pdf = pd.DataFrame(columns=["a", "b", "c", "a"])
    pdf["a"] = [1, 2, 3]

    with pytest.raises(
        ValueError, match="Duplicate column names are not allowed"
    ):
        cudf.from_pandas(pdf)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {"a": [1, 2, 3], "b": [10, 11, 20], "c": ["a", "bcd", "xyz"]}
        ),
        pd.DataFrame(),
    ],
)
@pytest.mark.parametrize(
    "columns",
    [
        None,
        ["a"],
        ["c", "a"],
        ["b", "a", "c"],
        [],
        pd.Index(["c", "a"]),
        cudf.Index(["c", "a"]),
        ["abc", "a"],
        ["column_not_exists1", "column_not_exists2"],
    ],
)
@pytest.mark.parametrize("index", [["abc", "def", "ghi"]])
def test_dataframe_constructor_columns(df, columns, index, request):
    def assert_local_eq(actual, df, expected, host_columns):
        check_index_type = not expected.empty
        if host_columns is not None and any(
            col not in df.columns for col in host_columns
        ):
            assert_eq(
                expected,
                actual,
                check_dtype=False,
                check_index_type=check_index_type,
            )
        else:
            assert_eq(
                expected,
                actual,
                check_index_type=check_index_type,
                check_column_type=False,
            )

    gdf = cudf.from_pandas(df)
    host_columns = (
        columns.to_pandas() if isinstance(columns, cudf.Index) else columns
    )

    expected = pd.DataFrame(df, columns=host_columns, index=index)
    actual = cudf.DataFrame(gdf, columns=columns, index=index)

    assert_local_eq(actual, df, expected, host_columns)


def test_dataframe_constructor_column_index_only():
    columns = ["a", "b", "c"]
    index = ["r1", "r2", "r3"]

    gdf = cudf.DataFrame(index=index, columns=columns)
    assert not id(gdf["a"]._column) == id(gdf["b"]._column) and not id(
        gdf["b"]._column
    ) == id(gdf["c"]._column)


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2.5, 3], "b": [3, 4.5, 5], "c": [2.0, 3.0, 4.0]},
        {"a": [1, 2.2, 3], "b": [2.0, 3.0, 4.0], "c": [5.0, 6.0, 4.0]},
    ],
)
@pytest.mark.parametrize(
    "aggs",
    [
        ["min", "sum", "max"],
        ("min", "sum", "max"),
        {"min", "sum", "max"},
        "sum",
        {"a": "sum", "b": "min", "c": "max"},
        {"a": ["sum"], "b": ["min"], "c": ["max"]},
        {"a": ("sum"), "b": ("min"), "c": ("max")},
        {"a": {"sum"}, "b": {"min"}, "c": {"max"}},
        {"a": ["sum", "min"], "b": ["sum", "max"], "c": ["min", "max"]},
        {"a": ("sum", "min"), "b": ("sum", "max"), "c": ("min", "max")},
        {"a": {"sum", "min"}, "b": {"sum", "max"}, "c": {"min", "max"}},
    ],
)
def test_agg_for_dataframes(data, aggs):
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    expect = pdf.agg(aggs).sort_index()
    got = gdf.agg(aggs).sort_index()

    assert_eq(expect, got, check_dtype=True)


@pytest_unmark_spilling
@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [3.0, 4.0, 5.0], "c": [True, True, False]},
        {"a": [1, 2, 3], "b": [True, True, False], "c": [False, True, False]},
    ],
)
@pytest.mark.parametrize(
    "aggs",
    [
        ["min", "sum", "max"],
        "sum",
        {"a": "sum", "b": "min", "c": "max"},
    ],
)
def test_agg_for_dataframes_error(data, aggs):
    gdf = cudf.DataFrame(data)

    with pytest.raises(TypeError):
        gdf.agg(aggs)


def test_agg_for_unsupported_function():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})

    with pytest.raises(NotImplementedError):
        gdf.agg({"a": np.sum, "b": np.min, "c": np.max})


def test_agg_for_dataframe_with_invalid_function():
    aggs = "asdf"
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})

    with pytest.raises(
        AttributeError,
        match=f"{aggs} is not a valid function for 'DataFrame' object",
    ):
        gdf.agg(aggs)


def test_agg_for_series_with_invalid_function():
    aggs = {"a": "asdf"}
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})

    with pytest.raises(
        AttributeError,
        match=f"{aggs['a']} is not a valid function for 'Series' object",
    ):
        gdf.agg(aggs)


@pytest.mark.parametrize(
    "aggs",
    [
        "sum",
        ["min", "sum", "max"],
        {"a": {"sum", "min"}, "b": {"sum", "max"}, "c": {"min", "max"}},
    ],
)
def test_agg_for_dataframe_with_string_columns(aggs):
    gdf = cudf.DataFrame(
        {"a": ["m", "n", "o"], "b": ["t", "u", "v"], "c": ["x", "y", "z"]},
        index=["a", "b", "c"],
    )

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "DataFrame.agg() is not supported for "
            "frames containing string columns"
        ),
    ):
        gdf.agg(aggs)


@pytest_unmark_spilling
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize(
    "left_keys,right_keys",
    [
        [("a", "b"), ("a", "b")],
        [("a", "b"), ("a", "c")],
        [("a", "b"), ("d", "e")],
    ],
)
@pytest.mark.parametrize(
    "data_left,data_right",
    [
        [([1, 2, 3], [3, 4, 5]), ([1, 2, 3], [3, 4, 5])],
        [
            ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]),
            ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]),
        ],
        [
            ([True, False, True], [False, False, False]),
            ([True, False, True], [False, False, False]),
        ],
        [
            ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]),
            ([np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]),
        ],
        [([1, 2, 3], [3, 4, 5]), ([1, 2, 4], [30, 40, 50])],
        [
            ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]),
            ([1.0, 2.0, 4.0], [30.0, 40.0, 50.0]),
        ],
        [([1, 2, 3], [3, 4, 5]), ([10, 20, 40], [30, 40, 50])],
        [
            ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]),
            ([10.0, 20.0, 40.0], [30.0, 40.0, 50.0]),
        ],
    ],
)
def test_update_for_dataframes(
    left_keys, right_keys, data_left, data_right, overwrite
):
    errors = "ignore"
    join = "left"
    left = dict(zip(left_keys, data_left, strict=True))
    right = dict(zip(right_keys, data_right, strict=True))
    pdf = pd.DataFrame(left)
    gdf = cudf.DataFrame(left, nan_as_null=False)

    other_pd = pd.DataFrame(right)
    other_gd = cudf.DataFrame(right, nan_as_null=False)

    pdf.update(other=other_pd, join=join, overwrite=overwrite, errors=errors)
    gdf.update(other=other_gd, join=join, overwrite=overwrite, errors=errors)

    assert_eq(pdf, gdf, check_dtype=False)


def test_update_for_right_join():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})
    other_gd = cudf.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2.0, 5.0]})

    with pytest.raises(
        NotImplementedError, match="Only left join is supported"
    ):
        gdf.update(other_gd, join="right")


def test_update_for_data_overlap():
    errors = "raise"
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})

    other_pd = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2.0, 5.0]})
    other_gd = cudf.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2.0, 5.0]})

    assert_exceptions_equal(
        lfunc=pdf.update,
        rfunc=gdf.update,
        lfunc_args_and_kwargs=([other_pd, errors], {}),
        rfunc_args_and_kwargs=([other_gd, errors], {}),
    )


@pytest.mark.parametrize(
    "data",
    [
        {"a": [[1], [2], [3]]},
        {
            "left-a": [0, 1, 2],
            "a": [[1], None, [3]],
            "right-a": ["abc", "def", "ghi"],
        },
        {
            "left-a": [[], None, None],
            "a": [[1], None, [3]],
            "right-a": ["abc", "def", "ghi"],
        },
    ],
)
def test_dataframe_roundtrip_arrow_list_dtype(data):
    gdf = cudf.DataFrame(data)
    table = gdf.to_arrow()
    expected = cudf.DataFrame.from_arrow(table)

    assert_eq(gdf, expected)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [{"one": 3, "two": 4, "three": 10}]},
        {
            "left-a": [0, 1, 2],
            "a": [{"x": 0.23, "y": 43}, None, {"x": 23.9, "y": 4.3}],
            "right-a": ["abc", "def", "ghi"],
        },
        {
            "left-a": [{"a": 1}, None, None],
            "a": [
                {"one": 324, "two": 23432, "three": 324},
                None,
                {"one": 3.24, "two": 1, "three": 324},
            ],
            "right-a": ["abc", "def", "ghi"],
        },
    ],
)
def test_dataframe_roundtrip_arrow_struct_dtype(data):
    gdf = cudf.DataFrame(data)
    table = gdf.to_arrow()
    expected = cudf.DataFrame.from_arrow(table)

    assert_eq(gdf, expected)


def test_dataframe_setitem_cupy_array():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.standard_normal(size=(10, 2)))
    gdf = cudf.from_pandas(pdf)

    gpu_array = cupy.array([True, False] * 5)
    pdf[gpu_array.get()] = 1.5
    gdf[gpu_array] = 1.5

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("level", ["x", 0])
def test_rename_for_level_MultiIndex_dataframe(level):
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    index = {0: 123, 1: 4, 2: 6}
    pdf = pd.DataFrame(
        data,
        index=pd.MultiIndex.from_tuples([(0, 1, 2), (1, 2, 3), (2, 3, 4)]),
    )
    pdf.index.names = ["x", "y", "z"]
    gdf = cudf.from_pandas(pdf)

    expect = pdf.rename(index=index, level=level)
    got = gdf.rename(index=index, level=level)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "columns",
    [{"a": "f", "b": "g"}, {1: 3, 2: 4}, lambda s: 2 * s],
)
@pytest.mark.parametrize("level", [0, 1])
def test_rename_for_level_MultiColumn_dataframe(columns, level):
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf.columns = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])

    pdf = gdf.to_pandas()

    expect = pdf.rename(columns=columns, level=level)
    got = gdf.rename(columns=columns, level=level)

    assert_eq(expect, got)


def test_rename_for_level_RangeIndex_dataframe():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    pdf = gdf.to_pandas()

    expect = pdf.rename(columns={"a": "f"}, index={0: 3, 1: 4}, level=0)
    got = gdf.rename(columns={"a": "f"}, index={0: 3, 1: 4}, level=0)

    assert_eq(expect, got)


def test_rename_for_level_is_None_MC():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    gdf.columns = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
    pdf = gdf.to_pandas()

    expect = pdf.rename(columns={"a": "f"}, level=None)
    got = gdf.rename(columns={"a": "f"}, level=None)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [
            [[1, 2, 3], 11, "a"],
            [None, 22, "e"],
            [[4], 33, "i"],
            [[], 44, "o"],
            [[5, 6], 55, "u"],
        ],  # nested
        [
            [1, 11, "a"],
            [2, 22, "e"],
            [3, 33, "i"],
            [4, 44, "o"],
            [5, 55, "u"],
        ],  # non-nested
    ],
)
@pytest.mark.parametrize(
    ("labels", "label_to_explode"),
    [
        (None, 0),
        (pd.Index(["a", "b", "c"]), "a"),
        (
            pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (1, "a")], names=["l0", "l1"]
            ),
            (0, "a"),
        ),
    ],
)
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize(
    "p_index",
    [
        None,
        ["ia", "ib", "ic", "id", "ie"],
        pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b")]
        ),
    ],
)
def test_explode(data, labels, ignore_index, p_index, label_to_explode):
    pdf = pd.DataFrame(data, index=p_index, columns=labels)
    gdf = cudf.from_pandas(pdf)

    expect = pdf.explode(label_to_explode, ignore_index)
    got = gdf.explode(label_to_explode, ignore_index)

    assert_eq(expect, got, check_dtype=False)


def test_explode_preserve_categorical():
    gdf = cudf.DataFrame(
        {
            "A": [[1, 2], None, [2, 3]],
            "B": cudf.Series([0, 1, 2], dtype="category"),
        }
    )
    result = gdf.explode("A")
    expected = cudf.DataFrame(
        {
            "A": [1, 2, None, 2, 3],
            "B": cudf.Series([0, 0, 1, 2, 2], dtype="category"),
        }
    )
    expected.index = cudf.Index([0, 0, 1, 2, 2])
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data,ascending,expected_data",
    [
        (
            {"a": [10, 0, 2], "b": [-10, 10, 1]},
            True,
            [1, 2, 0],
        ),
        (
            {"a": [10, 0, 2], "b": [-10, 10, 1]},
            False,
            [0, 2, 1],
        ),
    ],
)
def test_dataframe_argsort(data, ascending, expected_data):
    actual = cudf.DataFrame(data).argsort(ascending=ascending)
    expected = cupy.array(expected_data, dtype="int32")

    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        np.random.RandomState(seed=10).randint(-50, 50, (25, 30)),
        np.random.RandomState(seed=10).random_sample((4, 4)),
        np.array([1.123, 2.343, 5.890, 0.0]),
        [True, False, True, False, False],
        {"a": [1.123, 2.343, np.nan, np.nan], "b": [None, 3, 9.08, None]},
    ],
)
@pytest.mark.parametrize("periods", (-5, -1, 0, 1, 5))
def test_diff_numeric_dtypes(data, periods):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    actual = gdf.diff(periods=periods, axis=0)
    expected = pdf.diff(periods=periods, axis=0)

    assert_eq(
        expected,
        actual,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    ("precision", "scale"),
    [(5, 2), (8, 5)],
)
@pytest.mark.parametrize(
    "dtype",
    [cudf.Decimal32Dtype, cudf.Decimal64Dtype],
)
def test_diff_decimal_dtypes(precision, scale, dtype):
    gdf = cudf.DataFrame(
        np.random.default_rng(seed=42).uniform(10.5, 75.5, (10, 6)),
        dtype=dtype(precision=precision, scale=scale),
    )
    pdf = gdf.to_pandas()

    actual = gdf.diff()
    expected = pdf.diff()

    assert_eq(
        expected,
        actual,
        check_dtype=False,
    )


def test_diff_invalid_axis():
    gdf = cudf.DataFrame(np.array([1.123, 2.343, 5.890, 0.0]))
    with pytest.raises(NotImplementedError, match="Only axis=0 is supported."):
        gdf.diff(periods=1, axis=1)


@pytest.mark.parametrize(
    "data",
    [
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            "string_col": ["a", "b", "c", "d", "e"],
        },
        ["a", "b", "c", "d", "e"],
    ],
)
def test_diff_unsupported_dtypes(data):
    gdf = cudf.DataFrame(data)
    with pytest.raises(
        TypeError,
        match=r"unsupported operand type\(s\)",
    ):
        gdf.diff()


def test_diff_many_dtypes():
    pdf = pd.DataFrame(
        {
            "dates": pd.date_range("2020-01-01", "2020-01-06", freq="D"),
            "bools": [True, True, True, False, True, True],
            "floats": [1.0, 2.0, 3.5, np.nan, 5.0, -1.7],
            "ints": [1, 2, 3, 3, 4, 5],
            "nans_nulls": [np.nan, None, None, np.nan, np.nan, None],
        }
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.diff(), gdf.diff())
    assert_eq(pdf.diff(periods=2), gdf.diff(periods=2))


def test_dataframe_assign_cp_np_array():
    m, n = 5, 3
    cp_ndarray = cupy.random.randn(m, n)
    pdf = pd.DataFrame({f"f_{i}": range(m) for i in range(n)})
    gdf = cudf.DataFrame({f"f_{i}": range(m) for i in range(n)})
    pdf[[f"f_{i}" for i in range(n)]] = cupy.asnumpy(cp_ndarray)
    gdf[[f"f_{i}" for i in range(n)]] = cp_ndarray

    assert_eq(pdf, gdf)


def test_dataframe_nunique():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [1, 1, 0]})
    pdf = gdf.to_pandas()

    actual = gdf.nunique()
    expected = pdf.nunique()

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "columns",
    [
        pd.RangeIndex(2, name="foo"),
        pd.MultiIndex.from_arrays([[1, 2], [2, 3]], names=["foo", 1]),
        pd.Index([3, 5], dtype=np.int8, name="foo"),
    ],
)
def test_nunique_preserve_column_in_index(columns):
    df = cudf.DataFrame([[1, 2]], columns=columns)
    result = df.nunique().index.to_pandas()
    assert_eq(result, columns, exact=True)


def test_dataframe_nunique_index():
    gdf = cudf.DataFrame(
        {"key": [0, 1, 1, 0, 0, 1], "val": [1, 8, 3, 9, -3, 8]}
    )
    pdf = gdf.to_pandas()

    actual = gdf.index.nunique()
    expected = pdf.index.nunique()

    assert_eq(expected, actual)


def test_dataframe_rename_duplicate_column():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    with pytest.raises(
        ValueError, match="Duplicate column names are not allowed"
    ):
        gdf.rename(columns={"a": "b"}, inplace=True)


def test_dataframe_rename_columns_keep_type():
    gdf = cudf.DataFrame([[1, 2, 3]])
    gdf.columns = cudf.Index([4, 5, 6], dtype=np.int8)
    result = gdf.rename({4: 50}, axis="columns").columns
    expected = pd.Index([50, 5, 6], dtype=np.int8)
    assert_eq(result, expected)


@pytest_unmark_spilling
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "data",
    [
        np.random.RandomState(seed=10).randint(-50, 50, (10, 10)),
        np.random.RandomState(seed=10).random_sample((4, 4)),
        np.array([1.123, 2.343, 5.890, 0.0]),
        {"a": [1.123, 2.343, np.nan, np.nan], "b": [None, 3, 9.08, None]},
    ],
)
@pytest.mark.parametrize("periods", [-5, 0, 2])
@pytest.mark.parametrize(
    "fill_method", ["ffill", "bfill", "pad", "backfill", no_default]
)
def test_dataframe_pct_change(data, periods, fill_method):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    with expect_warning_if(fill_method is not no_default):
        actual = gdf.pct_change(periods=periods, fill_method=fill_method)
    with expect_warning_if(
        fill_method is not no_default or pdf.isna().any().any()
    ):
        expected = pdf.pct_change(periods=periods, fill_method=fill_method)

    assert_eq(expected, actual)


@pytest.mark.parametrize("numeric_only", [True, False])
def test_mean_timeseries(numeric_only):
    gdf = cudf.datasets.timeseries()
    if not numeric_only:
        gdf = gdf.select_dtypes(include="number")
    pdf = gdf.to_pandas()

    expected = pdf.mean(numeric_only=numeric_only)
    actual = gdf.mean(numeric_only=numeric_only)

    assert_eq(expected, actual)


@pytest.mark.parametrize("numeric_only", [True, False])
def test_std_different_dtypes(numeric_only):
    gdf = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "b", "c", "d", "e"],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    if not numeric_only:
        gdf = gdf.select_dtypes(include="number")
    pdf = gdf.to_pandas()

    expected = pdf.std(numeric_only=numeric_only)
    actual = gdf.std(numeric_only=numeric_only)

    assert_eq(expected, actual)


def test_empty_numeric_only():
    gdf = cudf.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "val1": ["v", "n", "k", "l", "m", "i", "y", "r", "w"],
            "val2": ["d", "d", "d", "e", "e", "e", "f", "f", "f"],
        }
    )
    pdf = gdf.to_pandas()
    expected = pdf.prod(numeric_only=True)
    actual = gdf.prod(numeric_only=True)
    assert_eq(expected, actual, check_dtype=True)


# Note that for now expressions do not automatically handle casting, so inputs
# need to be casted appropriately
@pytest.mark.parametrize(
    "expr, dtype",
    [
        ("a", int),
        ("+a", int),
        ("a + b", int),
        ("a == b", int),
        ("a / b", float),
        ("a * b", int),
        ("a > b", int),
        ("a >= b", int),
        ("a > b > c", int),
        ("a > b < c", int),
        ("a & b", int),
        ("a & b | c", int),
        ("sin(a)", float),
        ("exp(sin(abs(a)))", float),
        ("sqrt(floor(a))", float),
        ("ceil(arctanh(a))", float),
        ("(a + b) - (c * d)", int),
        ("~a", int),
        ("(a > b) and (c > d)", int),
        ("(a > b) or (c > d)", int),
        ("not (a > b)", int),
        ("a + 1", int),
        ("a + 1.0", float),
        ("-a + 1", int),
        ("+a + 1", int),
        ("e = a + 1", int),
        (
            """
            e = log(cos(a)) + 1.0
            f = abs(c) - exp(d)
            """,
            float,
        ),
        ("a_b_are_equal = (a == b)", int),
        ("a > b", str),
        ("a < '1'", str),
        ('a == "1"', str),
    ],
)
@pytest.mark.parametrize("nrows", [0, 10])
def test_dataframe_eval(nrows, expr, dtype):
    arr = np.ones(nrows)
    df_eval = cudf.DataFrame({"a": arr, "b": arr, "c": arr, "d": arr})
    df_eval = df_eval.astype(dtype)
    with _hide_ufunc_warnings(expr):
        expect = df_eval.to_pandas().eval(expr)
    got = df_eval.eval(expr)
    # In the specific case where the evaluated expression is a unary function
    # of a single column with no nesting, pandas will retain the name. This
    # level of compatibility is out of scope for now.
    assert_eq(expect, got, check_names=False)

    # Test inplace
    if re.search("[^=><]=[^=]", expr) is not None:
        pdf_eval = df_eval.to_pandas()
        with _hide_ufunc_warnings(expr):
            pdf_eval.eval(expr, inplace=True)
        df_eval.eval(expr, inplace=True)
        assert_eq(pdf_eval, df_eval)


@pytest.mark.parametrize(
    "expr",
    [
        """
        e = a + b
        a == b
        """,
        "a_b_are_equal = (a == b) = c",
    ],
)
def test_dataframe_eval_errors(expr):
    df = cudf.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError):
        df.eval(expr)


def test_dataframe_eval_misc():
    df = cudf.DataFrame({"a": [1, 2, 3, None, 5]})
    got = df.eval("isnull(a)")
    assert_eq(got, cudf.Series.isnull(df["a"]), check_names=False)

    df.eval("c = isnull(1)", inplace=True)
    assert_eq(df["c"], cudf.Series([False] * len(df), name="c"))


@pytest.mark.parametrize(
    "data",
    [
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        {
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", None, None, "Louise"],
        },
    ],
)
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("use_subset", [True, False])
def test_value_counts(
    data,
    sort,
    ascending,
    normalize,
    dropna,
    use_subset,
):
    subset = [next(iter(data.keys()))]
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    got = gdf.value_counts(
        subset=subset if (use_subset) else None,
        sort=sort,
        ascending=ascending,
        normalize=normalize,
        dropna=dropna,
    )
    expected = pdf.value_counts(
        subset=subset if (use_subset) else None,
        sort=sort,
        ascending=ascending,
        normalize=normalize,
        dropna=dropna,
    )

    if not dropna:
        # Convert the Pandas series to a cuDF one due to difference
        # in the handling of NaNs between the two (<NA> in cuDF and
        # NaN in Pandas) when dropna=False.
        assert_eq(got.sort_index(), cudf.from_pandas(expected).sort_index())
    else:
        assert_eq(got.sort_index(), expected.sort_index())


def test_value_counts_no_subset():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [1, 1, 0]})
    with pytest.raises(KeyError):
        gdf.value_counts(subset=["not_a_column_name"])


def test_multiindex_wildcard_selection_all():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2) for c1 in "abc" for c2 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(6)})
    df.columns = midx
    expect = df.to_pandas().loc[:, (slice(None), "b")]
    got = df.loc[:, (slice(None), "b")]
    assert_eq(expect, got)


@pytest_xfail(reason="Not yet properly supported.")
def test_multiindex_wildcard_selection_partial():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2) for c1 in "abc" for c2 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(6)})
    df.columns = midx
    expect = df.to_pandas().loc[:, (slice("a", "b"), "b")]
    got = df.loc[:, (slice("a", "b"), "b")]
    assert_eq(expect, got)


@pytest_xfail(reason="Not yet properly supported.")
def test_multiindex_wildcard_selection_three_level_all():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2, c3) for c1 in "abcd" for c2 in "abc" for c3 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(24)})
    df.columns = midx

    expect = df.to_pandas().loc[:, (slice("a", "c"), slice("a", "b"), "b")]
    got = df.loc[:, (slice(None), "b")]
    assert_eq(expect, got)


def test_dataframe_assign_scalar_to_empty_series():
    expected = pd.DataFrame({"a": []})
    actual = cudf.DataFrame({"a": []})
    expected.a = 0
    actual.a = 0
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        {0: [1, 2, 3], 2: [10, 11, 23]},
        {("a", "b"): [1, 2, 3], ("2",): [10, 11, 23]},
    ],
)
def test_non_string_column_name_to_arrow(data):
    df = cudf.DataFrame(data)

    expected = df.to_arrow()
    actual = pa.Table.from_pandas(df.to_pandas())

    assert expected.equals(actual)


def test_complex_types_from_arrow():
    expected = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3]),
            pa.array([10, 20, 30]),
            pa.array([{"a": 9}, {"b": 10}, {"c": 11}]),
            pa.array([[{"a": 1}], [{"b": 2}], [{"c": 3}]]),
            pa.array([10, 11, 12]).cast(pa.decimal128(21, 2)),
            pa.array([{"a": 9}, {"b": 10, "c": {"g": 43}}, {"c": {"a": 10}}]),
        ],
        names=["a", "b", "c", "d", "e", "f"],
    )

    df = cudf.DataFrame.from_arrow(expected)
    actual = df.to_arrow()

    assert expected.equals(actual)


@pytest.mark.parametrize(
    "data",
    [
        {
            "brand": ["Yum Yum", "Yum Yum", "Indomie", "Indomie", "Indomie"],
            "style": ["cup", "cup", "cup", "pack", "pack"],
            "rating": [4, 4, 3.5, 15, 5],
        },
        {
            "brand": ["Indomie", "Yum Yum", "Indomie", "Indomie", "Indomie"],
            "style": ["cup", "cup", "cup", "cup", "pack"],
            "rating": [4, 4, 3.5, 4, 5],
        },
    ],
)
@pytest.mark.parametrize(
    "subset", [None, ["brand"], ["rating"], ["style", "rating"]]
)
@pytest.mark.parametrize("keep", ["first", "last", False])
def test_dataframe_duplicated(data, subset, keep):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    expected = pdf.duplicated(subset=subset, keep=keep)
    actual = gdf.duplicated(subset=subset, keep=keep)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        {"col": [{"a": 1.1}, {"a": 2.1}, {"a": 10.0}, {"a": 11.2323}, None]},
        {"a": [[{"b": 567}], None] * 10},
        {"a": [decimal.Decimal(10), decimal.Decimal(20), None]},
    ],
)
def test_dataframe_values_complex_types(data):
    gdf = cudf.DataFrame(data)
    with pytest.raises(NotImplementedError):
        gdf.values


def test_dataframe_from_arrow_slice():
    table = pa.Table.from_pandas(
        pd.DataFrame.from_dict(
            {"a": ["aa", "bb", "cc"] * 3, "b": [1, 2, 3] * 3}
        )
    )
    table_slice = table.slice(3, 7)

    expected = table_slice.to_pandas()
    actual = cudf.DataFrame.from_arrow(table_slice)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data,index",
    [
        ({"a": [1, 2, 3], "b": ["x", "y", "z", "z"], "c": 4}, None),
        (
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            },
            [10, 11],
        ),
        (
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            },
            [10, 11],
        ),
        ([[10, 11], [12, 13]], ["a", "b", "c"]),
    ],
)
def test_dataframe_init_length_error(data, index):
    assert_exceptions_equal(
        lfunc=pd.DataFrame,
        rfunc=cudf.DataFrame,
        lfunc_args_and_kwargs=(
            [],
            {"data": data, "index": index},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"data": data, "index": index},
        ),
    )


def test_dataframe_binop_with_mixed_date_types():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        rng.random(size=(2, 2)),
        columns=pd.Index(["2000-01-03", "2000-01-04"], dtype="datetime64[ns]"),
    )
    ser = pd.Series(rng.random(size=3), index=[0, 1, 2])
    gdf = cudf.from_pandas(df)
    gser = cudf.from_pandas(ser)
    expected = df - ser
    got = gdf - gser
    assert_eq(expected, got)


def test_dataframe_binop_with_mixed_string_types():
    rng = np.random.default_rng(seed=0)
    df1 = pd.DataFrame(rng.random(size=(3, 3)), columns=pd.Index([0, 1, 2]))
    df2 = pd.DataFrame(
        rng.random(size=(6, 6)),
        columns=pd.Index([0, 1, 2, "VhDoHxRaqt", "X0NNHBIPfA", "5FbhPtS0D1"]),
    )
    gdf1 = cudf.from_pandas(df1)
    gdf2 = cudf.from_pandas(df2)

    expected = df2 + df1
    got = gdf2 + gdf1

    assert_eq(expected, got)


def test_dataframe_binop_and_where():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(rng.random(size=(2, 2)), columns=pd.Index([True, False]))
    gdf = cudf.from_pandas(df)

    expected = df > 1
    got = gdf > 1

    assert_eq(expected, got)

    expected = df[df > 1]
    got = gdf[gdf > 1]

    assert_eq(expected, got)


def test_dataframe_binop_with_datetime_index():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        rng.random(size=(2, 2)),
        columns=pd.Index(["2000-01-03", "2000-01-04"], dtype="datetime64[ns]"),
    )
    ser = pd.Series(
        rng.random(2),
        index=pd.Index(
            [
                "2000-01-04",
                "2000-01-03",
            ],
            dtype="datetime64[ns]",
        ),
    )
    gdf = cudf.from_pandas(df)
    gser = cudf.from_pandas(ser)
    expected = df - ser
    got = gdf - gser
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "dtype", ["datetime64[ns]", "timedelta64[ns]", "int64", "float32"]
)
def test_dataframe_mixed_dtype_error(dtype):
    pdf = pd.Series([1, 2, 3], dtype=dtype).to_frame().astype(object)
    with pytest.raises(TypeError):
        cudf.from_pandas(pdf)


@pytest.mark.parametrize(
    "index_data,name",
    [([10, 13], "a"), ([30, 40, 20], "b"), (["ef"], "c"), ([2, 3], "Z")],
)
def test_dataframe_reindex_with_index_names(index_data, name):
    gdf = cudf.DataFrame(
        {
            "a": [10, 12, 13],
            "b": [20, 30, 40],
            "c": cudf.Series(["ab", "cd", "ef"], dtype="category"),
        }
    )
    if name in gdf.columns:
        gdf = gdf.set_index(name)
    pdf = gdf.to_pandas()

    gidx = cudf.Index(index_data, name=name)
    actual = gdf.reindex(gidx)
    expected = pdf.reindex(gidx.to_pandas())

    assert_eq(actual, expected)

    actual = gdf.reindex(index_data)
    expected = pdf.reindex(index_data)

    assert_eq(actual, expected)


def test_dataframe_reduction_error():
    gdf = cudf.DataFrame(
        {
            "a": cudf.Series([1, 2, 3], dtype="float"),
            "d": cudf.Series([10, 20, 30], dtype="timedelta64[ns]"),
        }
    )

    with pytest.raises(TypeError):
        gdf.sum()


def test_dataframe_from_generator():
    pdf = pd.DataFrame((i for i in range(5)))
    gdf = cudf.DataFrame((i for i in range(5)))
    assert_eq(pdf, gdf)


@pytest.mark.parametrize("name", ["a", 0, None, np.nan, cudf.NA])
@pytest.mark.parametrize("contains", ["a", 0, None, np.nan, cudf.NA])
@pytest.mark.parametrize("other_names", [[], ["b", "c"], [1, 2]])
def test_dataframe_contains(name, contains, other_names):
    column_names = [name, *other_names]
    gdf = cudf.DataFrame({c: [0] for c in column_names})
    pdf = pd.DataFrame({c: [0] for c in column_names})

    assert_eq(gdf, pdf)

    if contains is cudf.NA or name is cudf.NA:
        expectation = contains is cudf.NA and name is cudf.NA
        assert (contains in pdf) == expectation
        assert (contains in gdf) == expectation
    elif gdf.columns.dtype.kind == "f":
        # In some cases, the columns are converted to an Index[float] based on
        # the other column names. That casts name values from None to np.nan.
        expectation = contains is np.nan and (name is None or name is np.nan)
        assert (contains in pdf) == expectation
        assert (contains in gdf) == expectation
    else:
        expectation = contains == name or (
            contains is np.nan and name is np.nan
        )
        assert (contains in pdf) == expectation
        assert (contains in gdf) == expectation

    assert (contains in pdf) == (contains in gdf)


def test_dataframe_series_dot():
    pser = pd.Series(range(2))
    gser = cudf.from_pandas(pser)

    expected = pser @ pser
    actual = gser @ gser

    assert_eq(expected, actual)

    pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list("ab"))
    gdf = cudf.from_pandas(pdf)

    expected = pser @ pdf
    actual = gser @ gdf

    assert_eq(expected, actual)

    assert_exceptions_equal(
        lfunc=pdf.dot,
        rfunc=gdf.dot,
        lfunc_args_and_kwargs=([pser], {}),
        rfunc_args_and_kwargs=([gser], {}),
    )

    assert_exceptions_equal(
        lfunc=pdf.dot,
        rfunc=gdf.dot,
        lfunc_args_and_kwargs=([pdf], {}),
        rfunc_args_and_kwargs=([gdf], {}),
    )

    pser = pd.Series(range(2), index=["a", "k"])
    gser = cudf.from_pandas(pser)

    pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list("ab"), index=["a", "k"])
    gdf = cudf.from_pandas(pdf)

    expected = pser @ pdf
    actual = gser @ gdf

    assert_eq(expected, actual)

    actual = gdf @ [2, 3]
    expected = pdf @ [2, 3]

    assert_eq(expected, actual)

    actual = pser @ [12, 13]
    expected = gser @ [12, 13]

    assert_eq(expected, actual)


def test_dataframe_reindex_keep_colname():
    gdf = cudf.DataFrame([1], columns=cudf.Index([1], name="foo"))
    result = gdf.reindex(index=[0, 1])
    expected = cudf.DataFrame(
        [1, None], columns=cudf.Index([1], name="foo"), index=[0, 1]
    )
    assert_eq(result, expected)


def test_dataframe_duplicate_index_reindex():
    gdf = cudf.DataFrame({"a": [0, 1, 2, 3]}, index=[0, 0, 1, 1])
    pdf = gdf.to_pandas()

    assert_exceptions_equal(
        gdf.reindex,
        pdf.reindex,
        lfunc_args_and_kwargs=([10, 11, 12, 13], {}),
        rfunc_args_and_kwargs=([10, 11, 12, 13], {}),
    )


@pytest.mark.parametrize(
    "expected",
    [
        pd.RangeIndex(1, 2, name="a"),
        pd.Index([1], dtype=np.int8, name="a"),
        pd.MultiIndex.from_arrays([[1]], names=["a"]),
    ],
)
@pytest.mark.parametrize("binop", [lambda df: df == df, lambda df: df - 1])
def test_dataframe_binop_preserves_column_metadata(expected, binop):
    df = cudf.DataFrame([1], columns=expected)
    result = binop(df).columns
    pd.testing.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
        [1],
        decimal.Decimal("1.0"),
    ],
)
def test_dataframe_to_pandas_arrow_type_nullable_raises(scalar):
    pa_array = pa.array([scalar, None])
    df = cudf.DataFrame({"a": pa_array})
    with pytest.raises(ValueError):
        df.to_pandas(nullable=True, arrow_type=True)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        {"1": 2},
        [1],
        decimal.Decimal("1.0"),
    ],
)
def test_dataframe_to_pandas_arrow_type(scalar):
    pa_array = pa.array([scalar, None])
    df = cudf.DataFrame({"a": pa_array})
    result = df.to_pandas(arrow_type=True)
    expected = pd.DataFrame({"a": pd.arrays.ArrowExtensionArray(pa_array)})
    pd.testing.assert_frame_equal(result, expected)


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


@pytest.mark.parametrize("names", [["abc", "def"], [1, 2], ["abc", 10]])
def test_dataframe_multiindex_column_names(names):
    arrays = [["A", "A", "B", "B"], ["one", "two", "one", "two"]]
    tuples = list(zip(*arrays, strict=True))
    index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

    pdf = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=index)
    df = cudf.from_pandas(pdf)

    assert_eq(df, pdf)
    assert_eq(df.columns.names, pdf.columns.names)
    pdf.columns.names = names
    df.columns.names = names
    assert_eq(df, pdf)
    assert_eq(df.columns.names, pdf.columns.names)


def test_roundtrip_dataframe_plc_table(na_data):
    pdf = na_data
    expect = cudf.DataFrame.from_pandas(pdf)
    actual = cudf.DataFrame.from_pylibcudf(*expect.to_pylibcudf())
    assert_eq(expect, actual)


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


def test_rename_reset_label_dtype():
    data = {1: [2]}
    col_mapping = {1: "a"}
    result = cudf.DataFrame(data).rename(columns=col_mapping)
    expected = pd.DataFrame(data).rename(columns=col_mapping)
    assert_eq(result, expected)


def test_insert_reset_label_dtype():
    result = cudf.DataFrame({1: [2]})
    expected = pd.DataFrame({1: [2]})
    result.insert(1, "a", [2])
    expected.insert(1, "a", [2])
    assert_eq(result, expected)


def test_setitem_reset_label_dtype():
    result = cudf.DataFrame({1: [2]})
    expected = pd.DataFrame({1: [2]})
    result["a"] = [2]
    expected["a"] = [2]
    assert_eq(result, expected)


def test_dataframe_midx_cols_getitem():
    df = cudf.DataFrame(
        {
            "a": ["a", "b", "c"],
            "b": ["b", "", ""],
            "c": [10, 11, 12],
        }
    )
    df.columns = df.set_index(["a", "b"]).index
    pdf = df.to_pandas()

    expected = df["c"]
    actual = pdf["c"]
    assert_eq(expected, actual)
    df = cudf.DataFrame(
        [[1, 0], [0, 1]],
        columns=[
            ["foo", "foo"],
            ["location", "location"],
            ["x", "y"],
        ],
    )
    df = df.assign(bools=cudf.Series([True, False], dtype="bool"))
    assert_eq(df["bools"], df.to_pandas()["bools"])
