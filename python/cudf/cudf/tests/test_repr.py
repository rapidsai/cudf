# Copyright (c) 2019-2020, NVIDIA CORPORATION.
import textwrap

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

import cudf
from cudf.core._compat import PANDAS_GE_110
from cudf.tests import utils
from cudf.utils.dtypes import cudf_dtypes_to_pandas_dtypes

repr_categories = utils.NUMERIC_TYPES + ["str", "category", "datetime64[ns]"]


@pytest.mark.parametrize("dtype", repr_categories)
@pytest.mark.parametrize("nrows", [0, 5, 10])
def test_null_series(nrows, dtype):
    size = 5
    mask = utils.random_bitmask(size)
    data = cudf.Series(np.random.randint(1, 9, size))
    column = data.set_mask(mask)
    sr = cudf.Series(column).astype(dtype)
    if dtype != "category" and np.dtype(dtype).kind in {"u", "i"}:
        ps = pd.Series(
            sr._column.data_array_view.copy_to_host(),
            dtype=cudf_dtypes_to_pandas_dtypes.get(
                np.dtype(dtype), np.dtype(dtype)
            ),
        )
        ps[sr.isnull().to_pandas()] = pd.NA
    else:
        ps = sr.to_pandas()

    pd.options.display.max_rows = int(nrows)
    psrepr = ps.__repr__()
    psrepr = psrepr.replace("NaN", "<NA>")
    psrepr = psrepr.replace("NaT", "<NA>")
    psrepr = psrepr.replace("None", "<NA>")
    if (
        dtype.startswith("int")
        or dtype.startswith("uint")
        or dtype.startswith("long")
    ):
        psrepr = psrepr.replace(
            str(sr._column.default_na_value()) + "\n", "<NA>\n"
        )
    if "UInt" in psrepr:
        psrepr = psrepr.replace("UInt", "uint")
    elif "Int" in psrepr:
        psrepr = psrepr.replace("Int", "int")
    assert psrepr.split() == sr.__repr__().split()
    pd.reset_option("display.max_rows")


dtype_categories = [
    "float32",
    "float64",
    "datetime64[ns]",
    "str",
    "category",
]


@pytest.mark.parametrize("ncols", [1, 2, 3, 4, 5, 10])
def test_null_dataframe(ncols):
    size = 20
    gdf = cudf.DataFrame()
    for idx, dtype in enumerate(dtype_categories):
        mask = utils.random_bitmask(size)
        data = cudf.Series(np.random.randint(0, 128, size))
        column = data.set_mask(mask)
        sr = cudf.Series(column).astype(dtype)
        gdf[dtype] = sr
    pdf = gdf.to_pandas()
    pd.options.display.max_columns = int(ncols)
    pdfrepr = pdf.__repr__()
    pdfrepr = pdfrepr.replace("NaN", "<NA>")
    pdfrepr = pdfrepr.replace("NaT", "<NA>")
    pdfrepr = pdfrepr.replace("None", "<NA>")
    assert pdfrepr.split() == gdf.__repr__().split()
    pd.reset_option("display.max_columns")


@pytest.mark.parametrize("dtype", repr_categories)
@pytest.mark.parametrize("nrows", [0, 1, 2, 9, 10, 11, 19, 20, 21])
def test_full_series(nrows, dtype):
    size = 20
    ps = pd.Series(np.random.randint(0, 100, size)).astype(dtype)
    sr = cudf.from_pandas(ps)
    pd.options.display.max_rows = int(nrows)
    assert ps.__repr__() == sr.__repr__()
    pd.reset_option("display.max_rows")


@pytest.mark.parametrize("dtype", repr_categories)
@pytest.mark.parametrize("nrows", [0, 1, 2, 9, 20 / 2, 11, 20 - 1, 20, 20 + 1])
@pytest.mark.parametrize("ncols", [0, 1, 2, 9, 20 / 2, 11, 20 - 1, 20, 20 + 1])
def test_full_dataframe_20(dtype, nrows, ncols):
    size = 20
    pdf = pd.DataFrame(
        {idx: np.random.randint(0, 100, size) for idx in range(size)}
    ).astype(dtype)
    gdf = cudf.from_pandas(pdf)

    ncols, nrows = gdf._repr_pandas025_formatting(ncols, nrows, dtype)
    pd.options.display.max_rows = int(nrows)
    pd.options.display.max_columns = int(ncols)

    assert pdf.__repr__() == gdf.__repr__()
    assert pdf._repr_html_() == gdf._repr_html_()
    assert pdf._repr_latex_() == gdf._repr_latex_()
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")


@pytest.mark.parametrize("dtype", repr_categories)
@pytest.mark.parametrize("nrows", [9, 21 / 2, 11, 21 - 1])
@pytest.mark.parametrize("ncols", [9, 21 / 2, 11, 21 - 1])
def test_full_dataframe_21(dtype, nrows, ncols):
    size = 21
    pdf = pd.DataFrame(
        {idx: np.random.randint(0, 100, size) for idx in range(size)}
    ).astype(dtype)
    gdf = cudf.from_pandas(pdf)

    pd.options.display.max_rows = int(nrows)
    pd.options.display.max_columns = int(ncols)
    assert pdf.__repr__() == gdf.__repr__()
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")


@given(
    st.lists(
        st.integers(-9223372036854775808, 9223372036854775807),
        min_size=1,
        max_size=10000,
    )
)
@settings(deadline=None)
def test_integer_dataframe(x):
    gdf = cudf.DataFrame({"x": x})
    pdf = gdf.to_pandas()
    pd.options.display.max_columns = 1
    assert gdf.__repr__() == pdf.__repr__()
    assert gdf.T.__repr__() == pdf.T.__repr__()
    pd.reset_option("display.max_columns")


@given(
    st.lists(
        st.integers(-9223372036854775808, 9223372036854775807), max_size=10000
    )
)
@settings(deadline=None)
def test_integer_series(x):
    sr = cudf.Series(x)
    ps = pd.Series(x)

    assert sr.__repr__() == ps.__repr__()


@given(st.lists(st.floats()))
@settings(deadline=None)
def test_float_dataframe(x):
    gdf = cudf.DataFrame({"x": cudf.Series(x, nan_as_null=False)})
    pdf = gdf.to_pandas()
    assert gdf.__repr__() == pdf.__repr__()


@given(st.lists(st.floats()))
@settings(deadline=None)
def test_float_series(x):
    sr = cudf.Series(x, nan_as_null=False)
    ps = pd.Series(x)
    assert sr.__repr__() == ps.__repr__()


@pytest.fixture
def mixed_pdf():
    pdf = pd.DataFrame()
    pdf["Integer"] = np.array([2345, 11987, 9027, 9027])
    pdf["Date"] = np.array(
        ["18/04/1995", "14/07/1994", "07/06/2006", "16/09/2005"]
    )
    pdf["Float"] = np.array([9.001, 8.343, 6, 2.781])
    pdf["Integer2"] = np.array([2345, 106, 2088, 789277])
    pdf["Category"] = np.array(["M", "F", "F", "F"])
    pdf["String"] = np.array(["Alpha", "Beta", "Gamma", "Delta"])
    pdf["Boolean"] = np.array([True, False, True, False])
    return pdf


@pytest.fixture
def mixed_gdf(mixed_pdf):
    return cudf.from_pandas(mixed_pdf)


def test_mixed_dataframe(mixed_pdf, mixed_gdf):
    assert mixed_gdf.__repr__() == mixed_pdf.__repr__()


def test_mixed_series(mixed_pdf, mixed_gdf):
    for col in mixed_gdf.columns:
        assert mixed_gdf[col].__repr__() == mixed_pdf[col].__repr__()


def test_MI():
    gdf = cudf.DataFrame(
        {
            "a": np.random.randint(0, 4, 10),
            "b": np.random.randint(0, 4, 10),
            "c": np.random.randint(0, 4, 10),
        }
    )
    levels = [["a", "b", "c", "d"], ["w", "x", "y", "z"], ["m", "n"]]
    codes = cudf.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 2, 2, 3, 3],
            "b": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
            "c": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 0
    gdf = gdf.set_index(cudf.MultiIndex(levels=levels, codes=codes))
    pdf = gdf.to_pandas()
    gdfT = gdf.T
    pdfT = pdf.T
    assert gdf.__repr__() == pdf.__repr__()
    assert gdf.index.__repr__() == pdf.index.__repr__()
    assert gdfT.__repr__() == pdfT.__repr__()
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")


@pytest.mark.parametrize("nrows", [0, 1, 3, 5, 10])
@pytest.mark.parametrize("ncols", [0, 1, 2, 3])
def test_groupby_MI(nrows, ncols):
    gdf = cudf.DataFrame(
        {"a": np.arange(10), "b": np.arange(10), "c": np.arange(10)}
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["a", "b"]).count()
    pdg = pdf.groupby(["a", "b"]).count()
    pd.options.display.max_rows = nrows
    pd.options.display.max_columns = ncols
    assert gdg.__repr__() == pdg.__repr__()
    assert gdg.index.__repr__() == pdg.index.__repr__()
    assert gdg.T.__repr__() == pdg.T.__repr__()
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")


@pytest.mark.parametrize("dtype", utils.NUMERIC_TYPES)
@pytest.mark.parametrize("length", [0, 1, 10, 100, 1000])
def test_generic_index(length, dtype):
    psr = pd.Series(
        range(length),
        index=np.random.randint(0, high=100, size=length).astype(dtype),
    )
    gsr = cudf.Series.from_pandas(psr)

    assert psr.index.__repr__() == gsr.index.__repr__()


@pytest.mark.parametrize(
    "gdf",
    [
        cudf.DataFrame({"a": range(10000)}),
        cudf.DataFrame({"a": range(10000), "b": range(10000)}),
        cudf.DataFrame({"a": range(20), "b": range(20)}),
        cudf.DataFrame(
            {
                "a": range(20),
                "b": range(20),
                "c": ["abc", "def", "xyz", "def", "pqr"] * 4,
            }
        ),
        cudf.DataFrame(index=[1, 2, 3]),
        cudf.DataFrame(index=range(10000)),
        cudf.DataFrame(columns=["a", "b", "c", "d"]),
        cudf.DataFrame(columns=["a"], index=range(10000)),
        cudf.DataFrame(columns=["a", "col2", "...col n"], index=range(10000)),
        cudf.DataFrame(index=cudf.Series(range(10000)).astype("str")),
        cudf.DataFrame(
            columns=["a", "b", "c", "d"],
            index=cudf.Series(range(10000)).astype("str"),
        ),
    ],
)
@pytest.mark.parametrize(
    "slice",
    [
        slice(2500, 5000),
        slice(2500, 2501),
        slice(5000),
        slice(1, 10),
        slice(10, 20),
        slice(15, 2400),
    ],
)
@pytest.mark.parametrize("max_seq_items", [1, 10, 60, 10000, None])
@pytest.mark.parametrize("max_rows", [1, 10, 60, 10000, None])
def test_dataframe_sliced(gdf, slice, max_seq_items, max_rows):
    pd.options.display.max_seq_items = max_seq_items
    pd.options.display.max_rows = max_rows
    pdf = gdf.to_pandas()

    sliced_gdf = gdf[slice]
    sliced_pdf = pdf[slice]

    expected_repr = sliced_pdf.__repr__().replace("None", "<NA>")
    actual_repr = sliced_gdf.__repr__()

    assert expected_repr == actual_repr
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_seq_items")


@pytest.mark.parametrize(
    "index,expected_repr",
    [
        (
            cudf.Index([1, 2, 3, None]),
            "Int64Index([1, 2, 3, <NA>], dtype='int64')",
        ),
        (
            cudf.Index([None, 2.2, 3.324342, None]),
            "Float64Index([<NA>, 2.2, 3.324342, <NA>], dtype='float64')",
        ),
        (
            cudf.Index([None, None, None], name="hello"),
            "Float64Index([<NA>, <NA>, <NA>], dtype='float64', name='hello')",
        ),
        (
            cudf.Index([None], name="hello"),
            "Float64Index([<NA>], dtype='float64', name='hello')",
        ),
        (
            cudf.Index([None], dtype="int8", name="hello"),
            "Int8Index([<NA>], dtype='int8', name='hello')",
        ),
        (
            cudf.Index([None] * 50, dtype="object"),
            "StringIndex([None None None None None None None None "
            "None None None None None None\n None None None None None None "
            "None None None None None None None None\n None None None None "
            "None None None None None None None None None None\n None None "
            "None None None None None None], dtype='object')",
        ),
        (
            cudf.Index([None] * 20, dtype="uint32"),
            "UInt32Index([<NA>, <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, "
            "<NA>,\n       <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, "
            "<NA>,\n       <NA>, <NA>],\n      dtype='uint32')",
        ),
        (
            cudf.Index(
                [None, 111, 22, 33, None, 23, 34, 2343, None], dtype="int16"
            ),
            "Int16Index([<NA>, 111, 22, 33, <NA>, 23, 34, 2343, <NA>], "
            "dtype='int16')",
        ),
        (
            cudf.Index([1, 2, 3, None], dtype="category"),
            "CategoricalIndex([1, 2, 3, <NA>], categories=[1, 2, 3], "
            "ordered=False, dtype='category')",
        ),
        (
            cudf.Index([None, None], dtype="category"),
            "CategoricalIndex([<NA>, <NA>], categories=[], ordered=False, "
            "dtype='category')",
        ),
        (
            cudf.Index(np.array([10, 20, 30, None], dtype="datetime64[ns]")),
            "DatetimeIndex([1970-01-01 00:00:00.000000010, "
            "1970-01-01 00:00:00.000000020,"
            "\n       1970-01-01 00:00:00.000000030, <NA>],\n      "
            "dtype='datetime64[ns]')",
        ),
        (
            cudf.Index(np.array([10, 20, 30, None], dtype="datetime64[s]")),
            "DatetimeIndex([1970-01-01 00:00:10, "
            "1970-01-01 00:00:20, 1970-01-01 00:00:30,\n"
            "       <NA>],\n      dtype='datetime64[s]')",
        ),
        (
            cudf.Index(np.array([10, 20, 30, None], dtype="datetime64[us]")),
            "DatetimeIndex([1970-01-01 00:00:00.000010, "
            "1970-01-01 00:00:00.000020,\n       "
            "1970-01-01 00:00:00.000030, <NA>],\n      "
            "dtype='datetime64[us]')",
        ),
        (
            cudf.Index(np.array([10, 20, 30, None], dtype="datetime64[ms]")),
            "DatetimeIndex([1970-01-01 00:00:00.010, "
            "1970-01-01 00:00:00.020,\n       "
            "1970-01-01 00:00:00.030, <NA>],\n      "
            "dtype='datetime64[ms]')",
        ),
        (
            cudf.Index(np.array([None] * 10, dtype="datetime64[ms]")),
            "DatetimeIndex([<NA>, <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, <NA>, "
            "<NA>,\n       <NA>],\n      dtype='datetime64[ms]')",
        ),
    ],
)
def test_generic_index_null(index, expected_repr):

    actual_repr = index.__repr__()

    assert expected_repr == actual_repr


@pytest.mark.parametrize(
    "df,pandas_special_case",
    [
        (pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, None]), False),
        (
            pd.DataFrame(
                {
                    "a": [1, None, 3],
                    "string_col": ["hello", "world", "rapids"],
                },
                index=[None, "a", "b"],
            ),
            True,
        ),
        (pd.DataFrame([], index=[None, "a", "b"]), False),
        (pd.DataFrame({"aa": [None, None]}, index=[None, None]), False),
        (pd.DataFrame({"aa": [1, 2, 3]}, index=[None, None, None]), False),
        (
            pd.DataFrame(
                {"aa": [None, 2, 3]},
                index=np.array([1, None, None], dtype="datetime64[ns]"),
            ),
            False,
        ),
        (
            pd.DataFrame(
                {"aa": [None, 2, 3]},
                index=np.array([100, None, None], dtype="datetime64[ns]"),
            ),
            False,
        ),
        (
            pd.DataFrame(
                {"aa": [None, None, None]},
                index=np.array([None, None, None], dtype="datetime64[ns]"),
            ),
            False,
        ),
        (
            pd.DataFrame(
                {"aa": [1, None, 3]},
                index=np.array([10, 15, None], dtype="datetime64[ns]"),
            ),
            False,
        ),
        (
            pd.DataFrame(
                {"a": [1, 2, None], "v": [10, None, 22], "p": [100, 200, 300]}
            ).set_index(["a", "v"]),
            False,
        ),
        (
            pd.DataFrame(
                {
                    "a": [1, 2, None],
                    "v": ["n", "c", "a"],
                    "p": [None, None, None],
                }
            ).set_index(["a", "v"]),
            False,
        ),
        (
            pd.DataFrame(
                {
                    "a": np.array([1, None, None], dtype="datetime64[ns]"),
                    "v": ["n", "c", "a"],
                    "p": [None, None, None],
                }
            ).set_index(["a", "v"]),
            False,
        ),
    ],
)
def test_dataframe_null_index_repr(df, pandas_special_case):
    pdf = df
    gdf = cudf.from_pandas(pdf)

    expected_repr = (
        pdf.__repr__()
        .replace("NaN", "<NA>")
        .replace("NaT", "<NA>")
        .replace("None", "<NA>")
    )
    actual_repr = gdf.__repr__()

    if pandas_special_case:
        # Pandas inconsistently print StringIndex null values
        # as `None` at some places and `NaN` at few other places
        # Whereas cudf is consistent with strings `null` values
        # to be printed as `None` everywhere.
        actual_repr = gdf.__repr__().replace("None", "<NA>")

    assert expected_repr.split() == actual_repr.split()


@pytest.mark.parametrize(
    "sr,pandas_special_case",
    [
        (pd.Series([1, 2, 3], index=[10, 20, None]), False),
        (pd.Series([1, None, 3], name="a", index=[None, "a", "b"]), True),
        (pd.Series(None, index=[None, "a", "b"], dtype="float"), True),
        (pd.Series([None, None], name="aa", index=[None, None]), False),
        (pd.Series([1, 2, 3], index=[None, None, None]), False),
        (
            pd.Series(
                [None, 2, 3],
                index=np.array([1, None, None], dtype="datetime64[ns]"),
            ),
            False,
        ),
        (
            pd.Series(
                [None, None, None],
                index=np.array([None, None, None], dtype="datetime64[ns]"),
            ),
            False,
        ),
        (
            pd.Series(
                [1, None, 3],
                index=np.array([10, 15, None], dtype="datetime64[ns]"),
            ),
            False,
        ),
        (
            pd.DataFrame(
                {"a": [1, 2, None], "v": [10, None, 22], "p": [100, 200, 300]}
            ).set_index(["a", "v"])["p"],
            False,
        ),
        (
            pd.DataFrame(
                {
                    "a": [1, 2, None],
                    "v": ["n", "c", "a"],
                    "p": [None, None, None],
                }
            ).set_index(["a", "v"])["p"],
            False,
        ),
        (
            pd.DataFrame(
                {
                    "a": np.array([1, None, None], dtype="datetime64[ns]"),
                    "v": ["n", "c", "a"],
                    "p": [None, None, None],
                }
            ).set_index(["a", "v"])["p"],
            False,
        ),
    ],
)
def test_series_null_index_repr(sr, pandas_special_case):
    psr = sr
    gsr = cudf.from_pandas(psr)

    expected_repr = (
        psr.__repr__()
        .replace("NaN", "<NA>")
        .replace("NaT", "<NA>")
        .replace("None", "<NA>")
    )
    actual_repr = gsr.__repr__()

    if pandas_special_case:
        # Pandas inconsistently print StringIndex null values
        # as `None` at some places and `NaN` at few other places
        # Whereas cudf is consistent with strings `null` values
        # to be printed as `None` everywhere.
        actual_repr = gsr.__repr__().replace("None", "<NA>")
    assert expected_repr.split() == actual_repr.split()


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [
            136457654,
            134736784,
            245345345,
            223432411,
            2343241,
            3634548734,
            23234,
        ],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize("dtype", ["timedelta64[s]", "timedelta64[us]"])
def test_timedelta_series_s_us_repr(data, dtype):
    if not PANDAS_GE_110:
        pytest.xfail(reason="pandas >= 1.1 requried")
    sr = cudf.Series(data, dtype=dtype)
    psr = sr.to_pandas()

    expected = (
        psr.__repr__().replace("timedelta64[ns]", dtype).replace("NaT", "<NA>")
    )
    actual = sr.__repr__()

    assert expected.split() == actual.split()


@pytest.mark.parametrize(
    "ser, expected_repr",
    [
        (
            cudf.Series([], dtype="timedelta64[ns]"),
            textwrap.dedent(
                """
            Series([], dtype: timedelta64[ns])
            """
            ),
        ),
        (
            cudf.Series([], dtype="timedelta64[ms]"),
            textwrap.dedent(
                """
            Series([], dtype: timedelta64[ms])
            """
            ),
        ),
        (
            cudf.Series([1000000, 200000, 3000000], dtype="timedelta64[ns]"),
            textwrap.dedent(
                """
            0    0 days 00:00:00.001000000
            1    0 days 00:00:00.000200000
            2    0 days 00:00:00.003000000
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            cudf.Series([1000000, 200000, 3000000], dtype="timedelta64[ms]"),
            textwrap.dedent(
                """
            0    0 days 00:16:40
            1    0 days 00:03:20
            2    0 days 00:50:00
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            cudf.Series([1000000, 200000, None], dtype="timedelta64[ns]"),
            textwrap.dedent(
                """
            0    0 days 00:00:00.001000000
            1    0 days 00:00:00.000200000
            2                         <NA>
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            cudf.Series([1000000, 200000, None], dtype="timedelta64[ms]"),
            textwrap.dedent(
                """
            0    0 days 00:16:40
            1    0 days 00:03:20
            2               <NA>
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            cudf.Series(
                [None, None, None, None, None], dtype="timedelta64[ns]"
            ),
            textwrap.dedent(
                """
            0    <NA>
            1    <NA>
            2    <NA>
            3    <NA>
            4    <NA>
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            cudf.Series(
                [None, None, None, None, None], dtype="timedelta64[ms]"
            ),
            textwrap.dedent(
                """
            0    <NA>
            1    <NA>
            2    <NA>
            3    <NA>
            4    <NA>
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            cudf.Series(
                [12, 12, 22, 343, 4353534, 435342], dtype="timedelta64[ns]"
            ),
            textwrap.dedent(
                """
            0    0 days 00:00:00.000000012
            1    0 days 00:00:00.000000012
            2    0 days 00:00:00.000000022
            3    0 days 00:00:00.000000343
            4    0 days 00:00:00.004353534
            5    0 days 00:00:00.000435342
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            cudf.Series(
                [12, 12, 22, 343, 4353534, 435342], dtype="timedelta64[ms]"
            ),
            textwrap.dedent(
                """
            0    0 days 00:00:00.012
            1    0 days 00:00:00.012
            2    0 days 00:00:00.022
            3    0 days 00:00:00.343
            4    0 days 01:12:33.534
            5    0 days 00:07:15.342
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            cudf.Series(
                [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
                dtype="timedelta64[ns]",
            ),
            textwrap.dedent(
                """
            0    0 days 00:00:00.000000001
            1    0 days 00:00:00.000001132
            2    0 days 00:00:00.023223231
            3    0 days 00:00:00.000000233
            4              0 days 00:00:00
            5    0 days 00:00:00.000000332
            6    0 days 00:00:00.000000323
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            cudf.Series(
                [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
                dtype="timedelta64[ms]",
            ),
            textwrap.dedent(
                """
            0    0 days 00:00:00.001
            1    0 days 00:00:01.132
            2    0 days 06:27:03.231
            3    0 days 00:00:00.233
            4        0 days 00:00:00
            5    0 days 00:00:00.332
            6    0 days 00:00:00.323
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            cudf.Series(
                [
                    13645765432432,
                    134736784,
                    245345345,
                    223432411,
                    999992343241,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[ms]",
            ),
            textwrap.dedent(
                """
            0    157937 days 02:23:52.432
            1         1 days 13:25:36.784
            2         2 days 20:09:05.345
            3         2 days 14:03:52.411
            4     11573 days 23:39:03.241
            5        42 days 01:35:48.734
            6         0 days 00:00:23.234
            dtype: timedelta64[ms]
            """
            ),
        ),
        (
            cudf.Series(
                [
                    13645765432432,
                    134736784,
                    245345345,
                    223432411,
                    999992343241,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[ns]",
            ),
            textwrap.dedent(
                """
            0    0 days 03:47:25.765432432
            1    0 days 00:00:00.134736784
            2    0 days 00:00:00.245345345
            3    0 days 00:00:00.223432411
            4    0 days 00:16:39.992343241
            5    0 days 00:00:03.634548734
            6    0 days 00:00:00.000023234
            dtype: timedelta64[ns]
            """
            ),
        ),
        (
            cudf.Series(
                [
                    13645765432432,
                    134736784,
                    245345345,
                    223432411,
                    999992343241,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[ms]",
                name="abc",
            ),
            textwrap.dedent(
                """
            0    157937 days 02:23:52.432
            1         1 days 13:25:36.784
            2         2 days 20:09:05.345
            3         2 days 14:03:52.411
            4     11573 days 23:39:03.241
            5        42 days 01:35:48.734
            6         0 days 00:00:23.234
            Name: abc, dtype: timedelta64[ms]
            """
            ),
        ),
        (
            cudf.Series(
                [
                    13645765432432,
                    134736784,
                    245345345,
                    223432411,
                    999992343241,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[ns]",
                index=["a", "b", "z", "x", "y", "l", "m"],
                name="hello",
            ),
            textwrap.dedent(
                """
            a    0 days 03:47:25.765432432
            b    0 days 00:00:00.134736784
            z    0 days 00:00:00.245345345
            x    0 days 00:00:00.223432411
            y    0 days 00:16:39.992343241
            l    0 days 00:00:03.634548734
            m    0 days 00:00:00.000023234
            Name: hello, dtype: timedelta64[ns]
            """
            ),
        ),
    ],
)
def test_timedelta_series_ns_ms_repr(ser, expected_repr):
    expected = expected_repr
    actual = ser.__repr__()

    assert expected.split() == actual.split()


@pytest.mark.parametrize(
    "df,expected_repr",
    [
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [1000000, 200000, 3000000], dtype="timedelta64[s]"
                    )
                }
            ),
            textwrap.dedent(
                """
                                  a
                0  11 days 13:46:40
                1   2 days 07:33:20
                2  34 days 17:20:00
                """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [
                            136457654,
                            None,
                            245345345,
                            223432411,
                            None,
                            3634548734,
                            23234,
                        ],
                        dtype="timedelta64[s]",
                    ),
                    "b": [10, 11, 22, 33, 44, 55, 66],
                }
            ),
            textwrap.dedent(
                """
                                     a   b
                0   1579 days 08:54:14  10
                1                 <NA>  11
                2   2839 days 15:29:05  22
                3   2586 days 00:33:31  33
                4                 <NA>  44
                5  42066 days 12:52:14  55
                6      0 days 06:27:14  66
                """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [
                            136457654,
                            None,
                            245345345,
                            223432411,
                            None,
                            3634548734,
                            23234,
                        ],
                        dtype="timedelta64[s]",
                        index=["a", "b", "c", "d", "e", "f", "g"],
                    )
                }
            ),
            textwrap.dedent(
                """
                                     a
                a   1579 days 08:54:14
                b                 <NA>
                c   2839 days 15:29:05
                d   2586 days 00:33:31
                e                 <NA>
                f  42066 days 12:52:14
                g      0 days 06:27:14
                """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series(
                        [1, 2, 3, 4, 5, 6, 7],
                        index=cudf.Index(
                            [
                                136457654,
                                None,
                                245345345,
                                223432411,
                                None,
                                3634548734,
                                23234,
                            ],
                            dtype="timedelta64[ms]",
                        ),
                    )
                }
            ),
            textwrap.dedent(
                """
                                      a
                1 days 13:54:17.654   1
                <NA>                  2
                2 days 20:09:05.345   3
                2 days 14:03:52.411   4
                <NA>                  5
                42 days 01:35:48.734  6
                0 days 00:00:23.234   7
                """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series(
                        ["a", "f", "q", "e", "w", "e", "t"],
                        index=cudf.Index(
                            [
                                136457654,
                                None,
                                245345345,
                                223432411,
                                None,
                                3634548734,
                                23234,
                            ],
                            dtype="timedelta64[ns]",
                        ),
                    )
                }
            ),
            textwrap.dedent(
                """
                                    a
                0 days 00:00:00.136457654  a
                <NA>                f
                0 days 00:00:00.245345345  q
                0 days 00:00:00.223432411  e
                <NA>                w
                0 days 00:00:03.634548734  e
                0 days 00:00:00.000023234  t
                """
            ),
        ),
    ],
)
def test_timedelta_dataframe_repr(df, expected_repr):
    actual_repr = df.__repr__()

    assert actual_repr.split() == expected_repr.split()


@pytest.mark.parametrize(
    "index, expected_repr",
    [
        (
            cudf.Index([1000000, 200000, 3000000], dtype="timedelta64[ms]"),
            "TimedeltaIndex(['0 days 00:16:40', "
            "'0 days 00:03:20', '0 days 00:50:00'], "
            "dtype='timedelta64[ms]')",
        ),
        (
            cudf.Index(
                [None, None, None, None, None], dtype="timedelta64[us]"
            ),
            "TimedeltaIndex([<NA>, <NA>, <NA>, <NA>, <NA>], "
            "dtype='timedelta64[us]')",
        ),
        (
            cudf.Index(
                [
                    136457654,
                    None,
                    245345345,
                    223432411,
                    None,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[us]",
            ),
            "TimedeltaIndex([0 days 00:02:16.457654, <NA>, "
            "0 days 00:04:05.345345, "
            "0 days 00:03:43.432411, <NA>,"
            "       0 days 01:00:34.548734, 0 days 00:00:00.023234],"
            "      dtype='timedelta64[us]')",
        ),
        (
            cudf.Index(
                [
                    136457654,
                    None,
                    245345345,
                    223432411,
                    None,
                    3634548734,
                    23234,
                ],
                dtype="timedelta64[s]",
            ),
            "TimedeltaIndex([1579 days 08:54:14, <NA>, 2839 days 15:29:05,"
            "       2586 days 00:33:31, <NA>, 42066 days 12:52:14, "
            "0 days 06:27:14],"
            "      dtype='timedelta64[s]')",
        ),
    ],
)
def test_timedelta_index_repr(index, expected_repr):
    if not PANDAS_GE_110:
        pytest.xfail(reason="pandas >= 1.1 requried")
    actual_repr = index.__repr__()

    assert actual_repr.split() == expected_repr.split()


@pytest.mark.parametrize(
    "pmi",
    [
        pd.MultiIndex.from_tuples(
            [(1, "red"), (1, "blue"), (2, "red"), (2, "blue")]
        ),
        pd.MultiIndex.from_tuples(
            [(1, "red"), (1, "blue"), (2, "red"), (2, "blue")] * 10
        ),
        pd.MultiIndex.from_tuples([(1, "red", 102, "sdf")]),
        pd.MultiIndex.from_tuples(
            [
                ("abc", 0.234, 1),
                ("a", -0.34, 0),
                ("ai", 111, 4385798),
                ("rapids", 0, 34534534),
            ],
            names=["alphabets", "floats", "ints"],
        ),
    ],
)
@pytest.mark.parametrize(
    "max_seq_items",
    [
        None,
        pytest.param(
            1,
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/38415"
            ),
        ),
        2,
        5,
        10,
        100,
    ],
)
def test_mulitIndex_repr(pmi, max_seq_items):
    pd.set_option("display.max_seq_items", max_seq_items)
    gmi = cudf.from_pandas(pmi)
    print(gmi)
    print(pmi)
    assert gmi.__repr__() == pmi.__repr__()
    pd.reset_option("display.max_seq_items")


@pytest.mark.parametrize(
    "gdi, expected_repr",
    [
        (
            cudf.DataFrame(
                {
                    "a": [None, 1, 2, 3],
                    "b": ["abc", None, "xyz", None],
                    "c": [0.345, np.nan, 100, 10],
                }
            )
            .set_index(["a", "b"])
            .index,
            textwrap.dedent(
                """
                MultiIndex([(<NA>, 'abc'),
                            (   1,  <NA>),
                            (   2, 'xyz'),
                            (   3,  <NA>)],
                        names=['a', 'b'])
                """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series([None, np.nan, 2, 3], nan_as_null=False),
                    "b": ["abc", None, "xyz", None],
                    "c": [0.345, np.nan, 100, 10],
                }
            )
            .set_index(["a", "b"])
            .index,
            textwrap.dedent(
                """
            MultiIndex([(<NA>, 'abc'),
                        ( nan,  <NA>),
                        ( 2.0, 'xyz'),
                        ( 3.0,  <NA>)],
                    names=['a', 'b'])
            """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series([None, 1, 2, 3], dtype="datetime64[ns]"),
                    "b": ["abc", None, "xyz", None],
                    "c": [0.345, np.nan, 100, 10],
                }
            )
            .set_index(["a", "b"])
            .index,
            textwrap.dedent(
                """
            MultiIndex([(                         '<NA>', 'abc'),
                        ('1970-01-01 00:00:00.000000001',  <NA>),
                        ('1970-01-01 00:00:00.000000002', 'xyz'),
                        ('1970-01-01 00:00:00.000000003',  <NA>)],
                    names=['a', 'b'])
            """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": cudf.Series([None, 1, 2, 3], dtype="datetime64[ns]"),
                    "b": ["abc", None, "xyz", None],
                    "c": [0.345, np.nan, 100, 10],
                }
            )
            .set_index(["a", "b", "c"])
            .index,
            textwrap.dedent(
                """
                MultiIndex([(                         '<NA>', 'abc', 0.345),
                            ('1970-01-01 00:00:00.000000001',  <NA>,  <NA>),
                            ('1970-01-01 00:00:00.000000002', 'xyz', 100.0),
                            ('1970-01-01 00:00:00.000000003',  <NA>,  10.0)],
                        names=['a', 'b', 'c'])
                """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": ["abc", None, "xyz", None],
                    "b": cudf.Series([None, 1, 2, 3], dtype="timedelta64[ns]"),
                    "c": [0.345, np.nan, 100, 10],
                }
            )
            .set_index(["a", "b", "c"])
            .index,
            textwrap.dedent(
                """
                MultiIndex([('abc',                      '<NA>', 0.345),
                            ( <NA>, '0 days 00:00:00.000000001',  <NA>),
                            ('xyz', '0 days 00:00:00.000000002', 100.0),
                            ( <NA>, '0 days 00:00:00.000000003',  10.0)],
                        names=['a', 'b', 'c'])
                """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": ["abc", None, "xyz", None],
                    "b": cudf.Series([None, 1, 2, 3], dtype="timedelta64[ns]"),
                    "c": [0.345, np.nan, 100, 10],
                }
            )
            .set_index(["c", "a"])
            .index,
            textwrap.dedent(
                """
                MultiIndex([(0.345, 'abc'),
                            ( <NA>,  <NA>),
                            (100.0, 'xyz'),
                            ( 10.0,  <NA>)],
                        names=['c', 'a'])
                """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": [None, None, None, None],
                    "b": cudf.Series(
                        [None, None, None, None], dtype="timedelta64[ns]"
                    ),
                    "c": [0.345, np.nan, 100, 10],
                }
            )
            .set_index(["b", "a"])
            .index,
            textwrap.dedent(
                """
            MultiIndex([('<NA>', <NA>),
                        ('<NA>', <NA>),
                        ('<NA>', <NA>),
                        ('<NA>', <NA>)],
                    names=['b', 'a'])
            """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": [1, 2, None, 3, 5],
                    "b": [
                        "abc",
                        "def, hi, bye",
                        None,
                        ", one, two, three, four",
                        None,
                    ],
                    "c": cudf.Series(
                        [0.3232, np.nan, 1, None, -0.34534], nan_as_null=False
                    ),
                    "d": [None, 100, 2000324, None, None],
                }
            )
            .set_index(["a", "b", "c", "d"])
            .index,
            textwrap.dedent(
                """
    MultiIndex([(   1,                     'abc',   0.3232,    <NA>),
                (   2,            'def, hi, bye',      nan,     100),
                (<NA>,                      <NA>,      1.0, 2000324),
                (   3, ', one, two, three, four',     <NA>,    <NA>),
                (   5,                      <NA>, -0.34534,    <NA>)],
            names=['a', 'b', 'c', 'd'])
    """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": [1, 2, None, 3, 5],
                    "b": [
                        "abc",
                        "def, hi, bye",
                        None,
                        ", one, two, three, four",
                        None,
                    ],
                    "c": cudf.Series(
                        [0.3232, np.nan, 1, None, -0.34534], nan_as_null=False
                    ),
                    "d": [None, 100, 2000324, None, None],
                }
            )
            .set_index(["b", "a", "c", "d"])
            .index,
            textwrap.dedent(
                """
    MultiIndex([(                    'abc',    1,   0.3232,    <NA>),
                (           'def, hi, bye',    2,      nan,     100),
                (                     <NA>, <NA>,      1.0, 2000324),
                (', one, two, three, four',    3,     <NA>,    <NA>),
                (                     <NA>,    5, -0.34534,    <NA>)],
            names=['b', 'a', 'c', 'd'])
    """
            ),
        ),
        (
            cudf.DataFrame(
                {
                    "a": ["(abc", "2", None, "3", "5"],
                    "b": [
                        "abc",
                        "def, hi, bye",
                        None,
                        ", one, two, three, four",
                        None,
                    ],
                    "c": cudf.Series(
                        [0.3232, np.nan, 1, None, -0.34534], nan_as_null=False
                    ),
                    "d": [None, 100, 2000324, None, None],
                }
            )
            .set_index(["a", "b", "c", "d"])
            .index,
            textwrap.dedent(
                """
    MultiIndex([('(abc',                     'abc',   0.3232,    <NA>),
                (   '2',            'def, hi, bye',      nan,     100),
                (  <NA>,                      <NA>,      1.0, 2000324),
                (   '3', ', one, two, three, four',     <NA>,    <NA>),
                (   '5',                      <NA>, -0.34534,    <NA>)],
            names=['a', 'b', 'c', 'd'])
    """
            ),
        ),
    ],
)
def test_mulitIndex_null_repr(gdi, expected_repr):
    actual_repr = gdi.__repr__()

    assert actual_repr.split() == expected_repr.split()
