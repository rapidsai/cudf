# Copyright (c) 2019-2020, NVIDIA CORPORATION.
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

import cudf
from cudf.tests import utils

repr_categories = utils.NUMERIC_TYPES + ["str", "category", "datetime64[ns]"]


@pytest.mark.parametrize("dtype", repr_categories)
@pytest.mark.parametrize("nrows", [0, 5, 10])
def test_null_series(nrows, dtype):
    size = 5
    mask = utils.random_bitmask(size)
    data = cudf.Series(np.random.randint(1, 9, size))
    column = data.set_mask(mask)
    sr = cudf.Series(column).astype(dtype)
    ps = sr.to_pandas(nullable_pd_dtype=False)
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

    print(psrepr)
    print(sr)
    assert psrepr.split() == sr.__repr__().split()


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
    print(pdf)
    print(gdf)
    assert pdfrepr.split() == gdf.__repr__().split()


@pytest.mark.parametrize("dtype", repr_categories)
@pytest.mark.parametrize("nrows", [0, 1, 2, 9, 10, 11, 19, 20, 21])
def test_full_series(nrows, dtype):
    size = 20
    ps = pd.Series(np.random.randint(0, 100, size)).astype(dtype)
    sr = cudf.from_pandas(ps)
    pd.options.display.max_rows = int(nrows)
    assert ps.__repr__() == sr.__repr__()


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


@given(
    st.lists(
        st.integers(-9223372036854775808, 9223372036854775807), max_size=10000
    )
)
@settings(deadline=None)
def test_integer_series(x):
    sr = cudf.Series(x)
    ps = pd.Series(x)
    print(sr)
    print(ps)
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
    assert gdfT.__repr__() == pdfT.__repr__()


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
    assert gdg.T.__repr__() == pdg.T.__repr__()


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
