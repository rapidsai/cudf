# Copyright (c) 2020, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core import DataFrame, Series
from cudf.tests.utils import INTEGER_TYPES, NUMERIC_TYPES, assert_eq


def test_series_replace():
    a1 = np.array([0, 1, 2, 3, 4])

    # Numerical
    a2 = np.array([5, 1, 2, 3, 4])
    sr1 = Series(a1)
    sr2 = sr1.replace(0, 5)
    assert_eq(a2, sr2.to_array())

    # Categorical
    psr3 = pd.Series(["one", "two", "three"], dtype="category")
    psr4 = psr3.replace("one", "two")
    sr3 = Series.from_pandas(psr3)
    sr4 = sr3.replace("one", "two")
    assert_eq(psr4, sr4)

    psr5 = psr3.replace("one", "five")
    sr5 = sr3.replace("one", "five")

    assert_eq(psr5, sr5)

    # List input
    a6 = np.array([5, 6, 2, 3, 4])
    sr6 = sr1.replace([0, 1], [5, 6])
    assert_eq(a6, sr6.to_array())

    with pytest.raises(TypeError):
        sr1.replace([0, 1], [5.5, 6.5])

    # Series input
    a8 = np.array([5, 5, 5, 3, 4])
    sr8 = sr1.replace(sr1[:3], 5)
    assert_eq(a8, sr8.to_array())

    # large input containing null
    sr9 = Series(list(range(400)) + [None])
    sr10 = sr9.replace([22, 323, 27, 0], None)
    assert sr10.null_count == 5
    assert len(sr10.to_array()) == (401 - 5)

    sr11 = sr9.replace([22, 323, 27, 0], -1)
    assert sr11.null_count == 1
    assert len(sr11.to_array()) == (401 - 1)

    # large input not containing nulls
    sr9 = sr9.fillna(-11)
    sr12 = sr9.replace([22, 323, 27, 0], None)
    assert sr12.null_count == 4
    assert len(sr12.to_array()) == (401 - 4)

    sr13 = sr9.replace([22, 323, 27, 0], -1)
    assert sr13.null_count == 0
    assert len(sr13.to_array()) == 401


def test_series_replace_with_nulls():
    a1 = np.array([0, 1, 2, 3, 4])

    # Numerical
    a2 = np.array([-10, 1, 2, 3, 4])
    sr1 = Series(a1)
    sr2 = sr1.replace(0, None).fillna(-10)
    assert_eq(a2, sr2.to_array())

    # List input
    a6 = np.array([-10, 6, 2, 3, 4])
    sr6 = sr1.replace([0, 1], [None, 6]).fillna(-10)
    assert_eq(a6, sr6.to_array())

    sr1 = Series([0, 1, 2, 3, 4, None])
    with pytest.raises(TypeError):
        sr1.replace([0, 1], [5.5, 6.5]).fillna(-10)

    # Series input
    a8 = np.array([-10, -10, -10, 3, 4, -10])
    sr8 = sr1.replace(sr1[:3], None).fillna(-10)
    assert_eq(a8, sr8.to_array())

    a9 = np.array([-10, 6, 2, 3, 4, -10])
    sr9 = sr1.replace([0, 1], [None, 6]).fillna(-10)
    assert_eq(a9, sr9.to_array())


def test_dataframe_replace():
    # numerical
    pdf1 = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]})
    gdf1 = DataFrame.from_pandas(pdf1)
    pdf2 = pdf1.replace(0, 4)
    gdf2 = gdf1.replace(0, 4)
    assert_eq(gdf2, pdf2)

    # categorical
    pdf4 = pd.DataFrame(
        {"a": ["one", "two", "three"], "b": ["one", "two", "three"]},
        dtype="category",
    )
    gdf4 = DataFrame.from_pandas(pdf4)
    pdf5 = pdf4.replace("two", "three")
    gdf5 = gdf4.replace("two", "three")
    assert_eq(gdf5, pdf5)

    # list input
    pdf6 = pdf1.replace([0, 1], [4, 5])
    gdf6 = gdf1.replace([0, 1], [4, 5])
    assert_eq(gdf6, pdf6)

    pdf7 = pdf1.replace([0, 1], 4)
    gdf7 = gdf1.replace([0, 1], 4)
    assert_eq(gdf7, pdf7)

    # dict input:
    pdf8 = pdf1.replace({"a": 0, "b": 0}, {"a": 4, "b": 5})
    gdf8 = gdf1.replace({"a": 0, "b": 0}, {"a": 4, "b": 5})
    assert_eq(gdf8, pdf8)

    pdf9 = pdf1.replace({"a": 0}, {"a": 4})
    gdf9 = gdf1.replace({"a": 0}, {"a": 4})
    assert_eq(gdf9, pdf9)


def test_dataframe_replace_with_nulls():
    # numerical
    pdf1 = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]})
    gdf1 = DataFrame.from_pandas(pdf1)
    pdf2 = pdf1.replace(0, 4)
    gdf2 = gdf1.replace(0, None).fillna(4)
    assert_eq(gdf2, pdf2)

    # list input
    pdf6 = pdf1.replace([0, 1], [4, 5])
    gdf6 = gdf1.replace([0, 1], [4, None]).fillna(5)
    assert_eq(gdf6, pdf6)

    pdf7 = pdf1.replace([0, 1], 4)
    gdf7 = gdf1.replace([0, 1], None).fillna(4)
    assert_eq(gdf7, pdf7)

    # dict input:
    pdf8 = pdf1.replace({"a": 0, "b": 0}, {"a": 4, "b": 5})
    gdf8 = gdf1.replace({"a": 0, "b": 0}, {"a": None, "b": 5}).fillna(4)
    assert_eq(gdf8, pdf8)

    gdf1 = DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, None]})
    gdf9 = gdf1.replace([0, 1], [4, 5]).fillna(3)
    assert_eq(gdf9, pdf6)


def test_replace_strings():
    pdf = pd.Series(["a", "b", "c", "d"])
    gdf = Series(["a", "b", "c", "d"])
    assert_eq(pdf.replace("a", "e"), gdf.replace("a", "e"))


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series([0, 1, None, 2, None], dtype=pd.Int8Dtype()),
        pd.Series([0, 1, np.nan, 2, np.nan]),
    ],
)
@pytest.mark.parametrize("data_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("fill_value", [10, pd.Series([10, 20, 30, 40, 50])])
@pytest.mark.parametrize("inplace", [True, False])
def test_series_fillna_numerical(psr, data_dtype, fill_value, inplace):
    test_psr = psr.copy(deep=True)
    # TODO: These tests should use Pandas' nullable int type
    # when we support a recent enough version of Pandas
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
    if np.dtype(data_dtype).kind not in ("f") and test_psr.dtype.kind == "i":
        test_psr = test_psr.astype(
            cudf.utils.dtypes.cudf_dtypes_to_pandas_dtypes[
                np.dtype(data_dtype)
            ]
        )

    gsr = cudf.from_pandas(test_psr)

    if isinstance(fill_value, pd.Series):
        fill_value_cudf = cudf.from_pandas(fill_value)
    else:
        fill_value_cudf = fill_value

    expected = test_psr.fillna(fill_value, inplace=inplace)
    actual = gsr.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        expected = test_psr
        actual = gsr

    # TODO: Remove check_dtype when we have support
    # to compare with pandas nullable dtypes
    assert_eq(expected, actual, check_dtype=False)


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series(["a", "b", "a", None, "c", None], dtype="category"),
        pd.Series(
            ["a", "b", "a", None, "c", None],
            dtype="category",
            index=["q", "r", "z", "a", "b", "c"],
        ),
        pd.Series(
            ["a", "b", "a", None, "c", None],
            dtype="category",
            index=["x", "t", "p", "q", "r", "z"],
        ),
        pd.Series(["a", "b", "a", np.nan, "c", np.nan], dtype="category"),
        pd.Series(
            [None, None, None, None, None, None, "a", "b", "c"],
            dtype="category",
        ),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        "c",
        pd.Series(["c", "c", "c", "c", "c", "a"], dtype="category"),
        pd.Series(
            ["a", "b", "a", None, "c", None],
            dtype="category",
            index=["x", "t", "p", "q", "r", "z"],
        ),
        pd.Series(
            ["a", "b", "a", None, "c", None],
            dtype="category",
            index=["q", "r", "z", "a", "b", "c"],
        ),
        pd.Series(["a", "b", "a", None, "c", None], dtype="category"),
        pd.Series(["a", "b", "a", np.nan, "c", np.nan], dtype="category"),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_categorical(psr, fill_value, inplace):

    gsr = Series.from_pandas(psr)

    if isinstance(fill_value, pd.Series):
        fill_value_cudf = cudf.from_pandas(fill_value)
    else:
        fill_value_cudf = fill_value

    expected = psr.fillna(fill_value, inplace=inplace)
    got = gsr.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        expected = psr
        got = gsr

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series(pd.date_range("2010-01-01", "2020-01-10", freq="1y")),
        pd.Series(["2010-01-01", None, "2011-10-10"], dtype="datetime64[ns]"),
        pd.Series(
            [
                None,
                None,
                None,
                None,
                None,
                None,
                "2011-10-10",
                "2010-01-01",
                "2010-01-02",
                "2010-01-04",
                "2010-11-01",
            ],
            dtype="datetime64[ns]",
        ),
        pd.Series(
            [
                None,
                None,
                None,
                None,
                None,
                None,
                "2011-10-10",
                "2010-01-01",
                "2010-01-02",
                "2010-01-04",
                "2010-11-01",
            ],
            dtype="datetime64[ns]",
            index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        ),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        pd.Timestamp("2010-01-02"),
        pd.Series(pd.date_range("2010-01-01", "2020-01-10", freq="1y"))
        + pd.Timedelta("1d"),
        pd.Series(["2010-01-01", None, "2011-10-10"], dtype="datetime64[ns]"),
        pd.Series(
            [
                None,
                None,
                None,
                None,
                None,
                None,
                "2011-10-10",
                "2010-01-01",
                "2010-01-02",
                "2010-01-04",
                "2010-11-01",
            ],
            dtype="datetime64[ns]",
        ),
        pd.Series(
            [
                None,
                None,
                None,
                None,
                None,
                None,
                "2011-10-10",
                "2010-01-01",
                "2010-01-02",
                "2010-01-04",
                "2010-11-01",
            ],
            dtype="datetime64[ns]",
            index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        ),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_datetime(psr, fill_value, inplace):
    gsr = cudf.from_pandas(psr)

    if isinstance(fill_value, pd.Series):
        fill_value_cudf = cudf.from_pandas(fill_value)
    else:
        fill_value_cudf = fill_value

    expected = psr.fillna(fill_value, inplace=inplace)
    got = gsr.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        got = gsr
        expected = psr

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"a": [1, 2, None], "b": [None, None, 5]}),
        pd.DataFrame(
            {"a": [1, 2, None], "b": [None, None, 5]}, index=["a", "p", "z"]
        ),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        10,
        pd.Series([10, 20, 30]),
        pd.Series([3, 4, 5]),
        pd.Series([10, 20, 30], index=["z", "a", "p"]),
        {"a": 5, "b": pd.Series([3, 4, 5])},
        {"a": 5001},
        {"b": pd.Series([11, 22, 33], index=["a", "p", "z"])},
        {"a": 5, "b": pd.Series([3, 4, 5], index=["a", "p", "z"])},
        {"c": 100},
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_dataframe(df, value, inplace):
    pdf = df.copy(deep=True)
    gdf = DataFrame.from_pandas(pdf)

    fill_value_pd = value
    if isinstance(fill_value_pd, (pd.Series, pd.DataFrame)):
        fill_value_cudf = cudf.from_pandas(fill_value_pd)
    elif isinstance(fill_value_pd, dict):
        fill_value_cudf = {}
        for key in fill_value_pd:
            temp_val = fill_value_pd[key]
            if isinstance(temp_val, pd.Series):
                temp_val = cudf.from_pandas(temp_val)
            fill_value_cudf[key] = temp_val
    else:
        fill_value_cudf = value

    expect = pdf.fillna(fill_value_pd, inplace=inplace)
    got = gdf.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        got = gdf
        expect = pdf

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series(["a", "b", "c", "d"]),
        pd.Series([None] * 4, dtype="object"),
        pd.Series(["z", None, "z", None]),
        pd.Series(["x", "y", None, None, None]),
        pd.Series([None, None, None, "i", "P"]),
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        "a",
        pd.Series(["a", "b", "c", "d"]),
        pd.Series(["z", None, "z", None]),
        pd.Series([None] * 4, dtype="object"),
        pd.Series(["x", "y", None, None, None]),
        pd.Series([None, None, None, "i", "P"]),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_string(psr, fill_value, inplace):
    gsr = cudf.from_pandas(psr)

    if isinstance(fill_value, pd.Series):
        fill_value_cudf = cudf.from_pandas(fill_value)
    else:
        fill_value_cudf = fill_value

    expected = psr.fillna(fill_value, inplace=inplace)
    got = gsr.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        expected = psr
        got = gsr

    assert_eq(expected, got)


@pytest.mark.parametrize("data_dtype", INTEGER_TYPES)
def test_series_fillna_invalid_dtype(data_dtype):
    gdf = Series([1, 2, None, 3], dtype=data_dtype)
    fill_value = 2.5
    with pytest.raises(TypeError) as raises:
        gdf.fillna(fill_value)
    raises.match(
        "Cannot safely cast non-equivalent {} to {}".format(
            type(fill_value).__name__, gdf.dtype.type.__name__
        )
    )


@pytest.mark.parametrize("data_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("fill_value", [100, 100.0, 128.5])
def test_series_where(data_dtype, fill_value):
    psr = pd.Series(list(range(10)), dtype=data_dtype)
    sr = Series.from_pandas(psr)

    if sr.dtype.type(fill_value) != fill_value:
        with pytest.raises(TypeError):
            sr.where(sr > 0, fill_value)
    else:
        # Cast back to original dtype as pandas automatically upcasts
        expect = psr.where(psr > 0, fill_value).astype(psr.dtype)
        got = sr.where(sr > 0, fill_value)
        assert_eq(expect, got)

    if sr.dtype.type(fill_value) != fill_value:
        with pytest.raises(TypeError):
            sr.where(sr < 0, fill_value)
    else:
        expect = psr.where(psr < 0, fill_value).astype(psr.dtype)
        got = sr.where(sr < 0, fill_value)
        assert_eq(expect, got)

    if sr.dtype.type(fill_value) != fill_value:
        with pytest.raises(TypeError):
            sr.where(sr == 0, fill_value)
    else:
        expect = psr.where(psr == 0, fill_value).astype(psr.dtype)
        got = sr.where(sr == 0, fill_value)
        assert_eq(expect, got)


@pytest.mark.parametrize("fill_value", [100, 100.0, 100.5])
def test_series_with_nulls_where(fill_value):
    psr = pd.Series([None] * 3 + list(range(5)))
    sr = Series.from_pandas(psr)

    expect = psr.where(psr > 0, fill_value)
    got = sr.where(sr > 0, fill_value)
    assert_eq(expect, got)

    expect = psr.where(psr < 0, fill_value)
    got = sr.where(sr < 0, fill_value)
    assert_eq(expect, got)

    expect = psr.where(psr == 0, fill_value)
    got = sr.where(sr == 0, fill_value)
    assert_eq(expect, got)


@pytest.mark.parametrize("fill_value", [[888, 999]])
def test_dataframe_with_nulls_where_with_scalars(fill_value):
    pdf = pd.DataFrame(
        {
            "A": [-1, 2, -3, None, 5, 6, -7, 0],
            "B": [4, -2, 3, None, 7, 6, 8, 0],
        }
    )
    gdf = DataFrame.from_pandas(pdf)

    expect = pdf.where(pdf % 3 == 0, fill_value)
    got = gdf.where(gdf % 3 == 0, fill_value)

    assert_eq(expect, got)


def test_dataframe_with_different_types():

    # Testing for int and float
    pdf = pd.DataFrame(
        {"A": [111, 22, 31, 410, 56], "B": [-10.12, 121.2, 45.7, 98.4, 87.6]}
    )
    gdf = DataFrame.from_pandas(pdf)
    expect = pdf.where(pdf > 50, -pdf)
    got = gdf.where(gdf > 50, -gdf)

    assert_eq(expect, got)

    # Testing for string
    pdf = pd.DataFrame({"A": ["a", "bc", "cde", "fghi"]})
    gdf = DataFrame.from_pandas(pdf)
    pdf_mask = pd.DataFrame({"A": [True, False, True, False]})
    gdf_mask = DataFrame.from_pandas(pdf_mask)
    expect = pdf.where(pdf_mask, ["cudf"])
    got = gdf.where(gdf_mask, ["cudf"])

    assert_eq(expect, got)

    # Testing for categoriacal
    pdf = pd.DataFrame({"A": ["a", "b", "b", "c"]})
    pdf["A"] = pdf["A"].astype("category")
    gdf = DataFrame.from_pandas(pdf)
    expect = pdf.where(pdf_mask, "c")
    got = gdf.where(gdf_mask, ["c"])

    assert_eq(expect, got)


def test_dataframe_where_with_different_options():
    pdf = pd.DataFrame({"A": [1, 2, 3], "B": [3, 4, 5]})
    gdf = DataFrame.from_pandas(pdf)

    # numpy array
    boolean_mask = np.array([[False, True], [True, False], [False, True]])

    expect = pdf.where(boolean_mask, -pdf)
    got = gdf.where(boolean_mask, -gdf)

    assert_eq(expect, got)

    # with single scalar
    expect = pdf.where(boolean_mask, 8)
    got = gdf.where(boolean_mask, 8)

    assert_eq(expect, got)

    # with multi scalar
    expect = pdf.where(boolean_mask, [8, 9])
    got = gdf.where(boolean_mask, [8, 9])

    assert_eq(expect, got)


def test_series_multiple_times_with_nulls():
    sr = Series([1, 2, 3, None])
    expected = Series([None, None, None, None], dtype=np.int64)

    for i in range(3):
        got = sr.replace([1, 2, 3], None)
        assert_eq(expected, got)
        # BUG: #2695
        # The following series will acquire a chunk of memory and update with
        # values, but these values may still linger even after the memory
        # gets released. This memory space might get used for replace in
        # subsequent calls and the memory used for mask may have junk values.
        # So, if it is not updated properly, the result would be wrong.
        # So, this will help verify that scenario.
        Series([1, 1, 1, None])


@pytest.mark.parametrize("series_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize(
    "replacement", [128, 128.0, 128.5, 32769, 32769.0, 32769.5]
)
def test_numeric_series_replace_dtype(series_dtype, replacement):
    psr = pd.Series([0, 1, 2, 3, 4, 5], dtype=series_dtype)
    sr = Series.from_pandas(psr)

    # Both Scalar
    if sr.dtype.type(replacement) != replacement:
        with pytest.raises(TypeError):
            sr.replace(1, replacement)
    else:
        expect = psr.replace(1, replacement).astype(psr.dtype)
        got = sr.replace(1, replacement)
        assert_eq(expect, got)

    # to_replace is a list, replacement is a scalar
    if sr.dtype.type(replacement) != replacement:
        with pytest.raises(TypeError):
            sr.replace([2, 3], replacement)
    else:
        expect = psr.replace([2, 3], replacement).astype(psr.dtype)
        got = sr.replace([2, 3], replacement)
        assert_eq(expect, got)

    # If to_replace is a scalar and replacement is a list
    with pytest.raises(TypeError):
        sr.replace(0, [replacement, 2])

    # Both list of unequal length
    with pytest.raises(ValueError):
        sr.replace([0, 1], [replacement])

    # Both lists of equal length
    if (
        np.dtype(type(replacement)).kind == "f" and sr.dtype.kind in {"i", "u"}
    ) or (sr.dtype.type(replacement) != replacement):
        with pytest.raises(TypeError):
            sr.replace([2, 3], [replacement, replacement])
    else:
        expect = psr.replace([2, 3], [replacement, replacement]).astype(
            psr.dtype
        )
        got = sr.replace([2, 3], [replacement, replacement])
        assert_eq(expect, got)


def test_replace_inplace():
    data = np.array([5, 1, 2, 3, 4])
    sr = Series(data)
    psr = pd.Series(data)

    sr_copy = sr.copy()
    psr_copy = psr.copy()

    assert_eq(sr, psr)
    assert_eq(sr_copy, psr_copy)
    sr.replace(5, 0, inplace=True)
    psr.replace(5, 0, inplace=True)
    assert_eq(sr, psr)
    assert_eq(sr_copy, psr_copy)

    sr = Series(data)
    psr = pd.Series(data)

    sr_copy = sr.copy()
    psr_copy = psr.copy()

    assert_eq(sr, psr)
    assert_eq(sr_copy, psr_copy)
    sr.replace({5: 0, 3: -5})
    psr.replace({5: 0, 3: -5})
    assert_eq(sr, psr)
    assert_eq(sr_copy, psr_copy)
    srr = sr.replace()
    psrr = psr.replace()
    assert_eq(srr, psrr)

    psr = pd.Series(["one", "two", "three"], dtype="category")
    sr = Series.from_pandas(psr)

    sr_copy = sr.copy()
    psr_copy = psr.copy()

    assert_eq(sr, psr)
    assert_eq(sr_copy, psr_copy)
    sr.replace("one", "two", inplace=True)
    psr.replace("one", "two", inplace=True)
    assert_eq(sr, psr)
    assert_eq(sr_copy, psr_copy)

    pdf = pd.DataFrame({"A": [0, 1, 2, 3, 4], "B": [5, 6, 7, 8, 9]})
    gdf = DataFrame.from_pandas(pdf)

    pdf_copy = pdf.copy()
    gdf_copy = gdf.copy()
    assert_eq(pdf, gdf)
    assert_eq(pdf_copy, gdf_copy)
    pdf.replace(5, 0, inplace=True)
    gdf.replace(5, 0, inplace=True)
    assert_eq(pdf, gdf)
    assert_eq(pdf_copy, gdf_copy)

    pds = pd.Series([1, 2, 3, 45])
    gds = Series.from_pandas(pds)
    vals = np.array([]).astype(int)

    assert_eq(pds.replace(vals, -1), gds.replace(vals, -1))

    pds.replace(vals, 77, inplace=True)
    gds.replace(vals, 77, inplace=True)
    assert_eq(pds, gds)

    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]})
    gdf = DataFrame.from_pandas(pdf)

    assert_eq(
        pdf.replace({"a": 2}, {"a": -33}), gdf.replace({"a": 2}, {"a": -33})
    )

    assert_eq(
        pdf.replace({"a": [2, 5]}, {"a": [9, 10]}),
        gdf.replace({"a": [2, 5]}, {"a": [9, 10]}),
    )

    assert_eq(
        pdf.replace([], []), gdf.replace([], []),
    )

    with pytest.raises(TypeError):
        pdf.replace(-1, [])

    with pytest.raises(TypeError):
        gdf.replace(-1, [])


@pytest.mark.parametrize(
    ("lower", "upper"),
    [([2, 7.4], [4, 7.9]), ([2, 7.4], None), (None, [4, 7.9],)],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_clip(lower, upper, inplace):
    pdf = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [7.1, 7.24, 7.5, 7.8, 8.11]}
    )
    gdf = DataFrame.from_pandas(pdf)

    got = gdf.clip(lower=lower, upper=upper, inplace=inplace)
    expect = pdf.clip(lower=lower, upper=upper, axis=1)

    if inplace is True:
        assert_eq(expect, gdf)
    else:
        assert_eq(expect, got)


@pytest.mark.parametrize(
    ("lower", "upper"), [("b", "d"), ("b", None), (None, "c"), (None, None)],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_category_clip(lower, upper, inplace):
    data = ["a", "b", "c", "d", "e"]
    pdf = pd.DataFrame({"a": data})
    gdf = DataFrame.from_pandas(pdf)
    gdf["a"] = gdf["a"].astype("category")

    expect = pdf.clip(lower=lower, upper=upper)
    got = gdf.clip(lower=lower, upper=upper, inplace=inplace)

    if inplace is True:
        assert_eq(expect, gdf.astype("str"))
    else:
        assert_eq(expect, got.astype("str"))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [([2, 7.4], [4, 7.9, "d"]), ([2, 7.4, "a"], [4, 7.9, "d"])],
)
def test_dataframe_exceptions_for_clip(lower, upper):
    gdf = DataFrame({"a": [1, 2, 3, 4, 5], "b": [7.1, 7.24, 7.5, 7.8, 8.11]})

    with pytest.raises(ValueError):
        gdf.clip(lower=lower, upper=upper)


@pytest.mark.parametrize(
    ("data", "lower", "upper"),
    [
        ([1, 2, 3, 4, 5], 2, 4),
        ([1, 2, 3, 4, 5], 2, None),
        ([1, 2, 3, 4, 5], None, 4),
        ([1, 2, 3, 4, 5], None, None),
        ([1, 2, 3, 4, 5], 4, 2),
        (["a", "b", "c", "d", "e"], "b", "d"),
        (["a", "b", "c", "d", "e"], "b", None),
        (["a", "b", "c", "d", "e"], None, "d"),
        (["a", "b", "c", "d", "e"], "d", "b"),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_series_clip(data, lower, upper, inplace):
    psr = pd.Series(data)
    gsr = Series.from_pandas(data)

    expect = psr.clip(lower=lower, upper=upper)
    got = gsr.clip(lower=lower, upper=upper, inplace=inplace)

    if inplace is True:
        assert_eq(expect, gsr)
    else:
        assert_eq(expect, got)


def test_series_exceptions_for_clip():

    with pytest.raises(ValueError):
        Series([1, 2, 3, 4]).clip([1, 2], [2, 3])

    with pytest.raises(NotImplementedError):
        Series([1, 2, 3, 4]).clip(1, 2, axis=0)


@pytest.mark.parametrize(
    ("data", "lower", "upper"),
    [
        ([1, 2, 3, 4, 5], 2, 4),
        ([1, 2, 3, 4, 5], 2, None),
        ([1, 2, 3, 4, 5], None, 4),
        ([1, 2, 3, 4, 5], None, None),
        (["a", "b", "c", "d", "e"], "b", "d"),
        (["a", "b", "c", "d", "e"], "b", None),
        (["a", "b", "c", "d", "e"], None, "d"),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_index_clip(data, lower, upper, inplace):
    pdf = pd.DataFrame({"a": data})
    index = DataFrame.from_pandas(pdf).set_index("a").index

    expect = pdf.clip(lower=lower, upper=upper)
    got = index.clip(lower=lower, upper=upper, inplace=inplace)

    if inplace is True:
        assert_eq(expect, index.to_frame(index=False))
    else:
        assert_eq(expect, got.to_frame(index=False))


@pytest.mark.parametrize(
    ("lower", "upper"), [([2, 3], [4, 5]), ([2, 3], None), (None, [4, 5],)],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_multiindex_clip(lower, upper, inplace):
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    gdf = DataFrame.from_pandas(df)

    index = gdf.set_index(["a", "b"]).index

    expected = df.clip(lower=lower, upper=upper, inplace=inplace, axis=1)
    got = index.clip(lower=lower, upper=upper, inplace=inplace)

    if inplace is True:
        assert_eq(df, index.to_frame(index=False))
    else:
        assert_eq(expected, got.to_frame(index=False))


@pytest.mark.parametrize(
    "data", [[1, 2.0, 3, 4, None, 1, None, 10, None], ["a", "b", "c"]]
)
@pytest.mark.parametrize(
    "index",
    [
        None,
        [1, 2, 3],
        ["a", "b", "z"],
        ["a", "b", "c", "d", "e", "f", "g", "l", "m"],
    ],
)
@pytest.mark.parametrize("value", [[1, 2, 3, 4, None, 1, None, 10, None]])
def test_series_fillna(data, index, value):
    psr = pd.Series(
        data,
        index=index if index is not None and len(index) == len(data) else None,
    )
    gsr = Series(
        data,
        index=index if index is not None and len(index) == len(data) else None,
    )

    expect = psr.fillna(pd.Series(value))
    got = gsr.fillna(Series(value))
    assert_eq(expect, got)


def test_series_fillna_error():
    psr = pd.Series([1, 2, None, 3, None])
    gsr = cudf.from_pandas(psr)

    try:
        psr.fillna(pd.DataFrame({"a": [1, 2, 3]}))
    except Exception as e:
        with pytest.raises(type(e), match=str(e)):
            gsr.fillna(cudf.DataFrame({"a": [1, 2, 3]}))
    else:
        raise AssertionError("Expected psr.fillna to fail")
