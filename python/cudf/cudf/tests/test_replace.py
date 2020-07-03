import numpy as np
import pandas as pd
import pytest

from cudf.core import DataFrame, Series
from cudf.tests.utils import INTEGER_TYPES, NUMERIC_TYPES, assert_eq


def test_series_replace():
    a1 = np.array([0, 1, 2, 3, 4])

    # Numerical
    a2 = np.array([5, 1, 2, 3, 4])
    sr1 = Series(a1)
    sr2 = sr1.replace(0, 5)
    np.testing.assert_equal(sr2.to_array(), a2)

    # Categorical
    psr3 = pd.Series(["one", "two", "three"], dtype="category")
    psr4 = psr3.replace("one", "two")
    sr3 = Series.from_pandas(psr3)
    sr4 = sr3.replace("one", "two")
    pd.testing.assert_series_equal(sr4.to_pandas(), psr4)

    psr5 = psr3.replace("one", "five")
    sr5 = sr3.replace("one", "five")

    pd.testing.assert_series_equal(sr5.to_pandas(), psr5)

    # List input
    a6 = np.array([5, 6, 2, 3, 4])
    sr6 = sr1.replace([0, 1], [5, 6])
    np.testing.assert_equal(sr6.to_array(), a6)

    with pytest.raises(TypeError):
        sr1.replace([0, 1], [5.5, 6.5])

    # Series input
    a8 = np.array([5, 5, 5, 3, 4])
    sr8 = sr1.replace(sr1[:3], 5)
    np.testing.assert_equal(sr8.to_array(), a8)

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
    np.testing.assert_equal(sr2.to_array(), a2)

    # List input
    a6 = np.array([-10, 6, 2, 3, 4])
    sr6 = sr1.replace([0, 1], [None, 6]).fillna(-10)
    np.testing.assert_equal(sr6.to_array(), a6)

    sr1 = Series([0, 1, 2, 3, 4, None])
    with pytest.raises(TypeError):
        sr1.replace([0, 1], [5.5, 6.5]).fillna(-10)

    # Series input
    a8 = np.array([-10, -10, -10, 3, 4, -10])
    sr8 = sr1.replace(sr1[:3], None).fillna(-10)
    np.testing.assert_equal(sr8.to_array(), a8)

    a9 = np.array([-10, 6, 2, 3, 4, -10])
    sr9 = sr1.replace([0, 1], [None, 6]).fillna(-10)
    np.testing.assert_equal(sr9.to_array(), a9)


def test_dataframe_replace():
    # numerical
    pdf1 = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]})
    gdf1 = DataFrame.from_pandas(pdf1)
    pdf2 = pdf1.replace(0, 4)
    gdf2 = gdf1.replace(0, 4)
    pd.testing.assert_frame_equal(gdf2.to_pandas(), pdf2)

    # categorical
    pdf4 = pd.DataFrame(
        {"a": ["one", "two", "three"], "b": ["one", "two", "three"]},
        dtype="category",
    )
    gdf4 = DataFrame.from_pandas(pdf4)
    pdf5 = pdf4.replace("two", "three")
    gdf5 = gdf4.replace("two", "three")
    pd.testing.assert_frame_equal(gdf5.to_pandas(), pdf5)

    # list input
    pdf6 = pdf1.replace([0, 1], [4, 5])
    gdf6 = gdf1.replace([0, 1], [4, 5])
    pd.testing.assert_frame_equal(gdf6.to_pandas(), pdf6)

    pdf7 = pdf1.replace([0, 1], 4)
    gdf7 = gdf1.replace([0, 1], 4)
    pd.testing.assert_frame_equal(gdf7.to_pandas(), pdf7)

    # dict input:
    pdf8 = pdf1.replace({"a": 0, "b": 0}, {"a": 4, "b": 5})
    gdf8 = gdf1.replace({"a": 0, "b": 0}, {"a": 4, "b": 5})
    pd.testing.assert_frame_equal(gdf8.to_pandas(), pdf8)

    pdf9 = pdf1.replace({"a": 0}, {"a": 4})
    gdf9 = gdf1.replace({"a": 0}, {"a": 4})
    pd.testing.assert_frame_equal(gdf9.to_pandas(), pdf9)


def test_dataframe_replace_with_nulls():
    # numerical
    pdf1 = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]})
    gdf1 = DataFrame.from_pandas(pdf1)
    pdf2 = pdf1.replace(0, 4)
    gdf2 = gdf1.replace(0, None).fillna(4)
    pd.testing.assert_frame_equal(gdf2.to_pandas(), pdf2)

    # list input
    pdf6 = pdf1.replace([0, 1], [4, 5])
    gdf6 = gdf1.replace([0, 1], [4, None]).fillna(5)
    pd.testing.assert_frame_equal(gdf6.to_pandas(), pdf6)

    pdf7 = pdf1.replace([0, 1], 4)
    gdf7 = gdf1.replace([0, 1], None).fillna(4)
    pd.testing.assert_frame_equal(gdf7.to_pandas(), pdf7)

    # dict input:
    pdf8 = pdf1.replace({"a": 0, "b": 0}, {"a": 4, "b": 5})
    gdf8 = gdf1.replace({"a": 0, "b": 0}, {"a": None, "b": 5}).fillna(4)
    pd.testing.assert_frame_equal(gdf8.to_pandas(), pdf8)

    gdf1 = DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, None]})
    gdf9 = gdf1.replace([0, 1], [4, 5]).fillna(3)
    pd.testing.assert_frame_equal(gdf9.to_pandas(), pdf6)


def test_replace_strings():
    pdf = pd.Series(["a", "b", "c", "d"])
    gdf = Series(["a", "b", "c", "d"])
    assert_eq(pdf.replace("a", "e"), gdf.replace("a", "e"))


@pytest.mark.parametrize("data_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("fill_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize("fill_type", ["scalar", "series"])
@pytest.mark.parametrize("null_value", [None, np.nan])
@pytest.mark.parametrize("inplace", [True, False])
def test_series_fillna_numerical(
    data_dtype, fill_dtype, fill_type, null_value, inplace
):
    # TODO: These tests should use Pandas' nullable int type
    # when we support a recent enough version of Pandas
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html

    if fill_type == "scalar":
        fill_value = np.random.randint(0, 5)
        expect = np.array([0, 1, fill_value, 2, fill_value], dtype=data_dtype)
    elif fill_type == "series":
        data = np.random.randint(0, 5, (5,))
        fill_value = pd.Series(data, dtype=data_dtype)
        expect = np.array(
            [0, 1, fill_value[2], 2, fill_value[4]], dtype=data_dtype
        )

    sr = Series([0, 1, null_value, 2, null_value], dtype=data_dtype)
    result = sr.fillna(fill_value, inplace=inplace)

    if inplace:
        result = sr

    got = result.to_array()

    np.testing.assert_equal(expect, got)


@pytest.mark.parametrize("fill_type", ["scalar", "series"])
@pytest.mark.parametrize("null_value", [None, np.nan])
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_categorical(fill_type, null_value, inplace):
    data = pd.Series(
        ["a", "b", "a", null_value, "c", null_value], dtype="category"
    )
    sr = Series.from_pandas(data)

    if fill_type == "scalar":
        fill_value = "c"
        expect = pd.Series(["a", "b", "a", "c", "c", "c"], dtype="category")
    elif fill_type == "series":
        fill_value = pd.Series(
            ["c", "c", "c", "c", "c", "a"], dtype="category"
        )
        expect = pd.Series(["a", "b", "a", "c", "c", "a"], dtype="category")

    got = sr.fillna(fill_value, inplace=inplace)

    if inplace:
        got = sr

    assert_eq(expect, got)


@pytest.mark.parametrize("fill_type", ["scalar", "series"])
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_datetime(fill_type, inplace):
    psr = pd.Series(pd.date_range("2010-01-01", "2020-01-10", freq="1y"))

    if fill_type == "scalar":
        fill_value = pd.Timestamp("2010-01-02")
    elif fill_type == "series":
        fill_value = psr + pd.Timedelta("1d")

    psr[[5, 9]] = None
    sr = Series.from_pandas(psr)

    expect = psr.fillna(fill_value)
    got = sr.fillna(fill_value, inplace=inplace)

    if inplace:
        got = sr

    assert_eq(expect, got)


@pytest.mark.parametrize("fill_type", ["scalar", "series", "dict"])
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_dataframe(fill_type, inplace):
    pdf = pd.DataFrame({"a": [1, 2, None], "b": [None, None, 5]})
    gdf = DataFrame.from_pandas(pdf)

    if fill_type == "scalar":
        fill_value_pd = 5
        fill_value_cudf = fill_value_pd
    elif fill_type == "series":
        fill_value_pd = pd.Series([3, 4, 5])
        fill_value_cudf = Series.from_pandas(fill_value_pd)
    else:
        fill_value_pd = {"a": 5, "b": pd.Series([3, 4, 5])}
        fill_value_cudf = {
            "a": fill_value_pd["a"],
            "b": Series.from_pandas(fill_value_pd["b"]),
        }

    # https://github.com/pandas-dev/pandas/issues/27197
    # pandas df.fill_value with series is not working

    if isinstance(fill_value_pd, pd.Series):
        expect = pd.DataFrame()
        for col in pdf.columns:
            expect[col] = pdf[col].fillna(fill_value_pd)
    else:
        expect = pdf.fillna(fill_value_pd)

    got = gdf.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        got = gdf

    assert_eq(expect, got)


@pytest.mark.parametrize("fill_type", ["scalar", "series"])
@pytest.mark.parametrize("inplace", [True, False])
def test_fillna_string(fill_type, inplace):
    psr = pd.Series(["z", None, "z", None])

    if fill_type == "scalar":
        fill_value_pd = "a"
        fill_value_cudf = fill_value_pd
    elif fill_type == "series":
        fill_value_pd = pd.Series(["a", "b", "c", "d"])
        fill_value_cudf = Series.from_pandas(fill_value_pd)

    sr = Series.from_pandas(psr)

    expect = psr.fillna(fill_value_pd)
    got = sr.fillna(fill_value_cudf, inplace=inplace)

    if inplace:
        got = sr

    assert_eq(expect, got)


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
