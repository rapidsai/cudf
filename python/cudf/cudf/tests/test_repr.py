# Copyright (c) 2019, NVIDIA CORPORATION.

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

import cudf
from cudf.tests import utils

repr_categories = [
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "datetime64[ns]",
    "str",
    "category",
]


@pytest.mark.parametrize("dtype", repr_categories)
@pytest.mark.parametrize("nrows", [0, 5, 10])
def test_null_series(nrows, dtype):
    size = 5
    mask = utils.random_bitmask(size)
    data = cudf.Series(np.random.randint(0, 128, size))
    column = data.set_mask(mask)
    sr = cudf.Series(column).astype(dtype)
    ps = sr.to_pandas()
    pd.options.display.max_rows = int(nrows)
    psrepr = ps.__repr__()
    psrepr = psrepr.replace("NaN", "null")
    psrepr = psrepr.replace("NaT", "null")
    psrepr = psrepr.replace("-1\n", "null\n")
    print(psrepr)
    print(sr)
    assert psrepr.split() == sr.__repr__().split()


@pytest.mark.parametrize("ncols", [1, 2, 3, 4, 5, 10])
def test_null_dataframe(ncols):
    size = 20
    gdf = cudf.DataFrame()
    for idx, dtype in enumerate(repr_categories):
        mask = utils.random_bitmask(size)
        data = cudf.Series(np.random.randint(0, 128, size))
        column = data.set_mask(mask)
        sr = cudf.Series(column).astype(dtype)
        gdf[dtype] = sr
    pdf = gdf.to_pandas()
    pd.options.display.max_columns = int(ncols)
    pdfrepr = pdf.__repr__()
    pdfrepr = pdfrepr.replace("NaN", "null")
    pdfrepr = pdfrepr.replace("NaT", "null")
    pdfrepr = pdfrepr.replace("-1", "null")
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
    pdf = pd.DataFrame(np.random.randint(0, 100, (size, size))).astype(dtype)
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
    pdf = pd.DataFrame(np.random.randint(0, 100, (size, size))).astype(dtype)
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
            "a": np.random.randint(0, 4, 10),
            "b": np.random.randint(0, 4, 10),
            "c": np.random.randint(0, 2, 10),
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
