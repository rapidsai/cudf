# Copyright (c) 2019, NVIDIA CORPORATION.

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

import cudf


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
def test_float_dataframe(x):
    gdf = cudf.DataFrame({"x": cudf.Series(x, nan_as_null=False)})
    pdf = gdf.to_pandas()
    assert gdf.__repr__() == pdf.__repr__()


@given(st.lists(st.floats()))
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
        {
            "a": np.random.randint(0, 4, 10),
            "b": np.random.randint(0, 4, 10),
            "c": np.random.randint(0, 4, 10),
        }
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["a", "b"]).count()
    pdg = pdf.groupby(["a", "b"]).count()
    pd.options.display.max_rows = nrows
    pd.options.display.max_columns = ncols
    assert gdg.__repr__() == pdg.__repr__()
    assert gdg.T.__repr__() == pdg.T.__repr__()
