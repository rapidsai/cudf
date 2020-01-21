import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq

_data = [
    {"a": [0, 1.0, 2.0, None, np.nan, None, 3, 5]},
    {"a": [0, 1.0, 2.0, None, 3, np.nan, None, 4]},
    {"a": [0, 1.0, 2.0, None, 3, np.nan, None, 4, None, 9]},
]
_queries = [
    "a == 3",
    # "a != 3", # incompatible with pandas
    "a < 3",
    "a <= 3",
    "a < 3",
    "a >= 3",
]


@pytest.mark.parametrize("data", _data)
@pytest.mark.parametrize("query", _queries)
def test_mask_0(data, query):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    pdf_q_res = pdf.query(query)
    gdf_q_res = gdf.query(query)

    assert_eq(pdf_q_res, gdf_q_res)


@pytest.mark.parametrize("data", _data)
@pytest.mark.parametrize("nan_as_null", [False, True])
@pytest.mark.parametrize("query", _queries)
def test_mask_1(data, nan_as_null, query):
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame.from_pandas(pdf, nan_as_null=nan_as_null)

    pdf_q_res = pdf.query(query)
    gdf_q_res = gdf.query(query)

    assert_eq(pdf_q_res, gdf_q_res)


@pytest.mark.parametrize("data", _data)
@pytest.mark.parametrize("query", _queries)
def test_mask_2(data, query):
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    pdf_q_res = pdf.query(query)
    gdf_q_res = gdf.query(query)

    assert_eq(pdf_q_res, gdf_q_res)


@pytest.mark.parametrize("data", _data)
@pytest.mark.parametrize("query", _queries)
def test_dataframe_initializer(data, query):
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    pdf_q_res = pdf.query(query)
    gdf_q_res = gdf.query(query)

    assert_eq(pdf_q_res, gdf_q_res)
