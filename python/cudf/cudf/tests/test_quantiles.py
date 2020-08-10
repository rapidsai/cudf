import pandas as pd

import cudf
from cudf.tests.utils import assert_eq


def test_single_q():
    q = 0.5

    pdf = pd.DataFrame({"a": [4, 24, 13, 8, 7]}, dtype=pd.Int64Dtype())
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantiles(q, interpolation="nearest")

    pdf_q = pdf_q.astype(pd.Int64Dtype())
    assert_eq(pdf_q, gdf_q, check_index_type=False)


def test_with_index():
    q = [0, 0.5, 1]

    pdf = pd.DataFrame({"a": [7, 4, 4, 9, 13]}, index=[0, 4, 3, 2, 7], dtype=pd.Int64Dtype())
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantiles(q, interpolation="nearest")
    pdf_q = pdf_q.astype(pd.Int64Dtype())
    assert_eq(pdf_q, gdf_q, check_index_type=False)


def test_with_multiindex():
    q = [0, 0.5, 1]

    pdf = pd.DataFrame(
        {
            "index_1": [3, 1, 9, 7, 5],
            "index_2": [2, 4, 3, 5, 1],
            "a": [8, 4, 2, 3, 8],
        }, dtype=pd.Int64Dtype()
    )
    pdf.set_index(["index_1", "index_2"], inplace=True)

    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantiles(q, interpolation="nearest")
    pdf_q = pdf_q.astype(pd.Int64Dtype())
    assert_eq(pdf_q, gdf_q, check_index_type=False)
