import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


def test_single_q():
    q = 0.5

    pdf = pd.DataFrame({"a": [4, 24, 13, 8, 7]})
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantiles(q, interpolation="nearest")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


def test_single_q_presorted():
    q = 0.5

    pdf = pd.DataFrame({"a": [4, 7, 8, 13, 24]})
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantiles(q, interpolation="nearest")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


def test_with_index():
    q = [0, 0.5, 1]

    pdf = pd.DataFrame({"a": [7, 4, 4, 9, 13]}, index=[0, 4, 3, 2, 7])
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantiles(q, interpolation="nearest")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


def test_with_multiindex():
    q = [0, 0.5, 1]

    pdf = pd.DataFrame(
        {
            "index_1": [3, 1, 9, 7, 5],
            "index_2": [2, 4, 3, 5, 1],
            "a": [8, 4, 2, 3, 8],
        }
    )
    pdf.set_index(["index_1", "index_2"], inplace=True)

    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantiles(q, interpolation="nearest")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


def test_dataframe_quantile():
    gdf = cudf.DataFrame({"x": [1, 2, 3]})
    pdf = gdf.to_pandas()

    expected = pdf.quantile()
    actual = gdf.quantile()

    assert_eq(expected, actual)


def test_series_quantile():
    gdf = cudf.Series([1, 2, 3])
    pdf = gdf.to_pandas()

    expected = pdf.quantile()
    actual = gdf.quantile()

    assert_eq(expected, actual)


@pytest.mark.parametrize("q", [0, 0.5, 1, [0, 0.5, 1]])
def test_series_quantile_multi(q):
    gdf = cudf.Series([1, 2, 3])
    pdf = gdf.to_pandas()

    expected = pdf.quantile(q)
    actual = gdf.quantile(q)

    assert_eq(expected, actual)


@pytest.mark.parametrize("q", [-1, 2, [-1], [2]])
@pytest.mark.parametrize("FrameType", [cudf.DataFrame, cudf.Series])
def test_quantile_out_of_range(q, FrameType):
    g = FrameType([0, 1, 2])
    p = g.to_pandas()

    with pytest.raises(ValueError):
        p.quantile(q)

    with pytest.raises(ValueError):
        g.quantile(q)
