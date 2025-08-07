# Copyright (c) 2025, NVIDIA CORPORATION.


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_single_q():
    q = 0.5

    pdf = pd.DataFrame({"a": [4, 24, 13, 8, 7]})
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantile(q, interpolation="nearest", method="table")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


def test_with_index():
    q = [0, 0.5, 1]

    pdf = pd.DataFrame({"a": [7, 4, 4, 9, 13]}, index=[0, 4, 3, 2, 7])
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantile(q, interpolation="nearest", method="table")

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
    gdf_q = gdf.quantile(q, interpolation="nearest", method="table")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [10, 11, 12]},
        {"a": [1, 0, 3], "b": [10, 11, 12]},
        {"a": [1, 2, 3], "b": [10, 11, None]},
        {
            "a": [],
        },
        {},
    ],
)
@pytest.mark.parametrize("op", ["all", "any"])
def test_any_all_axis_none(data, op):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    expected = getattr(pdf, op)(axis=None)
    actual = getattr(gdf, op)(axis=None)

    assert expected == actual
