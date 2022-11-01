# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import re

import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq, assert_exceptions_equal


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


@pytest.mark.parametrize("q", [2, [1, 2, 3]])
def test_quantile_range_error(q):
    ps = pd.Series([1, 2, 3])
    gs = cudf.from_pandas(ps)
    assert_exceptions_equal(
        lfunc=ps.quantile,
        rfunc=gs.quantile,
        lfunc_args_and_kwargs=([q],),
        rfunc_args_and_kwargs=([q],),
        expected_error_message=re.escape(
            "percentiles should all be in the interval [0, 1]"
        ),
    )


def test_quantile_q_type():
    gs = cudf.Series([1, 2, 3])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "q must be a scalar or array-like, got <class "
            "'cudf.core.dataframe.DataFrame'>"
        ),
    ):
        gs.quantile(cudf.DataFrame())
