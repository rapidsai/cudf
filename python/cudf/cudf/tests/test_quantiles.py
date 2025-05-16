# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import re

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


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


@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
)
def test_quantile_type_int_float(interpolation):
    data = [1, 3, 4]
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    expected = psr.quantile(0.5, interpolation=interpolation)
    actual = gsr.quantile(0.5, interpolation=interpolation)

    assert expected == actual
    assert type(expected) is type(actual)


@pytest.mark.parametrize(
    "data",
    [
        [float("nan"), float("nan"), 0.9],
        [float("nan"), float("nan"), float("nan")],
    ],
)
def test_ignore_nans(data):
    psr = pd.Series(data)
    gsr = cudf.Series(data, nan_as_null=False)

    expected = gsr.quantile(0.9)
    result = psr.quantile(0.9)
    assert_eq(result, expected)
