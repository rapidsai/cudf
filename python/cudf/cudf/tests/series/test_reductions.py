# Copyright (c) 2019-2025, NVIDIA CORPORATION.

import re

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("data", [[], [1, 2, 3]])
def test_series_pandas_methods(data, reduction_methods):
    arr = np.array(data)
    sr = cudf.Series(arr)
    psr = pd.Series(arr)
    np.testing.assert_equal(
        getattr(sr, reduction_methods)(), getattr(psr, reduction_methods)()
    )


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


@pytest.mark.parametrize("val", [0.9, float("nan")])
def test_quantile_ignore_nans(val):
    data = [float("nan"), float("nan"), val]
    psr = pd.Series(data)
    gsr = cudf.Series(data, nan_as_null=False)

    expected = gsr.quantile(0.9)
    result = psr.quantile(0.9)
    assert_eq(result, expected)
