# Copyright (c) 2025, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_create_interval_index_from_list():
    interval_list = [
        np.nan,
        pd.Interval(2.0, 3.0, closed="right"),
        pd.Interval(3.0, 4.0, closed="right"),
    ]

    expected = pd.Index(interval_list)
    actual = cudf.Index(interval_list)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
def test_infer_timedelta_index(data, timedelta_types_as_str):
    gdi = cudf.Index(data, dtype=timedelta_types_as_str)
    pdi = gdi.to_pandas()

    assert_eq(pdi, gdi)


def test_categorical_index_with_dtype():
    dtype = cudf.CategoricalDtype(categories=["a", "z", "c"])
    gi = cudf.Index(["z", "c", "a"], dtype=dtype)
    pi = pd.Index(["z", "c", "a"], dtype=dtype.to_pandas())

    assert_eq(gi, pi)
    assert_eq(gi.dtype, pi.dtype)
    assert_eq(gi.dtype.categories, pi.dtype.categories)
