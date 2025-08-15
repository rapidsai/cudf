# Copyright (c) 2025, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_cudf_factorize_array():
    data = [1, 2, 3, 4, 5]

    parr = np.array(data)
    garr = cp.array(data)

    expect = pd.factorize(parr)
    got = cudf.factorize(garr)

    assert len(expect) == len(got)

    np.testing.assert_array_equal(expect[0], got[0].get())
    np.testing.assert_array_equal(expect[1], got[1].get())


@pytest.mark.parametrize("pandas_compatibility", [True, False])
def test_factorize_code_pandas_compatibility(pandas_compatibility):
    psr = pd.Series([1, 2, 3, 4, 5])
    gsr = cudf.from_pandas(psr)

    expect = pd.factorize(psr)
    with cudf.option_context("mode.pandas_compatible", pandas_compatibility):
        got = cudf.factorize(gsr)
    assert_eq(got[0], expect[0])
    assert_eq(got[1], expect[1])
    if pandas_compatibility:
        assert got[0].dtype == expect[0].dtype
    else:
        assert got[0].dtype == cudf.dtype("int8")


def test_factorize_result_classes():
    data = [1, 2, 3]

    labels, cats = cudf.factorize(cudf.Series(data))

    assert isinstance(labels, cp.ndarray)
    assert isinstance(cats, cudf.Index)

    labels, cats = cudf.factorize(cudf.Index(data))

    assert isinstance(labels, cp.ndarray)
    assert isinstance(cats, cudf.Index)

    labels, cats = cudf.factorize(cp.array(data))

    assert isinstance(labels, cp.ndarray)
    assert isinstance(cats, cp.ndarray)
