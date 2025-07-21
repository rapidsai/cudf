# Copyright (c) 2018-2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture
def rng():
    return np.random.default_rng(seed=0)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.mean(x),
        lambda x: np.sum(x),
        lambda x: np.var(x, ddof=1),
        lambda x: np.unique(x),
        lambda x: np.dot(x, x),
        lambda x: np.linalg.norm(x),
    ],
)
def test_array_func_cudf_series(func, rng):
    np_ar = rng.random(100)
    cudf_ser = cudf.Series(np_ar)
    expect = func(np_ar)
    got = func(cudf_ser)
    if np.isscalar(expect):
        assert_eq(expect, got)
    else:
        assert_eq(expect, got.to_numpy())


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.mean(x, axis=0),
        lambda x: np.sum(x, axis=0),
        lambda x: np.var(x, ddof=1, axis=0),
        lambda x: np.dot(x, x.transpose()),
        lambda x: np.all(x),
        lambda x: np.any(x),
        lambda x: np.prod(x, axis=0),
        lambda x: np.prod(x, axis=1),
    ],
)
def test_array_func_cudf_dataframe(func, rng):
    pd_df = pd.DataFrame(rng.uniform(size=(100, 10)))
    cudf_df = cudf.from_pandas(pd_df)
    expect = func(pd_df)
    got = func(cudf_df)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.cov(x, x),
        lambda x: np.linalg.norm(x),
        lambda x: np.linalg.det(x),
    ],
)
def test_array_func_missing_cudf_dataframe(func, rng):
    pd_df = pd.DataFrame(rng.uniform(size=(100, 10)))
    cudf_df = cudf.from_pandas(pd_df)
    with pytest.raises(TypeError):
        func(cudf_df)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.unique(x),
    ],
)
def test_array_func_cudf_index(func, rng):
    np_ar = rng.random(100)
    cudf_index = cudf.Index(cudf.Series(np_ar))
    expect = func(np_ar)
    got = func(cudf_index)
    if np.isscalar(expect):
        assert_eq(expect, got)
    else:
        assert_eq(expect, got.to_numpy())


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.cov(x, x),
        lambda x: np.linalg.norm(x),
        lambda x: np.linalg.det(x),
    ],
)
def test_array_func_missing_cudf_index(func, rng):
    np_ar = rng.random(100)
    cudf_index = cudf.Index(cudf.Series(np_ar))
    with pytest.raises(TypeError):
        func(cudf_index)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.cov(x, x),
        lambda x: np.dot(x, x),
        lambda x: np.linalg.norm(x),
        lambda x: np.linalg.det(x),
    ],
)
def test_array_func_missing_cudf_multi_index(func):
    levels = [["a", "b"], ["c", "d"]]
    codes = [[0, 1], [1, 0]]

    cudf_multi_index = cudf.MultiIndex(levels, codes)
    with pytest.raises(TypeError):
        func(cudf_multi_index)


def test_list_input_array_func():
    ar = np.array([1, 2, 3])

    s = cudf.Series(ar)
    with pytest.raises(TypeError):
        np.concatenate([s, s, s])

    s = cudf.Series(ar, index=[1, 2, 3])
    with pytest.raises(TypeError):
        np.concatenate([s, s, s])
