# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import xgboost as xgb
from sklearn.datasets import make_regression
from xgboost.testing import IteratorForTest, make_categorical

n_samples = 128
n_features = 16


def xgboost_assert_equal(expect, got, rtol: float = 1e-7, atol: float = 0.0):
    if isinstance(expect, (tuple, list)):
        assert len(expect) == len(got)
        for e, g in zip(expect, got):
            xgboost_assert_equal(e, g, rtol, atol)
    elif isinstance(expect, scipy.sparse.csr_matrix):
        np.testing.assert_allclose(expect.data, got.data, rtol=rtol, atol=atol)
        np.testing.assert_equal(expect.indices, got.indices)
        np.testing.assert_equal(expect.indptr, got.indptr)
    else:
        pd._testing.assert_almost_equal(expect, got, rtol=rtol, atol=atol)


pytestmark = pytest.mark.assert_eq(fn=xgboost_assert_equal)


@pytest.fixture
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples, n_features, random_state=11)
    return X, y


@pytest.fixture
def reg_batches_data() -> tuple[list[pd.DataFrame], list[pd.Series]]:
    cov = []
    res = []
    for i in range(3):
        X, y = make_regression(n_samples, n_features, random_state=i + 1)
        cov.append(pd.DataFrame(X))
        res.append(pd.Series(y))
    return cov, res


def test_with_dmatrix(
    reg_data: tuple[np.ndarray, np.ndarray],
) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """DMatrix is the primary interface for XGBoost."""
    X, y = reg_data
    X_df = pd.DataFrame(X)
    y_ser = pd.Series(y)
    Xy = xgb.DMatrix(X_df, y_ser)
    assert Xy.feature_names == list(map(str, X_df.columns))
    csr_0 = Xy.get_data()

    Xc, yc = make_categorical(
        n_samples, n_features, n_categories=13, onehot=False
    )
    Xy = xgb.DMatrix(Xc, yc, enable_categorical=True)
    csr_1 = Xy.get_data()
    return csr_0, csr_1


def test_with_quantile_dmatrix(
    reg_data: tuple[np.ndarray, np.ndarray],
) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """QuantileDMatrix is an optimization for the `hist` tree method for XGBoost."""
    from xgboost.testing.data import memory

    memory.clear(warn=False)

    X, y = reg_data
    X_df = pd.DataFrame(X)
    y_ser = pd.Series(y)
    Xy = xgb.QuantileDMatrix(X_df, y_ser)
    assert Xy.feature_names == list(map(str, X_df.columns))
    csr_0 = Xy.get_data()

    Xc, yc = make_categorical(
        n_samples, n_features, n_categories=13, onehot=False
    )
    Xy = xgb.QuantileDMatrix(Xc, yc, enable_categorical=True)
    csr_1 = Xy.get_data()
    return csr_0, csr_1


def test_with_iter_quantile_dmatrix(
    reg_batches_data: tuple[list[pd.DataFrame], list[pd.DataFrame]],
) -> scipy.sparse.csr_matrix:
    """Using iterator to initialize QuantileDMatrix."""
    cov, res = reg_batches_data
    it = IteratorForTest(cov, res, w=None, cache=None)
    Xy = xgb.QuantileDMatrix(it)
    csr = Xy.get_data()
    return csr


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_with_external_memory(
    device: str,
    reg_batches_data: tuple[list[pd.DataFrame], list[pd.DataFrame]],
) -> np.ndarray:
    """Test with iterator-based external memory."""
    cov, res = reg_batches_data
    it = IteratorForTest(cov, res, w=None, cache="cache")
    Xy = xgb.DMatrix(it)
    predt = xgb.train({"device": device}, Xy, num_boost_round=1).predict(Xy)
    return predt


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_predict(device: str) -> np.ndarray:
    reg = xgb.XGBRegressor(n_estimators=2, device=device)
    X, y = make_regression(n_samples, n_features, random_state=11)
    X_df = pd.DataFrame(X)
    reg.fit(X_df, y)
    booster = reg.get_booster()

    predt0 = reg.predict(X_df)

    predt1 = booster.inplace_predict(X_df)
    # After https://github.com/dmlc/xgboost/pull/11014, .inplace_predict()
    # returns a real cupy array when called on a cudf.pandas proxy dataframe.
    # So we need to ensure we have a valid numpy array.
    if not isinstance(predt1, np.ndarray):
        predt1 = predt1.get()
    np.testing.assert_allclose(predt0, predt1)

    predt2 = booster.predict(xgb.DMatrix(X_df))
    np.testing.assert_allclose(predt0, predt2)

    predt3 = booster.inplace_predict(X)
    np.testing.assert_allclose(predt0, predt3)

    return predt0
