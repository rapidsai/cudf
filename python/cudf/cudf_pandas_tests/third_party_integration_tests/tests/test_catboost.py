# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.datasets import make_classification, make_regression

rng = np.random.default_rng(seed=42)


def assert_catboost_equal(expect, got, rtol=1e-7, atol=0.0):
    if isinstance(expect, (tuple, list)):
        assert len(expect) == len(got)
        for e, g in zip(expect, got, strict=True):
            assert_catboost_equal(e, g, rtol, atol)
    elif isinstance(expect, np.ndarray):
        np.testing.assert_allclose(expect, got, rtol=rtol, atol=atol)
    elif isinstance(expect, pd.DataFrame):
        pd.testing.assert_frame_equal(expect, got)
    elif isinstance(expect, pd.Series):
        pd.testing.assert_series_equal(expect, got)
    else:
        assert expect == got


pytestmark = pytest.mark.assert_eq(fn=assert_catboost_equal)


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    return pd.DataFrame(X), pd.Series(y)


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    categorical_feature = rng.choice(["0", "1", "2", "3"], size=(1000, 2))
    X = np.hstack([X, categorical_feature])
    df = pd.DataFrame(X)
    df.iloc[:, -2] = df.iloc[:, -2].astype("category")
    return df, pd.Series(y)


def test_catboost_regressor_with_dataframe(regression_data):
    X, y = regression_data
    model = CatBoostRegressor(iterations=50, verbose=0, random_seed=42)
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions


def test_catboost_regressor_with_numpy(regression_data):
    X, y = regression_data
    model = CatBoostRegressor(iterations=50, verbose=0, random_seed=42)
    model.fit(X.values, y.values)
    predictions = model.predict(X.values)
    return predictions


def test_catboost_classifier_with_dataframe(classification_data):
    X, y = classification_data
    cat_features = [X.columns[-2]]
    model = CatBoostClassifier(
        iterations=50, verbose=0, cat_features=cat_features, random_seed=42
    )
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions


def test_catboost_classifier_with_numpy(classification_data):
    X, y = classification_data
    model = CatBoostClassifier(iterations=50, verbose=0, random_seed=42)
    model.fit(X.values, y.values)
    predictions = model.predict(X.values)
    return predictions


def test_catboost_with_pool_and_dataframe(regression_data):
    X, y = regression_data
    train_pool = Pool(X, y)
    model = CatBoostRegressor(iterations=50, verbose=0, random_seed=42)
    model.fit(train_pool)
    predictions = model.predict(X)
    return predictions


def test_catboost_with_pool_and_numpy(regression_data):
    X, y = regression_data
    train_pool = Pool(X.values, y.values)
    model = CatBoostRegressor(iterations=50, verbose=0, random_seed=42)
    model.fit(train_pool)
    predictions = model.predict(X.values)
    return predictions


def test_catboost_with_categorical_features():
    data = {
        "feat_int_dtype": rng.standard_normal(1000),
        "cat_feat_int_dtype": rng.choice([0, 1, 2, 3], size=1000),
        "cat_feat_categorical_dtype": rng.choice(["A", "B", "C"], size=1000),
        "target": rng.integers(0, 2, size=1000),
    }
    df = pd.DataFrame(data)
    df["cat_feat_categorical_dtype"] = df["cat_feat_categorical_dtype"].astype(
        "category"
    )
    X = df[
        ["feat_int_dtype", "cat_feat_int_dtype", "cat_feat_categorical_dtype"]
    ]
    y = df["target"]
    cat_features = ["cat_feat_int_dtype", "cat_feat_categorical_dtype"]
    model = CatBoostClassifier(
        iterations=50, verbose=0, cat_features=cat_features, random_seed=42
    )
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions


@pytest.mark.parametrize(
    "X, y",
    [
        (
            pd.DataFrame(rng.standard_normal((1000, 5))),
            pd.Series(rng.standard_normal(1000)),
        ),
        (rng.standard_normal((1000, 5)), rng.standard_normal(1000)),
    ],
)
def test_catboost_train_test_split(X, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = CatBoostRegressor(iterations=50, verbose=0, random_seed=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return len(X_train), len(X_test), len(y_train), len(y_test), predictions
