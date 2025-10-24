# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def test_regression():
    data = {
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [2, 4, 1, 3, 5, 7, 6, 8, 10, 9],
        "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    }
    df = pd.DataFrame(data)

    X = df[["feature1", "feature2"]]
    y = df["target"]

    # Data Splitting
    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Basic deterministic LR model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # predction phase
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


@pytest.mark.assert_eq(fn=np.testing.assert_allclose)
def test_clustering():
    rng = np.random.default_rng(42)
    nsamps = 300
    X = rng.random((nsamps, 2))
    data = pd.DataFrame(X, columns=["x", "y"])

    # Create and fit a KMeans clustering model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data)
    return kmeans.cluster_centers_


def test_feature_selection():
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 10

    X = rng.random((n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)

    data = pd.DataFrame(
        X, columns=[f"feature{i}" for i in range(1, n_features + 1)]
    )
    data["target"] = y

    # Select the top k features
    k_best = SelectKBest(score_func=f_classif, k=5)
    k_best.fit_transform(X, y)

    feat_inds = k_best.get_support(indices=True)
    features = data.iloc[:, feat_inds]

    return sorted(features.columns.tolist())


@pytest.mark.assert_eq(fn=np.testing.assert_allclose)
def test_data_scaling():
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaled_data
