# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import cuml.experimental.accel
import pandas as pd

cuml.experimental.accel.install()
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.datasets import make_blobs  # noqa: E402


def test_workflow():
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)
    df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])

    # Contrived cudf-Pandas  work
    join_data = pd.DataFrame(
        {
            "id": range(len(df)),
            "Feature_1": [f"Group_{i % 3}" for i in range(len(df))],
        }
    )
    df["id"] = range(len(df))
    df = df.merge(join_data, on="id", suffixes=("", "_joined"))
    df.drop(columns=["id"], inplace=True)

    # Fit with cuML
    kmeans = KMeans(n_clusters=3, random_state=42)

    kmeans.fit(df[["Feature_2"]])  # Use only numerical data for clustering

    df["labels"] = kmeans.labels_
    assert any("cuml." in str(_) for _ in type(kmeans).mro())
    # confirm cupy array
    assert hasattr(df.labels.values, "__cuda_array_interface__")
