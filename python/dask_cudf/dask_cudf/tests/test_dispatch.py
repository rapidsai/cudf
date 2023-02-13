# Copyright (c) 2021-2022, NVIDIA CORPORATION.

import pandas as pd

from dask.dataframe.methods import is_categorical_dtype

import cudf


def test_is_categorical_dispatch():
    assert is_categorical_dtype(pd.CategoricalDtype([1, 2, 3]))
    assert is_categorical_dtype(cudf.CategoricalDtype([1, 2, 3]))

    assert is_categorical_dtype(cudf.Series([1, 2, 3], dtype="category"))
    assert is_categorical_dtype(pd.Series([1, 2, 3], dtype="category"))

    assert is_categorical_dtype(pd.Index([1, 2, 3], dtype="category"))
    assert is_categorical_dtype(cudf.Index([1, 2, 3], dtype="category"))
