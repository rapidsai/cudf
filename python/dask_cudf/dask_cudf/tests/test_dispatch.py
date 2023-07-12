# Copyright (c) 2021-2023, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest
from packaging import version

import dask
from dask.base import tokenize
from dask.dataframe import assert_eq
from dask.dataframe.methods import is_categorical_dtype

import cudf


def test_is_categorical_dispatch():
    assert is_categorical_dtype(pd.CategoricalDtype([1, 2, 3]))
    assert is_categorical_dtype(cudf.CategoricalDtype([1, 2, 3]))

    assert is_categorical_dtype(cudf.Series([1, 2, 3], dtype="category"))
    assert is_categorical_dtype(pd.Series([1, 2, 3], dtype="category"))

    assert is_categorical_dtype(pd.Index([1, 2, 3], dtype="category"))
    assert is_categorical_dtype(cudf.Index([1, 2, 3], dtype="category"))


@pytest.mark.skipif(
    version.parse(dask.__version__) <= version.parse("2023.6.0"),
    reason="Pyarrow-conversion dispatch requires dask>2023.6.0",
)
def test_pyarrow_conversion_dispatch():
    from dask.dataframe.dispatch import (
        from_pyarrow_table_dispatch,
        to_pyarrow_table_dispatch,
    )

    df1 = cudf.DataFrame(np.random.randn(10, 3), columns=list("abc"))
    df2 = from_pyarrow_table_dispatch(df1, to_pyarrow_table_dispatch(df1))

    assert type(df1) == type(df2)
    assert_eq(df1, df2)


def test_deterministic_tokenize():
    # Checks that `dask.base.normalize_token` correctly
    # dispatches to the logic defined in `backends.py`
    # when "tokenize.ensure-deterministic" == `True`
    # (making `tokenize(<cudf-data>)` deterministic).
    df = cudf.DataFrame({"A": list(range(10)), "B": ["dog", "cat"] * 5})

    with dask.config.set({"tokenize.ensure-deterministic": True}):
        assert tokenize(df) == tokenize(df)
        assert tokenize(df.A) == tokenize(df.A)
        assert tokenize(df.index) == tokenize(df.index)
