# Copyright (c) 2021-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("preserve_index", [True, False])
def test_pyarrow_conversion_dispatch(preserve_index):
    from dask.dataframe.dispatch import (
        from_pyarrow_table_dispatch,
        to_pyarrow_table_dispatch,
    )

    df1 = cudf.DataFrame(np.random.randn(10, 3), columns=list("abc"))
    df2 = from_pyarrow_table_dispatch(
        df1, to_pyarrow_table_dispatch(df1, preserve_index=preserve_index)
    )

    assert type(df1) == type(df2)
    assert_eq(df1, df2)

    # Check that preserve_index does not produce a RangeIndex
    if preserve_index:
        assert not isinstance(df2.index, cudf.RangeIndex)


@pytest.mark.parametrize("preserve_index", [True, False])
def test_pyarrow_schema_dispatch(preserve_index):
    from dask.dataframe.dispatch import (
        pyarrow_schema_dispatch,
        to_pyarrow_table_dispatch,
    )

    df = cudf.DataFrame(np.random.randn(10, 3), columns=list("abc"))
    df["d"] = cudf.Series(["cat", "dog"] * 5)
    table = to_pyarrow_table_dispatch(df, preserve_index=preserve_index)
    schema = pyarrow_schema_dispatch(df, preserve_index=preserve_index)

    assert schema.equals(table.schema)
