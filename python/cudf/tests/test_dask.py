# Copyright (c) 2019, NVIDIA CORPORATION.

import pytest
import cudf

dask = pytest.importorskip('dask')

from dask.dataframe.utils import (
    is_dataframe_like, is_series_like, is_index_like
)  # noqa: E402


def test_is_dataframe_like():
    df = cudf.DataFrame({'x': [1, 2, 3]})
    assert is_dataframe_like(df)
    assert is_series_like(df.x)
    assert is_index_like(df.index)
    assert not is_dataframe_like(df.x)
    assert not is_series_like(df)
    assert not is_index_like(df)
