# Copyright (c) 2024, NVIDIA CORPORATION.

import pytest

import cudf
import cudf.core.lazy
from cudf.pandas.fast_slow_proxy import _State
from cudf.testing._utils import assert_eq

dask_cudf = pytest.importorskip("dask_cudf")


def test_read_parquet(tmpdir):
    path = tmpdir.join("test_read_parquet.parquet")
    expect = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expect.to_parquet(path)

    df = cudf.read_parquet(path, lazy=True)
    assert isinstance(df, cudf.core.lazy.DataFrame)
    assert df._fsproxy_state is _State.FAST
    assert_eq(df, expect)
    assert df._fsproxy_state is _State.SLOW
