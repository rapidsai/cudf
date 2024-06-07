# Copyright (c) 2024, NVIDIA CORPORATION.

import pytest

import cudf
import cudf.core.lazy
from cudf.core.lazy import lazy_wrap_dataframe
from cudf.pandas.fast_slow_proxy import _State
from cudf.testing._utils import assert_eq

pytest.importorskip("dask_cudf")


def test_read_parquet(tmpdir):
    path = tmpdir.join("test_read_parquet.parquet")
    expect = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expect.to_parquet(path)

    df = cudf.read_parquet(path, lazy=True)
    assert isinstance(df, cudf.core.lazy.DataFrame)
    assert df._fsproxy_state is _State.FAST
    assert_eq(df, expect)
    assert df._fsproxy_state is _State.SLOW


def test_merge():
    df1 = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df2 = cudf.DataFrame({"a": [3, 2, 2], "b": [4, 5, 6]})
    lazy_df1 = lazy_wrap_dataframe(df1, noop_on_error=False)
    lazy_df2 = lazy_wrap_dataframe(df2, noop_on_error=False)
    assert isinstance(lazy_df1, cudf.core.lazy.DataFrame)
    assert isinstance(lazy_df2, cudf.core.lazy.DataFrame)
    expect = df1.merge(df2, on="a")
    got = lazy_df1.merge(lazy_df2, on="a")
    assert lazy_df1._fsproxy_state is _State.FAST
    assert lazy_df2._fsproxy_state is _State.FAST
    assert got._fsproxy_state is _State.FAST
    assert_eq(got, expect)
    assert got._fsproxy_state is _State.SLOW


def test_groupby_and_sum():
    df = cudf.DataFrame({"a": [1, 2, 2, 3, 3, 4]})
    lazy_df = lazy_wrap_dataframe(df, noop_on_error=False)
    assert isinstance(lazy_df, cudf.core.lazy.DataFrame)
    expect = df.groupby(by="a").sum()
    got = lazy_df.groupby(by="a").sum()
    assert got._fsproxy_state is _State.FAST
    assert_eq(got, expect)
    assert got._fsproxy_state is _State.SLOW
