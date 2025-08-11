# Copyright (c) 2018-2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest
from numba import cuda

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_groupby_results_equal


@pytest.fixture(params=["cudf", "jit"])
def engine(request):
    return request.param


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Include groups missing on old versions of pandas",
)
def test_groupby_as_index_apply(as_index, engine):
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = gdf.groupby("y", as_index=as_index).apply(
        lambda df: df["x"].mean(), engine=engine
    )
    kwargs = {"func": lambda df: df["x"].mean(), "include_groups": False}
    pdf = pdf.groupby("y", as_index=as_index).apply(**kwargs)
    assert_groupby_results_equal(pdf, gdf, as_index=as_index, by="y")


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply():
    rng = np.random.default_rng(seed=0)
    nelem = 20
    df = cudf.DataFrame(
        {
            "key1": rng.integers(0, 3, nelem),
            "key2": rng.integers(0, 2, nelem),
            "val1": rng.random(nelem),
            "val2": rng.random(nelem),
        }
    )

    expect_grpby = df.to_pandas().groupby(
        ["key1", "key2"], as_index=False, group_keys=False
    )
    got_grpby = df.groupby(["key1", "key2"])

    def foo(df):
        df["out"] = df["val1"] + df["val2"]
        return df

    expect = expect_grpby.apply(foo, include_groups=False)
    got = got_grpby.apply(foo, include_groups=False)
    assert_groupby_results_equal(expect, got)


def f1(df, k):
    df["out"] = df["val1"] + df["val2"] + k
    return df


def f2(df, k, L):
    df["out"] = df["val1"] - df["val2"] + (k / L)
    return df


def f3(df, k, L, m):
    df["out"] = ((k * df["val1"]) + (L * df["val2"])) / m
    return df


@pytest.mark.parametrize(
    "func,args", [(f1, (42,)), (f2, (42, 119)), (f3, (42, 119, 212.1))]
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_args(func, args):
    rng = np.random.default_rng(seed=0)
    nelem = 20
    df = cudf.DataFrame(
        {
            "key1": rng.integers(0, 3, nelem),
            "key2": rng.integers(0, 2, nelem),
            "val1": rng.random(nelem),
            "val2": rng.random(nelem),
        }
    )

    expect_grpby = df.to_pandas().groupby(
        ["key1", "key2"], as_index=False, group_keys=False
    )
    got_grpby = df.groupby(["key1", "key2"])
    expect = expect_grpby.apply(func, *args, include_groups=False)
    got = got_grpby.apply(func, *args, include_groups=False)
    assert_groupby_results_equal(expect, got)


def test_groupby_apply_grouped():
    df = cudf.DataFrame(
        {
            "key1": range(20),
            "key2": range(20),
            "val1": range(20),
            "val2": range(20),
        }
    )

    got_grpby = df.groupby(["key1", "key2"])

    def foo(key1, val1, com1, com2):
        for i in range(cuda.threadIdx.x, len(key1), cuda.blockDim.x):
            com1[i] = key1[i] * 10000 + val1[i]
            com2[i] = i

    got = got_grpby.apply_grouped(
        foo,
        incols=["key1", "val1"],
        outcols={"com1": np.float64, "com2": np.int32},
        tpb=8,
    )

    got = got.to_pandas()

    expect = df.copy()
    expect["com1"] = (expect["key1"] * 10000 + expect["key1"]).astype(
        np.float64
    )
    expect["com2"] = np.zeros(20, dtype=np.int32)

    assert_groupby_results_equal(expect, got)
