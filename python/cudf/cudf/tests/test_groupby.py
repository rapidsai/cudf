# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import collections
import datetime
import itertools
import operator
import string
import textwrap
from decimal import Decimal
from functools import partial

import numpy as np
import pandas as pd
import pytest
from numba import cuda
from numpy.testing import assert_array_equal

import rmm

import cudf
from cudf import DataFrame, Series
from cudf.api.extensions import no_default
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_GE_220,
    PANDAS_VERSION,
)
from cudf.core.udf._ops import arith_ops, comparison_ops, unary_ops
from cudf.core.udf.groupby_typing import SUPPORTED_GROUPBY_NUMPY_TYPES
from cudf.core.udf.utils import UDFError, precompiled
from cudf.testing import assert_eq
from cudf.testing._utils import (
    DATETIME_TYPES,
    SIGNED_TYPES,
    TIMEDELTA_TYPES,
    assert_exceptions_equal,
    expect_warning_if,
)
from cudf.testing.dataset_generator import rand_dataframe

_now = np.datetime64("now")
_tomorrow = _now + np.timedelta64(1, "D")
_now = np.int64(_now.astype("datetime64[ns]"))
_tomorrow = np.int64(_tomorrow.astype("datetime64[ns]"))
_index_type_aggs = {"count", "idxmin", "idxmax", "cumcount"}


def assert_groupby_results_equal(
    expect, got, sort=True, as_index=True, by=None, **kwargs
):
    # Because we don't sort by index by default in groupby,
    # sort expect and got by index before comparing.
    if sort:
        if as_index:
            expect = expect.sort_index()
            got = got.sort_index()
        else:
            assert by is not None
            if isinstance(expect, (pd.DataFrame, cudf.DataFrame)):
                expect = expect.sort_values(by=by).reset_index(drop=True)
            else:
                expect = expect.sort_values(by=by).reset_index(drop=True)

            if isinstance(got, cudf.DataFrame):
                got = got.sort_values(by=by).reset_index(drop=True)
            else:
                got = got.sort_values(by=by).reset_index(drop=True)

    assert_eq(expect, got, **kwargs)


def make_frame(
    dataframe_class,
    nelem,
    seed=0,
    extra_levels=(),
    extra_vals=(),
    with_datetime=False,
):
    rng = np.random.default_rng(seed=seed)

    df = dataframe_class()

    df["x"] = rng.integers(0, 5, nelem)
    df["y"] = rng.integers(0, 3, nelem)
    for lvl in extra_levels:
        df[lvl] = rng.integers(0, 2, nelem)

    df["val"] = rng.random(nelem)
    for val in extra_vals:
        df[val] = rng.random(nelem)

    if with_datetime:
        df["datetime"] = rng.integers(
            _now, _tomorrow, nelem, dtype=np.int64
        ).astype("datetime64[ns]")

    return df


@pytest.fixture
def gdf():
    return DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})


@pytest.fixture
def pdf(gdf):
    return gdf.to_pandas()


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
def test_groupby_mean(nelem):
    got_df = make_frame(DataFrame, nelem=nelem).groupby(["x", "y"]).mean()
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem).groupby(["x", "y"]).mean()
    )
    assert_groupby_results_equal(got_df, expect_df)


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
def test_groupby_mean_3level(nelem):
    lvls = "z"
    bys = list("xyz")
    got_df = (
        make_frame(DataFrame, nelem=nelem, extra_levels=lvls)
        .groupby(bys)
        .mean()
    )
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem, extra_levels=lvls)
        .groupby(bys)
        .mean()
    )
    assert_groupby_results_equal(got_df, expect_df)


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
def test_groupby_agg_mean_min(nelem):
    got_df = (
        make_frame(DataFrame, nelem=nelem)
        .groupby(["x", "y"])
        .agg(["mean", "min"])
    )
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem)
        .groupby(["x", "y"])
        .agg(["mean", "min"])
    )
    assert_groupby_results_equal(got_df, expect_df)


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
def test_groupby_agg_min_max_dictargs(nelem):
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem, extra_vals="ab")
        .groupby(["x", "y"])
        .agg({"a": "min", "b": "max"})
    )
    got_df = (
        make_frame(DataFrame, nelem=nelem, extra_vals="ab")
        .groupby(["x", "y"])
        .agg({"a": "min", "b": "max"})
    )
    assert_groupby_results_equal(expect_df, got_df)


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
def test_groupby_agg_min_max_dictlist(nelem):
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem, extra_vals="ab")
        .groupby(["x", "y"])
        .agg({"a": ["min", "max"], "b": ["min", "max"]})
    )
    got_df = (
        make_frame(DataFrame, nelem=nelem, extra_vals="ab")
        .groupby(["x", "y"])
        .agg({"a": ["min", "max"], "b": ["min", "max"]})
    )
    assert_groupby_results_equal(got_df, expect_df)


@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_as_index_single_agg(pdf, gdf, as_index):
    gdf = gdf.groupby("y", as_index=as_index).agg({"x": "mean"})
    pdf = pdf.groupby("y", as_index=as_index).agg({"x": "mean"})
    assert_groupby_results_equal(pdf, gdf, as_index=as_index, by="y")


@pytest.mark.parametrize("engine", ["cudf", "jit"])
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Include groups missing on old versions of pandas",
)
def test_groupby_as_index_apply(pdf, gdf, as_index, engine):
    gdf = gdf.groupby("y", as_index=as_index).apply(
        lambda df: df["x"].mean(), engine=engine
    )
    kwargs = {"func": lambda df: df["x"].mean(), "include_groups": False}
    pdf = pdf.groupby("y", as_index=as_index).apply(**kwargs)
    assert_groupby_results_equal(pdf, gdf, as_index=as_index, by="y")


@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_as_index_multiindex(pdf, gdf, as_index):
    pdf = pd.DataFrame(
        {"a": [1, 2, 1], "b": [3, 3, 3], "c": [2, 2, 3], "d": [3, 1, 2]}
    )
    gdf = cudf.from_pandas(pdf)

    gdf = gdf.groupby(["a", "b"], as_index=as_index, sort=True).agg(
        {"c": "mean"}
    )
    pdf = pdf.groupby(["a", "b"], as_index=as_index, sort=True).agg(
        {"c": "mean"}
    )

    if as_index:
        assert_eq(pdf, gdf)
    else:
        # column names don't match - check just the values
        for gcol, pcol in zip(gdf, pdf):
            assert_array_equal(gdf[gcol].to_numpy(), pdf[pcol].values)


def test_groupby_default(pdf, gdf):
    gdf = gdf.groupby("y").agg({"x": "mean"})
    pdf = pdf.groupby("y").agg({"x": "mean"})
    assert_groupby_results_equal(pdf, gdf)


def test_group_keys_true(pdf, gdf):
    gdf = gdf.groupby("y", group_keys=True).sum()
    pdf = pdf.groupby("y", group_keys=True).sum()
    assert_groupby_results_equal(pdf, gdf)


@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_getitem_getattr(as_index):
    pdf = pd.DataFrame({"x": [1, 3, 1], "y": [1, 2, 3], "z": [1, 4, 5]})
    gdf = cudf.from_pandas(pdf)
    assert_groupby_results_equal(
        pdf.groupby("x", as_index=as_index)["y"].sum(),
        gdf.groupby("x", as_index=as_index)["y"].sum(),
        as_index=as_index,
        by="x",
    )
    assert_groupby_results_equal(
        pdf.groupby("x", as_index=as_index).y.sum(),
        gdf.groupby("x", as_index=as_index).y.sum(),
        as_index=as_index,
        by="x",
    )
    assert_groupby_results_equal(
        pdf.groupby("x", as_index=as_index)[["y"]].sum(),
        gdf.groupby("x", as_index=as_index)[["y"]].sum(),
        as_index=as_index,
        by="x",
    )
    assert_groupby_results_equal(
        pdf.groupby(["x", "y"], as_index=as_index).sum(),
        gdf.groupby(["x", "y"], as_index=as_index).sum(),
        as_index=as_index,
        by=["x", "y"],
    )


def test_groupby_cats():
    rng = np.random.default_rng(seed=0)
    df = DataFrame(
        {"cats": pd.Categorical(list("aabaacaab")), "vals": rng.random(9)}
    )

    cats = df["cats"].values_host
    vals = df["vals"].to_numpy()

    grouped = df.groupby(["cats"], as_index=False).mean()

    got_vals = grouped["vals"]

    got_cats = grouped["cats"]

    for i in range(len(got_vals)):
        expect = vals[cats == got_cats[i]].mean()
        np.testing.assert_almost_equal(got_vals[i], expect)


def test_groupby_iterate_groups():
    rng = np.random.default_rng(seed=0)
    nelem = 20
    df = DataFrame(
        {
            "key1": rng.integers(0, 3, nelem),
            "key2": rng.integers(0, 2, nelem),
            "val1": rng.random(nelem),
            "val2": rng.random(nelem),
        }
    )

    def assert_values_equal(arr):
        np.testing.assert_array_equal(arr[0], arr)

    for name, grp in df.groupby(["key1", "key2"]):
        pddf = grp.to_pandas()
        for k in "key1,key2".split(","):
            assert_values_equal(pddf[k].values)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply():
    rng = np.random.default_rng(seed=0)
    nelem = 20
    df = DataFrame(
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


def create_test_groupby_apply_args_params():
    def f1(df, k):
        df["out"] = df["val1"] + df["val2"] + k
        return df

    def f2(df, k, L):
        df["out"] = df["val1"] - df["val2"] + (k / L)
        return df

    def f3(df, k, L, m):
        df["out"] = ((k * df["val1"]) + (L * df["val2"])) / m
        return df

    return [(f1, (42,)), (f2, (42, 119)), (f3, (42, 119, 212.1))]


@pytest.mark.parametrize("func,args", create_test_groupby_apply_args_params())
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_args(func, args):
    rng = np.random.default_rng(seed=0)
    nelem = 20
    df = DataFrame(
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
    df = DataFrame()
    nelem = 20
    df["key1"] = range(nelem)
    df["key2"] = range(nelem)
    df["val1"] = range(nelem)
    df["val2"] = range(nelem)

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
    expect["com2"] = np.zeros(nelem, dtype=np.int32)

    assert_groupby_results_equal(expect, got)


@pytest.fixture(scope="module")
def groupby_jit_data_small():
    """
    Return a small dataset for testing JIT Groupby Apply. The dataframe
    contains 4 groups of size 1, 2, 3, 4 as well as an additional key
    column that can be used to test subgroups within groups. This data
    is useful for smoke testing basic numeric results
    """
    rng = np.random.default_rng(42)
    df = DataFrame()
    key1 = [1] + [2] * 2 + [3] * 3 + [4] * 4
    key2 = [1, 2] * 5
    df["key1"] = key1
    df["key2"] = key2

    df["val1"] = rng.integers(0, 10, len(key1))
    df["val2"] = rng.integers(0, 10, len(key1))

    # randomly permute data
    df = df.sample(frac=1, ignore_index=True)
    return df


@pytest.fixture(scope="module")
def groupby_jit_data_large(groupby_jit_data_small):
    """
    Larger version of groupby_jit_data_small which contains enough data
    to require more than one block per group. This data is useful for
    testing if JIT GroupBy algorithms scale to larger dastasets without
    manifesting numerical issues such as overflow.
    """
    max_tpb = 1024
    factor = (
        max_tpb + 1
    )  # bigger than a block but not always an exact multiple
    df = cudf.concat([groupby_jit_data_small] * factor)

    return df


@pytest.fixture(scope="module")
def groupby_jit_data_nans(groupby_jit_data_small):
    """
    Returns a modified version of groupby_jit_data_small which contains
    nan values.
    """

    df = groupby_jit_data_small.sort_values(["key1", "key2"])
    df["val1"] = df["val1"].astype("float64")
    df["val1"][::2] = np.nan
    df = df.sample(frac=1, ignore_index=True)
    return df


@pytest.fixture(scope="module")
def groupby_jit_datasets(
    groupby_jit_data_small, groupby_jit_data_large, groupby_jit_data_nans
):
    return {
        "small": groupby_jit_data_small,
        "large": groupby_jit_data_large,
        "nans": groupby_jit_data_nans,
    }


def run_groupby_apply_jit_test(data, func, keys, *args):
    expect_groupby_obj = data.to_pandas().groupby(keys)
    got_groupby_obj = data.groupby(keys)

    # compare cuDF jit to pandas
    cudf_jit_result = got_groupby_obj.apply(
        func, *args, engine="jit", include_groups=False
    )
    pandas_result = expect_groupby_obj.apply(func, *args, include_groups=False)
    assert_groupby_results_equal(cudf_jit_result, pandas_result)


def groupby_apply_jit_reductions_test_inner(func, data, dtype):
    # ideally we'd just have:
    # lambda group: getattr(group, func)()
    # but the current kernel caching mechanism relies on pickle which
    # does not play nice with local functions. What's below uses
    # exec as a workaround to write the test functions dynamically

    funcstr = textwrap.dedent(
        f"""
        def func(df):
            return df['val1'].{func}()
        """
    )
    lcl = {}
    exec(funcstr, lcl)
    func = lcl["func"]

    data["val1"] = data["val1"].astype(dtype)
    data["val2"] = data["val2"].astype(dtype)

    run_groupby_apply_jit_test(data, func, ["key1"])


# test unary reductions
@pytest.mark.parametrize(
    "dtype",
    SUPPORTED_GROUPBY_NUMPY_TYPES,
    ids=[str(t) for t in SUPPORTED_GROUPBY_NUMPY_TYPES],
)
@pytest.mark.parametrize(
    "func", ["min", "max", "sum", "mean", "var", "std", "idxmin", "idxmax"]
)
@pytest.mark.parametrize("dataset", ["small", "large", "nans"])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Include groups missing on old versions of pandas",
)
def test_groupby_apply_jit_unary_reductions(
    func, dtype, dataset, groupby_jit_datasets
):
    dataset = groupby_jit_datasets[dataset]
    groupby_apply_jit_reductions_test_inner(func, dataset, dtype)


# test unary reductions for special values
def groupby_apply_jit_reductions_special_vals_inner(
    func, data, dtype, special_val
):
    funcstr = textwrap.dedent(
        f"""
        def func(df):
            return df['val1'].{func}()
        """
    )
    lcl = {}
    exec(funcstr, lcl)
    func = lcl["func"]

    data["val1"] = data["val1"].astype(dtype)
    data["val2"] = data["val2"].astype(dtype)
    data["val1"] = special_val
    data["val2"] = special_val

    run_groupby_apply_jit_test(data, func, ["key1"])


# test unary index reductions for special values
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def groupby_apply_jit_idx_reductions_special_vals_inner(
    func, data, dtype, special_val
):
    funcstr = textwrap.dedent(
        f"""
        def func(df):
            return df['val1'].{func}()
        """
    )
    lcl = {}
    exec(funcstr, lcl)
    func = lcl["func"]

    data["val1"] = data["val1"].astype(dtype)
    data["val2"] = data["val2"].astype(dtype)
    data["val1"] = special_val
    data["val2"] = special_val

    run_groupby_apply_jit_test(data, func, ["key1"])


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("func", ["min", "max", "sum", "mean", "var", "std"])
@pytest.mark.parametrize("special_val", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("dataset", ["small", "large", "nans"])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Include groups missing on old versions of pandas",
)
def test_groupby_apply_jit_reductions_special_vals(
    func, dtype, dataset, groupby_jit_datasets, special_val
):
    dataset = groupby_jit_datasets[dataset]
    with expect_warning_if(
        func in {"var", "std"} and not np.isnan(special_val), RuntimeWarning
    ):
        groupby_apply_jit_reductions_special_vals_inner(
            func, dataset, dtype, special_val
        )


@pytest.mark.parametrize("dtype", ["float64"])
@pytest.mark.parametrize("func", ["idxmax", "idxmin"])
@pytest.mark.parametrize(
    "special_val",
    [
        pytest.param(
            np.nan,
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/13832"
            ),
        ),
        np.inf,
        -np.inf,
    ],
)
@pytest.mark.parametrize("dataset", ["small", "large", "nans"])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="include_groups keyword new in pandas 2.2",
)
def test_groupby_apply_jit_idx_reductions_special_vals(
    func, dtype, dataset, groupby_jit_datasets, special_val
):
    dataset = groupby_jit_datasets[dataset]
    groupby_apply_jit_idx_reductions_special_vals_inner(
        func, dataset, dtype, special_val
    )


@pytest.mark.parametrize("dtype", ["int32"])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_jit_sum_integer_overflow(dtype):
    max = np.iinfo(dtype).max

    data = DataFrame(
        {
            "a": [0, 0, 0],
            "b": [max, max, max],
        }
    )

    def func(group):
        return group["b"].sum()

    run_groupby_apply_jit_test(data, func, ["a"])


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param(
            "small",
            marks=[
                pytest.mark.filterwarnings(
                    "ignore:Degrees of Freedom <= 0 for slice"
                ),
                pytest.mark.filterwarnings(
                    "ignore:divide by zero encountered in divide"
                ),
            ],
        ),
        "large",
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_jit_correlation(dataset, groupby_jit_datasets, dtype):
    dataset = groupby_jit_datasets[dataset]

    dataset["val1"] = dataset["val1"].astype(dtype)
    dataset["val2"] = dataset["val2"].astype(dtype)

    keys = ["key1"]

    def func(group):
        return group["val1"].corr(group["val2"])

    if np.dtype(dtype).kind == "f":
        # Correlation of floating types is not yet supported:
        # https://github.com/rapidsai/cudf/issues/13839
        m = (
            f"Series.corr\\(Series\\) is not "
            f"supported for \\({dtype}, {dtype}\\)"
        )
        with pytest.raises(UDFError, match=m):
            run_groupby_apply_jit_test(dataset, func, keys)
        return
    with expect_warning_if(dtype in {"int32", "int64"}, RuntimeWarning):
        run_groupby_apply_jit_test(dataset, func, keys)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_jit_correlation_zero_variance(dtype):
    # pearson correlation is undefined when the variance of either
    # variable is zero. This test ensures that the jit implementation
    # returns the same result as pandas in this case.
    data = DataFrame(
        {"a": [0, 0, 0, 0, 0], "b": [1, 1, 1, 1, 1], "c": [2, 2, 2, 2, 2]}
    )

    def func(group):
        return group["b"].corr(group["c"])

    with expect_warning_if(dtype in {"int32", "int64"}, RuntimeWarning):
        run_groupby_apply_jit_test(data, func, ["a"])


@pytest.mark.parametrize("op", unary_ops)
def test_groupby_apply_jit_invalid_unary_ops_error(groupby_jit_data_small, op):
    keys = ["key1"]

    def func(group):
        return op(group["val1"])

    with pytest.raises(
        UDFError,
        match=f"{op.__name__}\\(Series\\) is not supported by JIT GroupBy",
    ):
        run_groupby_apply_jit_test(groupby_jit_data_small, func, keys)


@pytest.mark.parametrize("op", arith_ops + comparison_ops)
def test_groupby_apply_jit_invalid_binary_ops_error(
    groupby_jit_data_small, op
):
    keys = ["key1"]

    def func(group):
        return op(group["val1"], group["val2"])

    with pytest.raises(
        UDFError,
        match=f"{op.__name__}\\(Series, Series\\) is not supported",
    ):
        run_groupby_apply_jit_test(groupby_jit_data_small, func, keys)


def test_groupby_apply_jit_no_df_ops(groupby_jit_data_small):
    # DataFrame level operations are not yet supported.
    def func(group):
        return group.sum()

    with pytest.raises(
        UDFError,
        match="JIT GroupBy.apply\\(\\) does not support DataFrame.sum\\(\\)",
    ):
        run_groupby_apply_jit_test(groupby_jit_data_small, func, ["key1"])


@pytest.mark.parametrize("dtype", ["uint8", "str"])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_unsupported_dtype(dtype):
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df["b"] = df["b"].astype(dtype)

    # a UDAF that doesn't actually use the input column
    # with the unsupported dtype should still succeed
    def func(group):
        return group["c"].sum()

    run_groupby_apply_jit_test(df, func, ["a"])

    # however a UDAF that does use the unsupported dtype
    # should fail
    def func(group):
        return group["b"].sum()

    with pytest.raises(UDFError, match="Only columns of the following dtypes"):
        run_groupby_apply_jit_test(df, func, ["a"])


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df["val1"].max() + df["val2"].min(),
        lambda df: df["val1"].sum() + df["val2"].var(),
        lambda df: df["val1"].mean() + df["val2"].std(),
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_jit_basic(func, groupby_jit_data_small):
    run_groupby_apply_jit_test(groupby_jit_data_small, func, ["key1", "key2"])


def create_test_groupby_apply_jit_args_params():
    def f1(df, k):
        return df["val1"].max() + df["val2"].min() + k

    def f2(df, k, L):
        return df["val1"].sum() - df["val2"].var() + (k / L)

    def f3(df, k, L, m):
        return ((k * df["val1"].mean()) + (L * df["val2"].std())) / m

    return [(f1, (42,)), (f2, (42, 119)), (f3, (42, 119, 212.1))]


@pytest.mark.parametrize(
    "func,args", create_test_groupby_apply_jit_args_params()
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_jit_args(func, args, groupby_jit_data_small):
    run_groupby_apply_jit_test(
        groupby_jit_data_small, func, ["key1", "key2"], *args
    )


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_jit_block_divergence():
    # https://github.com/rapidsai/cudf/issues/12686
    df = cudf.DataFrame(
        {
            "a": [0, 0, 0, 1, 1, 1],
            "b": [1, 1, 1, 2, 3, 4],
        }
    )

    def diverging_block(grp_df):
        if grp_df["b"].mean() > 1:
            return grp_df["b"].mean()
        return 0

    run_groupby_apply_jit_test(df, diverging_block, ["a"])


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_caching():
    # Make sure similar functions that differ
    # by simple things like constants actually
    # recompile

    # begin with a clear cache
    precompiled.clear()
    assert precompiled.currsize == 0

    data = cudf.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 4, 5, 6]})

    def f(group):
        return group["b"].mean() * 2

    # a single run should result in a cache size of 1
    run_groupby_apply_jit_test(data, f, ["a"])
    assert precompiled.currsize == 1

    # a second run with f should not increase the count
    run_groupby_apply_jit_test(data, f, ["a"])
    assert precompiled.currsize == 1

    # changing a constant value inside the UDF should miss
    def f(group):
        return group["b"].mean() * 3

    run_groupby_apply_jit_test(data, f, ["a"])
    assert precompiled.currsize == 2

    # changing the dtypes of the columns should miss
    data["b"] = data["b"].astype("float64")
    run_groupby_apply_jit_test(data, f, ["a"])

    assert precompiled.currsize == 3


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_no_bytecode_fallback():
    # tests that a function which contains no bytecode
    # attribute, but would still be executable using
    # the iterative groupby apply approach, still works.

    gdf = cudf.DataFrame({"a": [0, 1, 1], "b": [1, 2, 3]})
    pdf = gdf.to_pandas()

    def f(group):
        return group.sum()

    part = partial(f)

    expect = pdf.groupby("a").apply(part, include_groups=False)
    got = gdf.groupby("a").apply(part, engine="auto", include_groups=False)
    assert_groupby_results_equal(expect, got)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_return_col_from_df():
    # tests a UDF that consists of purely colwise
    # ops, such as `lambda group: group.x + group.y`
    # which returns a column
    df = cudf.DataFrame(
        {
            "id": range(10),
            "x": range(10),
            "y": range(10),
        }
    )
    pdf = df.to_pandas()

    def func(df):
        return df.x + df.y

    got = df.groupby("id").apply(func, include_groups=False)
    expect = pdf.groupby("id").apply(func, include_groups=False)
    # pandas seems to erroneously add an extra MI level of ids
    # TODO: Figure out how pandas groupby.apply determines the columns
    expect = pd.DataFrame(expect.droplevel(1), columns=got.columns)
    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize("func", [lambda group: group.sum()])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_return_df(func):
    # tests a UDF that reduces over a dataframe
    # and produces a series with the original column names
    # as its index, such as lambda group: group.sum() + group.min()
    df = cudf.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    pdf = df.to_pandas()

    expect = pdf.groupby("a").apply(func, include_groups=False)
    got = df.groupby("a").apply(func, include_groups=False)
    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_apply_return_reindexed_series(as_index):
    def gdf_func(df):
        return cudf.Series([df["a"].sum(), df["b"].min(), df["c"].max()])

    def pdf_func(df):
        return pd.Series([df["a"].sum(), df["b"].min(), df["c"].max()])

    df = cudf.DataFrame(
        {
            "key": [0, 0, 1, 1, 2, 2],
            "a": [1, 2, 3, 4, 5, 6],
            "b": [7, 8, 9, 10, 11, 12],
            "c": [13, 14, 15, 16, 17, 18],
        }
    )
    pdf = df.to_pandas()

    kwargs = {}
    if PANDAS_GE_220:
        kwargs["include_groups"] = False

    expect = pdf.groupby("key", as_index=as_index).apply(pdf_func, **kwargs)
    got = df.groupby("key", as_index=as_index).apply(gdf_func, **kwargs)
    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize("nelem", [2, 3, 100, 500, 1000])
@pytest.mark.parametrize(
    "func",
    [
        "mean",
        "std",
        "var",
        "min",
        "max",
        "idxmin",
        "idxmax",
        "count",
        "sum",
        "prod",
    ],
)
def test_groupby_2keys_agg(nelem, func):
    # gdf (Note: lack of multiIndex)
    expect_df = (
        make_frame(pd.DataFrame, nelem=nelem).groupby(["x", "y"]).agg(func)
    )
    got_df = make_frame(DataFrame, nelem=nelem).groupby(["x", "y"]).agg(func)

    check_dtype = func not in _index_type_aggs
    assert_groupby_results_equal(got_df, expect_df, check_dtype=check_dtype)


@pytest.mark.parametrize("num_groups", [2, 3, 10, 50, 100])
@pytest.mark.parametrize("nelem_per_group", [1, 10, 100])
@pytest.mark.parametrize(
    "func",
    ["min", "max", "count", "sum"],
    # TODO: Replace the above line with the one below once
    # https://github.com/pandas-dev/pandas/issues/40685 is resolved.
    # "func", ["min", "max", "idxmin", "idxmax", "count", "sum"],
)
def test_groupby_agg_decimal(num_groups, nelem_per_group, func):
    rng = np.random.default_rng(seed=0)
    # The number of digits after the decimal to use.
    decimal_digits = 2
    # The number of digits before the decimal to use.
    whole_digits = 2

    scale = 10**whole_digits
    nelem = num_groups * nelem_per_group

    # The unique is necessary because otherwise if there are duplicates idxmin
    # and idxmax may return different results than pandas (see
    # https://github.com/rapidsai/cudf/issues/7756). This is not relevant to
    # the current version of the test, because idxmin and idxmax simply don't
    # work with pandas Series composed of Decimal objects (see
    # https://github.com/pandas-dev/pandas/issues/40685). However, if that is
    # ever enabled, then this issue will crop up again so we may as well have
    # it fixed now.
    x = np.unique((rng.random(nelem) * scale).round(decimal_digits))
    y = np.unique((rng.random(nelem) * scale).round(decimal_digits))

    if x.size < y.size:
        total_elements = x.size
        y = y[: x.size]
    else:
        total_elements = y.size
        x = x[: y.size]

    # Note that this filtering can lead to one group with fewer elements, but
    # that shouldn't be a problem and is probably useful to test.
    idx_col = np.tile(np.arange(num_groups), nelem_per_group)[:total_elements]

    decimal_x = pd.Series([Decimal(str(d)) for d in x])
    decimal_y = pd.Series([Decimal(str(d)) for d in y])

    pdf = pd.DataFrame({"idx": idx_col, "x": decimal_x, "y": decimal_y})
    gdf = DataFrame(
        {
            "idx": idx_col,
            "x": cudf.Series(decimal_x),
            "y": cudf.Series(decimal_y),
        }
    )

    expect_df = pdf.groupby("idx", sort=True).agg(func)
    if rmm._cuda.gpu.runtimeGetVersion() < 11000:
        with pytest.raises(RuntimeError):
            got_df = gdf.groupby("idx", sort=True).agg(func)
    else:
        got_df = gdf.groupby("idx", sort=True).agg(func)
        assert_eq(expect_df["x"], got_df["x"], check_dtype=False)
        assert_eq(expect_df["y"], got_df["y"], check_dtype=False)


@pytest.mark.parametrize(
    "agg", ["min", "max", "idxmin", "idxmax", "count", "sum", "prod", "mean"]
)
def test_series_groupby(agg):
    s = pd.Series([1, 2, 3])
    g = Series([1, 2, 3])
    sg = s.groupby(s // 2)
    gg = g.groupby(g // 2)
    sa = getattr(sg, agg)()
    ga = getattr(gg, agg)()
    check_dtype = agg not in _index_type_aggs
    assert_groupby_results_equal(sa, ga, check_dtype=check_dtype)


@pytest.mark.parametrize(
    "agg", ["min", "max", "idxmin", "idxmax", "count", "sum", "prod", "mean"]
)
def test_series_groupby_agg(agg):
    s = pd.Series([1, 2, 3])
    g = Series([1, 2, 3])
    sg = s.groupby(s // 2).agg(agg)
    gg = g.groupby(g // 2).agg(agg)
    check_dtype = agg not in _index_type_aggs
    assert_groupby_results_equal(sg, gg, check_dtype=check_dtype)


@pytest.mark.parametrize(
    "agg",
    [
        "min",
        "max",
        "count",
        "sum",
        "prod",
        "mean",
        pytest.param(
            "idxmin",
            marks=pytest.mark.xfail(reason="gather needed for idxmin"),
        ),
        pytest.param(
            "idxmax",
            marks=pytest.mark.xfail(reason="gather needed for idxmax"),
        ),
    ],
)
def test_groupby_level_zero(agg):
    pdf = pd.DataFrame({"x": [1, 2, 3]}, index=[2, 5, 5])
    gdf = DataFrame.from_pandas(pdf)
    pdg = pdf.groupby(level=0)
    gdg = gdf.groupby(level=0)
    pdresult = getattr(pdg, agg)()
    gdresult = getattr(gdg, agg)()
    check_dtype = agg not in _index_type_aggs
    assert_groupby_results_equal(pdresult, gdresult, check_dtype=check_dtype)


@pytest.mark.parametrize(
    "agg",
    [
        "min",
        "max",
        "count",
        "sum",
        "prod",
        "mean",
        pytest.param(
            "idxmin",
            marks=pytest.mark.xfail(reason="gather needed for idxmin"),
        ),
        pytest.param(
            "idxmax",
            marks=pytest.mark.xfail(reason="gather needed for idxmax"),
        ),
    ],
)
def test_groupby_series_level_zero(agg):
    pdf = pd.Series([1, 2, 3], index=[2, 5, 5])
    gdf = Series.from_pandas(pdf)
    pdg = pdf.groupby(level=0)
    gdg = gdf.groupby(level=0)
    pdresult = getattr(pdg, agg)()
    gdresult = getattr(gdg, agg)()
    check_dtype = agg not in _index_type_aggs
    assert_groupby_results_equal(pdresult, gdresult, check_dtype=check_dtype)


def test_groupby_column_name():
    pdf = pd.DataFrame({"xx": [1.0, 2.0, 3.0], "yy": [1, 2, 3]})
    gdf = DataFrame.from_pandas(pdf)
    g = gdf.groupby("yy")
    p = pdf.groupby("yy")
    gxx = g["xx"].sum()
    pxx = p["xx"].sum()
    assert_groupby_results_equal(pxx, gxx)

    gxx = g["xx"].count()
    pxx = p["xx"].count()
    assert_groupby_results_equal(pxx, gxx, check_dtype=False)

    gxx = g["xx"].min()
    pxx = p["xx"].min()
    assert_groupby_results_equal(pxx, gxx)

    gxx = g["xx"].max()
    pxx = p["xx"].max()
    assert_groupby_results_equal(pxx, gxx)

    gxx = g["xx"].idxmin()
    pxx = p["xx"].idxmin()
    assert_groupby_results_equal(pxx, gxx, check_dtype=False)

    gxx = g["xx"].idxmax()
    pxx = p["xx"].idxmax()
    assert_groupby_results_equal(pxx, gxx, check_dtype=False)

    gxx = g["xx"].mean()
    pxx = p["xx"].mean()
    assert_groupby_results_equal(pxx, gxx)


def test_groupby_column_numeral():
    pdf = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [1, 2, 3]})
    gdf = DataFrame.from_pandas(pdf)
    p = pdf.groupby(1)
    g = gdf.groupby(1)
    pxx = p[0].sum()
    gxx = g[0].sum()
    assert_groupby_results_equal(pxx, gxx)

    pdf = pd.DataFrame({0.5: [1.0, 2.0, 3.0], 1.5: [1, 2, 3]})
    gdf = DataFrame.from_pandas(pdf)
    p = pdf.groupby(1.5)
    g = gdf.groupby(1.5)
    pxx = p[0.5].sum()
    gxx = g[0.5].sum()
    assert_groupby_results_equal(pxx, gxx)


@pytest.mark.parametrize(
    "series",
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
        [1, 2, 3],
        [4, 3, 2],
        [0, 2, 0],
        pd.Series([0, 2, 0]),
        pd.Series([0, 2, 0], index=[0, 2, 1]),
    ],
)
def test_groupby_external_series(series):
    pdf = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1, 2, 1]})
    gdf = DataFrame.from_pandas(pdf)
    pxx = pdf.groupby(pd.Series(series)).x.sum()
    gxx = gdf.groupby(cudf.Series(series)).x.sum()
    assert_groupby_results_equal(pxx, gxx)


@pytest.mark.parametrize("series", [[0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
def test_groupby_external_series_incorrect_length(series):
    pdf = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1, 2, 1]})
    gdf = DataFrame.from_pandas(pdf)
    pxx = pdf.groupby(pd.Series(series)).x.sum()
    gxx = gdf.groupby(cudf.Series(series)).x.sum()
    assert_groupby_results_equal(pxx, gxx)


@pytest.mark.parametrize(
    "level", [0, 1, "a", "b", [0, 1], ["a", "b"], ["a", 1], -1, [-1, -2]]
)
def test_groupby_levels(level):
    idx = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (2, 2)], names=("a", "b"))
    pdf = pd.DataFrame({"c": [1, 2, 3], "d": [2, 3, 4]}, index=idx)
    gdf = cudf.from_pandas(pdf)
    assert_groupby_results_equal(
        pdf.groupby(level=level).sum(),
        gdf.groupby(level=level).sum(),
    )


def test_advanced_groupby_levels():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 1], "z": [1, 1, 1]})
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby(["x", "y"]).sum()
    gdg = gdf.groupby(["x", "y"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdh = pdg.groupby(level=1).sum()
    gdh = gdg.groupby(level=1).sum()
    assert_groupby_results_equal(pdh, gdh)
    pdg = pdf.groupby(["x", "y", "z"]).sum()
    gdg = gdf.groupby(["x", "y", "z"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby(["z"]).sum()
    gdg = gdf.groupby(["z"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby(["y", "z"]).sum()
    gdg = gdf.groupby(["y", "z"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby(["x", "z"]).sum()
    gdg = gdf.groupby(["x", "z"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby(["y"]).sum()
    gdg = gdf.groupby(["y"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby(["x"]).sum()
    gdg = gdf.groupby(["x"]).sum()
    assert_groupby_results_equal(pdg, gdg)
    pdh = pdg.groupby(level=0).sum()
    gdh = gdg.groupby(level=0).sum()
    assert_groupby_results_equal(pdh, gdh)
    pdg = pdf.groupby(["x", "y"]).sum()
    gdg = gdf.groupby(["x", "y"]).sum()
    pdh = pdg.groupby(level=[0, 1]).sum()
    gdh = gdg.groupby(level=[0, 1]).sum()
    assert_groupby_results_equal(pdh, gdh)
    pdh = pdg.groupby(level=[1, 0]).sum()
    gdh = gdg.groupby(level=[1, 0]).sum()
    assert_groupby_results_equal(pdh, gdh)
    pdg = pdf.groupby(["x", "y"]).sum()
    gdg = gdf.groupby(["x", "y"]).sum()

    assert_exceptions_equal(
        lfunc=pdg.groupby,
        rfunc=gdg.groupby,
        lfunc_args_and_kwargs=([], {"level": 2}),
        rfunc_args_and_kwargs=([], {"level": 2}),
    )


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.groupby(["x", "y", "z"]).sum(),
        lambda df: df.groupby(["x", "y"]).sum(),
        lambda df: df.groupby(["x", "y"]).agg("sum"),
        lambda df: df.groupby(["y"]).sum(),
        lambda df: df.groupby(["y"]).agg("sum"),
        lambda df: df.groupby(["x"]).sum(),
        lambda df: df.groupby(["x"]).agg("sum"),
        lambda df: df.groupby(["x", "y"]).z.sum(),
        lambda df: df.groupby(["x", "y"]).z.agg("sum"),
    ],
)
def test_empty_groupby(func):
    pdf = pd.DataFrame({"x": [], "y": [], "z": []})
    gdf = cudf.from_pandas(pdf)
    assert_groupby_results_equal(func(pdf), func(gdf), check_index_type=False)


def test_groupby_unsupported_columns():
    rng = np.random.default_rng(seed=12)
    pd_cat = pd.Categorical(
        pd.Series(rng.choice(["a", "b", 1], 3), dtype="category")
    )
    pdf = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": ["a", "b", "c"],
            "z": ["d", "e", "f"],
            "a": [3, 4, 5],
        }
    )
    pdf["b"] = pd_cat
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby("x").sum(numeric_only=True)
    # cudf does not yet support numeric_only, so our default is False (unlike
    # pandas, which defaults to inferring and throws a warning about it).
    gdg = gdf.groupby("x").sum(numeric_only=True)
    assert_groupby_results_equal(pdg, gdg)


def test_list_of_series():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 1]})
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby([pdf.x]).y.sum()
    gdg = gdf.groupby([gdf.x]).y.sum()
    assert_groupby_results_equal(pdg, gdg)
    pdg = pdf.groupby([pdf.x, pdf.y]).y.sum()
    gdg = gdf.groupby([gdf.x, gdf.y]).y.sum()
    pytest.skip()
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_use_agg_column_as_index():
    pdf = pd.DataFrame()
    pdf["a"] = [1, 1, 1, 3, 5]
    gdf = cudf.DataFrame()
    gdf["a"] = [1, 1, 1, 3, 5]
    pdg = pdf.groupby("a").agg({"a": "count"})
    gdg = gdf.groupby("a").agg({"a": "count"})
    assert_groupby_results_equal(pdg, gdg, check_dtype=False)


def test_groupby_list_then_string():
    gdf = cudf.DataFrame()
    gdf["a"] = [0, 1, 0, 1, 2]
    gdf["b"] = [11, 2, 15, 12, 2]
    gdf["c"] = [6, 7, 6, 7, 6]
    pdf = gdf.to_pandas()
    gdg = gdf.groupby("a", as_index=True).agg(
        {"b": ["min", "max"], "c": "max"}
    )
    pdg = pdf.groupby("a", as_index=True).agg(
        {"b": ["min", "max"], "c": "max"}
    )
    assert_groupby_results_equal(gdg, pdg)


def test_groupby_different_unequal_length_column_aggregations():
    gdf = cudf.DataFrame()
    gdf["a"] = [0, 1, 0, 1, 2]
    gdf["b"] = [11, 2, 15, 12, 2]
    gdf["c"] = [11, 2, 15, 12, 2]
    pdf = gdf.to_pandas()
    gdg = gdf.groupby("a", as_index=True).agg(
        {"b": "min", "c": ["max", "min"]}
    )
    pdg = pdf.groupby("a", as_index=True).agg(
        {"b": "min", "c": ["max", "min"]}
    )
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_single_var_two_aggs():
    gdf = cudf.DataFrame()
    gdf["a"] = [0, 1, 0, 1, 2]
    gdf["b"] = [11, 2, 15, 12, 2]
    gdf["c"] = [11, 2, 15, 12, 2]
    pdf = gdf.to_pandas()
    gdg = gdf.groupby("a", as_index=True).agg({"b": ["min", "max"]})
    pdg = pdf.groupby("a", as_index=True).agg({"b": ["min", "max"]})
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_double_var_two_aggs():
    gdf = cudf.DataFrame()
    gdf["a"] = [0, 1, 0, 1, 2]
    gdf["b"] = [11, 2, 15, 12, 2]
    gdf["c"] = [11, 2, 15, 12, 2]
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["a", "b"], as_index=True).agg({"c": ["min", "max"]})
    pdg = pdf.groupby(["a", "b"], as_index=True).agg({"c": ["min", "max"]})
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_apply_basic_agg_single_column():
    gdf = DataFrame()
    gdf["key"] = [0, 0, 1, 1, 2, 2, 0]
    gdf["val"] = [0, 1, 2, 3, 4, 5, 6]
    gdf["mult"] = gdf["key"] * gdf["val"]
    pdf = gdf.to_pandas()

    gdg = gdf.groupby(["key", "val"]).mult.sum()
    pdg = pdf.groupby(["key", "val"]).mult.sum()
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_multi_agg_single_groupby_series():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "x": rng.integers(0, 5, size=10000),
            "y": rng.normal(size=10000),
        }
    )
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby("x").y.agg(["sum", "max"])
    gdg = gdf.groupby("x").y.agg(["sum", "max"])

    assert_groupby_results_equal(pdg, gdg)


def test_groupby_multi_agg_multi_groupby():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": rng.integers(0, 5, 10),
            "b": rng.integers(0, 5, 10),
            "c": rng.integers(0, 5, 10),
            "d": rng.integers(0, 5, 10),
        }
    )
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby(["a", "b"]).agg(["sum", "max"])
    gdg = gdf.groupby(["a", "b"]).agg(["sum", "max"])
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_datetime_multi_agg_multi_groupby():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": pd.date_range(
                datetime.datetime.now(),
                datetime.datetime.now() + datetime.timedelta(9),
                freq="D",
            ),
            "b": rng.integers(0, 5, 10),
            "c": rng.integers(0, 5, 10),
            "d": rng.integers(0, 5, 10),
        }
    )
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby(["a", "b"]).agg(["sum", "max"])
    gdg = gdf.groupby(["a", "b"]).agg(["sum", "max"])

    assert_groupby_results_equal(pdg, gdg)


@pytest.mark.parametrize(
    "agg",
    [
        ["min", "max", "count", "mean"],
        ["mean", "var", "std"],
        ["count", "mean", "var", "std"],
    ],
)
def test_groupby_multi_agg_hash_groupby(agg):
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    prefixes = alphabets[:10]
    coll_dict = dict()
    for prefix in prefixes:
        for this_name in alphabets:
            coll_dict[prefix + this_name] = float
    coll_dict["id"] = int
    gdf = cudf.datasets.timeseries(
        start="2000",
        end="2000-01-2",
        dtypes=coll_dict,
        freq="1s",
        seed=1,
    ).reset_index(drop=True)
    pdf = gdf.to_pandas()
    check_dtype = "count" not in agg
    pdg = pdf.groupby("id").agg(agg)
    gdg = gdf.groupby("id").agg(agg)
    assert_groupby_results_equal(pdg, gdg, check_dtype=check_dtype)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="previous verion of pandas throws a warning",
)
@pytest.mark.parametrize(
    "agg", ["min", "max", "idxmax", "idxmin", "sum", "prod", "count", "mean"]
)
def test_groupby_nulls_basic(agg):
    check_dtype = agg not in _index_type_aggs

    pdf = pd.DataFrame({"a": [0, 0, 1, 1, 2, 2], "b": [1, 2, 1, 2, 1, None]})
    gdf = cudf.from_pandas(pdf)
    assert_groupby_results_equal(
        getattr(pdf.groupby("a"), agg)(),
        getattr(gdf.groupby("a"), agg)(),
        check_dtype=check_dtype,
    )

    pdf = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2],
            "b": [1, 2, 1, 2, 1, None],
            "c": [1, 2, 1, None, 1, 2],
        }
    )
    gdf = cudf.from_pandas(pdf)
    assert_groupby_results_equal(
        getattr(pdf.groupby("a"), agg)(),
        getattr(gdf.groupby("a"), agg)(),
        check_dtype=check_dtype,
    )

    pdf = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2],
            "b": [1, 2, 1, 2, 1, None],
            "c": [1, 2, None, None, 1, 2],
        }
    )
    gdf = cudf.from_pandas(pdf)

    # TODO: fillna() used here since we don't follow
    # Pandas' null semantics. Should we change it?

    assert_groupby_results_equal(
        getattr(pdf.groupby("a"), agg)().fillna(0),
        getattr(gdf.groupby("a"), agg)().fillna(0 if agg != "prod" else 1),
        check_dtype=check_dtype,
    )


def test_groupby_nulls_in_index():
    pdf = pd.DataFrame({"a": [None, 2, 1, 1], "b": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").sum(), gdf.groupby("a").sum()
    )


def test_groupby_all_nulls_index():
    gdf = cudf.DataFrame(
        {
            "a": cudf.Series([None, None, None, None], dtype="object"),
            "b": [1, 2, 3, 4],
        }
    )
    pdf = gdf.to_pandas()
    assert_groupby_results_equal(
        pdf.groupby("a").sum(), gdf.groupby("a").sum()
    )

    gdf = cudf.DataFrame(
        {"a": cudf.Series([np.nan, np.nan, np.nan, np.nan]), "b": [1, 2, 3, 4]}
    )
    pdf = gdf.to_pandas()
    assert_groupby_results_equal(
        pdf.groupby("a").sum(), gdf.groupby("a").sum()
    )


@pytest.mark.parametrize("sort", [True, False])
def test_groupby_sort(sort):
    pdf = pd.DataFrame({"a": [2, 2, 1, 1], "b": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby("a", sort=sort).sum(),
        gdf.groupby("a", sort=sort).sum(),
        check_like=not sort,
    )

    pdf = pd.DataFrame(
        {"c": [-1, 2, 1, 4], "b": [1, 2, 3, 4], "a": [2, 2, 1, 1]}
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.groupby(["c", "b"], sort=sort).sum(),
        gdf.groupby(["c", "b"], sort=sort).sum(),
        check_like=not sort,
    )

    ps = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=[2, 2, 2, 3, 3, 1, 1, 1])
    gs = cudf.from_pandas(ps)

    assert_eq(
        ps.groupby(level=0, sort=sort).sum().to_frame(),
        gs.groupby(level=0, sort=sort).sum().to_frame(),
        check_like=not sort,
    )

    ps = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8],
        index=pd.MultiIndex.from_product([(1, 2), ("a", "b"), (42, 84)]),
    )
    gs = cudf.from_pandas(ps)

    assert_eq(
        ps.groupby(level=0, sort=sort).sum().to_frame(),
        gs.groupby(level=0, sort=sort).sum().to_frame(),
        check_like=not sort,
    )


def test_groupby_cat():
    pdf = pd.DataFrame(
        {"a": [1, 1, 2], "b": pd.Series(["b", "b", "a"], dtype="category")}
    )
    gdf = cudf.from_pandas(pdf)
    assert_groupby_results_equal(
        pdf.groupby("a").count(),
        gdf.groupby("a").count(),
        check_dtype=False,
    )


def test_groupby_index_type():
    df = cudf.DataFrame()
    df["string_col"] = ["a", "b", "c"]
    df["counts"] = [1, 2, 3]
    res = df.groupby(by="string_col").counts.sum()
    assert res.index.dtype == cudf.dtype("object")


@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize("q", [0.25, 0.4, 0.5, 0.7, 1])
def test_groupby_quantile(request, interpolation, q):
    request.applymarker(
        pytest.mark.xfail(
            condition=(q == 0.5 and interpolation == "nearest"),
            reason=(
                "Pandas NaN Rounding will fail nearest interpolation at 0.5"
            ),
        )
    )

    raw_data = {
        "y": [None, 1, 2, 3, 4, None, 6, 7, 8, 9],
        "x": [1, 2, 3, 1, 2, 2, 1, None, 3, 2],
    }
    # Pandas>0.25 now casts NaN in quantile operations as a float64
    # # so we are filling with zeros.
    pdf = pd.DataFrame(raw_data).fillna(0)
    gdf = DataFrame.from_pandas(pdf)

    pdg = pdf.groupby("x")
    gdg = gdf.groupby("x")

    pdresult = pdg.quantile(q, interpolation=interpolation)
    gdresult = gdg.quantile(q, interpolation=interpolation)

    assert_groupby_results_equal(pdresult, gdresult)


def test_groupby_std():
    raw_data = {
        "x": [1, 2, 3, 1, 2, 2, 1, None, 3, 2],
        "y": [None, 1, 2, 3, 4, None, 6, 7, 8, 9],
    }
    pdf = pd.DataFrame(raw_data)
    gdf = DataFrame.from_pandas(pdf)
    pdg = pdf.groupby("x")
    gdg = gdf.groupby("x")
    pdresult = pdg.std()
    gdresult = gdg.std()

    assert_groupby_results_equal(pdresult, gdresult)


def test_groupby_size():
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 3, 4],
            "b": ["bob", "bob", "alice", "cooper"],
            "c": [1, 2, 3, 4],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").size(),
        gdf.groupby("a").size(),
        check_dtype=False,
    )

    assert_groupby_results_equal(
        pdf.groupby(["a", "b", "c"]).size(),
        gdf.groupby(["a", "b", "c"]).size(),
        check_dtype=False,
    )

    sr = pd.Series(range(len(pdf)))
    assert_groupby_results_equal(
        pdf.groupby(sr).size(),
        gdf.groupby(sr).size(),
        check_dtype=False,
    )


@pytest.mark.parametrize("index", [None, [1, 2, 3, 4]])
def test_groupby_cumcount(index):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 3, 4],
            "b": ["bob", "bob", "alice", "cooper"],
            "c": [1, 2, 3, 4],
        },
        index=index,
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").cumcount(),
        gdf.groupby("a").cumcount(),
        check_dtype=False,
    )

    assert_groupby_results_equal(
        pdf.groupby(["a", "b", "c"]).cumcount(),
        gdf.groupby(["a", "b", "c"]).cumcount(),
        check_dtype=False,
    )

    sr = pd.Series(range(len(pdf)), index=index)
    assert_groupby_results_equal(
        pdf.groupby(sr).cumcount(),
        gdf.groupby(sr).cumcount(),
        check_dtype=False,
    )


@pytest.mark.parametrize("nelem", [2, 3, 1000])
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize(
    "agg", ["min", "max", "idxmin", "idxmax", "mean", "count"]
)
def test_groupby_datetime(nelem, as_index, agg):
    if agg == "mean" and as_index is True:
        return
    check_dtype = agg not in ("mean", "count", "idxmin", "idxmax")
    pdf = make_frame(pd.DataFrame, nelem=nelem, with_datetime=True)
    gdf = make_frame(cudf.DataFrame, nelem=nelem, with_datetime=True)
    pdg = pdf.groupby("datetime", as_index=as_index)
    gdg = gdf.groupby("datetime", as_index=as_index)
    if as_index is False:
        pdres = getattr(pdg, agg)()
        gdres = getattr(gdg, agg)()
    else:
        pdres = pdg.agg({"datetime": agg})
        gdres = gdg.agg({"datetime": agg})
    assert_groupby_results_equal(
        pdres,
        gdres,
        check_dtype=check_dtype,
        as_index=as_index,
        by=["datetime"],
    )


def test_groupby_dropna():
    df = cudf.DataFrame({"a": [1, 1, None], "b": [1, 2, 3]})
    expect = cudf.DataFrame(
        {"b": [3, 3]}, index=cudf.Series([1, None], name="a")
    )
    got = df.groupby("a", dropna=False).sum()
    assert_groupby_results_equal(expect, got)

    df = cudf.DataFrame(
        {"a": [1, 1, 1, None], "b": [1, None, 1, None], "c": [1, 2, 3, 4]}
    )
    idx = cudf.MultiIndex.from_frame(
        df[["a", "b"]].drop_duplicates().sort_values(["a", "b"]),
        names=["a", "b"],
    )
    expect = cudf.DataFrame({"c": [4, 2, 4]}, index=idx)
    got = df.groupby(["a", "b"], dropna=False).sum()

    assert_groupby_results_equal(expect, got)


def test_groupby_dropna_getattr():
    df = cudf.DataFrame()
    df["id"] = [0, 1, 1, None, None, 3, 3]
    df["val"] = [0, 1, 1, 2, 2, 3, 3]
    got = df.groupby("id", dropna=False).val.sum()

    expect = cudf.Series(
        [0, 2, 6, 4], name="val", index=cudf.Series([0, 1, 3, None], name="id")
    )

    assert_groupby_results_equal(expect, got)


def test_groupby_categorical_from_string():
    gdf = cudf.DataFrame()
    gdf["id"] = ["a", "b", "c"]
    gdf["val"] = [0, 1, 2]
    gdf["id"] = gdf["id"].astype("category")
    assert_groupby_results_equal(
        cudf.DataFrame({"val": gdf["val"]}).set_index(keys=gdf["id"]),
        gdf.groupby("id").sum(),
    )


def test_groupby_arbitrary_length_series():
    gdf = cudf.DataFrame({"a": [1, 1, 2], "b": [2, 3, 4]}, index=[4, 5, 6])
    gsr = cudf.Series([1.0, 2.0, 2.0], index=[3, 4, 5])

    pdf = gdf.to_pandas()
    psr = gsr.to_pandas()

    expect = pdf.groupby(psr).sum()
    got = gdf.groupby(gsr).sum()

    assert_groupby_results_equal(expect, got)


def test_groupby_series_same_name_as_dataframe_column():
    gdf = cudf.DataFrame({"a": [1, 1, 2], "b": [2, 3, 4]}, index=[4, 5, 6])
    gsr = cudf.Series([1.0, 2.0, 2.0], name="a", index=[3, 4, 5])

    pdf = gdf.to_pandas()
    psr = gsr.to_pandas()

    expect = pdf.groupby(psr).sum()
    got = gdf.groupby(gsr).sum()

    assert_groupby_results_equal(expect, got)


def test_group_by_series_and_column_name_in_by():
    gdf = cudf.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [1, 2, 1]}, index=[1, 2, 3]
    )
    gsr0 = cudf.Series([0.0, 1.0, 2.0], name="a", index=[1, 2, 3])
    gsr1 = cudf.Series([0.0, 1.0, 3.0], name="b", index=[3, 4, 5])

    pdf = gdf.to_pandas()
    psr0 = gsr0.to_pandas()
    psr1 = gsr1.to_pandas()

    expect = pdf.groupby(["x", psr0, psr1]).sum()
    got = gdf.groupby(["x", gsr0, gsr1]).sum()

    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize(
    "grouper",
    [
        "a",
        ["a"],
        ["a", "b"],
        np.array([0, 1, 1, 2, 3, 2]),
        {0: "a", 1: "a", 2: "b", 3: "a", 4: "b", 5: "c"},
        lambda x: x + 1,
        ["a", np.array([0, 1, 1, 2, 3, 2])],
    ],
)
def test_grouping(grouper):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 3],
            "b": [1, 2, 1, 2, 1, 2],
            "c": [1, 2, 3, 4, 5, 6],
        }
    )
    gdf = cudf.from_pandas(pdf)

    for pdf_group, gdf_group in zip(
        pdf.groupby(grouper), gdf.groupby(grouper)
    ):
        assert pdf_group[0] == gdf_group[0]
        assert_eq(pdf_group[1], gdf_group[1])


@pytest.mark.parametrize("agg", [lambda x: x.count(), "count"])
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_count(agg, by):
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 3], "b": [1, 2, 2, 2, 1], "c": [1, 2, None, 4, 5]}
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by).agg(agg)
    got = gdf.groupby(by).agg(agg)

    assert_groupby_results_equal(expect, got, check_dtype=True)


@pytest.mark.parametrize("agg", [lambda x: x.median(), "median"])
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_median(agg, by):
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 3], "b": [1, 2, 2, 2, 1], "c": [1, 2, None, 4, 5]}
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by).agg(agg)
    got = gdf.groupby(by).agg(agg)

    assert_groupby_results_equal(expect, got, check_dtype=False)


@pytest.mark.parametrize("agg", [lambda x: x.nunique(), "nunique"])
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_nunique(agg, by):
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 3], "b": [1, 2, 2, 2, 1], "c": [1, 2, None, 4, 5]}
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by).nunique()
    got = gdf.groupby(by).nunique()

    assert_groupby_results_equal(expect, got, check_dtype=False)


@pytest.mark.parametrize("dropna", [True, False])
def test_nunique_dropna(dropna):
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 2],
            "b": [4, None, 5],
            "c": [None, None, 7],
            "d": [1, 1, 3],
        }
    )
    pdf = gdf.to_pandas()

    result = gdf.groupby("a")["b"].nunique(dropna=dropna)
    expected = pdf.groupby("a")["b"].nunique(dropna=dropna)
    assert_groupby_results_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    "n",
    [0, 1, 2, 10],
)
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_nth(n, by):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 3],
            "b": [1, 2, 2, 2, 1],
            "c": [1, 2, None, 4, 5],
            "d": ["a", "b", "c", "d", "e"],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by).nth(n)
    got = gdf.groupby(by).nth(n)

    assert_groupby_results_equal(expect, got, check_dtype=False)


def test_raise_data_error():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
    gdf = cudf.from_pandas(pdf)

    assert_exceptions_equal(
        pdf.groupby("a").mean,
        gdf.groupby("a").mean,
    )


def test_multi_agg():
    gdf = cudf.DataFrame(
        {"a": [1, 1, 2, 2], "b": [1, 2, 3, 4], "c": ["a", "b", "c", "d"]}
    )
    pdf = gdf.to_pandas()
    assert_groupby_results_equal(
        pdf.groupby("a").agg({"b": ["count", "mean"], "c": ["count"]}),
        gdf.groupby("a").agg({"b": ["count", "mean"], "c": ["count"]}),
    )


@pytest.mark.parametrize(
    "agg",
    (
        [
            *itertools.combinations(["count", "max", "min", "nunique"], 2),
            {"b": "min", "c": "mean"},
            {"b": "max", "c": "mean"},
            {"b": "count", "c": "mean"},
            {"b": "nunique", "c": "mean"},
        ]
    ),
)
def test_groupby_agg_combinations(agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
            "b": ["a", "a", "b", "c", "d"],
            "c": [1, 2, 3, 4, 5],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").agg(agg),
        gdf.groupby("a").agg(agg),
        check_dtype=False,
    )


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Include groups missing on old versions of pandas",
)
def test_groupby_apply_noempty_group():
    pdf = pd.DataFrame(
        {"a": [1, 1, 2, 2], "b": [1, 2, 1, 2], "c": [1, 2, 3, 4]}
    )
    gdf = cudf.from_pandas(pdf)

    expect = (
        pdf.groupby("a", group_keys=False)
        .apply(lambda x: x.iloc[[0, 1]], include_groups=False)
        .reset_index(drop=True)
    )
    got = (
        gdf.groupby("a")
        .apply(lambda x: x.iloc[[0, 1]], include_groups=False)
        .reset_index(drop=True)
    )
    assert_groupby_results_equal(expect, got)


def test_reset_index_after_empty_groupby():
    # GH #5475
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").sum().reset_index(),
        gdf.groupby("a").sum().reset_index(),
        as_index=False,
        by="a",
    )


def test_groupby_attribute_error():
    err_msg = "Test error message"

    class TestGroupBy(cudf.core.groupby.GroupBy):
        @property
        def _groupby(self):
            raise AttributeError(err_msg)

    a = cudf.DataFrame({"a": [1, 2], "b": [2, 3]})
    gb = TestGroupBy(a, a["a"])

    with pytest.raises(AttributeError, match=err_msg):
        gb.sum()


@pytest.mark.parametrize(
    "by",
    [
        "a",
        "b",
        ["a"],
        ["b"],
        ["a", "b"],
        ["b", "a"],
        np.array([0, 0, 0, 1, 1, 1, 2]),
    ],
)
def test_groupby_groups(by):
    pdf = pd.DataFrame(
        {"a": [1, 2, 1, 2, 1, 2, 3], "b": [1, 2, 3, 4, 5, 6, 7]}
    )
    gdf = cudf.from_pandas(pdf)

    pdg = pdf.groupby(by)
    gdg = gdf.groupby(by)

    for key in pdg.groups:
        assert key in gdg.groups
        assert_eq(pdg.groups[key], gdg.groups[key])


@pytest.mark.parametrize(
    "by",
    [
        "a",
        "b",
        ["a"],
        ["b"],
        ["a", "b"],
        ["b", "a"],
        ["a", "c"],
        ["a", "b", "c"],
    ],
)
def test_groupby_groups_multi(by):
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 1, 2, 1, 2, 3],
            "b": ["a", "b", "a", "b", "b", "c", "c"],
            "c": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    gdf = cudf.from_pandas(pdf)

    pdg = pdf.groupby(by)
    gdg = gdf.groupby(by)

    for key in pdg.groups:
        assert key in gdg.groups
        assert_eq(pdg.groups[key], gdg.groups[key])


def test_groupby_nunique_series():
    pdf = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 1, 1, 2]})
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a")["b"].nunique(),
        gdf.groupby("a")["b"].nunique(),
        check_dtype=False,
    )


@pytest.mark.parametrize("list_agg", [list, "collect"])
def test_groupby_list_simple(list_agg):
    pdf = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, None, 4, 5, 6]})
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").agg({"b": list}),
        gdf.groupby("a").agg({"b": list_agg}),
        check_dtype=False,
    )


@pytest.mark.parametrize("list_agg", [list, "collect"])
def test_groupby_list_of_lists(list_agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [[1, 2], [3, None, 5], None, [], [7, 8], [9]],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").agg({"b": list}),
        gdf.groupby("a").agg({"b": list_agg}),
        check_dtype=False,
    )


@pytest.mark.parametrize("list_agg", [list, "collect"])
def test_groupby_list_of_structs(list_agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [
                {"c": "1", "d": 1},
                {"c": "2", "d": 2},
                {"c": "3", "d": 3},
                {"c": "4", "d": 4},
                {"c": "5", "d": 5},
                {"c": "6", "d": 6},
            ],
        }
    )
    gdf = cudf.from_pandas(pdf)
    grouped = gdf.groupby("a").agg({"b": list_agg})
    assert_groupby_results_equal(
        pdf.groupby("a").agg({"b": list}),
        grouped,
        check_dtype=True,
    )
    assert grouped["b"].dtype.element_type == gdf["b"].dtype


@pytest.mark.parametrize("list_agg", [list, "collect"])
def test_groupby_list_single_element(list_agg):
    pdf = pd.DataFrame({"a": [1, 2], "b": [3, None]})
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").agg({"b": list}),
        gdf.groupby("a").agg({"b": list_agg}),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "agg", [list, [list, "count"], {"b": list, "c": "sum"}]
)
def test_groupby_list_strings(agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": ["b", "a", None, "e", "d"],
            "c": [1, 2, 3, 4, 5],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").agg(agg),
        gdf.groupby("a").agg(agg),
        check_dtype=False,
    )


def test_groupby_list_columns_excluded():
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 2, 3, 4],
            "c": [[1, 2], [3, 4], [5, 6], [7, 8]],
        }
    )
    gdf = cudf.from_pandas(pdf)

    pandas_result = pdf.groupby("a").mean(numeric_only=True)
    pandas_agg_result = pdf.groupby("a").agg("mean", numeric_only=True)

    assert_groupby_results_equal(
        pandas_result,
        gdf.groupby("a").mean(numeric_only=True),
        check_dtype=False,
    )

    assert_groupby_results_equal(
        pandas_agg_result,
        gdf.groupby("a").agg("mean"),
        check_dtype=False,
    )


def test_groupby_pipe():
    pdf = pd.DataFrame({"A": "a b a b".split(), "B": [1, 2, 3, 4]})
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("A").pipe(lambda x: x.max() - x.min())
    actual = gdf.groupby("A").pipe(lambda x: x.max() - x.min())

    assert_groupby_results_equal(expected, actual)


def create_test_groupby_apply_return_scalars_params():
    def f0(x):
        x = x[~x["B"].isna()]
        ticker = x.shape[0]
        full = ticker / 10
        return full

    def f1(x, k):
        x = x[~x["B"].isna()]
        ticker = x.shape[0]
        full = ticker / k
        return full

    def f2(x, k, L):
        x = x[~x["B"].isna()]
        ticker = x.shape[0]
        full = L * (ticker / k)
        return full

    def f3(x, k, L, m):
        x = x[~x["B"].isna()]
        ticker = x.shape[0]
        full = L * (ticker / k) % m
        return full

    return [(f0, ()), (f1, (42,)), (f2, (42, 119)), (f3, (42, 119, 212.1))]


@pytest.mark.parametrize(
    "func,args", create_test_groupby_apply_return_scalars_params()
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_apply_return_scalars(func, args):
    pdf = pd.DataFrame(
        {
            "A": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "B": [
                0.01,
                np.nan,
                0.03,
                0.04,
                np.nan,
                0.06,
                0.07,
                0.08,
                0.09,
                1.0,
            ],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("A").apply(func, *args, include_groups=False)
    actual = gdf.groupby("A").apply(func, *args, include_groups=False)

    assert_groupby_results_equal(expected, actual)


def create_test_groupby_apply_return_series_dataframe_params():
    def f0(x):
        return x - x.max()

    def f1(x):
        return x.min() - x.max()

    def f2(x):
        return x.min()

    def f3(x, k):
        return x - x.max() + k

    def f4(x, k, L):
        return x.min() - x.max() + (k / L)

    def f5(x, k, L, m):
        return m * x.min() + (k / L)

    return [
        (f0, ()),
        (f1, ()),
        (f2, ()),
        (f3, (42,)),
        (f4, (42, 119)),
        (f5, (41, 119, 212.1)),
    ]


@pytest.mark.parametrize(
    "func,args", create_test_groupby_apply_return_series_dataframe_params()
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Include groups missing on old versions of pandas",
)
def test_groupby_apply_return_series_dataframe(func, args):
    pdf = pd.DataFrame(
        {"key": [0, 0, 1, 1, 2, 2, 2], "val": [0, 1, 2, 3, 4, 5, 6]}
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby(["key"], group_keys=False).apply(
        func, *args, include_groups=False
    )
    actual = gdf.groupby(["key"]).apply(func, *args, include_groups=False)

    assert_groupby_results_equal(expected, actual)


@pytest.mark.parametrize(
    "pdf",
    [pd.DataFrame(), pd.DataFrame({"a": []}), pd.Series([], dtype="float64")],
)
def test_groupby_no_keys(pdf):
    gdf = cudf.from_pandas(pdf)
    if isinstance(pdf, pd.DataFrame):
        kwargs = {"check_column_type": False}
    else:
        kwargs = {}
    assert_groupby_results_equal(
        pdf.groupby([]).max(),
        gdf.groupby([]).max(),
        check_dtype=False,
        check_index_type=False,  # Int64 v/s Float64
        **kwargs,
    )


@pytest.mark.parametrize(
    "pdf",
    [pd.DataFrame(), pd.DataFrame({"a": []}), pd.Series([], dtype="float64")],
)
def test_groupby_apply_no_keys(pdf):
    gdf = cudf.from_pandas(pdf)
    if isinstance(pdf, pd.DataFrame):
        kwargs = {"check_column_type": False}
    else:
        kwargs = {}
    assert_groupby_results_equal(
        pdf.groupby([], group_keys=False).apply(lambda x: x.max()),
        gdf.groupby([]).apply(lambda x: x.max()),
        check_index_type=False,  # Int64 v/s Float64
        **kwargs,
    )


@pytest.mark.parametrize(
    "pdf",
    [pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 2], "b": [2, 3]})],
)
def test_groupby_nonempty_no_keys(pdf):
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lambda: pdf.groupby([]),
        lambda: gdf.groupby([]),
    )


@pytest.mark.parametrize(
    "by,data",
    [
        # ([], []),  # error?
        ([1, 1, 2, 2], [0, 0, 1, 1]),
        ([1, 2, 3, 4], [0, 0, 0, 0]),
        ([1, 2, 1, 2], [0, 1, 1, 1]),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    SIGNED_TYPES + DATETIME_TYPES + TIMEDELTA_TYPES + ["string", "category"],
)
def test_groupby_unique(by, data, dtype):
    pdf = pd.DataFrame({"by": by, "data": data})
    pdf["data"] = pdf["data"].astype(dtype)
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby("by")["data"].unique()
    got = gdf.groupby("by")["data"].unique()
    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
@pytest.mark.parametrize(
    "func", ["cummin", "cummax", "cumcount", "cumsum", "cumprod"]
)
def test_groupby_2keys_scan(nelem, func):
    pdf = make_frame(pd.DataFrame, nelem=nelem)
    expect_df = pdf.groupby(["x", "y"], sort=True).agg(func)
    got_df = (
        make_frame(DataFrame, nelem=nelem)
        .groupby(["x", "y"], sort=True)
        .agg(func)
    )
    # pd.groupby.cumcount returns a series.
    if isinstance(expect_df, pd.Series):
        expect_df = expect_df.to_frame("val")

    check_dtype = func not in _index_type_aggs
    assert_groupby_results_equal(got_df, expect_df, check_dtype=check_dtype)


@pytest.mark.parametrize("nelem", [100, 1000])
@pytest.mark.parametrize("method", ["average", "min", "max", "first", "dense"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_option", ["keep", "top", "bottom"])
@pytest.mark.parametrize("pct", [False, True])
def test_groupby_2keys_rank(nelem, method, ascending, na_option, pct):
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    pdf = t.to_pandas()
    pdf.columns = ["x", "y", "z"]
    gdf = cudf.from_pandas(pdf)
    expect_df = pdf.groupby(["x", "y"], sort=True).rank(
        method=method, ascending=ascending, na_option=na_option, pct=pct
    )
    got_df = gdf.groupby(["x", "y"], sort=True).rank(
        method=method, ascending=ascending, na_option=na_option, pct=pct
    )

    assert_groupby_results_equal(got_df, expect_df, check_dtype=False)


def test_groupby_rank_fails():
    gdf = cudf.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "z": [1, 2, 3, 4]}
    )
    with pytest.raises(NotImplementedError):
        gdf.groupby(["x", "y"]).rank(method="min", axis=1)
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [[1, 2], [3, None, 5], None, [], [7, 8], [9]],
        }
    )
    with pytest.raises(NotImplementedError):
        gdf.groupby(["a"]).rank(method="min", axis=1)


@pytest.mark.parametrize(
    "with_nan", [False, True], ids=["just-NA", "also-NaN"]
)
@pytest.mark.parametrize("dropna", [False, True], ids=["keepna", "dropna"])
@pytest.mark.parametrize(
    "duplicate_index", [False, True], ids=["rangeindex", "dupindex"]
)
def test_groupby_scan_null_keys(with_nan, dropna, duplicate_index):
    key_col = [None, 1, 2, None, 3, None, 3, 1, None, 1]
    if with_nan:
        df = pd.DataFrame(
            {"key": pd.Series(key_col, dtype="float32"), "value": range(10)}
        )
    else:
        df = pd.DataFrame(
            {"key": pd.Series(key_col, dtype="Int32"), "value": range(10)}
        )

    if duplicate_index:
        # Non-default index with duplicates
        df.index = [1, 2, 3, 1, 3, 2, 4, 1, 6, 10]

    cdf = cudf.from_pandas(df)

    expect = df.groupby("key", dropna=dropna).cumsum()
    got = cdf.groupby("key", dropna=dropna).cumsum()
    assert_eq(expect, got)


def test_groupby_mix_agg_scan():
    err_msg = "Cannot perform both aggregation and scan in one operation"
    func = ["cumsum", "sum"]
    gb = make_frame(DataFrame, nelem=10).groupby(["x", "y"], sort=True)

    gb.agg(func[0])
    gb.agg(func[1])
    gb.agg(func[1:])
    with pytest.raises(NotImplementedError, match=err_msg):
        gb.agg(func)


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("fill_value", [None, np.nan, 42])
def test_groupby_shift_row(nelem, shift_perc, direction, fill_value):
    pdf = make_frame(pd.DataFrame, nelem=nelem, extra_vals=["val2"])
    gdf = cudf.from_pandas(pdf)
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["x", "y"]).shift(
        periods=n_shift, fill_value=fill_value
    )
    got = gdf.groupby(["x", "y"]).shift(periods=n_shift, fill_value=fill_value)

    assert_groupby_results_equal(
        expected[["val", "val2"]], got[["val", "val2"]]
    )


@pytest.mark.parametrize("nelem", [10, 50, 100, 1000])
@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize(
    "fill_value",
    [
        None,
        pytest.param(
            0,
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/10608"
            ),
        ),
        pytest.param(
            42,
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/10608"
            ),
        ),
    ],
)
def test_groupby_shift_row_mixed_numerics(
    nelem, shift_perc, direction, fill_value
):
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["0"]).shift(periods=n_shift, fill_value=fill_value)
    got = gdf.groupby(["0"]).shift(periods=n_shift, fill_value=fill_value)

    assert_groupby_results_equal(
        expected[["1", "2", "3", "4"]], got[["1", "2", "3", "4"]]
    )


# TODO: Shifting list columns is currently unsupported because we cannot
# construct a null list scalar in python. Support once it is added.
@pytest.mark.parametrize("nelem", [10, 50, 100, 1000])
@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
def test_groupby_shift_row_mixed(nelem, shift_perc, direction):
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "str", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["0"]).shift(periods=n_shift)
    got = gdf.groupby(["0"]).shift(periods=n_shift)

    assert_groupby_results_equal(
        expected[["1", "2", "3", "4"]], got[["1", "2", "3", "4"]]
    )


@pytest.mark.parametrize("nelem", [10, 50, 100, 1000])
@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize(
    "fill_value",
    [
        [
            42,
            "fill",
            np.datetime64(123, "ns"),
            cudf.Scalar(456, dtype="timedelta64[ns]"),
        ]
    ],
)
def test_groupby_shift_row_mixed_fill(
    nelem, shift_perc, direction, fill_value
):
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "str", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)
    n_shift = int(nelem * shift_perc) * direction

    # Pandas does not support specifying different fill_value by column, so we
    # simulate it column by column
    expected = pdf.copy()
    for col, single_fill in zip(pdf.iloc[:, 1:], fill_value):
        if isinstance(single_fill, cudf.Scalar):
            single_fill = single_fill._host_value
        expected[col] = (
            pdf[col]
            .groupby(pdf["0"])
            .shift(periods=n_shift, fill_value=single_fill)
        )

    got = gdf.groupby(["0"]).shift(periods=n_shift, fill_value=fill_value)

    assert_groupby_results_equal(
        expected[["1", "2", "3", "4"]], got[["1", "2", "3", "4"]]
    )


@pytest.mark.parametrize("nelem", [10, 50, 100, 1000])
@pytest.mark.parametrize("fill_value", [None, 0, 42])
def test_groupby_shift_row_zero_shift(nelem, fill_value):
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    gdf = cudf.from_pandas(t.to_pandas())

    expected = gdf
    got = gdf.groupby(["0"]).shift(periods=0, fill_value=fill_value)

    assert_groupby_results_equal(
        expected[["1", "2", "3", "4"]], got[["1", "2", "3", "4"]]
    )


@pytest.mark.parametrize("nelem", [2, 3, 100, 1000])
@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
def test_groupby_diff_row(nelem, shift_perc, direction):
    pdf = make_frame(pd.DataFrame, nelem=nelem, extra_vals=["val2"])
    gdf = cudf.from_pandas(pdf)
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["x", "y"]).diff(periods=n_shift)
    got = gdf.groupby(["x", "y"]).diff(periods=n_shift)

    assert_groupby_results_equal(
        expected[["val", "val2"]], got[["val", "val2"]]
    )


@pytest.mark.parametrize("nelem", [10, 50, 100, 1000])
@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
def test_groupby_diff_row_mixed_numerics(nelem, shift_perc, direction):
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "decimal64", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["0"]).diff(periods=n_shift)
    got = gdf.groupby(["0"]).diff(periods=n_shift)

    assert_groupby_results_equal(
        expected[["1", "2", "3", "4", "5"]], got[["1", "2", "3", "4", "5"]]
    )


@pytest.mark.parametrize("nelem", [10, 50, 100, 1000])
def test_groupby_diff_row_zero_shift(nelem):
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    gdf = cudf.from_pandas(t.to_pandas())

    expected = gdf
    got = gdf.groupby(["0"]).shift(periods=0)

    assert_groupby_results_equal(
        expected[["1", "2", "3", "4"]], got[["1", "2", "3", "4"]]
    )


# TODO: test for category columns when cudf.Scalar supports category type
@pytest.mark.parametrize("nelem", [10, 100, 1000])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
def test_groupby_fillna_multi_value(nelem):
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ms]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {"dtype": "decimal64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "str", "null_frequency": 0.4, "cardinality": 10},
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    key_col = "0"
    value_cols = ["1", "2", "3", "4", "5", "6"]
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)

    # fill the dataframe with the first non-null item in the column
    fill_values = {
        name: pdf[name].loc[pdf[name].first_valid_index()]
        for name in value_cols
    }
    # cudf can't fillna with a pandas.Timedelta type
    fill_values["4"] = fill_values["4"].to_numpy()
    with pytest.warns(FutureWarning):
        expect = pdf.groupby(key_col).fillna(value=fill_values)
    with pytest.warns(FutureWarning):
        got = gdf.groupby(key_col).fillna(value=fill_values)

    assert_groupby_results_equal(expect[value_cols], got[value_cols])


# TODO: test for category columns when cudf.Scalar supports category type
# TODO: cudf.fillna does not support decimal column to column fill yet
@pytest.mark.parametrize("nelem", [10, 100, 1000])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
def test_groupby_fillna_multi_value_df(nelem):
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ms]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {"dtype": "str", "null_frequency": 0.4, "cardinality": 10},
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    key_col = "0"
    value_cols = ["1", "2", "3", "4", "5"]
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)

    # fill the dataframe with the first non-null item in the column
    fill_values = {
        name: pdf[name].loc[pdf[name].first_valid_index()]
        for name in value_cols
    }
    # cudf can't fillna with a pandas.Timedelta type
    fill_values["4"] = fill_values["4"].to_numpy()
    fill_values = pd.DataFrame(fill_values, index=pdf.index)
    with pytest.warns(FutureWarning):
        expect = pdf.groupby(key_col).fillna(value=fill_values)

    fill_values = cudf.from_pandas(fill_values)
    with pytest.warns(FutureWarning):
        got = gdf.groupby(key_col).fillna(value=fill_values)

    assert_groupby_results_equal(expect[value_cols], got[value_cols])


@pytest.mark.parametrize(
    "by",
    [pd.Series([1, 1, 2, 2, 3, 4]), lambda x: x % 2 == 0, pd.Grouper(level=0)],
)
@pytest.mark.parametrize(
    "data", [[1, None, 2, None, 3, None], [1, 2, 3, 4, 5, 6]]
)
@pytest.mark.parametrize("args", [{"value": 42}, {"method": "ffill"}])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
def test_groupby_various_by_fillna(by, data, args):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    with pytest.warns(FutureWarning):
        expect = ps.groupby(by).fillna(**args)
    if isinstance(by, pd.Grouper):
        by = cudf.Grouper(level=by.level)
    with pytest.warns(FutureWarning):
        got = gs.groupby(by).fillna(**args)

    assert_groupby_results_equal(expect, got, check_dtype=False)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize("nelem", [10, 100, 1000])
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_groupby_fillna_method(nelem, method):
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "list",
                "null_frequency": 0.4,
                "cardinality": 10,
                "lists_max_length": 10,
                "nesting_max_depth": 3,
                "value_type": "int64",
            },
            {"dtype": "category", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "decimal64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "str", "null_frequency": 0.4, "cardinality": 10},
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    key_col = "0"
    value_cols = ["1", "2", "3", "4", "5", "6", "7", "8"]
    pdf = t.to_pandas()
    gdf = cudf.from_pandas(pdf)

    with pytest.warns(FutureWarning):
        expect = pdf.groupby(key_col).fillna(method=method)
    with pytest.warns(FutureWarning):
        got = gdf.groupby(key_col).fillna(method=method)

    assert_groupby_results_equal(
        expect[value_cols], got[value_cols], sort=False
    )


@pytest.mark.parametrize(
    "data",
    [
        {"Speed": [380.0, 370.0, 24.0, 26.0], "Score": [50, 30, 90, 80]},
        {
            "Speed": [380.0, 370.0, 24.0, 26.0],
            "Score": [50, 30, 90, 80],
            "Other": [10, 20, 30, 40],
        },
    ],
)
@pytest.mark.parametrize("group", ["Score", "Speed"])
def test_groupby_describe(data, group):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    got = gdf.groupby(group).describe()
    expect = pdf.groupby(group).describe()

    assert_groupby_results_equal(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [], "b": []},
        {"a": [2, 1, 2, 1, 1, 3], "b": [None, 1, 2, None, 2, None]},
        {"a": [None], "b": [None]},
        {"a": [2, 1, 1], "b": [None, 1, 0], "c": [None, 0, 1]},
    ],
)
@pytest.mark.parametrize("agg", ["first", "last", ["first", "last"]])
def test_groupby_first(data, agg):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)
    expect = pdf.groupby("a").agg(agg)
    got = gdf.groupby("a").agg(agg)
    assert_groupby_results_equal(expect, got, check_dtype=False)


def test_groupby_apply_series():
    def foo(x):
        return x.sum()

    got = make_frame(DataFrame, 100).groupby("x").y.apply(foo)
    expect = make_frame(pd.DataFrame, 100).groupby("x").y.apply(foo)

    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize(
    "func,args",
    [
        (lambda x, k: x + k, (42,)),
        (lambda x, k, L: x + k - L, (42, 191)),
        (lambda x, k, L, m: (x + k) / (L * m), (42, 191, 99.9)),
    ],
)
def test_groupby_apply_series_args(func, args):
    got = make_frame(DataFrame, 100).groupby("x").y.apply(func, *args)
    expect = (
        make_frame(pd.DataFrame, 100)
        .groupby("x", group_keys=False)
        .y.apply(func, *args)
    )

    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize("label", [None, "left", "right"])
@pytest.mark.parametrize("closed", [None, "left", "right"])
def test_groupby_freq_week(label, closed):
    pdf = pd.DataFrame(
        {
            "Publish date": [
                pd.Timestamp("2000-01-03"),
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-09"),
                pd.Timestamp("2000-01-02"),
                pd.Timestamp("2000-01-07"),
                pd.Timestamp("2000-01-16"),
            ],
            "ID": [0, 1, 2, 3, 4, 5],
            "Price": [10, 20, 30, 40, 50, 60],
        }
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.groupby(
        pd.Grouper(key="Publish date", freq="1W", label=label, closed=closed)
    ).mean()
    got = gdf.groupby(
        cudf.Grouper(key="Publish date", freq="1W", label=label, closed=closed)
    ).mean()
    assert_eq(
        expect,
        got,
        check_like=True,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize("label", [None, "left", "right"])
@pytest.mark.parametrize("closed", [None, "left", "right"])
def test_groupby_freq_day(label, closed):
    pdf = pd.DataFrame(
        {
            "Publish date": [
                pd.Timestamp("2000-01-03"),
                pd.Timestamp("2000-01-01"),
                pd.Timestamp("2000-01-09"),
                pd.Timestamp("2000-01-02"),
                pd.Timestamp("2000-01-07"),
                pd.Timestamp("2000-01-16"),
            ],
            "ID": [0, 1, 2, 3, 4, 5],
            "Price": [10, 20, 30, 40, 50, 60],
        }
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.groupby(
        pd.Grouper(key="Publish date", freq="3D", label=label, closed=closed)
    ).mean()
    got = gdf.groupby(
        cudf.Grouper(key="Publish date", freq="3D", label=label, closed=closed)
    ).mean()
    assert_eq(
        expect,
        got,
        check_like=True,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize("label", [None, "left", "right"])
@pytest.mark.parametrize("closed", [None, "left", "right"])
def test_groupby_freq_min(label, closed):
    pdf = pd.DataFrame(
        {
            "Publish date": [
                pd.Timestamp("2000-01-01 12:01:00"),
                pd.Timestamp("2000-01-01 12:05:00"),
                pd.Timestamp("2000-01-01 15:30:00"),
                pd.Timestamp("2000-01-02 00:00:00"),
                pd.Timestamp("2000-01-01 23:47:00"),
                pd.Timestamp("2000-01-02 00:05:00"),
            ],
            "ID": [0, 1, 2, 3, 4, 5],
            "Price": [10, 20, 30, 40, 50, 60],
        }
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.groupby(
        pd.Grouper(key="Publish date", freq="1h", label=label, closed=closed)
    ).mean()
    got = gdf.groupby(
        cudf.Grouper(key="Publish date", freq="1h", label=label, closed=closed)
    ).mean()
    assert_eq(
        expect,
        got,
        check_like=True,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize("label", [None, "left", "right"])
@pytest.mark.parametrize("closed", [None, "left", "right"])
def test_groupby_freq_s(label, closed):
    pdf = pd.DataFrame(
        {
            "Publish date": [
                pd.Timestamp("2000-01-01 00:00:02"),
                pd.Timestamp("2000-01-01 00:00:07"),
                pd.Timestamp("2000-01-01 00:00:02"),
                pd.Timestamp("2000-01-02 00:00:15"),
                pd.Timestamp("2000-01-01 00:00:05"),
                pd.Timestamp("2000-01-02 00:00:09"),
            ],
            "ID": [0, 1, 2, 3, 4, 5],
            "Price": [10, 20, 30, 40, 50, 60],
        }
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.groupby(
        pd.Grouper(key="Publish date", freq="3s", label=label, closed=closed)
    ).mean()
    got = gdf.groupby(
        cudf.Grouper(key="Publish date", freq="3s", label=label, closed=closed)
    ).mean()
    assert_eq(
        expect,
        got,
        check_like=True,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    "pdf, group, name, obj",
    [
        (
            pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]}),
            "X",
            "A",
            None,
        ),
        (
            pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]}),
            "X",
            "B",
            None,
        ),
        (
            pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]}),
            "X",
            "A",
            pd.DataFrame({"a": [1, 2, 4, 5, 10, 11]}),
        ),
        (
            pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]}),
            "Y",
            1,
            pd.DataFrame({"a": [1, 2, 4, 5, 10, 11]}),
        ),
        (
            pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]}),
            "Y",
            3,
            pd.DataFrame({"a": [1, 2, 0, 11]}),
        ),
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Warnings only given on newer versions.",
)
def test_groupby_get_group(pdf, group, name, obj):
    gdf = cudf.from_pandas(pdf)

    if isinstance(obj, pd.DataFrame):
        gobj = cudf.from_pandas(obj)
    else:
        gobj = obj

    pgb = pdf.groupby(group)
    ggb = gdf.groupby(group)
    with expect_warning_if(obj is not None):
        expected = pgb.get_group(name=name, obj=obj)
    with expect_warning_if(obj is not None):
        actual = ggb.get_group(name=name, obj=gobj)

    assert_groupby_results_equal(expected, actual)

    expected = pdf.iloc[pgb.indices.get(name)]
    actual = gdf.iloc[ggb.indices.get(name)]

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "by",
    [
        "a",
        ["a", "b"],
        pd.Series([2, 1, 1, 2, 2]),
        pd.Series(["b", "a", "a", "b", "b"]),
    ],
)
@pytest.mark.parametrize("agg", ["sum", "mean", lambda df: df.mean()])
def test_groupby_transform_aggregation(by, agg):
    gdf = cudf.DataFrame(
        {"a": [2, 2, 1, 2, 1], "b": [1, 1, 1, 2, 2], "c": [1, 2, 3, 4, 5]}
    )
    pdf = gdf.to_pandas()

    expected = pdf.groupby(by).transform(agg)
    actual = gdf.groupby(by).transform(agg)

    assert_groupby_results_equal(expected, actual)


def test_groupby_select_then_ffill():
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [1, None, None, 2, None],
            "c": [3, None, None, 4, None],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("a")["c"].ffill()
    actual = gdf.groupby("a")["c"].ffill()

    assert_groupby_results_equal(expected, actual)


def test_groupby_select_then_shift():
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5], "c": [3, 4, 5, 6, 7]}
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("a")["c"].shift(1)
    actual = gdf.groupby("a")["c"].shift(1)

    assert_groupby_results_equal(expected, actual)


def test_groupby_select_then_diff():
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5], "c": [3, 4, 5, 6, 7]}
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("a")["c"].diff(1)
    actual = gdf.groupby("a")["c"].diff(1)

    assert_groupby_results_equal(expected, actual)


# TODO: Add a test including datetime64[ms] column in input data


@pytest.mark.parametrize("by", ["a", ["a", "b"], pd.Series([1, 2, 1, 3])])
def test_groupby_transform_maintain_index(by):
    # test that we maintain the index after a groupby transform
    gdf = cudf.DataFrame(
        {"a": [1, 1, 1, 2], "b": [1, 2, 1, 2]}, index=[3, 2, 1, 0]
    )
    pdf = gdf.to_pandas()
    assert_groupby_results_equal(
        pdf.groupby(by).transform("max"), gdf.groupby(by).transform("max")
    )


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "data, gkey",
    [
        (
            {
                "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
                "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
                "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1],
            },
            ["id"],
        ),
        (
            {
                "id": [0, 0, 0, 0, 1, 1, 1],
                "a": [1, 3, 4, 2.0, -3.0, 9.0, 10.0],
                "b": [10.0, 23, -4.0, 2, -3.0, None, 19.0],
            },
            ["id", "a"],
        ),
        (
            {
                "id": ["a", "a", "b", "b", "c", "c"],
                "val1": [None, None, None, None, None, None],
            },
            ["id"],
        ),
    ],
)
@pytest.mark.parametrize("periods", [-5, -2, 0, 2, 5])
@pytest.mark.parametrize("fill_method", ["ffill", "bfill", no_default, None])
def test_groupby_pct_change(data, gkey, periods, fill_method):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    with expect_warning_if(fill_method not in (no_default, None)):
        actual = gdf.groupby(gkey).pct_change(
            periods=periods, fill_method=fill_method
        )
    with expect_warning_if(
        (
            fill_method not in (no_default, None)
            or (fill_method is not None and pdf.isna().any().any())
        )
    ):
        expected = pdf.groupby(gkey).pct_change(
            periods=periods, fill_method=fill_method
        )

    assert_eq(expected, actual)


@pytest.mark.parametrize("periods", [-5, 5])
def test_groupby_pct_change_multiindex_dataframe(periods):
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 1, 2, 3],
            "c": [2, 3, 4, 5],
            "d": [6, 8, 9, 1],
        }
    ).set_index(["a", "b"])

    actual = gdf.groupby(level=["a", "b"]).pct_change(periods)
    expected = gdf.to_pandas().groupby(level=["a", "b"]).pct_change(periods)

    assert_eq(expected, actual)


def test_groupby_pct_change_empty_columns():
    gdf = cudf.DataFrame(columns=["id", "val1", "val2"])
    pdf = gdf.to_pandas()

    actual = gdf.groupby("id").pct_change()
    expected = pdf.groupby("id").pct_change()

    assert_eq(expected, actual)


@pytest.mark.parametrize("group_keys", [None, True, False])
@pytest.mark.parametrize("by", ["A", ["A", "B"]])
def test_groupby_group_keys(group_keys, by):
    gdf = cudf.DataFrame(
        {
            "A": "a a a a b b".split(),
            "B": [1, 1, 2, 2, 3, 3],
            "C": [4, 6, 5, 9, 8, 7],
        }
    )
    pdf = gdf.to_pandas()

    g_group = gdf.groupby(by, group_keys=group_keys)
    p_group = pdf.groupby(by, group_keys=group_keys)

    actual = g_group[["B", "C"]].apply(lambda x: x / x.sum())
    expected = p_group[["B", "C"]].apply(lambda x: x / x.sum())
    assert_eq(actual, expected)


@pytest.fixture
def df_ngroup():
    df = cudf.DataFrame(
        {
            "a": [2, 2, 1, 1, 2, 3],
            "b": [1, 2, 1, 2, 1, 2],
            "c": ["a", "a", "b", "c", "d", "c"],
        },
        index=[1, 3, 5, 7, 4, 2],
    )
    df.index.name = "foo"
    return df


@pytest.mark.parametrize(
    "by",
    [
        lambda: "a",
        lambda: "b",
        lambda: ["a", "b"],
        lambda: "c",
        lambda: pd.Series([1, 2, 1, 2, 1, 2]),
        lambda: pd.Series(["x", "y", "y", "x", "z", "x"]),
    ],
)
@pytest.mark.parametrize("ascending", [True, False])
def test_groupby_ngroup(by, ascending, df_ngroup):
    by = by()
    expected = df_ngroup.to_pandas().groupby(by).ngroup(ascending=ascending)
    actual = df_ngroup.groupby(by).ngroup(ascending=ascending)
    assert_eq(expected, actual, check_dtype=False)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "groups", ["a", "b", "c", ["a", "c"], ["a", "b", "c"]]
)
def test_groupby_dtypes(groups):
    df = cudf.DataFrame(
        {"a": [1, 2, 3, 3], "b": ["x", "y", "z", "a"], "c": [10, 11, 12, 12]}
    )
    pdf = df.to_pandas()
    with pytest.warns(FutureWarning):
        expected = pdf.groupby(groups).dtypes
    with pytest.warns(FutureWarning):
        actual = df.groupby(groups).dtypes

    assert_eq(expected, actual)


@pytest.mark.parametrize("index_names", ["a", "b", "c", ["b", "c"]])
def test_groupby_by_index_names(index_names):
    gdf = cudf.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["a", "b", "a", "a"], "c": [1, 1, 2, 1]}
    ).set_index(index_names)
    pdf = gdf.to_pandas()

    assert_groupby_results_equal(
        pdf.groupby(index_names).min(), gdf.groupby(index_names).min()
    )


@pytest.mark.parametrize(
    "groups", ["a", "b", "c", ["a", "c"], ["a", "b", "c"]]
)
def test_group_by_pandas_compat(groups):
    with cudf.option_context("mode.pandas_compatible", True):
        df = cudf.DataFrame(
            {
                "a": [1, 3, 2, 3, 3],
                "b": ["x", "a", "y", "z", "a"],
                "c": [10, 13, 11, 12, 12],
            }
        )
        pdf = df.to_pandas()

        assert_eq(pdf.groupby(groups).max(), df.groupby(groups).max())


class TestSample:
    @pytest.fixture(params=["default", "rangeindex", "intindex", "strindex"])
    def index(self, request):
        n = 12
        if request.param == "rangeindex":
            return cudf.RangeIndex(2, n + 2)
        elif request.param == "intindex":
            return cudf.Index(
                [2, 3, 4, 1, 0, 5, 6, 8, 7, 9, 10, 13], dtype="int32"
            )
        elif request.param == "strindex":
            return cudf.Index(list(string.ascii_lowercase[:n]))
        elif request.param == "default":
            return None

    @pytest.fixture(
        params=[
            ["a", "a", "b", "b", "c", "c", "c", "d", "d", "d", "d", "d"],
            [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4],
        ],
        ids=["str-group", "int-group"],
    )
    def df(self, index, request):
        return cudf.DataFrame(
            {"a": request.param, "b": request.param, "v": request.param},
            index=index,
        )

    @pytest.fixture(params=["a", ["a", "b"]], ids=["single-col", "two-col"])
    def by(self, request):
        return request.param

    def expected(self, df, *, n=None, frac=None):
        value_counts = collections.Counter(df.a.values_host)
        if n is not None:
            values = list(
                itertools.chain.from_iterable(
                    itertools.repeat(v, n) for v in value_counts.keys()
                )
            )
        elif frac is not None:
            values = list(
                itertools.chain.from_iterable(
                    itertools.repeat(v, round(count * frac))
                    for v, count in value_counts.items()
                )
            )
        else:
            raise ValueError("Must provide either n or frac")
        values = cudf.Series(sorted(values), dtype=df.a.dtype)
        return cudf.DataFrame({"a": values, "b": values, "v": values})

    @pytest.mark.parametrize("n", [None, 0, 1, 2])
    def test_constant_n_no_replace(self, df, by, n):
        result = df.groupby(by).sample(n=n).sort_values("a")
        n = 1 if n is None else n
        assert_eq(self.expected(df, n=n), result.reset_index(drop=True))

    def test_constant_n_no_replace_too_large_raises(self, df):
        with pytest.raises(ValueError):
            df.groupby("a").sample(n=3)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_constant_n_replace(self, df, by, n):
        result = df.groupby(by).sample(n=n, replace=True).sort_values("a")
        assert_eq(self.expected(df, n=n), result.reset_index(drop=True))

    def test_invalid_arguments(self, df):
        with pytest.raises(ValueError):
            df.groupby("a").sample(n=1, frac=0.1)

    def test_not_implemented_arguments(self, df):
        with pytest.raises(NotImplementedError):
            # These are valid weights, but we don't implement this yet.
            df.groupby("a").sample(n=1, weights=[1 / len(df)] * len(df))

    @pytest.mark.parametrize("frac", [0, 1 / 3, 1 / 2, 2 / 3, 1])
    @pytest.mark.parametrize("replace", [False, True])
    def test_fraction_rounding(self, df, by, frac, replace):
        result = (
            df.groupby(by).sample(frac=frac, replace=replace).sort_values("a")
        )
        assert_eq(self.expected(df, frac=frac), result.reset_index(drop=True))


class TestHeadTail:
    @pytest.fixture(params=[-3, -2, -1, 0, 1, 2, 3], ids=lambda n: f"{n=}")
    def n(self, request):
        return request.param

    @pytest.fixture(
        params=[False, True], ids=["no-preserve-order", "preserve-order"]
    )
    def preserve_order(self, request):
        return request.param

    @pytest.fixture
    def df(self):
        return cudf.DataFrame(
            {
                "a": [1, 0, 1, 2, 2, 1, 3, 2, 3, 3, 3],
                "b": [0, 1, 2, 4, 3, 5, 6, 7, 9, 8, 10],
            }
        )

    @pytest.fixture(params=[True, False], ids=["head", "tail"])
    def take_head(self, request):
        return request.param

    @pytest.fixture
    def expected(self, df, n, take_head, preserve_order):
        if n == 0:
            # We'll get an empty dataframe in this case
            return df._empty_like(keep_index=True)
        else:
            if preserve_order:
                # Should match pandas here
                g = df.to_pandas().groupby("a")
                if take_head:
                    return g.head(n=n)
                else:
                    return g.tail(n=n)
            else:
                # We groupby "a" which is the first column. This
                # possibly relies on an implementation detail that for
                # integer group keys, cudf produces groups in sorted
                # (ascending) order.
                keyfunc = operator.itemgetter(0)
                if take_head or n == 0:
                    # Head does group[:n] as does tail for n == 0
                    slicefunc = operator.itemgetter(slice(None, n))
                else:
                    # Tail does group[-n:] except when n == 0
                    slicefunc = operator.itemgetter(
                        slice(-n, None) if n else slice(0)
                    )
                values_to_sort = np.hstack(
                    [df.values_host, np.arange(len(df)).reshape(-1, 1)]
                )
                expect_a, expect_b, index = zip(
                    *itertools.chain.from_iterable(
                        slicefunc(list(group))
                        for _, group in itertools.groupby(
                            sorted(values_to_sort.tolist(), key=keyfunc),
                            key=keyfunc,
                        )
                    )
                )
                return cudf.DataFrame(
                    {"a": expect_a, "b": expect_b}, index=index
                )

    def test_head_tail(self, df, n, take_head, expected, preserve_order):
        if take_head:
            actual = df.groupby("a").head(n=n, preserve_order=preserve_order)
        else:
            actual = df.groupby("a").tail(n=n, preserve_order=preserve_order)
        assert_eq(actual, expected)


def test_head_tail_empty():
    # GH #13397

    values = [1, 2, 3]
    pdf = pd.DataFrame({}, index=values)
    df = cudf.DataFrame({}, index=values)

    expected = pdf.groupby(pd.Series(values)).head()
    got = df.groupby(cudf.Series(values)).head()
    assert_eq(expected, got, check_column_type=False)

    expected = pdf.groupby(pd.Series(values)).tail()
    got = df.groupby(cudf.Series(values)).tail()

    assert_eq(expected, got, check_column_type=False)


@pytest.mark.parametrize(
    "groups", ["a", "b", "c", ["a", "c"], ["a", "b", "c"]]
)
@pytest.mark.parametrize("sort", [True, False])
def test_group_by_pandas_sort_order(groups, sort):
    with cudf.option_context("mode.pandas_compatible", True):
        df = cudf.DataFrame(
            {
                "a": [10, 1, 10, 3, 2, 1, 3, 3],
                "b": [5, 6, 7, 1, 2, 3, 4, 9],
                "c": [20, 20, 10, 11, 13, 11, 12, 12],
            }
        )
        pdf = df.to_pandas()

        assert_eq(
            pdf.groupby(groups, sort=sort).sum(),
            df.groupby(groups, sort=sort).sum(),
        )


@pytest.mark.parametrize(
    "dtype",
    ["int32", "int64", "float64", "datetime64[ns]", "timedelta64[ns]", "bool"],
)
@pytest.mark.parametrize(
    "reduce_op",
    [
        "min",
        "max",
        "idxmin",
        "idxmax",
        "first",
        "last",
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_group_by_empty_reduction(dtype, reduce_op):
    gdf = cudf.DataFrame({"a": [], "b": [], "c": []}, dtype=dtype)
    pdf = gdf.to_pandas()

    gg = gdf.groupby("a")["c"]
    pg = pdf.groupby("a")["c"]

    assert_eq(
        getattr(gg, reduce_op)(), getattr(pg, reduce_op)(), check_dtype=True
    )


@pytest.mark.parametrize(
    "dtype",
    ["int32", "int64", "float64", "datetime64[ns]", "timedelta64[ns]", "bool"],
)
@pytest.mark.parametrize(
    "apply_op",
    ["sum", "min", "max", "idxmax"],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_group_by_empty_apply(request, dtype, apply_op):
    request.applymarker(
        pytest.mark.xfail(
            condition=(dtype == "datetime64[ns]" and apply_op == "sum"),
            reason=("sum isn't supported for datetime64[ns]"),
        )
    )

    gdf = cudf.DataFrame({"a": [], "b": [], "c": []}, dtype=dtype)
    pdf = gdf.to_pandas()

    gg = gdf.groupby("a")["c"]
    pg = pdf.groupby("a")["c"]

    assert_eq(
        gg.apply(apply_op),
        pg.apply(apply_op),
        check_dtype=True,
        check_index_type=True,
    )


def test_groupby_consecutive_operations():
    df = cudf.DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    pdf = df.to_pandas()

    gg = df.groupby("A")
    pg = pdf.groupby("A")

    actual = gg.nth(-1)
    expected = pg.nth(-1)

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.nth(0)
    expected = pg.nth(0)

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.cumsum()
    expected = pg.cumsum()

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.cumcount()
    expected = pg.cumcount()

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.cumsum()
    expected = pg.cumsum()

    assert_groupby_results_equal(actual, expected, check_dtype=False)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Warning only given on newer versions.",
)
def test_categorical_grouping_pandas_compatibility():
    gdf = cudf.DataFrame(
        {
            "key": cudf.Series([2, 1, 3, 1, 1], dtype="category"),
            "a": [0, 1, 3, 2, 3],
        }
    )
    pdf = gdf.to_pandas()

    with cudf.option_context("mode.pandas_compatible", True):
        actual = gdf.groupby("key", sort=False).sum()
    with pytest.warns(FutureWarning):
        # observed param deprecation.
        expected = pdf.groupby("key", sort=False).sum()
    assert_eq(actual, expected)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("as_index", [True, False])
def test_group_by_value_counts(normalize, sort, ascending, dropna, as_index):
    # From Issue#12789
    df = cudf.DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", np.nan, "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )
    pdf = df.to_pandas()

    actual = df.groupby("gender", as_index=as_index).value_counts(
        normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
    )
    expected = pdf.groupby("gender", as_index=as_index).value_counts(
        normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
    )

    # TODO: Remove `check_names=False` once testing against `pandas>=2.0.0`
    assert_groupby_results_equal(
        actual,
        expected,
        check_names=False,
        check_index_type=False,
        as_index=as_index,
        by=["gender", "education"],
        sort=sort,
    )


def test_group_by_value_counts_subset():
    # From Issue#12789
    df = cudf.DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", "high", "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )
    pdf = df.to_pandas()

    actual = df.groupby("gender").value_counts(["education"])
    expected = pdf.groupby("gender").value_counts(["education"])

    # TODO: Remove `check_names=False` once testing against `pandas>=2.0.0`
    assert_groupby_results_equal(
        actual, expected, check_names=False, check_index_type=False
    )


def test_group_by_value_counts_clash_with_subset():
    df = cudf.DataFrame({"a": [1, 5, 3], "b": [2, 5, 2]})
    with pytest.raises(ValueError):
        df.groupby("a").value_counts(["a"])


def test_group_by_value_counts_subset_not_exists():
    df = cudf.DataFrame({"a": [1, 5, 3], "b": [2, 5, 2]})
    with pytest.raises(ValueError):
        df.groupby("a").value_counts(["c"])


def test_group_by_value_counts_with_count_column():
    df = cudf.DataFrame({"a": [1, 5, 3], "count": [2, 5, 2]})
    with pytest.raises(ValueError):
        df.groupby("a", as_index=False).value_counts()


def test_groupby_internal_groups_empty(gdf):
    # test that we don't segfault when calling the internal
    # .groups() method with an empty list:
    gb = gdf.groupby("y")._groupby
    _, _, grouped_vals = gb.groups([])
    assert grouped_vals == []


def test_groupby_shift_series_multiindex():
    idx = cudf.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["f", "s"]
    )
    ser = Series(range(4), index=idx)
    result = ser.groupby(level=0).shift(1)
    expected = ser.to_pandas().groupby(level=0).shift(1)
    assert_eq(expected, result)


@pytest.mark.parametrize(
    "func", ["min", "max", "sum", "mean", "idxmin", "idxmax"]
)
@pytest.mark.parametrize(
    "by,data",
    [
        ("a", {"a": [1, 2, 3]}),
        (["a", "id"], {"id": [0, 0, 1], "a": [1, 2, 3]}),
        ("a", {"a": [1, 2, 3], "b": ["A", "B", "C"]}),
        ("id", {"id": [0, 0, 1], "a": [1, 2, 3], "b": ["A", "B", "C"]}),
        (["b", "id"], {"id": [0, 0, 1], "b": ["A", "B", "C"]}),
        ("b", {"b": ["A", "B", "C"]}),
    ],
)
def test_group_by_reduce_numeric_only(by, data, func):
    # Test that simple groupby reductions support numeric_only=True
    df = cudf.DataFrame(data)
    expected = getattr(df.to_pandas().groupby(by, sort=True), func)(
        numeric_only=True
    )
    result = getattr(df.groupby(by, sort=True), func)(numeric_only=True)
    assert_eq(expected, result)


@pytest.mark.parametrize(
    "op", ["cummax", "cummin", "cumprod", "cumsum", "mean", "median"]
)
def test_group_by_raises_string_error(op):
    df = cudf.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})

    with pytest.raises(TypeError):
        df.groupby(df.a).agg(op)


@pytest.mark.parametrize(
    "op",
    [
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "mean",
        "median",
        "prod",
        "sum",
        list,
    ],
)
def test_group_by_raises_category_error(op):
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": cudf.Series(["a", "b", "c", "d", "e"], dtype="category"),
        }
    )

    with pytest.raises(TypeError):
        df.groupby(df.a).agg(op)


def test_ngroups():
    pdf = pd.DataFrame({"a": [1, 1, 3], "b": range(3)})
    gdf = cudf.DataFrame.from_pandas(pdf)

    pgb = pdf.groupby("a")
    ggb = gdf.groupby("a")
    assert pgb.ngroups == ggb.ngroups
    assert len(pgb) == len(ggb)


def test_ndim():
    pdf = pd.DataFrame({"a": [1, 1, 3], "b": range(3)})
    gdf = cudf.DataFrame.from_pandas(pdf)

    pgb = pdf.groupby("a")
    ggb = gdf.groupby("a")
    assert pgb.ndim == ggb.ndim

    pser = pd.Series(range(3))
    gser = cudf.Series.from_pandas(pser)
    pgb = pser.groupby([0, 0, 1])
    ggb = gser.groupby(cudf.Series([0, 0, 1]))
    assert pgb.ndim == ggb.ndim


@pytest.mark.skipif(
    not PANDAS_GE_220, reason="pandas behavior applicable in >=2.2"
)
def test_get_group_list_like():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.groupby(["a"]).get_group((1,))
    expected = df.to_pandas().groupby(["a"]).get_group((1,))
    assert_eq(result, expected)

    with pytest.raises(KeyError):
        df.groupby(["a"]).get_group((1, 2))

    with pytest.raises(KeyError):
        df.groupby(["a"]).get_group([1])


def test_size_as_index_false():
    df = pd.DataFrame({"a": [1, 2, 1], "b": [1, 2, 3]}, columns=["a", "b"])
    expected = df.groupby("a", as_index=False).size()
    result = cudf.from_pandas(df).groupby("a", as_index=False).size()
    assert_eq(result, expected)


def test_size_series_with_name():
    ser = pd.Series(range(3), name="foo")
    expected = ser.groupby(ser).size()
    result = cudf.from_pandas(ser).groupby(ser).size()
    assert_eq(result, expected)
