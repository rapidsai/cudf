# Copyright (c) 2022, NVIDIA CORPORATION.

"""Benchmarks of DataFrame methods."""

import string

import numpy
import pytest
from config import cudf, cupy
from utils import accepts_cudf_fixture


@pytest.mark.parametrize("N", [100, 1_000_000])
def bench_construction(benchmark, N):
    benchmark(cudf.DataFrame, {None: cupy.random.rand(N)})


@accepts_cudf_fixture(cls="dataframe", dtype="float", cols=6)
@pytest.mark.parametrize(
    "expr", ["a+b", "a+b+c+d+e", "a / (sin(a) + cos(b)) * tanh(d*e*f)"]
)
def bench_eval_func(benchmark, expr, dataframe):
    benchmark(dataframe.eval, expr)


@accepts_cudf_fixture(
    cls="dataframe", dtype="int", nulls=False, cols=6, name="df"
)
@pytest.mark.parametrize(
    "nkey_cols",
    [2, 3, 4],
)
def bench_merge(benchmark, df, nkey_cols):
    benchmark(df.merge, df, on=list(df.columns[:nkey_cols]))


# TODO: Some of these cases could be generalized to an IndexedFrame benchmark
# instead of a DataFrame benchmark.
@accepts_cudf_fixture(cls="dataframe", dtype="int")
@pytest.mark.parametrize(
    "values",
    [
        range(1000),
        {f"key{i}": range(1000) for i in range(10)},
        cudf.DataFrame({f"key{i}": range(1000) for i in range(10)}),
        cudf.Series(range(1000)),
    ],
)
def bench_isin(benchmark, dataframe, values):
    benchmark(dataframe.isin, values)


@pytest.fixture(
    params=[0, numpy.random.RandomState, cupy.random.RandomState],
    ids=["Seed", "NumpyRandomState", "CupyRandomState"],
)
def random_state(request):
    rs = request.param
    return rs if isinstance(rs, int) else rs(seed=42)


@accepts_cudf_fixture(cls="dataframe", dtype="int")
@pytest.mark.parametrize("frac", [0.5])
def bench_sample(benchmark, dataframe, axis, frac, random_state):
    if axis == 1 and isinstance(random_state, cupy.random.RandomState):
        pytest.skip("Unsupported params.")
    benchmark(
        dataframe.sample, frac=frac, axis=axis, random_state=random_state
    )


@accepts_cudf_fixture(cls="dataframe", dtype="int", nulls=False, cols=6)
@pytest.mark.parametrize(
    "nkey_cols",
    [2, 3, 4],
)
def bench_groupby(benchmark, dataframe, nkey_cols):
    benchmark(dataframe.groupby, by=list(dataframe.columns[:nkey_cols]))


@accepts_cudf_fixture(cls="dataframe", dtype="int", nulls=False, cols=6)
@pytest.mark.parametrize(
    "agg",
    [
        "sum",
        ["sum", "mean"],
        {
            f"{string.ascii_lowercase[i]}": ["sum", "mean", "count"]
            for i in range(6)
        },
    ],
)
@pytest.mark.parametrize(
    "nkey_cols",
    [2, 3, 4],
)
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def bench_groupby_agg(benchmark, dataframe, agg, nkey_cols, as_index, sort):
    by = list(dataframe.columns[:nkey_cols])
    benchmark(dataframe.groupby(by=by, as_index=as_index, sort=sort).agg, agg)


@accepts_cudf_fixture(cls="dataframe", dtype="int")
@pytest.mark.parametrize("ncol_sort", [1])
def bench_sort_values(benchmark, dataframe, ncol_sort):
    benchmark(dataframe.sort_values, list(dataframe.columns[:ncol_sort]))


@accepts_cudf_fixture(cls="dataframe", dtype="int")
@pytest.mark.parametrize("ncol_sort", [1])
@pytest.mark.parametrize("n", [10])
def bench_nsmallest(benchmark, dataframe, ncol_sort, n):
    by = list(dataframe.columns[:ncol_sort])
    benchmark(dataframe.nsmallest, n, by)
