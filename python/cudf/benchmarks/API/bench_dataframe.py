# Copyright (c) 2022, NVIDIA CORPORATION.

"""Benchmarks of DataFrame methods."""

import string

import numpy
import pytest
import pytest_cases
from config import cudf, cupy
from utils import benchmark_with_object


@pytest.mark.parametrize("N", [100, 1_000_000])
def bench_construction(benchmark, N):
    benchmark(cudf.DataFrame, {None: cupy.random.rand(N)})


@benchmark_with_object(cls="dataframe", dtype="float", cols=6)
@pytest.mark.parametrize(
    "expr", ["a+b", "a+b+c+d+e", "a / (sin(a) + cos(b)) * tanh(d*e*f)"]
)
def bench_eval_func(benchmark, expr, dataframe):
    benchmark(dataframe.eval, expr)


@benchmark_with_object(cls="dataframe", dtype="int", nulls=False, cols=6)
@pytest.mark.parametrize(
    "num_key_cols",
    [2, 3, 4],
)
def bench_merge(benchmark, dataframe, num_key_cols):
    benchmark(
        dataframe.merge, dataframe, on=list(dataframe.columns[:num_key_cols])
    )


# TODO: Some of these cases could be generalized to an IndexedFrame benchmark
# instead of a DataFrame benchmark.
@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize(
    "values",
    [
        lambda: range(50),
        lambda: {f"{string.ascii_lowercase[i]}": range(50) for i in range(10)},
        lambda: cudf.DataFrame(
            {f"{string.ascii_lowercase[i]}": range(50) for i in range(10)}
        ),
        lambda: cudf.Series(range(50)),
    ],
)
def bench_isin(benchmark, dataframe, values):
    benchmark(dataframe.isin, values())


@pytest.fixture(
    params=[0, numpy.random.RandomState, cupy.random.RandomState],
    ids=["Seed", "NumpyRandomState", "CupyRandomState"],
)
def random_state(request):
    rs = request.param
    return rs if isinstance(rs, int) else rs(seed=42)


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize("frac", [0.5])
def bench_sample(benchmark, dataframe, axis, frac, random_state):
    if axis == 1 and isinstance(random_state, cupy.random.RandomState):
        pytest.skip("Unsupported params.")
    benchmark(
        dataframe.sample, frac=frac, axis=axis, random_state=random_state
    )


@benchmark_with_object(cls="dataframe", dtype="int", nulls=False, cols=6)
@pytest.mark.parametrize(
    "num_key_cols",
    [2, 3, 4],
)
def bench_groupby(benchmark, dataframe, num_key_cols):
    benchmark(dataframe.groupby, by=list(dataframe.columns[:num_key_cols]))


@benchmark_with_object(cls="dataframe", dtype="int", nulls=False, cols=6)
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
    "num_key_cols",
    [2, 3, 4],
)
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def bench_groupby_agg(benchmark, dataframe, agg, num_key_cols, as_index, sort):
    by = list(dataframe.columns[:num_key_cols])
    benchmark(dataframe.groupby(by=by, as_index=as_index, sort=sort).agg, agg)


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize("num_cols_to_sort", [1])
def bench_sort_values(benchmark, dataframe, num_cols_to_sort):
    benchmark(
        dataframe.sort_values, list(dataframe.columns[:num_cols_to_sort])
    )


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize("num_cols_to_sort", [1])
@pytest.mark.parametrize("n", [10])
def bench_nsmallest(benchmark, dataframe, num_cols_to_sort, n):
    by = list(dataframe.columns[:num_cols_to_sort])
    benchmark(dataframe.nsmallest, n, by)


@pytest_cases.parametrize_with_cases("dataframe, cond, other", prefix="where")
def bench_where(benchmark, dataframe, cond, other):
    benchmark(dataframe.where, cond, other)
