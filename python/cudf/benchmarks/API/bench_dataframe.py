# Copyright (c) 2022-2024, NVIDIA CORPORATION.

"""Benchmarks of DataFrame methods."""

import string

import numba.cuda
import numpy
import pytest
import pytest_cases
from config import cudf, cupy
from utils import benchmark_with_object


@pytest.mark.parametrize("N", [100, 1_000_000])
def bench_construction(benchmark, N):
    benchmark(cudf.DataFrame, {None: cupy.random.rand(N)})


@pytest.mark.parametrize("N", [100, 100_000])
@pytest.mark.pandas_incompatible
def bench_construction_numba_device_array(benchmark, N):
    benchmark(cudf.DataFrame, numba.cuda.to_device(numpy.ones((100, N))))


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


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize("frac", [0, 0.25, 0.5, 0.75, 1])
def bench_iloc_getitem_indices(benchmark, dataframe, frac):
    rs = numpy.random.RandomState(seed=42)
    n = int(len(dataframe) * frac)
    values = rs.choice(len(dataframe), size=n, replace=False)
    benchmark(dataframe.iloc.__getitem__, values)


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize("frac", [0, 0.25, 0.5, 0.75, 1])
def bench_iloc_getitem_mask(benchmark, dataframe, frac):
    rs = numpy.random.RandomState(seed=42)
    n = int(len(dataframe) * frac)
    values = rs.choice(len(dataframe), size=n, replace=False)
    mask = numpy.zeros(len(dataframe), dtype=bool)
    mask[values] = True
    benchmark(dataframe.iloc.__getitem__, mask)


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize(
    "slice",
    [slice(None), slice(0, 0, 1), slice(1, None, 10), slice(None, -1, -1)],
)
def bench_iloc_getitem_slice(benchmark, dataframe, slice):
    benchmark(dataframe.iloc.__getitem__, slice)


@benchmark_with_object(cls="dataframe", dtype="int")
def bench_iloc_getitem_scalar(benchmark, dataframe):
    benchmark(dataframe.iloc.__getitem__, len(dataframe) // 2)


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


@benchmark_with_object(cls="dataframe", dtype="int", nulls=False, cols=6)
@pytest.mark.parametrize(
    "num_key_cols",
    [2, 3, 4],
)
@pytest.mark.parametrize("use_frac", [True, False])
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("target_sample_frac", [0.1, 0.5, 1])
def bench_groupby_sample(
    benchmark, dataframe, num_key_cols, use_frac, replace, target_sample_frac
):
    grouper = dataframe.groupby(by=list(dataframe.columns[:num_key_cols]))
    if use_frac:
        kwargs = {"frac": target_sample_frac, "replace": replace}
    else:
        minsize = grouper.size().min()
        target_size = numpy.round(
            target_sample_frac * minsize, decimals=0
        ).astype(int)
        kwargs = {"n": target_size, "replace": replace}

    benchmark(grouper.sample, **kwargs)


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


@pytest_cases.parametrize_with_cases(
    "dataframe, cond, other", prefix="where", cases="cases_dataframe"
)
def bench_where(benchmark, dataframe, cond, other):
    benchmark(dataframe.where, cond, other)
