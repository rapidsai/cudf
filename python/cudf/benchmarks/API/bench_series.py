# Copyright (c) 2022, NVIDIA CORPORATION.

"""Benchmarks of Series methods."""

import pytest
from config import cudf, cupy
from utils import benchmark_with_object


@pytest.mark.parametrize("N", [100, 1_000_000])
def bench_construction(benchmark, N):
    benchmark(cudf.Series, cupy.random.rand(N))


@benchmark_with_object(cls="series", dtype="int")
def bench_sort_values(benchmark, series):
    benchmark(series.sort_values)


@benchmark_with_object(cls="series", dtype="int")
@pytest.mark.parametrize("n", [10])
def bench_series_nsmallest(benchmark, series, n):
    benchmark(series.nsmallest, n)
