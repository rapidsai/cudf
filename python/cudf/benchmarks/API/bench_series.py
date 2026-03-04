# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of Series methods."""

import pytest
from config import NUM_ROWS, cudf, cupy
from utils import benchmark_with_object


@pytest.mark.parametrize("N", NUM_ROWS)
def bench_construction(benchmark, N):
    benchmark(cudf.Series, cupy.random.rand(N))


@benchmark_with_object(cls="series", dtype="int")
def bench_sort_values(benchmark, series):
    benchmark(series.sort_values)


@benchmark_with_object(cls="series", dtype="int")
@pytest.mark.parametrize("n", [10])
def bench_series_nsmallest(benchmark, series, n):
    benchmark(series.nsmallest, n)


@benchmark_with_object(cls="series", dtype="int", nulls=False)
def bench_series_cp_asarray(benchmark, series):
    benchmark(cupy.asarray, series)


@benchmark_with_object(cls="series", dtype="int", nulls=False)
@pytest.mark.pandas_incompatible
def bench_to_cupy(benchmark, series):
    benchmark(lambda: series.values)


@benchmark_with_object(cls="series", dtype="int", nulls=False)
def bench_series_values(benchmark, series):
    benchmark(lambda: series.values)
