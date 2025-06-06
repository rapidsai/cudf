# Copyright (c) 2022-2025, NVIDIA CORPORATION.

"""Benchmarks of Index methods."""

import pytest
from config import cudf, cupy
from utils import benchmark_with_object


@pytest.mark.parametrize("N", [100, 1_000_000])
def bench_construction(benchmark, N):
    benchmark(cudf.Index, cupy.random.rand(N))


@benchmark_with_object(cls="index", dtype="int", nulls=False)
def bench_sort_values(benchmark, index):
    benchmark(index.sort_values)


def bench_large_unique_categories_repr(benchmark):
    pi = cudf.CategoricalIndex(range(100_000_000))
    benchmark(repr, pi)
