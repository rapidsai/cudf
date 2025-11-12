# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of Index methods."""

import pytest
from config import NUM_ROWS, cudf, cupy
from utils import benchmark_with_object


@pytest.mark.parametrize("N", NUM_ROWS)
def bench_construction(benchmark, N):
    benchmark(cudf.Index, cupy.random.rand(N))


@benchmark_with_object(cls="index", dtype="int", nulls=False)
def bench_sort_values(benchmark, index):
    benchmark(index.sort_values)


@pytest.mark.parametrize("N", NUM_ROWS)
def bench_large_unique_categories_repr(benchmark, N):
    pi = cudf.CategoricalIndex(range(N))
    benchmark(repr, pi)
