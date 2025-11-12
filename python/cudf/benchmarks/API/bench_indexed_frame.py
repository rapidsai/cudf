# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of IndexedFrame methods."""

import pytest
from utils import benchmark_with_object


@benchmark_with_object(cls="indexedframe", dtype="int")
@pytest.mark.parametrize("op", ["cumsum", "cumprod", "cummax"])
def bench_scans(benchmark, op, indexedframe):
    benchmark(getattr(indexedframe, op))


@benchmark_with_object(cls="indexedframe", dtype="int")
@pytest.mark.parametrize("op", ["sum", "product", "mean"])
def bench_reductions(benchmark, op, indexedframe):
    benchmark(getattr(indexedframe, op))


@benchmark_with_object(cls="indexedframe", dtype="int")
def bench_drop_duplicates(benchmark, indexedframe):
    benchmark(indexedframe.drop_duplicates)


@benchmark_with_object(cls="indexedframe", dtype="int")
def bench_rangeindex_replace(benchmark, indexedframe):
    # TODO: Consider adding more DataFrame-specific benchmarks for different
    # types of valid inputs (dicts, etc).
    benchmark(indexedframe.replace, 0, 2)
