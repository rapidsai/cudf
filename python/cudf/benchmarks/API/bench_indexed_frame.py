# Copyright (c) 2022, NVIDIA CORPORATION.

"""Benchmarks of IndexedFrame methods."""

import pytest

from ..common.utils import accepts_cudf_fixture


@accepts_cudf_fixture(cls="indexedframe", dtype="int")
@pytest.mark.parametrize("op", ["cumsum", "cumprod", "cummax"])
def bench_scans(benchmark, op, indexedframe):
    benchmark(getattr(indexedframe, op))


@accepts_cudf_fixture(cls="indexedframe", dtype="int")
@pytest.mark.parametrize("op", ["sum", "product", "mean"])
def bench_reductions(benchmark, op, indexedframe):
    benchmark(getattr(indexedframe, op))


@accepts_cudf_fixture(cls="indexedframe", dtype="int")
def bench_drop_duplicates(benchmark, indexedframe):
    benchmark(indexedframe.drop_duplicates)


@accepts_cudf_fixture(cls="indexedframe", dtype="int")
def bench_rangeindex_replace(benchmark, indexedframe):
    # TODO: Consider adding more DataFrame-specific benchmarks for different
    # types of valid inputs (dicts, etc).
    benchmark(indexedframe.replace, 0, 2)
