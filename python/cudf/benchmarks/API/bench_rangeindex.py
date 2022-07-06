# Copyright (c) 2022, NVIDIA CORPORATION.

from utils import benchmark_with_object


@benchmark_with_object(cls="rangeindex", dtype="int")
def bench_values_host(benchmark, rangeindex):
    benchmark(rangeindex.values_host)


@benchmark_with_object(cls="rangeindex", dtype="int")
def bench_to_numpy(benchmark, rangeindex):
    benchmark(rangeindex.to_numpy)


@benchmark_with_object(cls="rangeindex", dtype="int")
def bench_to_arrow(benchmark, rangeindex):
    benchmark(rangeindex.to_arrow)


@benchmark_with_object(cls="rangeindex", dtype="int")
def bench_argsort(benchmark, rangeindex):
    benchmark(rangeindex.argsort)


@benchmark_with_object(cls="rangeindex", dtype="int")
def bench_nunique(benchmark, rangeindex):
    benchmark(rangeindex.nunique)


@benchmark_with_object(cls="rangeindex", dtype="int")
def bench_isna(benchmark, rangeindex):
    benchmark(rangeindex.isna)


@benchmark_with_object(cls="rangeindex", dtype="int")
def bench_max(benchmark, rangeindex):
    benchmark(rangeindex.max)


@benchmark_with_object(cls="rangeindex", dtype="int")
def bench_min(benchmark, rangeindex):
    benchmark(rangeindex.min)


@benchmark_with_object(cls="rangeindex", dtype="int")
def bench_replace(benchmark, rangeindex):
    benchmark(rangeindex.replace, 0, 2)


@benchmark_with_object(cls="rangeindex", dtype="int")
def bench_where(benchmark, rangeindex):
    cond = rangeindex % 2 == 0
    benchmark(rangeindex.where, cond, 0)
