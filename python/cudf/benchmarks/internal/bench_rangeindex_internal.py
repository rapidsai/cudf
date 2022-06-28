# Copyright (c) 2022, NVIDIA CORPORATION.

"""Benchmarks of internal RangeIndex methods."""


def bench_column(benchmark, rangeindex):
    benchmark(lambda: rangeindex._column)


def bench_columns(benchmark, rangeindex):
    benchmark(lambda: rangeindex._columns)
