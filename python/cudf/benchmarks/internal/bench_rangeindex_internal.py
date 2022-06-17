# Copyright (c) 2022, NVIDIA CORPORATION.


def bench_column(benchmark, rangeindex):
    benchmark(lambda: rangeindex._column)


def bench_columns(benchmark, rangeindex):
    benchmark(lambda: rangeindex._columns)
