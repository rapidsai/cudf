# Copyright (c) 2022, NVIDIA CORPORATION.

"""Benchmarks of internal DataFrame methods."""

from utils import benchmark_with_object, make_boolean_mask_column


@benchmark_with_object(cls="dataframe", dtype="int")
def bench_apply_boolean_mask(benchmark, dataframe):
    mask = make_boolean_mask_column(len(dataframe))
    benchmark(dataframe._apply_boolean_mask, mask)
