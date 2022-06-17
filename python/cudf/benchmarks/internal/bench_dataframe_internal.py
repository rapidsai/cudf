# Copyright (c) 2022, NVIDIA CORPORATION.

from utils import accepts_cudf_fixture, make_boolean_mask_column


@accepts_cudf_fixture(cls="dataframe", dtype="int")
def bench_apply_boolean_mask(benchmark, dataframe):
    mask = make_boolean_mask_column(len(dataframe))
    benchmark(dataframe._apply_boolean_mask, mask)
