# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of internal DataFrame methods."""

from utils import benchmark_with_object, make_boolean_mask_column

from cudf.core.copy_types import BooleanMask


@benchmark_with_object(cls="dataframe", dtype="int")
def bench_apply_boolean_mask(benchmark, dataframe):
    mask = make_boolean_mask_column(len(dataframe))
    benchmark(dataframe._apply_boolean_mask, BooleanMask(mask, len(dataframe)))
