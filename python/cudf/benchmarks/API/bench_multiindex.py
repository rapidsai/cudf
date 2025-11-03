# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of MultiIndex methods."""

import numpy as np
import pandas as pd
import pytest
from config import cudf


@pytest.fixture
def midx():
    num_elements = int(1e3)
    rng = np.random.default_rng(seed=0)
    a = rng.integers(0, num_elements // 10, num_elements)
    b = rng.integers(0, num_elements // 10, num_elements)
    pidx = pd.MultiIndex.from_arrays([a, b], names=("a", "b"))
    return cudf.MultiIndex(
        codes=pidx.codes, levels=pidx.levels, names=pidx.names
    )


def bench_constructor(benchmark, midx):
    benchmark(
        cudf.MultiIndex, codes=midx.codes, levels=midx.levels, names=midx.names
    )


def bench_from_frame(benchmark, midx):
    benchmark(cudf.MultiIndex.from_frame, midx.to_frame(index=False))


def bench_copy(benchmark, midx):
    benchmark(midx.copy, deep=False)
