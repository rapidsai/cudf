# Copyright (c) 2022-2024, NVIDIA CORPORATION.

"""Benchmarks of MultiIndex methods."""

import numpy as np
import pandas as pd
import pytest
from config import cudf


@pytest.fixture
def pidx():
    num_elements = int(1e3)
    rng = np.random.default_rng(seed=0)
    a = rng.integers(0, num_elements // 10, num_elements)
    b = rng.integers(0, num_elements // 10, num_elements)
    return pd.MultiIndex.from_arrays([a, b], names=("a", "b"))


@pytest.fixture
def midx(pidx):
    num_elements = int(1e3)
    rng = np.random.default_rng(seed=0)
    a = rng.integers(0, num_elements // 10, num_elements)
    b = rng.integers(0, num_elements // 10, num_elements)
    df = cudf.DataFrame({"a": a, "b": b})
    return cudf.MultiIndex.from_frame(df)


@pytest.mark.pandas_incompatible
def bench_from_pandas(benchmark, pidx):
    benchmark(cudf.MultiIndex.from_pandas, pidx)


def bench_constructor(benchmark, midx):
    benchmark(
        cudf.MultiIndex, codes=midx.codes, levels=midx.levels, names=midx.names
    )


def bench_from_frame(benchmark, midx):
    benchmark(cudf.MultiIndex.from_frame, midx.to_frame(index=False))


def bench_copy(benchmark, midx):
    benchmark(midx.copy, deep=False)
