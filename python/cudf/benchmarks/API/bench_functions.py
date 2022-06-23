# Copyright (c) 2022, NVIDIA CORPORATION.

"""Benchmarks of free functions that accept cudf objects."""

import pytest
import pytest_cases
from config import cudf, cupy


@pytest_cases.parametrize_with_cases("objs", prefix="concat")
@pytest.mark.parametrize(
    "axis",
    [
        1,
    ],
)
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("ignore_index", [True, False])
def bench_concat_axis_1(benchmark, objs, axis, join, ignore_index):
    benchmark(
        cudf.concat, objs=objs, axis=axis, join=join, ignore_index=ignore_index
    )


@pytest.mark.parametrize("size", [10_000, 100_000])
@pytest.mark.parametrize("cardinality", [10, 100, 1000])
@pytest.mark.parametrize("dtype", [cupy.bool_, cupy.float64])
def bench_get_dummies_high_cardinality(benchmark, size, cardinality, dtype):
    """Benchmark when the cardinality of column to encode is high."""
    df = cudf.DataFrame(
        {
            "col": cudf.Series(
                cupy.random.randint(low=0, high=cardinality, size=size)
            ).astype("category")
        }
    )
    benchmark(cudf.get_dummies, df, columns=["col"], dtype=dtype)


@pytest.mark.parametrize("prefix", [None, "pre"])
def bench_get_dummies_simple(benchmark, prefix):
    """Benchmark with small input to test the efficiency of the API itself."""
    df = cudf.DataFrame(
        {
            "col1": list(range(10)),
            "col2": list("abcdefghij"),
            "col3": cudf.Series(list(range(100, 110)), dtype="category"),
        }
    )
    benchmark(
        cudf.get_dummies, df, columns=["col1", "col2", "col3"], prefix=prefix
    )
