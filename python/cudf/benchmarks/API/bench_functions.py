# Copyright (c) 2022-2024, NVIDIA CORPORATION.

"""Benchmarks of free functions that accept cudf objects."""

import numpy as np
import pytest
import pytest_cases
from config import NUM_ROWS, cudf, cupy
from utils import benchmark_with_object


@pytest_cases.parametrize_with_cases(
    "objs", prefix="concat", cases="cases_functions"
)
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


@benchmark_with_object(cls="dataframe", dtype="int", cols=6)
def bench_pivot_table_simple(benchmark, dataframe):
    values = ["d", "e"]
    index = ["a", "b"]
    columns = ["c"]
    benchmark(
        cudf.pivot_table,
        data=dataframe,
        values=values,
        index=index,
        columns=columns,
    )


@pytest_cases.parametrize("nr", NUM_ROWS)
def bench_crosstab_simple(benchmark, nr):
    rng = np.random.default_rng(seed=0)
    series_a = np.array(["foo", "bar"] * nr)
    series_b = np.array(["one", "two"] * nr)
    series_c = np.array(["dull", "shiny"] * nr)
    rng.shuffle(series_a)
    rng.shuffle(series_b)
    rng.shuffle(series_c)
    series_a = cudf.Series(series_a)
    series_b = cudf.Series(series_b)
    series_c = cudf.Series(series_c)
    benchmark(cudf.crosstab, index=series_a, columns=[series_b, series_c])
