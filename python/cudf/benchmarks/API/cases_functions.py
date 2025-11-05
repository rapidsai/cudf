# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for benchmarks in bench_functions.py."""

import pytest_cases
from config import NUM_ROWS, cudf, cupy


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_default_index(nr):
    return [
        cudf.DataFrame({"a": cupy.tile([1, 2, 3], nr)}),
        cudf.DataFrame({"b": cupy.tile([4, 5, 7], nr)}),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_contiguous_indexes(nr):
    return [
        cudf.DataFrame({"a": cupy.tile([1, 2, 3], nr)}),
        cudf.DataFrame(
            {"b": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_contiguous_indexes_different_cols(nr):
    return [
        cudf.DataFrame(
            {"a": cupy.tile([1, 2, 3], nr), "b": cupy.tile([4, 5, 7], nr)}
        ),
        cudf.DataFrame(
            {"c": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_string_index(nr):
    return [
        cudf.DataFrame(
            {"a": cupy.tile([1, 2, 3], nr), "b": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"c": [4, 5, 7] * nr},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_contiguous_string_index_different_col(nr):
    return [
        cudf.DataFrame(
            {"a": cupy.tile([1, 2, 3], nr), "b": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"c": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3).astype("str"),
        ),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_complex_string_index(nr):
    return [
        cudf.DataFrame(
            {"a": cupy.tile([1, 2, 3], nr), "b": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"c": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"d": cupy.tile([1, 2, 3], nr), "e": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"f": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"g": cupy.tile([1, 2, 3], nr), "h": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=0, stop=nr * 3).astype("str"),
        ),
        cudf.DataFrame(
            {"i": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3).astype("str"),
        ),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_unique_columns(nr):
    # To avoid any edge case bugs, always use at least 10 rows per DataFrame.
    nr_actual = max(10, nr // 20)
    return [
        cudf.DataFrame({"a": cupy.tile([1, 2, 3], nr_actual)}),
        cudf.DataFrame({"b": cupy.tile([4, 5, 7], nr_actual)}),
        cudf.DataFrame({"c": cupy.tile([1, 2, 3], nr_actual)}),
        cudf.DataFrame({"d": cupy.tile([4, 5, 7], nr_actual)}),
        cudf.DataFrame({"e": cupy.tile([1, 2, 3], nr_actual)}),
        cudf.DataFrame({"f": cupy.tile([4, 5, 7], nr_actual)}),
        cudf.DataFrame({"g": cupy.tile([1, 2, 3], nr_actual)}),
        cudf.DataFrame({"h": cupy.tile([4, 5, 7], nr_actual)}),
        cudf.DataFrame({"i": cupy.tile([1, 2, 3], nr_actual)}),
        cudf.DataFrame({"j": cupy.tile([4, 5, 7], nr_actual)}),
    ]


@pytest_cases.parametrize("nr", NUM_ROWS)
def concat_case_unique_columns_with_different_range_index(nr):
    return [
        cudf.DataFrame(
            {"a": cupy.tile([1, 2, 3], nr), "b": cupy.tile([4, 5, 7], nr)}
        ),
        cudf.DataFrame(
            {"c": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
        cudf.DataFrame(
            {"d": cupy.tile([1, 2, 3], nr), "e": cupy.tile([4, 5, 7], nr)}
        ),
        cudf.DataFrame(
            {"f": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
        cudf.DataFrame(
            {"g": cupy.tile([1, 2, 3], nr), "h": cupy.tile([4, 5, 7], nr)}
        ),
        cudf.DataFrame(
            {"i": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
        cudf.DataFrame(
            {"j": cupy.tile([1, 2, 3], nr), "k": cupy.tile([4, 5, 7], nr)}
        ),
        cudf.DataFrame(
            {"l": cupy.tile([4, 5, 7], nr)},
            index=cudf.RangeIndex(start=nr * 3, stop=nr * 2 * 3),
        ),
    ]
