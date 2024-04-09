# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import itertools
import string

import numpy as np
import pytest

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.testing._utils import assert_eq


@pytest.fixture(params=[False, True], ids=["unsorted", "sorted"])
def sort(request):
    return request.param


@pytest.fixture
def left():
    left_key = [1, 3, 2, 1, 1, 2, 5, 1, 4, 5, 8, 12, 12312, 1] * 100
    left_val = list(range(len(left_key)))
    return cudf.DataFrame({"key": left_key, "val": left_val})


@pytest.fixture
def right():
    right_key = [12312, 12312, 3, 2, 1, 1, 5, 7, 2] * 200
    right_val = list(
        itertools.islice(itertools.cycle(string.ascii_letters), len(right_key))
    )
    return cudf.DataFrame({"key": right_key, "val": right_val})


# Behaviour in sort=False case didn't match documentation in many
# cases prior to https://github.com/pandas-dev/pandas/pull/54611
# (released as part of pandas 2.2)
def expected(left, right, sort, *, how):
    left = left.to_pandas()
    right = right.to_pandas()
    return left.merge(right, on="key", how=how, sort=sort)


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_join_ordering_pandas_compat(request, left, right, sort, how):
    request.applymarker(
        pytest.mark.xfail(
            PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION
            and how == "right",
            reason="TODO: Result ording of suffix'ed columns is incorrect",
        )
    )
    with cudf.option_context("mode.pandas_compatible", True):
        actual = left.merge(right, on="key", how=how, sort=sort)
    expect = expected(left, right, sort, how=how)
    assert_eq(expect, actual)


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("on_index", [True, False])
@pytest.mark.parametrize("left_unique", [True, False])
@pytest.mark.parametrize("left_monotonic", [True, False])
@pytest.mark.parametrize("right_unique", [True, False])
@pytest.mark.parametrize("right_monotonic", [True, False])
def test_merge_combinations(
    request,
    how,
    sort,
    on_index,
    left_unique,
    left_monotonic,
    right_unique,
    right_monotonic,
):
    request.applymarker(
        pytest.mark.xfail(
            condition=how == "outer"
            and on_index
            and left_unique
            and not left_monotonic
            and right_unique
            and not right_monotonic,
            reason="https://github.com/pandas-dev/pandas/issues/55992",
        )
    )
    left = [2, 3]
    if left_unique:
        left.append(4 if left_monotonic else 1)
    else:
        left.append(3 if left_monotonic else 2)

    right = [2, 3]
    if right_unique:
        right.append(4 if right_monotonic else 1)
    else:
        right.append(3 if right_monotonic else 2)

    left = cudf.DataFrame({"key": left})
    right = cudf.DataFrame({"key": right})

    if on_index:
        left = left.set_index("key")
        right = right.set_index("key")
        on_kwargs = {"left_index": True, "right_index": True}
    else:
        on_kwargs = {"on": "key"}

    with cudf.option_context("mode.pandas_compatible", True):
        result = cudf.merge(left, right, how=how, sort=sort, **on_kwargs)
    if on_index:
        left = left.reset_index()
        right = right.reset_index()

    if how in ["left", "right", "inner"]:
        if how in ["left", "inner"]:
            expected, other, other_unique = left, right, right_unique
        else:
            expected, other, other_unique = right, left, left_unique
        if how == "inner":
            keep_values = set(left["key"].values_host).intersection(
                right["key"].values_host
            )
            keep_mask = expected["key"].isin(keep_values)
            expected = expected[keep_mask]
        if sort:
            expected = expected.sort_values("key")
        if not other_unique:
            other_value_counts = other["key"].value_counts()
            repeats = other_value_counts.reindex(
                expected["key"].values, fill_value=1
            )
            repeats = repeats.astype(np.intp)
            expected = expected["key"].repeat(repeats.values)
            expected = expected.to_frame()
    elif how == "outer":
        if on_index and left_unique and left["key"].equals(right["key"]):
            expected = cudf.DataFrame({"key": left["key"]})
        else:
            left_counts = left["key"].value_counts()
            right_counts = right["key"].value_counts()
            expected_counts = left_counts.mul(right_counts, fill_value=1)
            expected_counts = expected_counts.astype(np.intp)
            expected = expected_counts.index.values_host.repeat(
                expected_counts.values_host
            )
            expected = cudf.DataFrame({"key": expected})
            expected = expected.sort_values("key")

    if on_index:
        expected = expected.set_index("key")
    else:
        expected = expected.reset_index(drop=True)

    assert_eq(result, expected)
