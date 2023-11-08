# Copyright (c) 2023, NVIDIA CORPORATION.

import itertools
import operator
import string
from collections import defaultdict

import pytest

import cudf
from cudf.core._compat import PANDAS_GE_220
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


if PANDAS_GE_220:
    # Behaviour in sort=False case didn't match documentation in many
    # cases prior to https://github.com/pandas-dev/pandas/pull/54611
    # (released as part of pandas 2.2)
    def expected(left, right, sort, *, how):
        left = left.to_pandas()
        right = right.to_pandas()
        return left.merge(right, on="key", how=how, sort=sort)

else:

    def expect_inner(left, right, sort):
        left_key = left.key.values_host.tolist()
        left_val = left.val.values_host.tolist()
        right_key = right.key.values_host.tolist()
        right_val = right.val.values_host.tolist()

        right_have = defaultdict(list)
        for i, k in enumerate(right_key):
            right_have[k].append(i)
        keys = []
        val_x = []
        val_y = []
        for k, v in zip(left_key, left_val):
            if k not in right_have:
                continue
            for i in right_have[k]:
                keys.append(k)
                val_x.append(v)
                val_y.append(right_val[i])

        if sort:
            # Python sort is stable, so this will preserve input order for
            # equal items.
            keys, val_x, val_y = zip(
                *sorted(zip(keys, val_x, val_y), key=operator.itemgetter(0))
            )
        return cudf.DataFrame({"key": keys, "val_x": val_x, "val_y": val_y})

    def expect_left(left, right, sort):
        left_key = left.key.values_host.tolist()
        left_val = left.val.values_host.tolist()
        right_key = right.key.values_host.tolist()
        right_val = right.val.values_host.tolist()

        right_have = defaultdict(list)
        for i, k in enumerate(right_key):
            right_have[k].append(i)
        keys = []
        val_x = []
        val_y = []
        for k, v in zip(left_key, left_val):
            if k not in right_have:
                right_vals = [None]
            else:
                right_vals = [right_val[i] for i in right_have[k]]

            for rv in right_vals:
                keys.append(k)
                val_x.append(v)
                val_y.append(rv)

        if sort:
            # Python sort is stable, so this will preserve input order for
            # equal items.
            keys, val_x, val_y = zip(
                *sorted(zip(keys, val_x, val_y), key=operator.itemgetter(0))
            )
        return cudf.DataFrame({"key": keys, "val_x": val_x, "val_y": val_y})

    def expect_outer(left, right, sort):
        left_key = left.key.values_host.tolist()
        left_val = left.val.values_host.tolist()
        right_key = right.key.values_host.tolist()
        right_val = right.val.values_host.tolist()
        right_have = defaultdict(list)
        for i, k in enumerate(right_key):
            right_have[k].append(i)
        keys = []
        val_x = []
        val_y = []
        for k, v in zip(left_key, left_val):
            if k not in right_have:
                right_vals = [None]
            else:
                right_vals = [right_val[i] for i in right_have[k]]
            for rv in right_vals:
                keys.append(k)
                val_x.append(v)
                val_y.append(rv)
        left_have = set(left_key)
        for k, v in zip(right_key, right_val):
            if k not in left_have:
                keys.append(k)
                val_x.append(None)
                val_y.append(v)

        # Python sort is stable, so this will preserve input order for
        # equal items.
        # outer joins are always sorted, but we test both sort values
        keys, val_x, val_y = zip(
            *sorted(zip(keys, val_x, val_y), key=operator.itemgetter(0))
        )
        return cudf.DataFrame({"key": keys, "val_x": val_x, "val_y": val_y})

    def expected(left, right, sort, *, how):
        if how == "inner":
            return expect_inner(left, right, sort)
        elif how == "outer":
            return expect_outer(left, right, sort)
        elif how == "left":
            return expect_left(left, right, sort)
        elif how == "right":
            return expect_left(right, left, sort).rename(
                {"val_x": "val_y", "val_y": "val_x"}, axis=1
            )
        else:
            raise NotImplementedError()


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_join_ordering_pandas_compat(left, right, sort, how):
    with cudf.option_context("mode.pandas_compatible", True):
        actual = left.merge(right, on="key", how=how, sort=sort)
    expect = expected(left, right, sort, how=how)
    assert_eq(expect, actual)
