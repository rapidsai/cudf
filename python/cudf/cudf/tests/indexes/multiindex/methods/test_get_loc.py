# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@pytest.mark.parametrize(
    "data",
    [
        [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 3), (2, 1, 1), (2, 2, 1)],
        [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 2), (2, 2, 1), (1, 1, 1)],
        [(1, 1, 1), (1, 1, 2), (1, 1, 2), (1, 2, 3), (2, 1, 1), (2, 2, 1)],
    ],
)
@pytest.mark.parametrize("key", [1, (1, 2), (1, 2, 3), (2, 1, 1), (9, 9, 9)])
def test_get_loc_multi_numeric(data, key):
    idx = pd.MultiIndex.from_tuples(data)
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "key, result",
    [
        (1, slice(1, 5, 1)),  # deviates
        ((1, 2), slice(1, 3, 1)),
        ((1, 2, 3), slice(1, 2, None)),
        ((2, 1, 1), slice(0, 1, None)),
        ((9, 9, 9), None),
    ],
)
def test_get_loc_multi_numeric_deviate(key, result):
    pi = pd.MultiIndex.from_tuples(
        [(2, 1, 1), (1, 2, 3), (1, 2, 1), (1, 1, 1), (1, 1, 1), (2, 2, 1)]
    )
    gi = cudf.from_pandas(pi)

    with expect_warning_if(
        isinstance(key, tuple), pd.errors.PerformanceWarning
    ):
        key_flag = key not in pi

    if key_flag:
        with expect_warning_if(
            isinstance(key, tuple), pd.errors.PerformanceWarning
        ):
            assert_exceptions_equal(
                lfunc=pi.get_loc,
                rfunc=gi.get_loc,
                lfunc_args_and_kwargs=([], {"key": key}),
                rfunc_args_and_kwargs=([], {"key": key}),
            )
    else:
        expected = result
        got = gi.get_loc(key)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        [
            ("a", "a", "a"),
            ("a", "a", "b"),
            ("a", "b", "a"),
            ("a", "b", "c"),
            ("b", "a", "a"),
            ("b", "c", "a"),
        ],
        [
            ("a", "a", "b"),
            ("a", "b", "c"),
            ("b", "a", "a"),
            ("a", "a", "a"),
            ("a", "b", "a"),
            ("b", "c", "a"),
        ],
        [
            ("a", "a", "a"),
            ("a", "b", "c"),
            ("b", "a", "a"),
            ("a", "a", "b"),
            ("a", "b", "a"),
            ("b", "c", "a"),
        ],
        [
            ("a", "a", "a"),
            ("a", "a", "b"),
            ("a", "a", "b"),
            ("a", "b", "c"),
            ("b", "a", "a"),
            ("b", "c", "a"),
        ],
        [
            ("a", "a", "b"),
            ("b", "a", "a"),
            ("b", "a", "a"),
            ("a", "a", "a"),
            ("a", "b", "a"),
            ("b", "c", "a"),
        ],
    ],
)
@pytest.mark.parametrize(
    "key", ["a", ("a", "a"), ("a", "b", "c"), ("b", "c", "a"), ("z", "z", "z")]
)
def test_get_loc_multi_string(data, key):
    idx = pd.MultiIndex.from_tuples(data)
    pi = idx.sort_values()
    gi = cudf.from_pandas(pi)

    if key not in pi:
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key}),
            rfunc_args_and_kwargs=([], {"key": key}),
        )
    else:
        expected = pi.get_loc(key)
        got = gi.get_loc(key)

        assert_eq(expected, got)
