# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "idx",
    [
        [1, 3, 3, 6],
        [6, 1, 3, 3],
        [4, 3, 2, 1, 0],
    ],
    ids=["monotonic increasing", "non-monotonic", "monotonic decreasing"],
)
@pytest.mark.parametrize("key", [0, 3, 6, 7, 4])
def test_get_loc_duplicate_numeric(idx, key):
    pi = pd.Index(idx)
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


@pytest.mark.parametrize("idx", [["b", "f", "m", "q"], ["m", "f", "b", "q"]])
@pytest.mark.parametrize("key", ["a", "f", "n", "z"])
def test_get_loc_single_unique_string(idx, key):
    pi = pd.Index(idx)
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


@pytest.mark.parametrize("idx", [["b", "m", "m", "q"], ["m", "f", "m", "q"]])
@pytest.mark.parametrize("key", ["a", "f", "n", "z"])
def test_get_loc_single_duplicate_string(idx, key):
    pi = pd.Index(idx)
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
