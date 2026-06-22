# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "names", [[None, None], ["a", None], ["new name", "another name"]]
)
@pytest.mark.parametrize("inplace", [True, False])
def test_multiindex_set_names(names, inplace):
    pi = pd.MultiIndex.from_product([["python", "cobra"], [2018, 2019]])
    gi = cudf.from_pandas(pi)

    expected = pi.set_names(names=names, inplace=inplace)
    actual = gi.set_names(names=names, inplace=inplace)

    if inplace:
        expected, actual = pi, gi

    assert_eq(expected, actual)


@pytest.mark.parametrize("idx_names", [[None, None, None], [1, 0, 2]])
@pytest.mark.parametrize(
    "level, names",
    [
        (0, "abc"),
        (1, "xyz"),
        ([2, 1], ["a", "b"]),
        ([0, 1], ["aa", "bb"]),
        (None, ["a", "b", "c"]),
        (None, ["a", None, "c"]),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_multiindex_set_names_default_and_int_names(
    idx_names, level, names, inplace
):
    pi = pd.MultiIndex.from_product(
        [["python", "cobra"], [2018, 2019], ["aab", "bcd"]], names=idx_names
    )
    gi = cudf.from_pandas(pi)

    expected = pi.set_names(names=names, level=level, inplace=inplace)
    actual = gi.set_names(names=names, level=level, inplace=inplace)

    if inplace:
        expected, actual = pi, gi

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "level, names",
    [
        ([None], "abc"),
        (["three", "one"], ["a", "b"]),
        (["three", 1], ["a", "b"]),
        ([0, "three", 1], ["a", "b", "z"]),
        (["one", 1, "three"], ["a", "b", "z"]),
        (["one", None, "three"], ["a", "b", "z"]),
        ([2, 1], ["a", "b"]),
        (1, "xyz"),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_multiindex_set_names_string_names(level, names, inplace):
    pi = pd.MultiIndex.from_product(
        [["python", "cobra"], [2018, 2019], ["aab", "bcd"]],
        names=["one", None, "three"],
    )
    gi = cudf.from_pandas(pi)

    expected = pi.set_names(names=names, level=level, inplace=inplace)
    actual = gi.set_names(names=names, level=level, inplace=inplace)

    if inplace:
        expected, actual = pi, gi

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "level, names", [(1, ["a"]), (None, "a"), ([1, 2], ["a"]), (None, ["a"])]
)
def test_multiindex_set_names_error(level, names):
    pi = pd.MultiIndex.from_product(
        [["python", "cobra"], [2018, 2019], ["aab", "bcd"]]
    )
    gi = cudf.from_pandas(pi)

    assert_exceptions_equal(
        lfunc=pi.set_names,
        rfunc=gi.set_names,
        lfunc_args_and_kwargs=([], {"names": names, "level": level}),
        rfunc_args_and_kwargs=([], {"names": names, "level": level}),
    )
