# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "data", [[1, 3, 6], [6, 1, 3]], ids=["monotonic", "non-monotonic"]
)
@pytest.mark.parametrize("method", [None, "ffill", "bfill", "nearest"])
def test_get_indexer_single_unique_numeric(data, method):
    key = list(range(0, 8))
    pi = pd.Index(data)
    gi = cudf.from_pandas(pi)

    if (
        # `method` only applicable to monotonic index
        not pi.is_monotonic_increasing and method is not None
    ):
        assert_exceptions_equal(
            lfunc=pi.get_loc,
            rfunc=gi.get_loc,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
        )
    else:
        expected = pi.get_indexer(key, method=method)
        got = gi.get_indexer(key, method=method)

        assert_eq(expected, got)

        with cudf.option_context("mode.pandas_compatible", True):
            got = gi.get_indexer(key, method=method)
        assert_eq(expected, got, check_dtype=True)


@pytest.mark.parametrize(
    "idx",
    [
        [-1, 2, 3, 6],
        [6, 1, 3, 4],
    ],
    ids=["monotonic", "non-monotonic"],
)
@pytest.mark.parametrize("key", [[0, 3, 1], [6, 7]])
@pytest.mark.parametrize("method", [None, "ffill", "bfill", "nearest"])
@pytest.mark.parametrize("tolerance", [None, 1, 2])
def test_get_indexer_single_duplicate_numeric(idx, key, method, tolerance):
    pi = pd.Index(idx)
    gi = cudf.from_pandas(pi)

    if not pi.is_monotonic_increasing and method is not None:
        assert_exceptions_equal(
            lfunc=pi.get_indexer,
            rfunc=gi.get_indexer,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
        )
    else:
        expected = pi.get_indexer(
            key, method=method, tolerance=None if method is None else tolerance
        )
        got = gi.get_indexer(
            key, method=method, tolerance=None if method is None else tolerance
        )

        assert_eq(expected, got)


@pytest.mark.parametrize("idx", [["b", "f", "m", "q"], ["m", "f", "b", "q"]])
@pytest.mark.parametrize("key", [["a", "f", "n", "z"], ["p", "p", "b"]])
@pytest.mark.parametrize("method", [None, "ffill", "bfill"])
def test_get_indexer_single_unique_string(idx, key, method):
    pi = pd.Index(idx)
    gi = cudf.from_pandas(pi)

    if not pi.is_monotonic_increasing and method is not None:
        assert_exceptions_equal(
            lfunc=pi.get_indexer,
            rfunc=gi.get_indexer,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
        )
    else:
        expected = pi.get_indexer(key, method=method)
        got = gi.get_indexer(key, method=method)

        assert_eq(expected, got)


@pytest.mark.parametrize("idx", [["b", "m", "m", "q"], ["a", "f", "m", "q"]])
@pytest.mark.parametrize("key", [["a"], ["f", "n", "z"]])
@pytest.mark.parametrize("method", [None, "ffill", "bfill"])
def test_get_indexer_single_duplicate_string(idx, key, method):
    pi = pd.Index(idx)
    gi = cudf.from_pandas(pi)

    if (
        # `method` only applicable to monotonic index
        (not pi.is_monotonic_increasing and method is not None)
        or not pi.is_unique
    ):
        assert_exceptions_equal(
            lfunc=pi.get_indexer,
            rfunc=gi.get_indexer,
            lfunc_args_and_kwargs=([], {"key": key, "method": method}),
            rfunc_args_and_kwargs=([], {"key": key, "method": method}),
        )
    else:
        expected = pi.get_indexer(key, method=method)
        got = gi.get_indexer(key, method=method)

        assert_eq(expected, got)

        with cudf.option_context("mode.pandas_compatible", True):
            got = gi.get_indexer(key, method=method)

        assert_eq(expected, got, check_dtype=True)


@pytest.mark.parametrize(
    "idx1",
    [
        lambda: cudf.Index(["a", "b", "c"]),
        lambda: cudf.RangeIndex(0, 10),
        lambda: cudf.Index([1, 2, 3], dtype="category"),
        lambda: cudf.Index(["a", "b", "c", "d"], dtype="category"),
        lambda: cudf.MultiIndex.from_tuples(
            [
                ("a", "a", "a"),
                ("a", "b", "c"),
                ("b", "a", "a"),
                ("a", "a", "b"),
                ("a", "b", "a"),
                ("b", "c", "a"),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "idx2",
    [
        lambda: cudf.Index(["a", "b", "c"]),
        lambda: cudf.RangeIndex(0, 10),
        lambda: cudf.Index([1, 2, 3], dtype="category"),
        lambda: cudf.Index(["a", "b", "c", "d"], dtype="category"),
    ],
)
def test_get_indexer_invalid(idx1, idx2):
    idx1 = idx1()
    idx2 = idx2()
    assert_eq(
        idx1.get_indexer(idx2), idx1.to_pandas().get_indexer(idx2.to_pandas())
    )
