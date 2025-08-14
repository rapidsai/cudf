# Copyright (c) 2025, NVIDIA CORPORATION.

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
