# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing._utils import (
    assert_exceptions_equal,
)


@pytest.mark.parametrize("attr", ["nlargest", "nsmallest"])
def test_series_nlargest_nsmallest_str_error(attr):
    gs = cudf.Series(["a", "b", "c", "d", "e"])
    ps = gs.to_pandas()

    assert_exceptions_equal(
        getattr(gs, attr), getattr(ps, attr), ([], {"n": 1}), ([], {"n": 1})
    )
