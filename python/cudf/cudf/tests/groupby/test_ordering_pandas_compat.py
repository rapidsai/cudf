# Copyright (c) 2024, NVIDIA CORPORATION.
import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture(params=[False, True], ids=["without_nulls", "with_nulls"])
def with_nulls(request):
    return request.param


@pytest.mark.parametrize("nrows", [30, 300, 300_000])
@pytest.mark.parametrize("nkeys", [1, 2, 4])
def test_groupby_maintain_order_random(nrows, nkeys, with_nulls):
    rng = np.random.default_rng(seed=0)
    key_names = [f"key{key}" for key in range(nkeys)]
    key_values = [rng.integers(100, size=nrows) for _ in key_names]
    value = rng.integers(-100, 100, size=nrows)
    df = cudf.DataFrame(dict(zip(key_names, key_values), value=value))
    if with_nulls:
        for key in key_names:
            df.loc[df[key] == 1, key] = None
    with cudf.option_context("mode.pandas_compatible", True):
        got = df.groupby(key_names, sort=False).agg({"value": "sum"})
    expect = (
        df.to_pandas().groupby(key_names, sort=False).agg({"value": "sum"})
    )
    assert_eq(expect, got, check_index_type=not with_nulls)
