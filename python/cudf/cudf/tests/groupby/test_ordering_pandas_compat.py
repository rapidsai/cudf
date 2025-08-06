# Copyright (c) 2024-2025, NVIDIA CORPORATION.
import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("with_nulls", [False, True])
def test_groupby_maintain_order_random(with_nulls):
    nrows = 20
    nkeys = 3
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
