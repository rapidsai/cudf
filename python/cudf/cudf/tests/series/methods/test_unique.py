# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_datetime_unique(data, nulls):
    rng = np.random.default_rng(seed=0)
    psr = data.copy()

    if len(data) > 0:
        if nulls == "some":
            p = rng.integers(0, len(data), 2)
            psr[p] = None

    gsr = cudf.from_pandas(psr)
    expected = psr.unique()
    got = gsr.unique()

    # Unique does not provide a guarantee on ordering.
    assert_eq(
        pd.Series(expected).sort_values(ignore_index=True),
        got.sort_values(ignore_index=True).to_pandas(),
    )
