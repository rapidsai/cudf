# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("count", [1, 10])
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_tile(nulls, all_supported_types_as_str, count):
    if (
        all_supported_types_as_str not in ["float32", "float64"]
        and nulls == "some"
    ):
        pytest.skip(
            reason=f"nulls not supported in {all_supported_types_as_str}"
        )

    num_cols = 2
    num_rows = 10
    pdf = pd.DataFrame(dtype=all_supported_types_as_str)
    rng = np.random.default_rng(seed=0)
    for i in range(num_cols):
        colname = str(i)
        data = pd.Series(rng.integers(num_cols, 26, num_rows)).astype(
            all_supported_types_as_str
        )

        if nulls == "some":
            idx = rng.choice(num_rows, size=int(num_rows / 2), replace=False)
            data[idx] = np.nan
        pdf[colname] = data

    gdf = cudf.from_pandas(pdf)

    got = gdf.tile(count)
    expect = pd.DataFrame(pd.concat([pdf] * count))

    assert_eq(expect, got)
