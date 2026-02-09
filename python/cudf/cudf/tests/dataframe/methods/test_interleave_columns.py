# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("nulls", ["none", "some"])
def test_interleave_columns(nulls, all_supported_types_as_str):
    if (
        all_supported_types_as_str not in ["float32", "float64"]
        and nulls == "some"
    ):
        pytest.skip(
            reason=f"nulls not supported in {all_supported_types_as_str}"
        )

    num_rows = 10
    num_cols = 2
    pdf = pd.DataFrame(dtype=all_supported_types_as_str)
    rng = np.random.default_rng(seed=0)
    for i in range(num_cols):
        colname = str(i)
        data = pd.Series(rng.integers(0, 26, num_rows)).astype(
            all_supported_types_as_str
        )

        if nulls == "some":
            idx = rng.choice(num_rows, size=int(num_rows / 2), replace=False)
            data[idx] = np.nan
        pdf[colname] = data

    gdf = cudf.from_pandas(pdf)

    if all_supported_types_as_str == "category":
        with pytest.raises(ValueError):
            assert gdf.interleave_columns()
    else:
        got = gdf.interleave_columns()

        expect = pd.Series(np.vstack(pdf.to_numpy()).reshape((-1,))).astype(
            all_supported_types_as_str
        )

        assert_eq(expect, got)
