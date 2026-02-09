# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_df_corr(corr_method):
    gdf = cudf.DataFrame(np.random.default_rng(seed=0).normal(-100, 100, 10))
    pdf = gdf.to_pandas()
    got = gdf.corr(corr_method)
    expected = pdf.corr(corr_method)
    assert_eq(got, expected)


def test_cov():
    gdf = cudf.DataFrame(np.random.default_rng(seed=0).normal(-100, 100, 10))
    pdf = gdf.to_pandas()

    assert_eq(pdf.cov(), gdf.cov())


@pytest.mark.xfail(
    raises=NotImplementedError, reason="cupy-based cov does not support nulls"
)
def test_cov_nans():
    pdf = pd.DataFrame(
        {
            "a": [None, None, None, 2.00758632, None],
            "b": [0.36403686, None, None, None, None],
            "c": [None, None, None, 0.64882227, None],
            "d": [None, -1.46863125, None, 1.22477948, -0.06031689],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.cov(), gdf.cov())
