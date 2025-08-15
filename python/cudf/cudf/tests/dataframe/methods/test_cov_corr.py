# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np

import cudf
from cudf.testing import assert_eq


def test_df_corr(corr_method):
    gdf = cudf.DataFrame(np.random.default_rng(seed=0).normal(-100, 100, 10))
    pdf = gdf.to_pandas()
    got = gdf.corr(corr_method)
    expected = pdf.corr(corr_method)
    assert_eq(got, expected)
