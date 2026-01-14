# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
@pytest.mark.parametrize("elem1", [np.nan, np.inf, -np.inf, 1.43])
@pytest.mark.parametrize("elem2", [np.nan, np.inf, -np.inf, 1.43])
def test_series_rank_combinations(elem1, elem2, numeric_types_as_str):
    aa = np.array([elem1, elem2], dtype=np.float64).astype(
        numeric_types_as_str, copy=False
    )
    pd_ser = pd.Series(aa)
    cudf_ser = cudf.Series(aa)
    ranked_gs = cudf_ser.rank(method="first")
    ranked_ps = pd_ser.rank(method="first")
    assert_eq(ranked_ps, ranked_gs)


@pytest.mark.parametrize("klass", ["Series", "DataFrame"])
def test_int_nan_pandas_compatible(klass):
    data = [3, 6, 1, 1, None, 6]
    pd_obj = getattr(pd, klass)(data)
    cudf_obj = getattr(cudf, klass)(data)
    with cudf.option_context("mode.pandas_compatible", True):
        result = cudf_obj.rank()
    expected = pd_obj.rank()
    assert_eq(result, expected)
