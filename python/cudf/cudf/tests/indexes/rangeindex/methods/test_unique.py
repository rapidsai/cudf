# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_rangeindex_unique_shallow_copy():
    ri_pandas = pd.RangeIndex(1)
    result = ri_pandas.unique()
    assert result is not ri_pandas

    ri_cudf = cudf.RangeIndex(1)
    result = ri_cudf.unique()
    assert result is not ri_cudf
    assert_eq(result, ri_cudf)
