# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_unique_level():
    pd_mi = pd.MultiIndex.from_arrays([[1, 1, 2], [3, 3, 2]])
    cudf_mi = cudf.MultiIndex(
        levels=pd_mi.levels, codes=pd_mi.codes, names=pd_mi.names
    )

    result = pd_mi.unique(level=1)
    expected = cudf_mi.unique(level=1)
    assert_eq(result, expected)
