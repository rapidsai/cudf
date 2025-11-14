# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_index_remove_categories():
    pd_ci = pd.CategoricalIndex([1, 2, 3], categories=[1, 2, 3, 4])
    cudf_ci = cudf.from_pandas(pd_ci)

    expected = pd_ci.remove_categories([4])
    result = cudf_ci.remove_categories([4])
    assert_eq(result, expected)
