# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("ordered", [True, False])
def test_index_reorder_categories(ordered):
    pd_ci = pd.CategoricalIndex([1, 2, 3], categories=[1, 3, 2, 4])
    cudf_ci = cudf.from_pandas(pd_ci)

    expected = pd_ci.reorder_categories([1, 2, 3, 4], ordered=ordered)
    result = cudf_ci.reorder_categories([1, 2, 3, 4], ordered=ordered)
    assert_eq(result, expected)
