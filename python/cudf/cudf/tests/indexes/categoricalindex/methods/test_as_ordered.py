# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("method", ["as_ordered", "as_unordered"])
@pytest.mark.parametrize("ordered", [True, False])
def test_index_as_ordered(method, ordered):
    pd_ci = pd.CategoricalIndex([1, 2, 3], ordered=ordered)
    cudf_ci = cudf.from_pandas(pd_ci)

    expected = getattr(pd_ci, method)()
    result = getattr(cudf_ci, method)()
    assert_eq(result, expected)
