# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("data", [[True, False, None], [10, 200, 300]])
@pytest.mark.parametrize("index", [None, [10, 20, 30]])
def test_series_contains(data, index):
    ps = pd.Series(data, index=index)
    gs = cudf.Series(data, index=index)

    assert_eq(1 in ps, 1 in gs)
    assert_eq(10 in ps, 10 in gs)
    assert_eq(True in ps, True in gs)
    assert_eq(False in ps, False in gs)
