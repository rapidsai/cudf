# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_index_isna_notna():
    idx = [1, None, 3, None, 5]
    pidx = pd.Index(idx, name="idx")
    gidx = cudf.Index(idx, name="idx")
    assert_eq(gidx.isna(), pidx.isna())
    assert_eq(gidx.notna(), pidx.notna())
