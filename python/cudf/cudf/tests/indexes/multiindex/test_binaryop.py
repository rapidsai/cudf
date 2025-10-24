# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

import cudf
from cudf.testing import assert_eq


def test_multiindex_eq_other_multiindex():
    idx = cudf.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])
    result = idx == idx
    expected = np.array([True, True])
    assert_eq(result, expected)
