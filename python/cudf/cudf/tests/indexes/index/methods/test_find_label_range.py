# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

import cudf


def test_index_find_label_range_index():
    # Monotonic Index
    idx = cudf.Index(np.asarray([4, 5, 6, 10]))
    assert idx.find_label_range(slice(4, 6)) == slice(0, 3, 1)
    assert idx.find_label_range(slice(5, 10)) == slice(1, 4, 1)
    assert idx.find_label_range(slice(0, 6)) == slice(0, 3, 1)
    assert idx.find_label_range(slice(4, 11)) == slice(0, 4, 1)

    # Non-monotonic Index
    idx_nm = cudf.Index(np.asarray([5, 4, 6, 10]))
    assert idx_nm.find_label_range(slice(4, 6)) == slice(1, 3, 1)
    assert idx_nm.find_label_range(slice(5, 10)) == slice(0, 4, 1)
    # Last value not found
    with pytest.raises(KeyError, match="not in index"):
        idx_nm.find_label_range(slice(0, 6))
    # Last value not found
    with pytest.raises(KeyError, match="not in index"):
        idx_nm.find_label_range(slice(4, 11))
