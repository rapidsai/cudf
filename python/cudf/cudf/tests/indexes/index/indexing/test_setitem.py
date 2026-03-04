# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf


def test_index_immutable():
    start, stop = 10, 34
    rg = cudf.RangeIndex(start, stop)
    with pytest.raises(TypeError):
        rg[1] = 5
    gi = cudf.Index(np.arange(start, stop))
    with pytest.raises(TypeError):
        gi[1] = 5
