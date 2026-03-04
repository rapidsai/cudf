# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_repeat_index():
    rng = np.random.default_rng(seed=0)
    arrays = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
    psr = pd.MultiIndex.from_arrays(arrays, names=("number", "color"))
    gsr = cudf.from_pandas(psr)
    repeats = rng.integers(10, size=4)

    assert_eq(psr.repeat(repeats), gsr.repeat(repeats))
