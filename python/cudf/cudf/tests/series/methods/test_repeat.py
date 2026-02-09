# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_repeat(all_supported_types_as_str):
    rng = np.random.default_rng(seed=0)
    arr = rng.random(10) * 10
    repeats = rng.integers(10, size=10)
    psr = pd.Series(arr).astype(all_supported_types_as_str)
    gsr = cudf.from_pandas(psr)

    assert_eq(psr.repeat(repeats), gsr.repeat(repeats))


def test_repeat_scalar(numeric_types_as_str):
    rng = np.random.default_rng(seed=0)
    arr = rng.random(10) * 10
    repeats = 10
    psr = pd.Series(arr).astype(numeric_types_as_str)
    gsr = cudf.from_pandas(psr)

    assert_eq(psr.repeat(repeats), gsr.repeat(repeats))
