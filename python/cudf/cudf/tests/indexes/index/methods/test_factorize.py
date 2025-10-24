# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd

import cudf
from cudf import Index


def test_factorize_index_obj():
    rng = np.random.default_rng(seed=0)

    arr = rng.integers(2, size=10, dtype=np.int32)
    ser = cudf.Index(arr)

    uvals, labels = ser.factorize()
    unique_values, indices = np.unique(arr, return_index=True)
    expected_values = unique_values[np.argsort(indices)]

    np.testing.assert_array_equal(labels.values.get(), expected_values)
    assert isinstance(uvals, cp.ndarray)
    assert isinstance(labels, Index)

    encoder = {labels[idx]: idx for idx in range(len(labels))}
    handcoded = [encoder[v] for v in arr]
    np.testing.assert_array_equal(uvals.get(), handcoded)


def test_cudf_factorize_index():
    data = [1, 2, 3, 4, 5]

    pi = pd.Index(data)
    gi = cudf.Index(data)

    expect = pd.factorize(pi)
    got = cudf.factorize(gi)

    assert len(expect) == len(got)

    np.testing.assert_array_equal(expect[0], got[0].get())
    np.testing.assert_array_equal(expect[1], got[1].values.get())
