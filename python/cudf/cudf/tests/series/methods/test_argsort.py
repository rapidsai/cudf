# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf


def test_series_argsort(numeric_types_as_str, ascending):
    sr = cudf.Series([1, 3, 2, 5, 4]).astype(numeric_types_as_str)
    res = sr.argsort(ascending=ascending)

    if ascending:
        expected = np.argsort(sr.to_numpy(), kind="mergesort")
    else:
        # -1 multiply works around missing desc sort (may promote to float64)
        expected = np.argsort(sr.to_numpy() * np.int8(-1), kind="mergesort")
    np.testing.assert_array_equal(expected, res.to_numpy())
