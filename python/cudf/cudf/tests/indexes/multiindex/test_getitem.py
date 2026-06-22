# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_multiindex_empty_slice_pandas_compatibility():
    expected = pd.MultiIndex.from_tuples([("a", "b")])[:0]
    with cudf.option_context("mode.pandas_compatible", True):
        actual = cudf.from_pandas(expected)
    assert_eq(expected, actual, exact=False)
