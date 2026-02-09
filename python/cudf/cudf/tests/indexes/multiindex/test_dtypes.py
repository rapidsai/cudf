# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cudf
from cudf.testing import assert_eq


def test_multiindex_dtypes():
    mi = cudf.MultiIndex.from_tuples(
        [("a", 1), ("b", 2), ("c", 3)], names=["letters", "numbers"]
    )
    assert_eq(mi.dtypes, mi.to_pandas().dtypes)
