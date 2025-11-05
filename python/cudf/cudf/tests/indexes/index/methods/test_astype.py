# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_index_astype(all_supported_types_as_str, copy):
    pdi = pd.Index([1, 2, 3])
    gdi = cudf.from_pandas(pdi)

    actual = gdi.astype(dtype=all_supported_types_as_str, copy=copy)
    expected = pdi.astype(dtype=all_supported_types_as_str, copy=copy)

    assert_eq(expected, actual)
    assert_eq(pdi, gdi)
