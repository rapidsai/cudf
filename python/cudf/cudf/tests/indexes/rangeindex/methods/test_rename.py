# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

import cudf


def test_rename_shallow_copy():
    idx = pd.Index([1])
    result = idx.rename("a")
    assert idx.to_numpy(copy=False) is result.to_numpy(copy=False)

    idx = cudf.Index([1])
    result = idx.rename("a")
    assert idx._column.base_data.get_ptr(
        mode="read"
    ) == result._column.base_data.get_ptr(mode="read")
