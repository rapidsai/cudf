# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf


def test_series_init_none():
    # test for creating empty series
    # 1: without initializing
    sr1 = cudf.Series()
    got = sr1.to_string()

    expect = sr1.to_pandas().to_string()
    assert got == expect

    # 2: Using `None` as an initializer
    sr2 = cudf.Series(None)
    got = sr2.to_string()

    expect = sr2.to_pandas().to_string()
    assert got == expect
