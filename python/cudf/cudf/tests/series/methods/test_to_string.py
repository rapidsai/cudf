# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf


def test_series_init_none():
    # test for creating empty series
    # 1: without initializing
    sr1 = cudf.Series()
    got = sr1.to_string()

    expect = repr(sr1.to_pandas())
    assert got == expect

    # 2: Using `None` as an initializer
    sr2 = cudf.Series(None)
    got = sr2.to_string()

    expect = repr(sr2.to_pandas())
    assert got == expect
