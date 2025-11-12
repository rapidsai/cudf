# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf


@pytest.mark.parametrize("value", [1, 1.1])
def test_nans_to_nulls_noop_copies_column(value):
    ser1 = cudf.Series([value])
    ser2 = ser1.nans_to_nulls()
    assert ser1._column is not ser2._column
