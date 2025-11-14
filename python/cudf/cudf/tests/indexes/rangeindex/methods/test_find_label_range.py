# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf


def test_index_find_label_range_rangeindex():
    """Cudf specific"""
    # step > 0
    # 3, 8, 13, 18
    ridx = cudf.RangeIndex(3, 20, 5)
    assert ridx.find_label_range(slice(3, 8)) == slice(0, 2, 1)
    assert ridx.find_label_range(slice(0, 7)) == slice(0, 1, 1)
    assert ridx.find_label_range(slice(3, 19)) == slice(0, 4, 1)
    assert ridx.find_label_range(slice(2, 21)) == slice(0, 4, 1)

    # step < 0
    # 20, 15, 10, 5
    ridx = cudf.RangeIndex(20, 3, -5)
    assert ridx.find_label_range(slice(15, 10)) == slice(1, 3, 1)
    assert ridx.find_label_range(slice(10, 15, -1)) == slice(2, 0, -1)
    assert ridx.find_label_range(slice(10, 0)) == slice(2, 4, 1)
    assert ridx.find_label_range(slice(30, 13)) == slice(0, 2, 1)
    assert ridx.find_label_range(slice(30, 0)) == slice(0, 4, 1)
