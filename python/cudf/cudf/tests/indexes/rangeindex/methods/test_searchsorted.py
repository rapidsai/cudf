# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf


def test_index_rangeindex_searchsorted():
    # step > 0
    ridx = cudf.RangeIndex(-13, 17, 4)
    for i in range(len(ridx)):
        assert i == ridx.searchsorted(ridx[i], side="left")
        assert i + 1 == ridx.searchsorted(ridx[i], side="right")
