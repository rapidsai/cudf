# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from cudf_polars.utils.sorting import sort_order


@pytest.mark.parametrize(
    "descending,nulls_last,num_keys",
    [
        ([True], [False, True], 3),
        ([True, True], [False, True, False], 3),
        ([False, True], [True], 3),
    ],
)
def test_sort_order_raises_mismatch(descending, nulls_last, num_keys):
    with pytest.raises(ValueError):
        _ = sort_order(descending, nulls_last=nulls_last, num_keys=num_keys)
