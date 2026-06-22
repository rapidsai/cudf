# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("data", [range(2), range(2, -1, -1)])
def test_rangeindex_factorize(sort, data):
    res_codes, res_uniques = cudf.RangeIndex(data).factorize(sort=sort)
    exp_codes, exp_uniques = cudf.Index(list(data)).factorize(sort=sort)
    assert_eq(res_codes, exp_codes)
    assert_eq(res_uniques, exp_uniques)
