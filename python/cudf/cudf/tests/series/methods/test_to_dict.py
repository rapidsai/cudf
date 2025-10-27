# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict, defaultdict

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("into", [dict, OrderedDict, defaultdict(list)])
def test_series_to_dict(into):
    gs = cudf.Series(["ab", "de", "zx"], index=[10, 20, 100])
    ps = gs.to_pandas()

    actual = gs.to_dict(into=into)
    expected = ps.to_dict(into=into)

    assert_eq(expected, actual)
