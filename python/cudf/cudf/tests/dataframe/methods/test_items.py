# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cudf.testing import assert_eq


def test_iteritems():
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    for k, v in gdf.items():
        assert k in gdf.columns
        assert isinstance(v, cudf.Series)
        assert_eq(v, gdf[k])
