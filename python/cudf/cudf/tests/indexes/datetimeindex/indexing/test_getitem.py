# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cudf.testing import assert_eq


def test_index_getitem_time_duration(temporal_types_as_str):
    gidx = cudf.Index([1, 2, 3, 4, None], dtype=temporal_types_as_str)
    pidx = gidx.to_pandas()
    with cudf.option_context("mode.pandas_compatible", True):
        for i in range(len(gidx)):
            if i == 4:
                assert gidx[i] is pidx[i]
            else:
                assert_eq(gidx[i], pidx[i])
