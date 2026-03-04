# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import cudf
from cudf.testing import assert_eq


def test_multiindex_swaplevel():
    midx = cudf.MultiIndex(
        levels=[
            ["lama", "cow", "falcon"],
            ["speed", "weight", "length"],
            ["first", "second"],
        ],
        codes=[
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
        ],
        names=["Col1", "Col2", "Col3"],
    )
    pd_midx = midx.to_pandas()

    assert_eq(pd_midx.swaplevel(-1, -2), midx.swaplevel(-1, -2))
    assert_eq(pd_midx.swaplevel(2, 1), midx.swaplevel(2, 1))
    assert_eq(midx.swaplevel(2, 1), midx.swaplevel(1, 2))
    assert_eq(pd_midx.swaplevel(0, 2), midx.swaplevel(0, 2))
    assert_eq(pd_midx.swaplevel(2, 0), midx.swaplevel(2, 0))
    assert_eq(midx.swaplevel(1, 1), midx.swaplevel(1, 1))
