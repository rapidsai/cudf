# Copyright (c) 2024, NVIDIA CORPORATION.

import numpy as np
import pyarrow as pa
import pylibcudf as plc
from utils import assert_table_eq


def test_cross_join():
    left = pa.Table.from_arrays([[0, 1, 2], [3, 4, 5]], names=["a", "b"])
    right = pa.Table.from_arrays(
        [[6, 7, 8, 9], [10, 11, 12, 13]], names=["c", "d"]
    )

    pleft = plc.interop.from_arrow(left)
    pright = plc.interop.from_arrow(right)

    expect = pa.Table.from_arrays(
        [
            *(np.repeat(c.to_numpy(), len(right)) for c in left.columns),
            *(np.tile(c.to_numpy(), len(left)) for c in right.columns),
        ],
        names=["a", "b", "c", "d"],
    )

    got = plc.join.cross_join(pleft, pright)

    assert_table_eq(expect, got)
