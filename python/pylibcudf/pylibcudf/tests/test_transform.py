# Copyright (c) 2024, NVIDIA CORPORATION.

import math

import pyarrow as pa
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc


def test_nans_to_nulls(has_nans):
    if has_nans:
        values = [1, float("nan"), float("nan"), None, 3, None]
    else:
        values = [1, 4, 5, None, 3, None]

    replaced = [
        None if (v is None or (v is not None and math.isnan(v))) else v
        for v in values
    ]

    h_input = pa.array(values, type=pa.float32())
    input = plc.interop.from_arrow(h_input)
    assert input.null_count() == h_input.null_count
    expect = pa.array(replaced, type=pa.float32())

    mask, null_count = plc.transform.nans_to_nulls(input)

    assert null_count == expect.null_count
    got = input.with_mask(mask, null_count)

    assert_column_eq(expect, got)
