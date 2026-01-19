# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.

# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa

from pylibcudf import Column, Scalar
from pylibcudf.strings import split


def test_split_part():
    # 1. Setup Data using PyArrow (Standard way for pylibcudf tests)
    data = pa.array(["a_b_c", "d_e", "f", None])
    col = Column.from_arrow(data)

    delimiter = Scalar("_")

    # 2. Call the Cython wrapper directly
    # Case: Index 1
    result = split.split_part(col, delimiter, 1)

    # 3. Verify
    # Convert back to Arrow to check values
    # Expected: "b", "e", null, null
    result_pa = result.to_arrow()
    expected = pa.array(["b", "e", None, None])

    assert result_pa.equals(expected)


def test_split_part_out_of_bounds():
    data = pa.array(["a_b", "c"])
    col = Column.from_arrow(data)
    delimiter = Scalar("_")

    # Case: Index 10 (Out of bounds)
    result = split.split_part(col, delimiter, 10)

    # Expected: null, null
    result_pa = result.to_arrow()
    expected = pa.array([None, None], type=pa.string())

    assert result_pa.equals(expected)
