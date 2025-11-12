# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_reverse():
    arr = pa.array(
        [
            "bunny",
            "rabbit",
            "hare",
            "dog",
            "",
            "a",
            "  leading",
            "trailing  ",
            " mid dle ",
            "123!@#",
            None,
        ]
    )
    got = plc.strings.reverse.reverse(
        plc.Column.from_arrow(arr),
    )
    expect = pa.compute.utf8_reverse(arr)
    assert_column_eq(expect, got)
