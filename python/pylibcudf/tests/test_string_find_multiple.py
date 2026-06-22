# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_find_multiple():
    arr = pa.array(["abc", "def"])
    targets = pa.array(["a", "c", "e"])
    got = plc.strings.find_multiple.find_multiple(
        plc.Column.from_arrow(arr),
        plc.Column.from_arrow(targets),
    )
    expect = pa.array(
        [
            [elem.find(target) for target in targets.to_pylist()]
            for elem in arr.to_pylist()
        ],
        type=pa.list_(pa.int32()),
    )
    assert_column_eq(expect, got)
