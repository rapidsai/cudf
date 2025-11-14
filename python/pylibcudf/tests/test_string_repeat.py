# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.mark.parametrize("repeats", [pa.array([2, 2]), 2])
def test_repeat_strings(repeats):
    arr = pa.array(["1", None])
    got = plc.strings.repeat.repeat_strings(
        plc.Column.from_arrow(arr),
        plc.Column.from_arrow(repeats)
        if not isinstance(repeats, int)
        else repeats,
    )
    expect = pa.array(pc.binary_repeat(arr, repeats))
    assert_column_eq(expect, got)
