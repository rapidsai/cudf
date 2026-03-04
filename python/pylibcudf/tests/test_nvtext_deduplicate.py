# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def input_col():
    arr = ["0123456789"] * 6
    return pa.array(arr)


@pytest.mark.parametrize("min_width", [10, 20])
def test_substring_duplicates(input_col, min_width):
    sa = plc.nvtext.deduplicate.build_suffix_array(
        plc.Column.from_arrow(input_col),
        min_width,
    )
    result = plc.nvtext.deduplicate.resolve_duplicates(
        plc.Column.from_arrow(input_col),
        sa,
        min_width,
    )
    expected = pa.array(["01234567890123456789012345678901234567890123456789"])
    assert_column_eq(result, expected)
