# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.mark.parametrize("na_rep", [None, pa.scalar("")])
@pytest.mark.parametrize("separators", [None, pa.array([",", "[", "]"])])
def test_format_list_column(na_rep, separators):
    got = plc.strings.convert.convert_lists.format_list_column(
        plc.Column.from_arrow(pa.array([["1", "A"], None])),
        na_rep if na_rep is None else plc.Scalar.from_arrow(na_rep),
        separators
        if separators is None
        else plc.Column.from_arrow(separators),
    )
    expect = pa.array(["[1,A]", ""])
    assert_column_eq(expect, got)
