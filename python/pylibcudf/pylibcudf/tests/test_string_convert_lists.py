# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.mark.parametrize("na_rep", [None, pa.scalar("")])
@pytest.mark.parametrize("separators", [None, pa.array([",", "[", "]"])])
def test_format_list_column(na_rep, separators):
    arr = pa.array([["1", "A"], None])
    result = plc.strings.convert.convert_lists.format_list_column(
        plc.interop.from_arrow(arr),
        na_rep if na_rep is None else plc.interop.from_arrow(na_rep),
        separators
        if separators is None
        else plc.interop.from_arrow(separators),
    )
    expected = pa.array(["[1,A]", ""])
    assert_column_eq(result, expected)
