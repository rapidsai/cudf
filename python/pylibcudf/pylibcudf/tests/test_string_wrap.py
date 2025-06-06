# Copyright (c) 2024-2025, NVIDIA CORPORATION.
import textwrap

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_wrap():
    width = 12
    pa_array = pa.array(
        [
            "the quick brown fox jumped over the lazy brown dog",
            "hello, world",
            None,
        ]
    )
    got = plc.strings.wrap.wrap(plc.Column.from_arrow(pa_array), width)
    expect = pa.array(
        [
            textwrap.fill(val, width) if isinstance(val, str) else val
            for val in pa_array.to_pylist()
        ]
    )
    assert_column_eq(expect, got)
