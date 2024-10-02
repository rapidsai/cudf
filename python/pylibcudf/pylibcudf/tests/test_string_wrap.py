# Copyright (c) 2024, NVIDIA CORPORATION.
import textwrap

import pyarrow as pa
import pylibcudf as plc
from utils import assert_column_eq


def test_wrap():
    pa_array = pa.array(
        [
            "the quick brown fox jumped over the lazy brown dog",
            "hello, world",
            None,
        ]
    )
    result = plc.strings.wrap.wrap(plc.interop.from_arrow(pa_array), 12)
    expected = pa.array(
        [
            textwrap.fill(val, 12) if isinstance(val, str) else val
            for val in pa_array.to_pylist()
        ]
    )
    assert_column_eq(expected, result)
