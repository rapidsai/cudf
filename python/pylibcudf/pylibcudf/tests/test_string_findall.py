# Copyright (c) 2024, NVIDIA CORPORATION.
import re

import pyarrow as pa
import pylibcudf as plc
from utils import assert_column_eq


def test_findall():
    arr = pa.array(["bunny", "rabbit", "hare", "dog"])
    pattern = "[ab]"
    plc_result = plc.strings.findall.findall(
        plc.interop.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
    )
    result = plc.interop.to_arrow(plc_result)
    expected = pa.chunked_array(
        [
            pa.array(
                [re.findall(pattern, elem) for elem in arr.to_pylist()],
                type=result.type,
            )
        ]
    )
    assert assert_column_eq(result, expected)
