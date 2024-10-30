# Copyright (c) 2024, NVIDIA CORPORATION.
import re

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_findall():
    arr = pa.array(["bunny", "rabbit", "hare", "dog"])
    pattern = "[ab]"
    result = plc.strings.findall.findall(
        plc.interop.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
    )
    pa_result = plc.interop.to_arrow(result)
    expected = pa.array(
        [re.findall(pattern, elem) for elem in arr.to_pylist()],
        type=pa_result.type,
    )
    assert_column_eq(result, expected)


def test_find_re():
    arr = pa.array(["bunny", "rabbit", "hare", "dog"])
    pattern = "[eb]"
    result = plc.strings.findall.find_re(
        plc.interop.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
    )
    pa_result = plc.interop.to_arrow(result)
    expected = pa.array(
        [0, 2, 3, -1],
        type=pa_result.type,
    )
    assert_column_eq(result, expected)
