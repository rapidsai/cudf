# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.compute as pc

import pylibcudf as plc


def test_extract():
    pattern = "([ab])(\\d)"
    pa_pattern = "(?P<letter>[ab])(?P<digit>\\d)"
    arr = pa.array(["a1", "b2", "c3"])
    result = plc.strings.extract.extract(
        plc.Column.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
    ).to_arrow()
    expected = pc.extract_regex(arr, pa_pattern)
    for i, result_col in enumerate(result.itercolumns()):
        expected_col = pa.chunked_array(expected.field(i))
        assert result_col.fill_null("").equals(expected_col)


def test_extract_all_record():
    pattern = "([ab])(\\d)"
    arr = pa.array(["a1", "b2", "c3"])
    result = plc.strings.extract.extract_all_record(
        plc.Column.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
    ).to_arrow()
    expected = pa.array([["a", "1"], ["b", "2"], None], type=result.type)
    assert result.equals(expected)
