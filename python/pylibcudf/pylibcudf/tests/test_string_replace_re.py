# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.mark.parametrize("max_replace_count", [-1, 1])
def test_replace_re_regex_program_scalar(max_replace_count):
    arr = pa.array(["foo", "fuz", None])
    pat = "f."
    repl = "ba"
    result = plc.strings.replace_re.replace_re(
        plc.interop.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pat, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
        plc.interop.from_arrow(pa.scalar(repl)),
        max_replace_count=max_replace_count,
    )
    expected = pc.replace_substring_regex(
        arr,
        pat,
        repl,
        max_replacements=max_replace_count
        if max_replace_count != -1
        else None,
    )
    assert_column_eq(result, expected)


@pytest.mark.parametrize(
    "flags",
    [
        plc.strings.regex_flags.RegexFlags.DEFAULT,
        plc.strings.regex_flags.RegexFlags.DOTALL,
    ],
)
def test_replace_re_list_str_columns(flags):
    arr = pa.array(["foo", "fuz", None])
    pats = ["oo", "uz"]
    repls = ["a", "b"]
    result = plc.strings.replace_re.replace_re(
        plc.interop.from_arrow(arr),
        pats,
        plc.interop.from_arrow(pa.array(repls)),
        flags=flags,
    )
    expected = arr
    for pat, repl in zip(pats, repls):
        expected = pc.replace_substring_regex(
            expected,
            pat,
            repl,
        )
    assert_column_eq(result, expected)


def test_replace_with_backrefs():
    arr = pa.array(["Z756", None])
    result = plc.strings.replace_re.replace_with_backrefs(
        plc.interop.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            "(\\d)(\\d)", plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
        "V\\2\\1",
    )
    expected = pa.array(["ZV576", None])
    assert_column_eq(result, expected)
