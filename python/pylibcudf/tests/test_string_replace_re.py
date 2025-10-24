# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            pat, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
        plc.Scalar.from_arrow(pa.scalar(repl)),
        max_replace_count=max_replace_count,
    )
    expect = pc.replace_substring_regex(
        arr,
        pat,
        repl,
        max_replacements=max_replace_count
        if max_replace_count != -1
        else None,
    )
    assert_column_eq(expect, got)


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
    got = plc.strings.replace_re.replace_re(
        plc.Column.from_arrow(arr),
        pats,
        plc.Column.from_arrow(pa.array(repls)),
        flags=flags,
    )
    expect = arr
    for pat, repl in zip(pats, repls, strict=True):
        expect = pc.replace_substring_regex(
            expect,
            pat,
            repl,
        )
    assert_column_eq(expect, got)


def test_replace_with_backrefs():
    arr = pa.array(["Z756", None])
    got = plc.strings.replace_re.replace_with_backrefs(
        plc.Column.from_arrow(arr),
        plc.strings.regex_program.RegexProgram.create(
            "(\\d)(\\d)", plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
        "V\\2\\1",
    )
    expect = pa.array(["ZV576", None])
    assert_column_eq(expect, got)
