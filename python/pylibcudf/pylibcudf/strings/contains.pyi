# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.regex_program import RegexProgram

def contains_re(
    input: Column, prog: RegexProgram, stream: Stream | None = None
) -> Column: ...
def count_re(
    input: Column, prog: RegexProgram, stream: Stream | None = None
) -> Column: ...
def matches_re(
    input: Column, prog: RegexProgram, stream: Stream | None = None
) -> Column: ...
def like(
    input: Column,
    pattern: Column | Scalar,
    escape_character: Scalar | None = None,
    stream: Stream | None = None,
) -> Column: ...
