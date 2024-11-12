# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.regex_program import RegexProgram

def contains_re(input: Column, prog: RegexProgram) -> Column: ...
def count_re(input: Column, prog: RegexProgram) -> Column: ...
def matches_re(input: Column, prog: RegexProgram) -> Column: ...
def like(
    input: Column,
    pattern: Column | Scalar,
    escape_character: Scalar | None = None,
) -> Column: ...
