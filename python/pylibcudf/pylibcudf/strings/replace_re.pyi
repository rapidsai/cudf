# Copyright (c) 2024, NVIDIA CORPORATION.

from typing import overload

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.regex_flags import RegexFlags
from pylibcudf.strings.regex_program import RegexProgram

@overload
def replace_re(
    input: Column,
    pattern: RegexProgram,
    replacement: Scalar,
    max_replace_count: int = -1,
) -> Column: ...
@overload
def replace_re(
    input: Column,
    patterns: list[str],
    replacement: Column,
    max_replace_count: int = -1,
    flags: RegexFlags = RegexFlags.DEFAULT,
) -> Column: ...
def replace_with_backrefs(
    input: Column, prog: RegexProgram, replacement: str
) -> Column: ...
