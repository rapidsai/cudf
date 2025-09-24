# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.strings.regex_program import RegexProgram

def find_re(
    input: Column, pattern: RegexProgram, stream: Stream | None = None
) -> Column: ...
def findall(
    input: Column, pattern: RegexProgram, stream: Stream | None = None
) -> Column: ...
