# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.strings.regex_program import RegexProgram
from pylibcudf.table import Table

def extract(
    input: Column, prog: RegexProgram, stream: Stream | None = None
) -> Table: ...
def extract_all_record(
    input: Column, prog: RegexProgram, stream: Stream | None = None
) -> Column: ...
def extract_single(
    input: Column, prog: RegexProgram, group: int, stream: Stream | None = None
) -> Column: ...
