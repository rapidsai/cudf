# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.strings.regex_program import RegexProgram
from pylibcudf.table import Table

def extract(input: Column, prog: RegexProgram) -> Table: ...
def extract_all_record(input: Column, prog: RegexProgram) -> Column: ...
def extract_single(
    input: Column, prog: RegexProgram, group: int
) -> Column: ...
