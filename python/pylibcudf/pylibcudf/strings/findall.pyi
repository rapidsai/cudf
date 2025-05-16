# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.strings.regex_program import RegexProgram

def find_re(input: Column, pattern: RegexProgram) -> Column: ...
def findall(input: Column, pattern: RegexProgram) -> Column: ...
