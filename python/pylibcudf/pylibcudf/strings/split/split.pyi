# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.regex_program import RegexProgram
from pylibcudf.table import Table

def split(
    strings_column: Column, delimiter: Scalar, maxsplit: int
) -> Table: ...
def rsplit(
    strings_column: Column, delimiter: Scalar, maxsplit: int
) -> Table: ...
def split_record(
    strings: Column, delimiter: Scalar, maxsplit: int
) -> Column: ...
def rsplit_record(
    strings: Column, delimiter: Scalar, maxsplit: int
) -> Column: ...
def split_re(input: Column, prog: RegexProgram, maxsplit: int) -> Table: ...
def rsplit_re(input: Column, prog: RegexProgram, maxsplit: int) -> Table: ...
def split_record_re(
    input: Column, prog: RegexProgram, maxsplit: int
) -> Column: ...
def rsplit_record_re(
    input: Column, prog: RegexProgram, maxsplit: int
) -> Column: ...
