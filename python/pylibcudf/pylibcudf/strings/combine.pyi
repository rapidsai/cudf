# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum, auto

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table

class SeparatorOnNulls(IntEnum):
    YES = auto()
    NO = auto()

class OutputIfEmptyList(IntEnum):
    EMPTY_STRING = auto()
    NULL_ELEMENT = auto()

def concatenate(
    strings_columns: Table,
    separator: Column | Scalar,
    narep: Scalar | None = None,
    col_narep: Scalar | None = None,
    separate_nulls: SeparatorOnNulls = SeparatorOnNulls.YES,
) -> Column: ...
def join_strings(
    input: Column, separator: Scalar, narep: Scalar
) -> Column: ...
def join_list_elements(
    source_strings: Column,
    separator: Column | Scalar,
    separator_narep: Scalar,
    string_narep: Scalar,
    separate_nulls: SeparatorOnNulls,
    empty_list_policy: OutputIfEmptyList,
) -> Column: ...
