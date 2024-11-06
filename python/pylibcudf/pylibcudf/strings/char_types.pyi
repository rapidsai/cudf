# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum, auto

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

class StringCharacterTypes(IntEnum):
    DECIMAL = auto()
    NUMERIC = auto()
    DIGIT = auto()
    ALPHA = auto()
    SPACE = auto()
    UPPER = auto()
    LOWER = auto()
    ALPHANUM = auto()
    CASE_TYPES = auto()
    ALL_TYPES = auto()

def all_characters_of_type(
    source_strings: Column,
    types: StringCharacterTypes,
    verify_types: StringCharacterTypes,
) -> Column: ...
def filter_characters_of_type(
    source_strings: Column,
    types_to_remove: StringCharacterTypes,
    replacement: Scalar,
    types_to_keep: StringCharacterTypes,
) -> Column: ...
