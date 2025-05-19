# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.char_types import StringCharacterTypes

def capitalize(input: Column, delimiters: Scalar | None = None) -> Column: ...
def title(
    input: Column,
    sequence_type: StringCharacterTypes = StringCharacterTypes.ALPHA,
) -> Column: ...
def is_title(input: Column) -> Column: ...
