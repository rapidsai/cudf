# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.strings.char_types import StringCharacterTypes

def capitalize(
    input: Column,
    delimiters: Scalar | None = None,
    stream: Stream | None = None,
) -> Column: ...
def title(
    input: Column,
    sequence_type: StringCharacterTypes = StringCharacterTypes.ALPHA,
    stream: Stream | None = None,
) -> Column: ...
def is_title(input: Column, stream: Stream | None = None) -> Column: ...
