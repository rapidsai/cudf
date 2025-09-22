# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

def is_letter(
    input: Column,
    check_vowels: bool,
    indices: Column | int,
    stream: Stream | None = None,
) -> Column: ...
def porter_stemmer_measure(
    input: Column, stream: Stream | None = None
) -> Column: ...
