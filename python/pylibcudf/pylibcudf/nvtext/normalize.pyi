# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

class CharacterNormalizer:
    def __init__(
        self,
        do_lower_case: bool,
        special_tokens: Column,
        stream: Stream | None = None,
    ): ...

def normalize_spaces(
    input: Column, stream: Stream | None = None
) -> Column: ...
def normalize_characters(
    input: Column,
    normalizer: CharacterNormalizer,
    stream: Stream | None = None,
) -> Column: ...
