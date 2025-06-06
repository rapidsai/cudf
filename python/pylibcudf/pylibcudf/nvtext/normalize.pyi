# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column import Column

class CharacterNormalizer:
    def __init__(self, do_lower_case: bool, special_tokens: Column): ...

def normalize_spaces(input: Column) -> Column: ...
def normalize_characters(
    input: Column, normalizer: CharacterNormalizer
) -> Column: ...
