# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def generate_ngrams(
    input: Column, ngrams: int, separator: Scalar, stream: Stream | None = None
) -> Column: ...
def generate_character_ngrams(
    input: Column, ngrams: int = 2, stream: Stream | None = None
) -> Column: ...
def hash_character_ngrams(
    input: Column, ngrams: int, seed: int, stream: Stream | None = None
) -> Column: ...
