# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def generate_ngrams(
    input: Column, ngrams: int, separator: Scalar
) -> Column: ...
def generate_character_ngrams(input: Column, ngrams: int = 2) -> Column: ...
def hash_character_ngrams(input: Column, ngrams: int = 2) -> Column: ...
