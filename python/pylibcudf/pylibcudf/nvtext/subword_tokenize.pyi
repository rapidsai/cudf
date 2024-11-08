# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column

class HashedVocabulary:
    def __init__(self, hash_file: str): ...

def subword_tokenize(
    input: Column,
    vocabulary_table: HashedVocabulary,
    max_sequence_length: int,
    stride: int,
    do_lower_case: bool,
    do_truncate: bool,
) -> tuple[Column, Column, Column]: ...
