# Copyright (c) 2025, NVIDIA CORPORATION.

from pylibcudf.column import Column

class WordPieceVocabulary:
    def __init__(self, vocab: Column): ...

def wordpiece_tokenize(
    input: Column,
    vocabulary: WordPieceVocabulary,
    max_words_per_row: int,
) -> Column: ...
