# Copyright (c) 2025, NVIDIA CORPORATION.

from __future__ import annotations

import pylibcudf as plc

import cudf


class WordPieceVocabulary:
    """
    A vocabulary object used to tokenize input text.

    Parameters
    ----------
    vocabulary : str
        Strings column of vocabulary terms
    """

    def __init__(self, vocabulary: cudf.Series) -> None:
        self.vocabulary = plc.nvtext.wordpiece_tokenize.WordPieceVocabulary(
            vocabulary._column.to_pylibcudf(mode="read")
        )

    def tokenize(self, text, max_words_per_row: int) -> cudf.Series:
        """
        Parameters
        ----------
        text : cudf string series
            The strings to be tokenized.
        max_words_per_row : int
            Maximum number of words to tokenize per row

        Returns
        -------
        Token values
        """
        result = text._column.wordpiece_tokenize(
            self.vocabulary, max_words_per_row
        )

        return cudf.Series._from_column(result)
