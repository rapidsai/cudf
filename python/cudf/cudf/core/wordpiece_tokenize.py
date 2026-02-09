# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pylibcudf as plc

from cudf.core.series import Series


class WordPieceVocabulary:
    """
    A vocabulary object used to tokenize input text.

    Parameters
    ----------
    vocabulary : cudf.Series
        Strings column of vocabulary terms
    """

    def __init__(self, vocabulary: Series) -> None:
        self.vocabulary = plc.nvtext.wordpiece_tokenize.WordPieceVocabulary(
            vocabulary._column.plc_column
        )

    def tokenize(self, text: Series, max_words_per_row: int = 0) -> Series:
        """
        Produces tokens for the input strings.
        The input is expected to be the output of NormalizeCharacters or a
        similar normalizer.

        Parameters
        ----------
        text : cudf.Series
            Normalized strings to be tokenized.
        max_words_per_row : int
            Maximum number of words to tokenize per row.
            Default 0 tokenizes all words.

        Returns
        -------
        cudf.Series
            Token values
        """
        result = text._column.wordpiece_tokenize(
            self.vocabulary, max_words_per_row
        )

        return Series._from_column(result)
