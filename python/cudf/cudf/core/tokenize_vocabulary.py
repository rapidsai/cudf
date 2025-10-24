# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pylibcudf as plc

from cudf.core.series import Series


class TokenizeVocabulary:
    """
    A vocabulary object used to tokenize input text.

    Parameters
    ----------
    vocabulary : str
        Strings column of vocabulary terms
    """

    def __init__(self, vocabulary: Series) -> None:
        self.vocabulary = plc.nvtext.tokenize.TokenizeVocabulary(
            vocabulary._column.to_pylibcudf(mode="read")
        )

    def tokenize(
        self, text: Series, delimiter: str = "", default_id: int = -1
    ) -> Series:
        """
        Parameters
        ----------
        text : cudf string series
            The strings to be tokenized.
        delimiter : str
            Delimiter to identify tokens. Default is whitespace.
        default_id : int
            Value to use for tokens not found in the vocabulary.
            Default is -1.

        Returns
        -------
        Tokenized strings
        """
        if delimiter is None:
            delimiter = ""
        result = text._column.tokenize_with_vocabulary(
            self.vocabulary, delimiter, default_id
        )

        return Series._from_column(result)
