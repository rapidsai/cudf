# Copyright (c) 2023, NVIDIA CORPORATION.

from __future__ import annotations

import cudf
from cudf._lib.nvtext.tokenize import (
    Tokenize_Vocabulary as cpp_tokenize_vocabulary,
    tokenize_with_vocabulary as cpp_tokenize_with_vocabulary,
)


class TokenizeVocabulary:
    """
    Parameters
    ----------
    vocabulary : str
        Strings column of vocabulary terms
    Returns
    -------
    TokenizeVocabulary
    """

    def __init__(self, vocabulary: "cudf.Series"):
        self.vocabulary = cpp_tokenize_vocabulary(vocabulary._column)

    def __call__(self, text, delimiter: str = "", default_id: int = -1):
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
        delim = cudf.Scalar(delimiter, dtype="str")
        result = cpp_tokenize_with_vocabulary(
            text._column, self.vocabulary, delim, default_id
        )

        return cudf.Series(result)
