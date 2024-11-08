# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from __future__ import annotations

import pylibcudf as plc

import cudf
from cudf._lib.nvtext.tokenize import (
    tokenize_with_vocabulary as cpp_tokenize_with_vocabulary,
)


class TokenizeVocabulary:
    """
    A vocabulary object used to tokenize input text.

    Parameters
    ----------
    vocabulary : str
        Strings column of vocabulary terms
    """

    def __init__(self, vocabulary: "cudf.Series"):
        self.vocabulary = plc.nvtext.tokenize.TokenizeVocabulary(
            vocabulary._column.to_pylibcudf(mode="read")
        )

    def tokenize(
        self, text, delimiter: str = "", default_id: int = -1
    ) -> cudf.Series:
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

        return cudf.Series._from_column(result)
