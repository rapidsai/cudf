# Copyright (c) 2023, NVIDIA CORPORATION.

from __future__ import annotations

import cudf
from cudf._lib.nvtext.bpe_tokenize import (
    BPE_Merge_Pairs as cpp_merge_pairs,
    byte_pair_encoding as cpp_byte_pair_encoding,
)


class BytePairEncoder:
    """

    Parameters
    ----------
    merges_file : str
        Path to file containing merge pairs.

    Returns
    -------
    BytePairEncoder
    """

    def __init__(self, merges_file: str):
        self.merge_pairs = cpp_merge_pairs(merges_file)

    def __call__(self, text):
        """

        Parameters
        ----------
        text : cudf string series
            The strings to be encoded.

        Returns
        -------
        Encoded strings

        Examples
        --------
        >>> import cudf
        >>> from cudf.core.byte_pair_encoding import BytePairEncoder
        >>> bpe = BytePairEncoder('merges.txt')
        >>> str_series = cudf.Series(['This is the sentence', 'thisisit'])
        >>> bpe(str_series)
        0    This is a sent ence
        1             this is it
        dtype: object
        """

        result = cpp_byte_pair_encoding(text._column, self.merge_pairs)

        return cudf.Series(result)
