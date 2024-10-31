# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from __future__ import annotations

import pylibcudf as plc

import cudf
from cudf._lib.nvtext.byte_pair_encode import (
    byte_pair_encoding as cpp_byte_pair_encoding,
)


class BytePairEncoder:
    """
    Given a merge pairs strings series, performs byte pair encoding on
    a strings series using the provided separator.

    Parameters
    ----------
    merges_pairs : str
        Strings column of merge pairs

    Returns
    -------
    BytePairEncoder
    """

    def __init__(self, merges_pair: "cudf.Series"):
        self.merge_pairs = plc.nvtext.byte_pair_encode.BPEMergePairs(
            merges_pair._column.to_pylibcudf(mode="read")
        )

    def __call__(self, text, separator: str = " ") -> cudf.Series:
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
        >>> mps = cudf.Series(["e n", "i t", "i s", "e s", "en t",
        ...                    "c e", "es t", "en ce", "T h", "Th is",
        ...                    "t est", "s ent", "t h", "th is"])
        >>> bpe = BytePairEncoder(mps)
        >>> str_series = cudf.Series(['This is the sentence', 'thisisit'])
        >>> bpe(str_series)
        0    This is a sent ence
        1             this is it
        dtype: object
        """
        sep = cudf.Scalar(separator, dtype="str")
        result = cpp_byte_pair_encoding(text._column, self.merge_pairs, sep)

        return cudf.Series._from_column(result)
