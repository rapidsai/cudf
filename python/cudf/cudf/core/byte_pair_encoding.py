# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pylibcudf as plc

from cudf.core.series import Series


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

    def __init__(self, merges_pair: Series) -> None:
        self.merge_pairs = plc.nvtext.byte_pair_encode.BPEMergePairs(
            merges_pair._column.to_pylibcudf(mode="read")
        )

    def __call__(self, text: Series, separator: str = " ") -> Series:
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
        return Series._from_column(
            text._column.byte_pair_encoding(self.merge_pairs, separator)
        )
