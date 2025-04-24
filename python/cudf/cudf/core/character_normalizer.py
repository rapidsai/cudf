# Copyright (c) 2025, NVIDIA CORPORATION.

from __future__ import annotations

import pylibcudf as plc

import cudf


class CharacterNormalizer:
    """
    A normalizer object used to normalize input text.

    Parameters
    ----------
    do_lower : bool
        If True, the normalizer should also lower-case
        while normalizing.
    special_tokens : cudf.Series
        Series of special tokens.
    """

    def __init__(
        self,
        do_lower: bool,
        special_tokens: cudf.Series = cudf.Series([], dtype="object"),
    ) -> None:
        self.normalizer = plc.nvtext.normalize.CharacterNormalizer(
            do_lower, special_tokens._column.to_pylibcudf(mode="read")
        )

    def normalize(self, text: cudf.Series) -> cudf.Series:
        """
        Parameters
        ----------
        text : cudf.Series
            The strings to be normalized.

        Returns
        -------
        cudf.Series
            Normalized strings
        """
        result = text._column.normalize_characters(self.normalizer)

        return cudf.Series._from_column(result)
