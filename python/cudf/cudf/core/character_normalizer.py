# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pylibcudf as plc

from cudf.core.series import Series


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
        These are expected to be all upper case and
        include the bracket ``[]`` characters.
    """

    def __init__(
        self,
        do_lower: bool,
        special_tokens: Series | None = None,
    ) -> None:
        if special_tokens is None:
            special_tokens = Series([], dtype="object")
        self.normalizer = plc.nvtext.normalize.CharacterNormalizer(
            do_lower, special_tokens._column.plc_column
        )

    def normalize(self, text: Series) -> Series:
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

        return Series._from_column(result)
