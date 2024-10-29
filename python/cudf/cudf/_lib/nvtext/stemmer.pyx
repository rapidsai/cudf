# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from enum import IntEnum

from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.nvtext.stemmer cimport (
    letter_type,
    underlying_type_t_letter_type,
)
from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column

from pylibcudf import nvtext


class LetterType(IntEnum):
    CONSONANT = <underlying_type_t_letter_type> letter_type.CONSONANT
    VOWEL = <underlying_type_t_letter_type> letter_type.VOWEL


@acquire_spill_lock()
def porter_stemmer_measure(Column strings):
    return Column.from_pylibcudf(
        nvtext.stemmer.porter_stemmer_measure(
            strings.to_pylibcudf(mode="read"),
        )
    )


@acquire_spill_lock()
def is_letter(Column strings,
              object ltype,
              size_type index):
    return Column.from_pylibcudf(
        nvtext.stemmer.is_letter(
            strings.to_pylibcudf(mode="read"),
            ltype==LetterType.VOWEL,
            index,
        )
    )


@acquire_spill_lock()
def is_letter_multi(Column strings,
                    object ltype,
                    Column indices):
    return Column.from_pylibcudf(
        nvtext.stemmer.is_letter(
            strings.to_pylibcudf(mode="read"),
            ltype==LetterType.VOWEL,
            indices.to_pylibcudf(mode="read"),
        )
    )
