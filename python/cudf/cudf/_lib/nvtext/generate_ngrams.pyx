# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from pylibcudf import nvtext


@acquire_spill_lock()
def generate_ngrams(Column strings, int ngrams, object py_separator):
    result = nvtext.generate_ngrams.generate_ngrams(
        strings.to_pylibcudf(mode="read"),
        ngrams,
        py_separator.device_value.c_value
    )
    return Column.from_pylibcudf(result)


@acquire_spill_lock()
def generate_character_ngrams(Column strings, int ngrams):
    result = nvtext.generate_ngrams.generate_character_ngrams(
        strings.to_pylibcudf(mode="read"),
        ngrams
    )
    return Column.from_pylibcudf(result)


@acquire_spill_lock()
def hash_character_ngrams(Column strings, int ngrams):
    result = nvtext.generate_ngrams.hash_character_ngrams(
        strings.to_pylibcudf(mode="read"),
        ngrams
    )
    return Column.from_pylibcudf(result)
