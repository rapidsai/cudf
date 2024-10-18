# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from pylibcudf import nvtext


@acquire_spill_lock()
def ngrams_tokenize(
    Column input,
    int ngrams,
    object py_delimiter,
    object py_separator
):
    return Column.from_pylibcudf(
        nvtext.ngrams_tokenize.ngrams_tokenize(
            input.to_pylibcudf(mode="read"),
            ngrams,
            py_delimiter.device_value.c_value,
            py_separator.device_value.c_value
        )
    )
