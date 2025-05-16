# Copyright (c) 2024, NVIDIA CORPORATION.
from pylibcudf.column cimport Column
from pylibcudf.libcudf.strings.translate cimport filter_type
from pylibcudf.scalar cimport Scalar


cpdef Column translate(Column input, dict chars_table)

cpdef Column filter_characters(
    Column input,
    dict characters_to_filter,
    filter_type keep_characters,
    Scalar replacement
)
