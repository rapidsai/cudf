# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.stemmer cimport letter_type
from pylibcudf.libcudf.types cimport size_type

ctypedef fused ColumnOrSize:
    Column
    size_type

cpdef Column is_letter(Column input, bool check_vowels, ColumnOrSize indices)

cpdef Column porter_stemmer_measure(Column input)
