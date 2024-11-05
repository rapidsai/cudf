# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.tokenize cimport tokenize_vocabulary
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar

cdef class TokenizeVocabulary:
    cdef unique_ptr[tokenize_vocabulary] c_obj

cpdef Column tokenize_scalar(Column input, Scalar delimiter=*)

cpdef Column tokenize_column(Column input, Column delimiters)

cpdef Column count_tokens_scalar(Column input, Scalar delimiter=*)

cpdef Column count_tokens_column(Column input, Column delimiters)

cpdef Column character_tokenize(Column input)

cpdef Column detokenize(Column input, Column row_indices, Scalar separator=*)

cpdef Column tokenize_with_vocabulary(
    Column input,
    TokenizeVocabulary vocabulary,
    Scalar delimiter,
    size_type default_id=*
)
