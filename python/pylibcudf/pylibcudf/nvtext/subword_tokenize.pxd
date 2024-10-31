# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.subword_tokenize cimport hashed_vocabulary


cdef class HashedVocabulary:
    cdef unique_ptr[hashed_vocabulary] c_obj

cpdef tuple[Column, Column, Column] subword_tokenize(
    Column input,
    HashedVocabulary vocabulary_table,
    uint32_t max_sequence_length,
    uint32_t stride,
    bool do_lower_case,
    bool do_truncate,
)
