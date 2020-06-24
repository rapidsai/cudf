# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libc.stdint cimport uint32_t

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view


cdef extern from "nvtext/subword_tokenize.hpp" namespace "nvtext" nogil:
    cdef cppclass tokenizer_result "nvtext::tokenizer_result":
        uint32_t nrows_tensor
        uint32_t sequence_length
        unique_ptr[column] tensor_token_ids
        unique_ptr[column] tensor_attention_mask
        unique_ptr[column] tensor_metadata

    cdef tokenizer_result subword_tokenize(
        const column_view &strings,
        const string &filename_hashed_vocabulary,
        uint32_t max_sequence_length,
        uint32_t stride,
        bool do_lower,
        bool do_truncate,
        uint32_t max_num_stings,
        uint32_t max_num_chars,
        uint32_t max_rows_tensor
    ) except +

cdef extern from "<utility>" namespace "std" nogil:
    cdef tokenizer_result move(tokenizer_result)
