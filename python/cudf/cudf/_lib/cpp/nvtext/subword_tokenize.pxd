# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libc.stdint cimport uint32_t

from cudf._lib.cpp.column.column_view cimport column_view

cdef extern from "nvtext/subword_tokenize.hpp" namespace "nvtext" nogil:
    struct tokenizer_result:
        uint32_t nrows_tensor
        uint32_t* device_tensor_tokenIDS
        uint32_t* device_attention_mask
        uint32_t* device_tensor_metadata

    cdef unique_ptr[tokenizer_result] subword_tokenize(
        const column_view &strings,
        const string &filename_hashed_vocabulary,
        uint32_t max_sequence_length,
        uint32_t stride,
        bool do_lower,
        bool do_truncate,
        uint32_t max_num_sentences,
        uint32_t max_num_chars,
        uint32_t max_rows_tensor
    ) except +

