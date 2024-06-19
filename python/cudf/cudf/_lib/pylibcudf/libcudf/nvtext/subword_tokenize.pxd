# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint16_t, uint32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view


cdef extern from "nvtext/subword_tokenize.hpp" namespace "nvtext" nogil:
    cdef cppclass tokenizer_result "nvtext::tokenizer_result":
        uint32_t nrows_tensor
        uint32_t sequence_length
        unique_ptr[column] tensor_token_ids
        unique_ptr[column] tensor_attention_mask
        unique_ptr[column] tensor_metadata

    cdef struct hashed_vocabulary "nvtext::hashed_vocabulary":
        uint16_t first_token_id
        uint16_t separator_token_id
        uint16_t unknown_token_id
        uint32_t outer_hash_a
        uint32_t outer_hash_b
        uint16_t num_bin
        unique_ptr[column] table
        unique_ptr[column] bin_coefficients
        unique_ptr[column] bin_offsets

    cdef unique_ptr[hashed_vocabulary] load_vocabulary_file(
        const string &filename_hashed_vocabulary
    ) except +

    cdef tokenizer_result subword_tokenize(
        const column_view & strings,
        hashed_vocabulary & hashed_vocablary_obj,
        uint32_t max_sequence_length,
        uint32_t stride,
        bool do_lower,
        bool do_truncate
    ) except +

    cdef tokenizer_result subword_tokenize(
        const column_view &strings,
        const string &filename_hashed_vocabulary,
        uint32_t max_sequence_length,
        uint32_t stride,
        bool do_lower,
        bool do_truncate
    ) except +

cdef extern from "<utility>" namespace "std" nogil:
    cdef tokenizer_result move(tokenizer_result)
