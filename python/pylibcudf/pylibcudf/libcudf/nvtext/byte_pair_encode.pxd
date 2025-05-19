# Copyright (c) 2023-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar


cdef extern from "nvtext/byte_pair_encoding.hpp" namespace "nvtext" nogil:

    cdef struct bpe_merge_pairs "nvtext::bpe_merge_pairs":
        pass

    cdef unique_ptr[bpe_merge_pairs] load_merge_pairs(
        const column_view &merge_pairs
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] byte_pair_encoding(
        const column_view &strings,
        const bpe_merge_pairs &merge_pairs,
        const string_scalar &separator
    ) except +libcudf_exception_handler
