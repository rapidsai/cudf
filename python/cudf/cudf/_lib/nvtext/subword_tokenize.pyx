# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libc.stdint cimport uint32_t
from libc.stdint cimport uintptr_t

from cudf._lib.move cimport move
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.subword_tokenize cimport (
    subword_tokenize as cpp_subword_tokenize,
    tokenizer_result as cpp_tokenizer_result,
    move as tr_move
)
from cudf._lib.column cimport Column


def subword_tokenize(
    Column strings,
    hash_file,
    max_sequence_length=64,
    stride=48,
    do_lower=True,
    do_truncate=False,
    max_num_strings=100,
    max_num_chars=100000,
    max_rows_tensor=500
):
    cdef column_view c_strings = strings.view()
    cdef string c_hash_file = hash_file.encode()
    cdef uint32_t c_max_sequence_length = max_sequence_length
    cdef uint32_t c_stride = stride
    cdef bool c_do_lower = do_lower
    cdef bool c_do_truncate = do_truncate
    cdef uint32_t c_max_num_strings = max_num_strings
    cdef uint32_t c_max_num_chars = max_num_chars
    cdef uint32_t c_max_rows_tensor = max_rows_tensor
    cdef cpp_tokenizer_result c_result

    with nogil:
        c_result = tr_move(
            cpp_subword_tokenize(
                c_strings,
                c_hash_file,
                c_max_sequence_length,
                c_stride,
                c_do_lower,
                c_do_truncate,
                c_max_num_strings,
                c_max_num_chars,
                c_max_rows_tensor
            )
        )

    # return 3 tensor components
    return [Column.from_unique_ptr(move(c_result.tensor_token_ids)),
            Column.from_unique_ptr(move(c_result.tensor_attention_mask)),
            Column.from_unique_ptr(move(c_result.tensor_metadata))]
