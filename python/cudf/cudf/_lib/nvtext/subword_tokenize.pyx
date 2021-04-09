# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.string cimport string
from libc.stdint cimport uint32_t
from libc.stdint cimport uintptr_t

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.subword_tokenize cimport (
    subword_tokenize as cpp_subword_tokenize,
    hashed_vocabulary as cpp_hashed_vocabulary,
    load_vocabulary_file as cpp_load_vocabulary_file,
    tokenizer_result as cpp_tokenizer_result,
    move as tr_move,
)
from cudf._lib.column cimport Column


cdef class Hashed_Vocabulary:
    cdef unique_ptr[cpp_hashed_vocabulary] c_obj

    def __cinit__(self, hash_file):
        cdef string c_hash_file = <string>str(hash_file).encode()
        with nogil:
            self.c_obj = move(cpp_load_vocabulary_file(c_hash_file))


## merge code with allready existing one
def subword_tokenize_inmem_hash(Column strings,
    Hashed_Vocabulary hashed_vocablary,
    uint32_t max_sequence_length=64,
    uint32_t stride=48,
    bool do_lower=True,
    bool do_truncate=False,
    uint32_t max_rows_tensor=500):
    
    cdef column_view c_strings = strings.view()
    cdef unique_ptr[cpp_hashed_vocabulary] c_vocab = move(hashed_vocablary.c_obj)
    cdef cpp_tokenizer_result c_result
    with nogil:
        c_result = tr_move(
            cpp_subword_tokenize(
                c_strings,
                c_vocab.get()[0],
                max_sequence_length,
                stride,
                do_lower,
                do_truncate,
                max_rows_tensor
            )
        )
    # return the 3 tensor components
    tokens = Column.from_unique_ptr(move(c_result.tensor_token_ids))
    masks = Column.from_unique_ptr(move(c_result.tensor_attention_mask))
    metadata = Column.from_unique_ptr(move(c_result.tensor_metadata))
    return tokens, masks, metadata


def subword_tokenize_vocab_file(
    Column strings,
    object   hash_file,
    uint32_t max_sequence_length=64,
    uint32_t stride=48,
    bool do_lower=True,
    bool do_truncate=False,
    uint32_t max_rows_tensor=500
):
    cdef column_view c_strings = strings.view()
    cdef cpp_tokenizer_result c_result
    cdef string c_hash_file = <string>str(hash_file).encode()
    with nogil:
        c_result = tr_move(
            cpp_subword_tokenize(
                c_strings,
                c_hash_file,
                max_sequence_length,
                stride,
                do_lower,
                do_truncate,
                max_rows_tensor
            )
        )
    # return the 3 tensor components
    tokens = Column.from_unique_ptr(move(c_result.tensor_token_ids))
    masks = Column.from_unique_ptr(move(c_result.tensor_attention_mask))
    metadata = Column.from_unique_ptr(move(c_result.tensor_metadata))
    return tokens, masks, metadata
