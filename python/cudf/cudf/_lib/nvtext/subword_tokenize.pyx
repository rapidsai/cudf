# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t, uintptr_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.nvtext.subword_tokenize cimport (
    hashed_vocabulary as cpp_hashed_vocabulary,
    load_vocabulary_file as cpp_load_vocabulary_file,
    move as tr_move,
    subword_tokenize as cpp_subword_tokenize,
    tokenizer_result as cpp_tokenizer_result,
)


cdef class Hashed_Vocabulary:
    cdef unique_ptr[cpp_hashed_vocabulary] c_obj

    def __cinit__(self, hash_file):
        cdef string c_hash_file = <string>str(hash_file).encode()
        with nogil:
            self.c_obj = move(cpp_load_vocabulary_file(c_hash_file))


def subword_tokenize_inmem_hash(
    Column strings,
    Hashed_Vocabulary hashed_vocabulary,
    uint32_t max_sequence_length=64,
    uint32_t stride=48,
    bool do_lower=True,
    bool do_truncate=False,
    uint32_t max_rows_tensor=500
):
    """
    Subword tokenizes text series by using the pre-loaded hashed vocabulary
    """
    cdef column_view c_strings = strings.view()
    cdef cpp_tokenizer_result c_result
    with nogil:
        c_result = tr_move(
            cpp_subword_tokenize(
                c_strings,
                hashed_vocabulary.c_obj.get()[0],
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
    """
        Subword tokenizes text series by using the hashed vocabulary
        stored on disk
    """
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
