# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libc.stdint cimport uint32_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.subword_tokenize cimport (
    load_vocabulary_file as cpp_load_vocabulary_file,
    move as tr_move,
    subword_tokenize as cpp_subword_tokenize,
    tokenizer_result as cpp_tokenizer_result,
)

__all__ = ["HashedVocabulary", "subword_tokenize"]

cdef class HashedVocabulary:
    """The vocabulary data for use with the subword_tokenize function.

    For details, see :cpp:class:`cudf::nvtext::hashed_vocabulary`.
    """
    def __cinit__(self, hash_file):
        cdef string c_hash_file = <string>str(hash_file).encode()
        with nogil:
            self.c_obj = move(cpp_load_vocabulary_file(c_hash_file))

    __hash__ = None

cpdef tuple[Column, Column, Column] subword_tokenize(
    Column input,
    HashedVocabulary vocabulary_table,
    uint32_t max_sequence_length,
    uint32_t stride,
    bool do_lower_case,
    bool do_truncate,
):
    """
    Creates a tokenizer that cleans the text, splits it into
    tokens and returns token-ids from an input vocabulary.

    For details, see cpp:func:`subword_tokenize`

    Parameters
    ----------
    input : Column
        The input strings to tokenize.
    vocabulary_table : HashedVocabulary
        The vocabulary table pre-loaded into this object.
    max_sequence_length : uint32_t
        Limit of the number of token-ids per row in final tensor for each string.
    stride : uint32_t
        Each row in the output token-ids will replicate
        ``max_sequence_length`` - ``stride`` the token-ids
        from the previous row, unless it is the first string.
    do_lower_case : bool
        If true, the tokenizer will convert uppercase characters in the
        input stream to lower-case and strip accents from those characters.
        If false, accented and uppercase characters are not transformed.
    do_truncate : bool
        If true, the tokenizer will discard all the token-ids after
        ``max_sequence_length`` for each input string. If false, it
        will use a new row in the output token-ids to continue
        generating the output.

    Returns
    -------
    tuple[Column, Column, Column]
        A tuple of three columns containing the
        tokens, masks, and metadata.
    """
    cdef cpp_tokenizer_result c_result
    with nogil:
        c_result = tr_move(
            cpp_subword_tokenize(
                input.view(),
                dereference(vocabulary_table.c_obj.get()),
                max_sequence_length,
                stride,
                do_lower_case,
                do_truncate,
            )
        )
    cdef Column tokens = Column.from_libcudf(move(c_result.tensor_token_ids))
    cdef Column masks = Column.from_libcudf(move(c_result.tensor_attention_mask))
    cdef Column metadata = Column.from_libcudf(move(c_result.tensor_metadata))
    return tokens, masks, metadata
