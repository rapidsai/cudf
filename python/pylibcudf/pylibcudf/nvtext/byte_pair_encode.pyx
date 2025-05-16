# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.byte_pair_encode cimport (
    byte_pair_encoding as cpp_byte_pair_encoding,
    load_merge_pairs as cpp_load_merge_pairs,
)
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.scalar cimport Scalar

__all__ = ["BPEMergePairs", "byte_pair_encoding"]

cdef class BPEMergePairs:
    """The table of merge pairs for the BPE encoder.

    For details, see :cpp:class:`cudf::nvtext::bpe_merge_pairs`.
    """
    def __cinit__(self, Column merge_pairs):
        cdef column_view c_pairs = merge_pairs.view()
        with nogil:
            self.c_obj = move(cpp_load_merge_pairs(c_pairs))

    __hash__ = None

cpdef Column byte_pair_encoding(
    Column input,
    BPEMergePairs merge_pairs,
    Scalar separator=None
):
    """
    Byte pair encode the input strings.

    For details, see cpp:func:`cudf::nvtext::byte_pair_encoding`

    Parameters
    ----------
    input : Column
        Strings to encode.
    merge_pairs : BPEMergePairs
       Substrings to rebuild each string on.
    separator : Scalar
        String used to build the output after encoding. Default is a space.

    Returns
    -------
    Column
        An encoded column of strings.
    """
    cdef unique_ptr[column] c_result

    if separator is None:
        separator = Scalar.from_libcudf(
            cpp_make_string_scalar(" ".encode())
        )

    with nogil:
        c_result = move(
            cpp_byte_pair_encoding(
                input.view(),
                dereference(merge_pairs.c_obj.get()),
                dereference(<const string_scalar*>separator.c_obj.get()),
            )
        )

    return Column.from_libcudf(move(c_result))
