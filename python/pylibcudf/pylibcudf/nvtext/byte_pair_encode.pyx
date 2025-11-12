# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

__all__ = ["BPEMergePairs", "byte_pair_encoding"]

cdef class BPEMergePairs:
    """The table of merge pairs for the BPE encoder.

    For details, see :cpp:class:`cudf::nvtext::bpe_merge_pairs`.
    """
    def __cinit__(
        self,
        Column merge_pairs,
        Stream stream=None,
        DeviceMemoryResource mr=None
    ):
        cdef column_view c_pairs = merge_pairs.view()
        stream = _get_stream(stream)
        mr = _get_memory_resource(mr)
        with nogil:
            self.c_obj = move(cpp_load_merge_pairs(c_pairs, stream.view(), mr.get_mr()))

    __hash__ = None

cpdef Column byte_pair_encoding(
    Column input,
    BPEMergePairs merge_pairs,
    Scalar separator=None,
    Stream stream=None,
    DeviceMemoryResource mr=None,
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
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        An encoded column of strings.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    if separator is None:
        separator = Scalar.from_libcudf(
            cpp_make_string_scalar(" ".encode(), stream.view(), mr.get_mr())
        )

    with nogil:
        c_result = move(
            cpp_byte_pair_encoding(
                input.view(),
                dereference(merge_pairs.c_obj.get()),
                dereference(<const string_scalar*>separator.c_obj.get()),
                stream.view(),
                mr.get_mr()
            )
        )

    return Column.from_libcudf(move(c_result), stream, mr)
