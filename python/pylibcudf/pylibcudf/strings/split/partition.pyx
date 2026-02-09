# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings.split cimport partition as cpp_partition
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.scalar cimport Scalar
from pylibcudf.table cimport Table
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from cython.operator import dereference

__all__ = ["partition", "rpartition"]

cpdef Table partition(
    Column input,
    Scalar delimiter=None,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a set of 3 columns by splitting each string using the
    specified delimiter.

    For details, see :cpp:func:`partition`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation

    delimiter : Scalar
        UTF-8 encoded string indicating where to split each string.

    Returns
    -------
    Table
        New table of strings columns
    """
    cdef unique_ptr[table] c_result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    if delimiter is None:
        delimiter = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), stream.view(), mr.get_mr())
        )

    cdef const string_scalar* c_delimiter = <const string_scalar*>(
        delimiter.c_obj.get()
    )

    with nogil:
        c_result = cpp_partition.partition(
            input.view(),
            dereference(c_delimiter),
            stream.view(),
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), stream, mr)

cpdef Table rpartition(
    Column input,
    Scalar delimiter=None,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a set of 3 columns by splitting each string using the
    specified delimiter starting from the end of each string.

    For details, see :cpp:func:`rpartition`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation

    delimiter : Scalar
        UTF-8 encoded string indicating where to split each string.

    Returns
    -------
    Table
       New strings columns
    """
    cdef unique_ptr[table] c_result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    if delimiter is None:
        delimiter = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), stream.view(), mr.get_mr())
        )

    cdef const string_scalar* c_delimiter = <const string_scalar*>(
        delimiter.c_obj.get()
    )

    with nogil:
        c_result = cpp_partition.rpartition(
            input.view(),
            dereference(c_delimiter),
            stream.view(),
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), stream, mr)
