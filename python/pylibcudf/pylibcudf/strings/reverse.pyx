# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport reverse as cpp_reverse
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

__all__ = ["reverse"]

cpdef Column reverse(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """Reverses the characters within each string.

    Any null string entries return corresponding null output column entries.

    For details, see :cpp:func:`reverse`.

    Parameters
    ----------
    input : Column
        Strings column for this operation
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        New strings column
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = cpp_reverse.reverse(input.view(), stream.view(), mr.get_mr())

    return Column.from_libcudf(move(c_result), stream, mr)
