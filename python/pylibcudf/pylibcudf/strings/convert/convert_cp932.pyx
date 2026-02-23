# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings.convert cimport convert_cp932 as cpp_convert_cp932
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

__all__ = ["utf8_to_cp932"]

cpdef Column utf8_to_cp932(
    Column input, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Converts UTF-8 encoded strings to CP932 (Shift-JIS) encoding.

    For details, see :cpp:func:`cudf::strings::utf8_to_cp932`

    Parameters
    ----------
    input : Column
        Strings column containing UTF-8 encoded text.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    mr : DeviceMemoryResource | None
        Device memory resource used for allocations.

    Returns
    -------
    Column
        New column with CP932 encoded strings.

    Raises
    ------
    RuntimeError
        If any character cannot be represented in CP932 encoding.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_convert_cp932.utf8_to_cp932(
            input.view(),
            stream.view(),
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)
