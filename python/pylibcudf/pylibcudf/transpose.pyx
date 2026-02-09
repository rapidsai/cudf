# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from pylibcudf.libcudf cimport transpose as cpp_transpose
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table_view cimport table_view

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource

__all__ = ["transpose"]

cpdef Table transpose(
    Table input_table, Stream stream=None, DeviceMemoryResource mr=None
):
    """Transpose a Table.

    For details, see :cpp:func:`transpose`.

    Parameters
    ----------
    input_table : Table
        Table to transpose
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    Table
        Transposed table.
    """
    cdef pair[unique_ptr[column], table_view] c_result
    cdef Table owner_table
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_transpose.transpose(
            input_table.view(), stream.view(), mr.get_mr()
        )

    owner_table = Table(
        [Column.from_libcudf(move(c_result.first), stream, mr)] *
        c_result.second.num_columns()
    )

    return Table.from_table_view(c_result.second, owner_table)
