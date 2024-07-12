# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move, pair

from rmm._lib.device_buffer cimport DeviceBuffer, device_buffer

from cudf._lib.pylibcudf.libcudf cimport transform as cpp_transform
from cudf._lib.pylibcudf.libcudf.types cimport size_type

from .column cimport Column
from .gpumemoryview cimport gpumemoryview


cpdef tuple[gpumemoryview, int] nans_to_nulls(Column input):
    """Create a null mask preserving existing nulls and converting nans to null.

    Parameters
    ----------
    input : Column
        Column to produce new mask from.

    Returns
    -------
    Two-tuple of a gpumemoryview wrapping the null mask and the new null count.
    """
    cdef pair[unique_ptr[device_buffer], size_type] c_result

    with nogil:
        c_result = move(cpp_transform.nans_to_nulls(input.view()))

    return (
        gpumemoryview(DeviceBuffer.c_from_unique_ptr(move(c_result.first))),
        c_result.second
    )
