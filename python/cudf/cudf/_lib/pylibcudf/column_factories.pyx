# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_factories cimport (
    make_duration_column as cpp_make_duration_column,
    make_empty_column as cpp_make_empty_column,
    make_fixed_point_column as cpp_make_fixed_point_column,
    make_fixed_width_column as cpp_make_fixed_width_column,
    make_numeric_column as cpp_make_numeric_column,
    make_timestamp_column as cpp_make_timestamp_column,
)
from cudf._lib.cpp.types cimport size_type, mask_state
from cudf._lib.cpp.types import mask_state as MaskState
from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer
from .gpumemoryview import gpumemoryview

from .types cimport DataType, Id as TypeId

cpdef Column make_empty_column(MakeEmptyColumnOperand type_or_id):
    cdef unique_ptr[column] result
    with nogil:
        result = move(
            cpp_make_empty_column(
                type_or_id.c_obj
            )
        )
    return Column.from_libcudf(move(result))


cpdef Column make_numeric_column(
    DataType type_,
    size_type size,
    MaskArg mstate
):

    cdef unique_ptr[column] result
    cdef mask_state state

    cdef DeviceBuffer mask_buf
    cdef device_buffer mask
    cdef size_type null_count

    if MaskArg is object:
        if isinstance(mstate, MaskState):
            state = mstate
            with nogil:
                result = move(
                    cpp_make_numeric_column(
                        type_.c_obj,
                        size,
                        state
                    )
                )
    elif MaskArg is tuple:
        mask_buf = mstate[0]
        mask = move(mask_buf.c_release())
        null_count = mstate[1]



        with nogil:
            result = move(
                cpp_make_numeric_column(
                    type_.c_obj,
                    size,
                    move(mask),
                    null_count
                )
            )
    else:
        raise TypeError("Invalid mask argument")
 

    return Column.from_libcudf(move(result))

