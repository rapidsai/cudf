# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_factories cimport (
    make_duration_column as cpp_make_duration_column,
    make_empty_column as cpp_make_empty_column,
    make_fixed_point_column as cpp_make_fixed_point_column,
    make_fixed_width_column as cpp_make_fixed_width_column,
    make_numeric_column as cpp_make_numeric_column,
    make_timestamp_column as cpp_make_timestamp_column,
)
from pylibcudf.libcudf.types cimport mask_state, size_type

from .types cimport DataType, type_id

from .types import MaskState, TypeId


__all__ = [
    "make_duration_column",
    "make_empty_column",
    "make_fixed_point_column",
    "make_fixed_width_column",
    "make_numeric_column",
    "make_timestamp_column",
]

cpdef Column make_empty_column(MakeEmptyColumnOperand type_or_id):
    """Creates an empty column of the specified type.

    For details, see :cpp:func::`make_empty_column`.

    Parameters
    ----------
    type_or_id : Union[DataType, type_id, object]
        The column data type.

    Returns
    -------
    Column
        An empty Column
    """
    cdef unique_ptr[column] result
    cdef type_id id

    if MakeEmptyColumnOperand is object:
        if isinstance(type_or_id, TypeId):
            id = type_or_id
            with nogil:
                result = cpp_make_empty_column(id)
        else:
            raise TypeError(
                "Must pass a TypeId or DataType"
            )
    elif MakeEmptyColumnOperand is DataType:
        with nogil:
            result = cpp_make_empty_column(type_or_id.c_obj)
    elif MakeEmptyColumnOperand is type_id:
        with nogil:
            result = cpp_make_empty_column(type_or_id)
    else:
        raise TypeError(
            "Must pass a TypeId or DataType"
        )
    return Column.from_libcudf(move(result))


cpdef Column make_numeric_column(
    DataType type_,
    size_type size,
    MaskArg mstate
):
    """Creates an empty numeric column.

    For details, see :cpp:func::`make_numeric_column`.

    """
    cdef unique_ptr[column] result
    cdef mask_state state

    if MaskArg is object:
        if isinstance(mstate, MaskState):
            state = mstate
        else:
            raise TypeError("Invalid mask argument")
    elif MaskArg is mask_state:
        state = mstate
    else:
        raise TypeError("Invalid mask argument")
    with nogil:
        result = cpp_make_numeric_column(
            type_.c_obj,
            size,
            state
        )

    return Column.from_libcudf(move(result))

cpdef Column make_fixed_point_column(
    DataType type_,
    size_type size,
    MaskArg mstate
):

    cdef unique_ptr[column] result
    cdef mask_state state

    if MaskArg is object:
        if isinstance(mstate, MaskState):
            state = mstate
        else:
            raise TypeError("Invalid mask argument")
    elif MaskArg is mask_state:
        state = mstate
    else:
        raise TypeError("Invalid mask argument")
    with nogil:
        result = cpp_make_fixed_point_column(
            type_.c_obj,
            size,
            state
        )

    return Column.from_libcudf(move(result))


cpdef Column make_timestamp_column(
    DataType type_,
    size_type size,
    MaskArg mstate
):

    cdef unique_ptr[column] result
    cdef mask_state state

    if MaskArg is object:
        if isinstance(mstate, MaskState):
            state = mstate
        else:
            raise TypeError("Invalid mask argument")
    elif MaskArg is mask_state:
        state = mstate
    else:
        raise TypeError("Invalid mask argument")
    with nogil:
        result = cpp_make_timestamp_column(
            type_.c_obj,
            size,
            state
        )

    return Column.from_libcudf(move(result))


cpdef Column make_duration_column(
    DataType type_,
    size_type size,
    MaskArg mstate
):

    cdef unique_ptr[column] result
    cdef mask_state state

    if MaskArg is object:
        if isinstance(mstate, MaskState):
            state = mstate
        else:
            raise TypeError("Invalid mask argument")
    elif MaskArg is mask_state:
        state = mstate
    else:
        raise TypeError("Invalid mask argument")
    with nogil:
        result = cpp_make_duration_column(
            type_.c_obj,
            size,
            state
        )

    return Column.from_libcudf(move(result))


cpdef Column make_fixed_width_column(
    DataType type_,
    size_type size,
    MaskArg mstate
):

    cdef unique_ptr[column] result
    cdef mask_state state

    if MaskArg is object:
        if isinstance(mstate, MaskState):
            state = mstate
        else:
            raise TypeError("Invalid mask argument")
    elif MaskArg is mask_state:
        state = mstate
    else:
        raise TypeError("Invalid mask argument")
    with nogil:
        result = cpp_make_fixed_width_column(
            type_.c_obj,
            size,
            state
        )

    return Column.from_libcudf(move(result))
