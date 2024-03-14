# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cpython cimport pycapsule
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pyarrow.lib cimport CTable, pyarrow_unwrap_table, pyarrow_wrap_table

from cudf._lib.cpp.interop cimport (
    DLManagedTensor,
    column_metadata,
    from_arrow as cpp_from_arrow,
    from_dlpack as cpp_from_dlpack,
    to_arrow as cpp_to_arrow,
    to_dlpack as cpp_to_dlpack,
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.utils cimport columns_from_unique_ptr, table_view_from_columns

from cudf.core.buffer import acquire_spill_lock
from cudf.core.dtypes import ListDtype, StructDtype


def from_dlpack(dlpack_capsule):
    """
    Converts a DLPack Tensor PyCapsule into a list of columns.

    DLPack Tensor PyCapsule is expected to have the name "dltensor".
    """
    cdef DLManagedTensor* dlpack_tensor = <DLManagedTensor*>pycapsule.\
        PyCapsule_GetPointer(dlpack_capsule, 'dltensor')
    pycapsule.PyCapsule_SetName(dlpack_capsule, 'used_dltensor')

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_from_dlpack(dlpack_tensor)
        )

    res = columns_from_unique_ptr(move(c_result))
    dlpack_tensor.deleter(dlpack_tensor)
    return res


def to_dlpack(list source_columns):
    """
    Converts a list of columns into a DLPack Tensor PyCapsule.

    DLPack Tensor PyCapsule will have the name "dltensor".
    """
    if any(column.null_count for column in source_columns):
        raise ValueError(
            "Cannot create a DLPack tensor with null values. \
                Input is required to have null count as zero."
        )

    cdef DLManagedTensor *dlpack_tensor
    cdef table_view source_table_view = table_view_from_columns(source_columns)

    with nogil:
        dlpack_tensor = cpp_to_dlpack(
            source_table_view
        )

    return pycapsule.PyCapsule_New(
        dlpack_tensor,
        'dltensor',
        dlmanaged_tensor_pycapsule_deleter
    )


cdef void dlmanaged_tensor_pycapsule_deleter(object pycap_obj) noexcept:
    cdef DLManagedTensor* dlpack_tensor = <DLManagedTensor*>0
    try:
        dlpack_tensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(
            pycap_obj, 'used_dltensor')
        return  # we do not call a used capsule's deleter
    except Exception:
        dlpack_tensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(
            pycap_obj, 'dltensor')
    dlpack_tensor.deleter(dlpack_tensor)


cdef vector[column_metadata] gather_metadata(object cols_dtypes) except *:
    """
    Generates a column_metadata vector for each column.

    Parameters
    ----------
    cols_dtypes : iterable
        An iterable of ``(column_name, dtype)`` pairs.
    """
    cdef vector[column_metadata] cpp_metadata
    cpp_metadata.reserve(len(cols_dtypes))

    if cols_dtypes is not None:
        for idx, (col_name, col_dtype) in enumerate(cols_dtypes):
            cpp_metadata.push_back(column_metadata(col_name.encode()))
            if isinstance(col_dtype, (ListDtype, StructDtype)):
                _set_col_children_metadata(col_dtype, cpp_metadata[idx])
    else:
        raise TypeError(
            "An iterable of (column_name, dtype) pairs is required to "
            "construct column_metadata"
        )
    return cpp_metadata


cdef _set_col_children_metadata(dtype,
                                column_metadata& col_meta):

    cdef column_metadata element_metadata

    if isinstance(dtype, StructDtype):
        for name, value in dtype.fields.items():
            element_metadata = column_metadata(name.encode())
            _set_col_children_metadata(
                value, element_metadata
            )
            col_meta.children_meta.push_back(element_metadata)
    elif isinstance(dtype, ListDtype):
        col_meta.children_meta.reserve(2)
        # Offsets - child 0
        col_meta.children_meta.push_back(column_metadata())

        # Element column - child 1
        element_metadata = column_metadata()
        _set_col_children_metadata(
            dtype.element_type, element_metadata
        )
        col_meta.children_meta.push_back(element_metadata)
    else:
        col_meta.children_meta.push_back(column_metadata())


@acquire_spill_lock()
def to_arrow(list source_columns, object column_dtypes):
    """Convert a list of columns from
    cudf Frame to a PyArrow Table.

    Parameters
    ----------
    source_columns : a list of columns to convert
    column_dtypes : Iterable of ``(column_name, column_dtype)`` pairs

    Returns
    -------
    pyarrow table
    """
    cdef vector[column_metadata] cpp_metadata = gather_metadata(column_dtypes)
    cdef table_view input_table_view = table_view_from_columns(source_columns)

    cdef shared_ptr[CTable] cpp_arrow_table
    with nogil:
        cpp_arrow_table = cpp_to_arrow(
            input_table_view, cpp_metadata
        )

    return pyarrow_wrap_table(cpp_arrow_table)


@acquire_spill_lock()
def from_arrow(object input_table):
    """Convert from PyArrow Table to a list of columns.

    Parameters
    ----------
    input_table : PyArrow table

    Returns
    -------
    A list of columns to construct Frame object
    """
    cdef shared_ptr[CTable] cpp_arrow_table = (
        pyarrow_unwrap_table(input_table)
    )
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_from_arrow(cpp_arrow_table.get()[0]))

    return columns_from_unique_ptr(move(c_result))
