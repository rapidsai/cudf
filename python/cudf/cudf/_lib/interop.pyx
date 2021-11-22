# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf

from cpython cimport pycapsule
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
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
from cudf._lib.utils cimport data_from_unique_ptr, table_view_from_table


def from_dlpack(dlpack_capsule):
    """
    Converts a DLPack Tensor PyCapsule into a cudf Frame object.

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

    res = data_from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )
    dlpack_tensor.deleter(dlpack_tensor)
    return res


def to_dlpack(source_table):
    """
    Converts a cudf Frame into a DLPack Tensor PyCapsule.

    DLPack Tensor PyCapsule will have the name "dltensor".
    """
    for column in source_table._columns:
        if column.null_count:
            raise ValueError(
                "Cannot create a DLPack tensor with null values. \
                    Input is required to have null count as zero."
            )

    cdef DLManagedTensor *dlpack_tensor
    cdef table_view source_table_view = table_view_from_table(
        source_table, ignore_index=True
    )

    with nogil:
        dlpack_tensor = cpp_to_dlpack(
            source_table_view
        )

    return pycapsule.PyCapsule_New(
        dlpack_tensor,
        'dltensor',
        dlmanaged_tensor_pycapsule_deleter
    )


cdef void dlmanaged_tensor_pycapsule_deleter(object pycap_obj):
    cdef DLManagedTensor* dlpack_tensor = <DLManagedTensor*>0
    try:
        dlpack_tensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(
            pycap_obj, 'used_dltensor')
        return  # we do not call a used capsule's deleter
    except Exception:
        dlpack_tensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(
            pycap_obj, 'dltensor')
    dlpack_tensor.deleter(dlpack_tensor)


cdef vector[column_metadata] gather_metadata(object metadata) except *:
    """
    Metadata is stored as lists, and expected format is as follows,
    [["a", [["b"], ["c"], ["d"]]],       [["e"]],        ["f", ["", ""]]].
    First value signifies name of the main parent column,
    and adjacent list will signify child column.
    """
    cdef vector[column_metadata] cpp_metadata
    if isinstance(metadata, list):
        cpp_metadata.reserve(len(metadata))
        for i, val in enumerate(metadata):
            cpp_metadata.push_back(column_metadata(str.encode(str(val[0]))))
            if len(val) == 2:
                cpp_metadata[i].children_meta = gather_metadata(val[1])

        return cpp_metadata
    else:
        raise ValueError("Malformed metadata has been encountered")


def to_arrow(input_table,
             object metadata,
             bool keep_index=True):
    """Convert from cudf Frame to PyArrow Table.

    Parameters
    ----------
    input_table : cudf table
    column_names : names for the pyarrow arrays
    field_names : field names for nested type arrays
    keep_index : whether index needs to be part of arrow table

    Returns
    -------
    pyarrow table
    """

    cdef vector[column_metadata] cpp_metadata = gather_metadata(metadata)
    cdef table_view input_table_view = (
        table_view_from_table(input_table, not keep_index)
    )

    cdef shared_ptr[CTable] cpp_arrow_table
    with nogil:
        cpp_arrow_table = cpp_to_arrow(
            input_table_view, cpp_metadata
        )

    return pyarrow_wrap_table(cpp_arrow_table)


def from_arrow(
    object input_table,
    object column_names=None,
    object index_names=None
):
    """Convert from PyArrow Table to cudf Frame.

    Parameters
    ----------
    input_table : PyArrow table
    column_names : names for the cudf table data columns
    index_names : names for the cudf table index columns

    Returns
    -------
    cudf Frame
    """
    cdef shared_ptr[CTable] cpp_arrow_table = (
        pyarrow_unwrap_table(input_table)
    )
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_from_arrow(cpp_arrow_table.get()[0]))

    return data_from_unique_ptr(
        move(c_result),
        column_names=column_names,
        index_names=index_names
    )
