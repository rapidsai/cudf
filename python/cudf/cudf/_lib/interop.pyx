# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
import warnings

from cudf._lib.table cimport Table
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.utility cimport move

from cpython cimport pycapsule

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from pyarrow.lib cimport CTable, pyarrow_wrap_table, pyarrow_unwrap_table
from cudf._lib.cpp.interop cimport (
    to_arrow as cpp_to_arrow,
    from_arrow as cpp_from_arrow,
    from_dlpack as cpp_from_dlpack,
    to_dlpack as cpp_to_dlpack,
    DLManagedTensor
)


def from_dlpack(dlpack_capsule):
    """
    Converts a DLPack Tensor PyCapsule into a cudf Table object.

    DLPack Tensor PyCapsule is expected to have the name "dltensor".
    """
    warnings.warn("WARNING: cuDF from_dlpack() assumes column-major (Fortran"
                  " order) input. If the input tensor is row-major, transpose"
                  " it before passing it to this function.")

    cdef DLManagedTensor* dlpack_tensor = <DLManagedTensor*>pycapsule.\
        PyCapsule_GetPointer(dlpack_capsule, 'dltensor')
    pycapsule.PyCapsule_SetName(dlpack_capsule, 'used_dltensor')

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_from_dlpack(dlpack_tensor)
        )

    res = Table.from_unique_ptr(
        move(c_result),
        column_names=range(0, c_result.get()[0].num_columns())
    )
    dlpack_tensor.deleter(dlpack_tensor)
    return res


def to_dlpack(Table source_table):
    """
    Converts a Table cudf object into a DLPack Tensor PyCapsule.

    DLPack Tensor PyCapsule will have the name "dltensor".
    """

    warnings.warn("WARNING: cuDF to_dlpack() produces column-major (Fortran "
                  "order) output. If the output tensor needs to be row major, "
                  "transpose the output of this function.")

    for column in source_table._columns:
        if column.null_count:
            raise ValueError(
                "Cannot create a DLPack tensor with null values. \
                    Input is required to have null count as zero."
            )

    cdef DLManagedTensor *dlpack_tensor
    cdef table_view source_table_view = source_table.data_view()

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


def to_arrow(Table input_table, object column_names, bool keep_index=True):
    """Convert from cudf Table to PyArrow Table.

    Parameters
    ----------
    input_table : cudf table
    column_names : names for the pyarrow arrays
    keep_index : whether index needs to be part of arrow table

    Returns
    -------
    pyarrow table
    """

    cdef vector[string] cpp_column_names
    cdef table_view input = (
        input_table.view() if keep_index else input_table.data_view()
    )
    cpp_column_names.reserve(len(column_names))
    for name in column_names:
        cpp_column_names.push_back(str.encode(str(name)))

    cdef shared_ptr[CTable] cpp_arrow_table
    with nogil:
        cpp_arrow_table = cpp_to_arrow(input, cpp_column_names)

    return pyarrow_wrap_table(cpp_arrow_table)


def from_arrow(
    object input_table,
    object column_names=None,
    object index_names=None
):
    """Convert from PyArrow Table to cudf Table.

    Parameters
    ----------
    input_table : PyArrow table
    column_names : names for the cudf table data columns
    index_names : names for the cudf table index columns

    Returns
    -------
    cudf Table
    """
    cdef shared_ptr[CTable] cpp_arrow_table = (
        pyarrow_unwrap_table(input_table)
    )
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_from_arrow(cpp_arrow_table.get()[0]))

    out_table = Table.from_unique_ptr(
        move(c_result),
        column_names=column_names,
        index_names=index_names
    )

    return out_table
