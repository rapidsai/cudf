# Copyright (c) 2018-2020, NVIDIA CORPORATION.

import cudf
from cudf._lib.table cimport Table

from libcpp.memory cimport unique_ptr

from cudf._lib.move cimport move
from cpython cimport pycapsule

import warnings

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.dlpack cimport (
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
