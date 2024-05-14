# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.types cimport (
    bitmask_type,
    data_type,
    size_type,
)


cdef extern from "cudf/column/column_view.hpp" namespace "cudf" nogil:
    cdef cppclass column_view:
        column_view() except +
        column_view(const column_view& other) except +

        column_view& operator=(const column_view&) except +

        column_view(
            data_type type,
            size_type size,
            const void* data
        ) except +

        column_view(
            data_type type,
            size_type size,
            const void* data,
            const bitmask_type* null_mask
        ) except +

        column_view(
            data_type type,
            size_type size,
            const void* data,
            const bitmask_type* null_mask,
            size_type null_count
        ) except +

        column_view(
            data_type type,
            size_type size,
            const void* data,
            const bitmask_type* null_mask,
            size_type null_count,
            size_type offset
        ) except +

        column_view(
            data_type type,
            size_type size,
            const void* data,
            const bitmask_type* null_mask,
            size_type null_count,
            size_type offset,
            vector[column_view] children
        ) except +

        const T* data[T]() except +
        const T* head[T]() except +
        const bitmask_type* null_mask() except +
        size_type size() except +
        data_type type() except +
        bool nullable() except +
        size_type null_count() except +
        bool has_nulls() except +
        size_type offset() except +
        size_type num_children() except +
        column_view child(size_type) except +

    cdef cppclass mutable_column_view:
        mutable_column_view() except +
        mutable_column_view(const mutable_column_view&) except +
        mutable_column_view& operator=(const mutable_column_view&) except +

        mutable_column_view(
            data_type type,
            size_type size,
            const void* data
        ) except +

        mutable_column_view(
            data_type type,
            size_type size,
            const void* data,
            const bitmask_type* null_mask
        ) except +

        mutable_column_view(
            data_type type,
            size_type size,
            const void* data,
            const bitmask_type* null_mask,
            size_type null_count
        ) except +

        mutable_column_view(
            data_type type,
            size_type size,
            const void* data,
            const bitmask_type* null_mask,
            size_type null_count,
            size_type offset
        ) except +

        mutable_column_view(
            data_type type, size_type size, const void* data,
            const bitmask_type* null_mask, size_type null_count,
            size_type offset, vector[mutable_column_view] children
        ) except +

        T* data[T]() except +
        T* head[T]() except +
        bitmask_type* null_mask() except +
        size_type size() except +
        data_type type() except +
        bool nullable() except +
        size_type null_count() except +
        bool has_nulls() except +
        size_type offset() except +
        size_type num_children() except +
        mutable_column_view& child(size_type) except +
