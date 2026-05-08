# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.string_view cimport string_view

from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref

from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/io/experimental/variant.hpp" \
        namespace "cudf::io::parquet::experimental" nogil:
    unique_ptr[column] get_variant_field(
        const column_view& variant_column,
        string_view path,
        cudaStream_t stream,
        device_async_resource_ref mr,
    ) except +libcudf_exception_handler

    unique_ptr[column] cast_variant(
        const column_view& variant_column,
        data_type desired_type,
        cudaStream_t stream,
        device_async_resource_ref mr,
    ) except +libcudf_exception_handler

    unique_ptr[column] extract_variant_field(
        const column_view& variant_column,
        string_view path,
        data_type desired_type,
        cudaStream_t stream,
        device_async_resource_ref mr,
    ) except +libcudf_exception_handler
