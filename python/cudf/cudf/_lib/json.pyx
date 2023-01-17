# Copyright (c) 2019-2023, NVIDIA CORPORATION.

# cython: boundscheck = False

import io
import os
from collections import abc

import cudf

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

cimport cudf._lib.cpp.io.types as cudf_io_types
from cudf._lib.cpp.io.json cimport (
    json_reader_options,
    read_json as libcudf_read_json,
    schema_element,
)
from cudf._lib.cpp.types cimport data_type, size_type
from cudf._lib.io.utils cimport make_source_info, update_struct_field_names
from cudf._lib.types cimport dtype_to_data_type
from cudf._lib.utils cimport data_from_unique_ptr


cpdef read_json(object filepaths_or_buffers,
                object dtype,
                bool lines,
                object compression,
                object byte_range,
                bool experimental,
                bool keep_quotes):
    """
    Cython function to call into libcudf API, see `read_json`.

    See Also
    --------
    cudf.io.json.read_json
    cudf.io.json.to_json
    """

    # If input data is a JSON string (or StringIO), hold a reference to
    # the encoded memoryview externally to ensure the encoded buffer
    # isn't destroyed before calling libcudf `read_json()`
    for idx in range(len(filepaths_or_buffers)):
        if isinstance(filepaths_or_buffers[idx], io.StringIO):
            filepaths_or_buffers[idx] = \
                filepaths_or_buffers[idx].read().encode()
        elif isinstance(filepaths_or_buffers[idx], str) and \
                not os.path.isfile(filepaths_or_buffers[idx]):
            filepaths_or_buffers[idx] = filepaths_or_buffers[idx].encode()

    # Setup arguments
    cdef vector[data_type] c_dtypes_list
    cdef map[string, schema_element] c_dtypes_schema_map
    cdef cudf_io_types.compression_type c_compression
    # Determine byte read offsets if applicable
    cdef size_type c_range_offset = (
        byte_range[0] if byte_range is not None else 0
    )
    cdef size_type c_range_size = (
        byte_range[1] if byte_range is not None else 0
    )
    cdef bool c_lines = lines

    if compression is not None:
        if compression == 'gzip':
            c_compression = cudf_io_types.compression_type.GZIP
        elif compression == 'bz2':
            c_compression = cudf_io_types.compression_type.BZIP2
        else:
            c_compression = cudf_io_types.compression_type.AUTO
    else:
        c_compression = cudf_io_types.compression_type.NONE
    is_list_like_dtypes = False
    if dtype is False:
        raise ValueError("False value is unsupported for `dtype`")
    elif dtype is not True:
        if isinstance(dtype, abc.Mapping):
            for k, v in dtype.items():
                c_dtypes_schema_map[str(k).encode()] = \
                    _get_cudf_schema_element_from_dtype(v)
        elif isinstance(dtype, abc.Collection):
            is_list_like_dtypes = True
            c_dtypes_list.reserve(len(dtype))
            for col_dtype in dtype:
                c_dtypes_list.push_back(
                    _get_cudf_data_type_from_dtype(
                        col_dtype))
        else:
            raise TypeError("`dtype` must be 'list like' or 'dict'")

    cdef json_reader_options opts = move(
        json_reader_options.builder(make_source_info(filepaths_or_buffers))
        .compression(c_compression)
        .lines(c_lines)
        .byte_range_offset(c_range_offset)
        .byte_range_size(c_range_size)
        .legacy(not experimental)
        .build()
    )
    if is_list_like_dtypes:
        opts.set_dtypes(c_dtypes_list)
    else:
        opts.set_dtypes(c_dtypes_schema_map)

    opts.enable_keep_quotes(keep_quotes)
    # Read JSON
    cdef cudf_io_types.table_with_metadata c_result

    with nogil:
        c_result = move(libcudf_read_json(opts))

    meta_names = [info.name.decode() for info in c_result.metadata.schema_info]
    df = cudf.DataFrame._from_data(*data_from_unique_ptr(
        move(c_result.tbl),
        column_names=meta_names
    ))

    update_struct_field_names(df, c_result.metadata.schema_info)

    return df


cdef schema_element _get_cudf_schema_element_from_dtype(object dtype) except +:
    cdef schema_element s_element
    cdef data_type lib_type
    if cudf.api.types.is_categorical_dtype(dtype):
        raise NotImplementedError(
            "CategoricalDtype as dtype is not yet "
            "supported in JSON reader"
        )

    dtype = cudf.dtype(dtype)
    lib_type = dtype_to_data_type(dtype)
    s_element.type = lib_type
    if isinstance(dtype, cudf.StructDtype):
        for name, child_type in dtype.fields.items():
            s_element.child_types[name.encode()] = \
                _get_cudf_schema_element_from_dtype(child_type)
    elif isinstance(dtype, cudf.ListDtype):
        s_element.child_types["offsets".encode()] = \
            _get_cudf_schema_element_from_dtype(cudf.dtype("int32"))
        s_element.child_types["element".encode()] = \
            _get_cudf_schema_element_from_dtype(dtype.element_type)

    return s_element


cdef data_type _get_cudf_data_type_from_dtype(object dtype) except +:
    if cudf.api.types.is_categorical_dtype(dtype):
        raise NotImplementedError(
            "CategoricalDtype as dtype is not yet "
            "supported in JSON reader"
        )

    dtype = cudf.dtype(dtype)
    return dtype_to_data_type(dtype)
