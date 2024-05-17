# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import io
import os
from collections import abc

import cudf
from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

cimport cudf._lib.pylibcudf.libcudf.io.types as cudf_io_types
from cudf._lib.column cimport Column
from cudf._lib.io.utils cimport (
    make_sink_info,
    make_source_info,
    update_struct_field_names,
)
from cudf._lib.pylibcudf.libcudf.io.data_sink cimport data_sink
from cudf._lib.pylibcudf.libcudf.io.json cimport (
    json_reader_options,
    json_writer_options,
    read_json as libcudf_read_json,
    schema_element,
    write_json as libcudf_write_json,
)
from cudf._lib.pylibcudf.libcudf.io.types cimport (
    column_name_info,
    compression_type,
    sink_info,
    table_metadata,
    table_with_metadata,
)
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.types cimport data_type, size_type
from cudf._lib.types cimport dtype_to_data_type
from cudf._lib.utils cimport data_from_unique_ptr, table_view_from_table


cpdef read_json(object filepaths_or_buffers,
                object dtype,
                bool lines,
                object compression,
                object byte_range,
                bool legacy,
                bool keep_quotes,
                bool mixed_types_as_string,
                bool prune_columns):
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
        elif compression == 'zip':
            c_compression = cudf_io_types.compression_type.ZIP
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
        .legacy(legacy)
        .build()
    )
    if is_list_like_dtypes:
        opts.set_dtypes(c_dtypes_list)
    else:
        opts.set_dtypes(c_dtypes_schema_map)

    opts.enable_keep_quotes(keep_quotes)
    opts.enable_mixed_types_as_string(mixed_types_as_string)
    opts.enable_prune_columns(prune_columns)
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


@acquire_spill_lock()
def write_json(
    table,
    object path_or_buf=None,
    object na_rep="null",
    bool include_nulls=True,
    bool lines=False,
    bool index=False,
    int rows_per_chunk=1024*64,  # 64K rows
):
    """
    Cython function to call into libcudf API, see `write_json`.

    See Also
    --------
    cudf.to_json
    """
    cdef table_view input_table_view = table_view_from_table(
        table, ignore_index=True
    )

    cdef unique_ptr[data_sink] data_sink_c
    cdef sink_info sink_info_c = make_sink_info(path_or_buf, data_sink_c)
    cdef string na_c = na_rep.encode()
    cdef bool include_nulls_c = include_nulls
    cdef bool lines_c = lines
    cdef int rows_per_chunk_c = rows_per_chunk
    cdef string true_value_c = 'true'.encode()
    cdef string false_value_c = 'false'.encode()
    cdef table_metadata tbl_meta

    num_index_cols_meta = 0
    cdef column_name_info child_info
    for i, name in enumerate(table._column_names, num_index_cols_meta):
        child_info.name = name.encode()
        tbl_meta.schema_info.push_back(child_info)
        _set_col_children_metadata(
            table[name]._column,
            tbl_meta.schema_info[i]
        )

    cdef json_writer_options options = move(
        json_writer_options.builder(sink_info_c, input_table_view)
        .metadata(tbl_meta)
        .na_rep(na_c)
        .include_nulls(include_nulls_c)
        .lines(lines_c)
        .rows_per_chunk(rows_per_chunk_c)
        .true_value(true_value_c)
        .false_value(false_value_c)
        .build()
    )

    try:
        with nogil:
            libcudf_write_json(options)
    except OverflowError:
        raise OverflowError(
            f"Writing JSON file with rows_per_chunk={rows_per_chunk} failed. "
            "Consider providing a smaller rows_per_chunk argument."
        )


cdef schema_element _get_cudf_schema_element_from_dtype(object dtype) except *:
    cdef schema_element s_element
    cdef data_type lib_type
    dtype = cudf.dtype(dtype)
    if isinstance(dtype, cudf.CategoricalDtype):
        raise NotImplementedError(
            "CategoricalDtype as dtype is not yet "
            "supported in JSON reader"
        )
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


cdef data_type _get_cudf_data_type_from_dtype(object dtype) except *:
    dtype = cudf.dtype(dtype)
    if isinstance(dtype, cudf.CategoricalDtype):
        raise NotImplementedError(
            "CategoricalDtype as dtype is not yet "
            "supported in JSON reader"
        )
    return dtype_to_data_type(dtype)

cdef _set_col_children_metadata(Column col,
                                column_name_info& col_meta):
    cdef column_name_info child_info
    if isinstance(col.dtype, cudf.StructDtype):
        for i, (child_col, name) in enumerate(
            zip(col.children, list(col.dtype.fields))
        ):
            child_info.name = name.encode()
            col_meta.children.push_back(child_info)
            _set_col_children_metadata(
                child_col, col_meta.children[i]
            )
    elif isinstance(col.dtype, cudf.ListDtype):
        for i, child_col in enumerate(col.children):
            col_meta.children.push_back(child_info)
            _set_col_children_metadata(
                child_col, col_meta.children[i]
            )
    else:
        return
