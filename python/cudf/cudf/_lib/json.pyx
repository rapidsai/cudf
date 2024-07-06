# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import io
import os
from collections import abc

import cudf
from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

cimport cudf._lib.pylibcudf.libcudf.io.types as cudf_io_types
from cudf._lib.io.utils cimport make_source_info, update_struct_field_names
from cudf._lib.pylibcudf.libcudf.io.json cimport (
    json_reader_options,
    json_recovery_mode_t,
    read_json as libcudf_read_json,
    schema_element,
)
from cudf._lib.pylibcudf.libcudf.io.types cimport (
    compression_type,
    table_with_metadata,
)
from cudf._lib.pylibcudf.libcudf.types cimport data_type, size_type
from cudf._lib.types cimport dtype_to_data_type
from cudf._lib.utils cimport (
    columns_from_unique_ptr,
    data_from_pylibcudf_table,
    data_from_unique_ptr,
)

import cudf._lib.pylibcudf as plc
from cudf._lib.concat import concat_columns


cdef json_recovery_mode_t _get_json_recovery_mode(object on_bad_lines):
    if on_bad_lines.lower() == "error":
        return json_recovery_mode_t.FAIL
    elif on_bad_lines.lower() == "recover":
        return json_recovery_mode_t.RECOVER_WITH_NULL
    else:
        raise TypeError(f"Invalid parameter for {on_bad_lines=}")

cdef json_reader_options _setup_json_reader_options(
        object filepaths_or_buffers,
        object dtype,
        object compression,
        bool keep_quotes,
        bool mixed_types_as_string,
        bool prune_columns,
        object on_bad_lines,
        bool lines,
        size_type byte_range_offset,
        size_type byte_range_size):
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
        .lines(lines)
        .byte_range_offset(byte_range_offset)
        .byte_range_size(byte_range_size)
        .recovery_mode(_get_json_recovery_mode(on_bad_lines))
        .build()
    )
    if is_list_like_dtypes:
        opts.set_dtypes(c_dtypes_list)
    else:
        opts.set_dtypes(c_dtypes_schema_map)

    opts.enable_keep_quotes(keep_quotes)
    opts.enable_mixed_types_as_string(mixed_types_as_string)
    opts.enable_prune_columns(prune_columns)

    return opts

cpdef read_json(object filepaths_or_buffers,
                object dtype,
                bool lines,
                object compression,
                object byte_range,
                bool keep_quotes,
                bool mixed_types_as_string,
                bool prune_columns,
                object on_bad_lines):
    """
    Cython function to call into libcudf API, see `read_json`.

    See Also
    --------
    cudf.io.json.read_json
    cudf.io.json.to_json
    """
    # Determine byte read offsets if applicable
    cdef size_type c_range_offset = (
        byte_range[0] if byte_range is not None else 0
    )
    cdef size_type c_range_size = (
        byte_range[1] if byte_range is not None else 0
    )
    cdef json_reader_options opts = _setup_json_reader_options(
        filepaths_or_buffers, dtype, compression, keep_quotes,
        mixed_types_as_string, prune_columns, on_bad_lines,
        lines, c_range_offset, c_range_size
    )

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

cpdef chunked_read_json(object filepaths_or_buffers,
                        object dtype,
                        object compression,
                        bool keep_quotes,
                        bool mixed_types_as_string,
                        bool prune_columns,
                        object on_bad_lines,
                        int chunk_size=100_000_000):
    """
    Cython function to call into libcudf API, see `read_json`.

    See Also
    --------
    cudf.io.json.read_json
    cudf.io.json.to_json
    """
    cdef size_type c_range_size = (
        chunk_size if chunk_size is not None else 0
    )
    cdef json_reader_options opts = _setup_json_reader_options(
        filepaths_or_buffers, dtype, compression, keep_quotes,
        mixed_types_as_string, prune_columns, on_bad_lines,
        True, 0, c_range_size
    )

    # Read JSON
    cdef cudf_io_types.table_with_metadata c_result
    final_columns = []
    meta_names = None
    i = 0
    while True:
        opts.set_byte_range_offset(c_range_size * i)
        opts.set_byte_range_size(c_range_size)

        try:
            with nogil:
                c_result = move(libcudf_read_json(opts))
        except (ValueError, OverflowError):
            break
        if meta_names is None:
            meta_names = [info.name.decode() for info in c_result.metadata.schema_info]
        new_chunk = columns_from_unique_ptr(move(c_result.tbl))
        if len(final_columns) == 0:
            final_columns = new_chunk
        else:
            for col_idx in range(len(meta_names)):
                final_columns[col_idx] = concat_columns(
                    [final_columns[col_idx], new_chunk[col_idx]]
                )
                # Must drop any residual GPU columns to save memory
                new_chunk[col_idx] = None
        i += 1
    df = cudf.DataFrame._from_data(
            *data_from_pylibcudf_table(
                plc.Table(
                    [col.to_pylibcudf(mode="read") for col in final_columns]
                ),
                column_names=meta_names,
                index_names=None
            )
        )
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
    cdef list colnames = []

    for name in table._column_names:
        colnames.append((name, _dtype_to_names_list(table[name]._column)))

    try:
        plc.io.json.write_json(
            plc.io.SinkInfo([path_or_buf]),
            plc.io.TableWithMetadata(
                plc.Table([
                    c.to_pylibcudf(mode="read") for c in table._columns
                ]),
                colnames
            ),
            na_rep,
            include_nulls,
            lines,
            rows_per_chunk,
            true_value="true",
            false_value="false"
        )
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


def _dtype_to_names_list(col):
    if isinstance(col.dtype, cudf.StructDtype):
        return [(name, _dtype_to_names_list(child))
                for name, child in zip(col.dtype.fields, col.children)]
    elif isinstance(col.dtype, cudf.ListDtype):
        return [("", _dtype_to_names_list(child))
                for child in col.children]
    return []
