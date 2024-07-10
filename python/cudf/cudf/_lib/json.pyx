# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import io
import os
from collections import abc

import cudf
from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool

cimport cudf._lib.pylibcudf.libcudf.io.types as cudf_io_types
from cudf._lib.io.utils cimport add_df_col_struct_names
from cudf._lib.pylibcudf.io.types cimport compression_type
from cudf._lib.pylibcudf.libcudf.io.json cimport json_recovery_mode_t
from cudf._lib.pylibcudf.libcudf.io.types cimport compression_type
from cudf._lib.pylibcudf.libcudf.types cimport data_type, type_id
from cudf._lib.pylibcudf.types cimport DataType
from cudf._lib.types cimport dtype_to_data_type
from cudf._lib.utils cimport data_from_pylibcudf_io

import cudf._lib.pylibcudf as plc


cdef json_recovery_mode_t _get_json_recovery_mode(object on_bad_lines):
    if on_bad_lines.lower() == "error":
        return json_recovery_mode_t.FAIL
    elif on_bad_lines.lower() == "recover":
        return json_recovery_mode_t.RECOVER_WITH_NULL
    else:
        raise TypeError(f"Invalid parameter for {on_bad_lines=}")


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

    processed_dtypes = None

    if dtype is False:
        raise ValueError("False value is unsupported for `dtype`")
    elif dtype is not True:
        processed_dtypes = []
        if isinstance(dtype, abc.Mapping):
            for k, v in dtype.items():
                # Make sure keys are string
                k = str(k)
                lib_type, child_types = _get_cudf_schema_element_from_dtype(v)
                processed_dtypes.append((k, lib_type, child_types))
        elif isinstance(dtype, abc.Collection):
            for col_dtype in dtype:
                processed_dtypes.append(
                    # Ignore child columns since we cannot specify their dtypes
                    # when passing a list
                    _get_cudf_schema_element_from_dtype(col_dtype)[0]
                )
        else:
            raise TypeError("`dtype` must be 'list like' or 'dict'")

    table_w_meta = plc.io.json.read_json(
        plc.io.SourceInfo(filepaths_or_buffers),
        processed_dtypes,
        c_compression,
        lines,
        byte_range_offset = byte_range[0] if byte_range is not None else 0,
        byte_range_size = byte_range[1] if byte_range is not None else 0,
        keep_quotes = keep_quotes,
        mixed_types_as_string = mixed_types_as_string,
        prune_columns = prune_columns,
        recovery_mode = _get_json_recovery_mode(on_bad_lines)
    )

    df = cudf.DataFrame._from_data(
        *data_from_pylibcudf_io(
            table_w_meta
        )
    )

    # Post-processing to add in struct column names
    add_df_col_struct_names(df, table_w_meta.child_names)
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


cdef _get_cudf_schema_element_from_dtype(object dtype) except *:
    dtype = cudf.dtype(dtype)
    if isinstance(dtype, cudf.CategoricalDtype):
        raise NotImplementedError(
            "CategoricalDtype as dtype is not yet "
            "supported in JSON reader"
        )

    lib_type = DataType.from_libcudf(dtype_to_data_type(dtype))
    child_types = []

    if isinstance(dtype, cudf.StructDtype):
        for name, child_type in dtype.fields.items():
            child_lib_type, grandchild_types = \
                _get_cudf_schema_element_from_dtype(child_type)
            child_types.append((name, child_lib_type, grandchild_types))
    elif isinstance(dtype, cudf.ListDtype):
        child_lib_type, grandchild_types = \
            _get_cudf_schema_element_from_dtype(dtype.element_type)

        child_types = [
            ("offsets", DataType.from_libcudf(data_type(type_id.INT32)), []),
            ("element", child_lib_type, grandchild_types)
        ]

    return lib_type, child_types


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
