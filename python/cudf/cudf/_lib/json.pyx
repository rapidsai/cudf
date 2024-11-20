# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import os
from collections import abc

import cudf
from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool

from cudf._lib.column cimport Column
from cudf._lib.io.utils cimport add_df_col_struct_names
from cudf._lib.types cimport dtype_to_pylibcudf_type
from cudf._lib.utils cimport _data_from_columns, data_from_pylibcudf_io

import pylibcudf as plc


cpdef read_json(object filepaths_or_buffers,
                object dtype,
                bool lines,
                object compression,
                object byte_range,
                bool keep_quotes,
                bool mixed_types_as_string,
                bool prune_columns,
                str on_bad_lines):
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

    for idx, source in enumerate(filepaths_or_buffers):
        if isinstance(source, str) and not os.path.isfile(source):
            filepaths_or_buffers[idx] = source.encode()

    # Setup arguments
    if compression is not None:
        if compression == 'gzip':
            c_compression = plc.io.types.CompressionType.GZIP
        elif compression == 'bz2':
            c_compression = plc.io.types.CompressionType.BZIP2
        elif compression == 'zip':
            c_compression = plc.io.types.CompressionType.ZIP
        else:
            c_compression = plc.io.types.CompressionType.AUTO
    else:
        c_compression = plc.io.types.CompressionType.NONE

    if on_bad_lines.lower() == "error":
        c_on_bad_lines = plc.io.types.JSONRecoveryMode.FAIL
    elif on_bad_lines.lower() == "recover":
        c_on_bad_lines = plc.io.types.JSONRecoveryMode.RECOVER_WITH_NULL
    else:
        raise TypeError(f"Invalid parameter for {on_bad_lines=}")

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

    if cudf.get_option("io.json.low_memory") and lines:
        res_cols, res_col_names, res_child_names = plc.io.json.chunked_read_json(
            plc.io.SourceInfo(filepaths_or_buffers),
            processed_dtypes,
            c_compression,
            keep_quotes = keep_quotes,
            mixed_types_as_string = mixed_types_as_string,
            prune_columns = prune_columns,
            recovery_mode = c_on_bad_lines
        )
        df = cudf.DataFrame._from_data(
            *_data_from_columns(
                columns=[Column.from_pylibcudf(col) for col in res_cols],
                column_names=res_col_names,
                index_names=None
               )
            )
        add_df_col_struct_names(df, res_child_names)
        return df
    else:
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
            recovery_mode = c_on_bad_lines
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


def _get_cudf_schema_element_from_dtype(object dtype):
    dtype = cudf.dtype(dtype)
    if isinstance(dtype, cudf.CategoricalDtype):
        raise NotImplementedError(
            "CategoricalDtype as dtype is not yet "
            "supported in JSON reader"
        )

    lib_type = dtype_to_pylibcudf_type(dtype)
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
            ("offsets", plc.DataType(plc.TypeId.INT32), []),
            ("element", child_lib_type, grandchild_types)
        ]

    return lib_type, child_types


def _dtype_to_names_list(col):
    if isinstance(col.dtype, cudf.StructDtype):
        return [(name, _dtype_to_names_list(child))
                for name, child in zip(col.dtype.fields, col.children)]
    elif isinstance(col.dtype, cudf.ListDtype):
        return [("", _dtype_to_names_list(child))
                for child in col.children]
    return []
