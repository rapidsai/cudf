# Copyright (c) 2019-2024, NVIDIA CORPORATION.
from __future__ import annotations

import os
import warnings
from collections import abc
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

import pylibcudf as plc

import cudf
from cudf._lib.column import Column
from cudf._lib.types import dtype_to_pylibcudf_type
from cudf.core.buffer import acquire_spill_lock
from cudf.utils import ioutils
from cudf.utils.dtypes import _maybe_convert_to_default_type

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase


def _get_cudf_schema_element_from_dtype(
    dtype,
) -> tuple[plc.DataType, list[tuple[str, plc.DataType, Any]]]:
    dtype = cudf.dtype(dtype)
    if isinstance(dtype, cudf.CategoricalDtype):
        raise NotImplementedError(
            "CategoricalDtype as dtype is not yet " "supported in JSON reader"
        )

    lib_type = dtype_to_pylibcudf_type(dtype)
    child_types = []

    if isinstance(dtype, cudf.StructDtype):
        for name, child_type in dtype.fields.items():
            child_lib_type, grandchild_types = (
                _get_cudf_schema_element_from_dtype(child_type)
            )
            child_types.append((name, child_lib_type, grandchild_types))
    elif isinstance(dtype, cudf.ListDtype):
        child_lib_type, grandchild_types = _get_cudf_schema_element_from_dtype(
            dtype.element_type
        )
        child_types = [
            ("offsets", plc.DataType(plc.TypeId.INT32), []),
            ("element", child_lib_type, grandchild_types),
        ]

    return lib_type, child_types


def _to_plc_compression(
    compression: Literal["infer", "gzip", "bz2", "zip", "xz", None],
) -> plc.io.types.CompressionType:
    if compression is not None:
        if compression == "gzip":
            return plc.io.types.CompressionType.GZIP
        elif compression == "bz2":
            return plc.io.types.CompressionType.BZIP2
        elif compression == "zip":
            return plc.io.types.CompressionType.ZIP
        else:
            return plc.io.types.CompressionType.AUTO
    else:
        return plc.io.types.CompressionType.NONE


@ioutils.doc_read_json()
def read_json(
    path_or_buf,
    engine: Literal["auto", "pandas", "cudf"] = "auto",
    orient=None,
    dtype=None,
    lines: bool = False,
    compression="infer",
    byte_range: None | list[int] = None,
    keep_quotes: bool = False,
    storage_options=None,
    mixed_types_as_string: bool = False,
    prune_columns: bool = False,
    on_bad_lines: Literal["error", "recover"] = "error",
    *args,
    **kwargs,
) -> cudf.DataFrame:
    """{docstring}"""

    if dtype is not None and not isinstance(dtype, (abc.Mapping, bool)):
        raise TypeError(
            "'dtype' parameter only supports "
            "a dict of column names and types as key-value pairs, "
            f"or a bool, or None. Got {type(dtype)}"
        )

    if engine == "auto":
        engine = "cudf" if lines else "pandas"
    if engine != "cudf" and keep_quotes:
        raise ValueError(
            "keep_quotes='True' is supported only with engine='cudf'"
        )

    if engine == "cudf":
        if dtype is None:
            dtype = True

        if args:
            raise ValueError(
                "cudf engine doesn't support the "
                f"following positional arguments: {list(args)}"
            )

        filepaths_or_buffers = ioutils.get_reader_filepath_or_buffer(
            path_or_buf,
            iotypes=(BytesIO, StringIO),
            allow_raw_text_input=True,
            storage_options=storage_options,
            warn_on_raw_text_input=True,
            warn_meta=("json", "read_json"),
            expand_dir_pattern="*.json",
        )

        # If input data is a JSON string (or StringIO), hold a reference to
        # the encoded memoryview externally to ensure the encoded buffer
        # isn't destroyed before calling pylibcudf `read_json()`

        for idx, source in enumerate(filepaths_or_buffers):
            if isinstance(source, str) and not os.path.isfile(source):
                filepaths_or_buffers[idx] = source.encode()

        c_compression = _to_plc_compression(compression)

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
                    lib_type, child_types = (
                        _get_cudf_schema_element_from_dtype(v)
                    )
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
            res_cols, res_col_names, res_child_names = (
                plc.io.json.chunked_read_json(
                    plc.io.json._setup_json_reader_options(
                        plc.io.SourceInfo(filepaths_or_buffers),
                        processed_dtypes,
                        c_compression,
                        keep_quotes=keep_quotes,
                        mixed_types_as_string=mixed_types_as_string,
                        prune_columns=prune_columns,
                        recovery_mode=c_on_bad_lines,
                    )
                )
            )
            data = {
                name: Column.from_pylibcudf(col)
                for name, col in zip(res_col_names, res_cols, strict=True)
            }
            df = cudf.DataFrame._from_data(data)
            ioutils._add_df_col_struct_names(df, res_child_names)
            return df
        else:
            table_w_meta = plc.io.json.read_json(
                plc.io.json._setup_json_reader_options(
                    plc.io.SourceInfo(filepaths_or_buffers),
                    processed_dtypes,
                    c_compression,
                    lines,
                    byte_range_offset=byte_range[0]
                    if byte_range is not None
                    else 0,
                    byte_range_size=byte_range[1]
                    if byte_range is not None
                    else 0,
                    keep_quotes=keep_quotes,
                    mixed_types_as_string=mixed_types_as_string,
                    prune_columns=prune_columns,
                    recovery_mode=c_on_bad_lines,
                    extra_parameters=kwargs,
                )
            )
            data = {
                name: Column.from_pylibcudf(col)
                for name, col in zip(
                    table_w_meta.column_names(include_children=False),
                    table_w_meta.columns,
                    strict=True,
                )
            }
            df = cudf.DataFrame._from_data(data)

            # Post-processing to add in struct column names
            ioutils._add_df_col_struct_names(df, table_w_meta.child_names)
    else:
        warnings.warn(
            "Using CPU via Pandas to read JSON dataset, this may "
            "be GPU accelerated in the future"
        )

        filepath_or_buffer = ioutils.get_reader_filepath_or_buffer(
            path_or_data=path_or_buf,
            iotypes=(BytesIO, StringIO),
            allow_raw_text_input=True,
            storage_options=storage_options,
        )
        filepath_or_buffer = ioutils._select_single_source(
            filepath_or_buffer, "read_json (via pandas)"
        )

        pd_value = pd.read_json(
            filepath_or_buffer,
            lines=lines,
            dtype=dtype,
            compression=compression,
            storage_options=storage_options,
            orient=orient,
            *args,
            **kwargs,
        )
        df = cudf.from_pandas(pd_value)

    if dtype is None:
        dtype = True

    if dtype is True or isinstance(dtype, abc.Mapping):
        # There exists some dtypes in the result columns that is inferred.
        # Find them and map them to the default dtypes.
        specified_dtypes = {} if dtype is True else dtype
        unspecified_dtypes = {
            name: dtype
            for name, dtype in df._dtypes
            if name not in specified_dtypes
        }
        default_dtypes = {}

        for name, dt in unspecified_dtypes.items():
            if dt == np.dtype("i1"):
                # csv reader reads all null column as int8.
                # The dtype should remain int8.
                default_dtypes[name] = dt
            else:
                default_dtypes[name] = _maybe_convert_to_default_type(dt)
        df = df.astype(default_dtypes)

    return df


def _maybe_return_nullable_pd_obj(
    cudf_obj: cudf.DataFrame | cudf.Series,
) -> pd.DataFrame | pd.Series:
    try:
        return cudf_obj.to_pandas(nullable=True)
    except NotImplementedError:
        return cudf_obj.to_pandas(nullable=False)


def _dtype_to_names_list(col: ColumnBase) -> list[tuple[abc.Hashable, Any]]:
    if isinstance(col.dtype, cudf.StructDtype):
        return [
            (name, _dtype_to_names_list(child))
            for name, child in zip(col.dtype.fields, col.children)
        ]
    elif isinstance(col.dtype, cudf.ListDtype):
        return [("", _dtype_to_names_list(child)) for child in col.children]
    return []


@acquire_spill_lock()
def _plc_write_json(
    table: cudf.Series | cudf.DataFrame,
    colnames: list[tuple[abc.Hashable, Any]],
    path_or_buf,
    na_rep: str = "null",
    include_nulls: bool = True,
    lines: bool = False,
    rows_per_chunk: int = 1024 * 64,  # 64K rows
    compression: Literal["infer", "gzip", "bz2", "zip", "xz", None] = None,
) -> None:
    try:
        tbl_w_meta = plc.io.TableWithMetadata(
            plc.Table(
                [col.to_pylibcudf(mode="read") for col in table._columns]
            ),
            colnames,
        )
        options = (
            plc.io.json.JsonWriterOptions.builder(
                plc.io.SinkInfo([path_or_buf]), tbl_w_meta.tbl
            )
            .metadata(tbl_w_meta)
            .na_rep(na_rep)
            .include_nulls(include_nulls)
            .lines(lines)
            .compression(_to_plc_compression(compression))
            .build()
        )
        if rows_per_chunk != np.iinfo(np.int32).max:
            options.set_rows_per_chunk(rows_per_chunk)
        plc.io.json.write_json(options)
    except OverflowError as err:
        raise OverflowError(
            f"Writing JSON file with rows_per_chunk={rows_per_chunk} failed. "
            "Consider providing a smaller rows_per_chunk argument."
        ) from err


@ioutils.doc_to_json()
def to_json(
    cudf_val: cudf.DataFrame | cudf.Series,
    path_or_buf=None,
    engine: Literal["auto", "pandas", "cudf"] = "auto",
    orient=None,
    storage_options=None,
    *args,
    **kwargs,
):
    """{docstring}"""

    if engine == "auto":
        engine = "pandas"

    if engine == "cudf":
        if orient not in {"records", None}:
            raise ValueError(
                f"Only the `orient='records'` is supported for JSON writer"
                f" with `engine='cudf'`, got {orient}"
            )

        if path_or_buf is None:
            path_or_buf = StringIO()
            return_as_string = True
        else:
            path_or_buf = ioutils.get_writer_filepath_or_buffer(
                path_or_data=path_or_buf,
                mode="w",
                storage_options=storage_options,
            )
            return_as_string = False

        colnames = [
            (name, _dtype_to_names_list(col))
            for name, col in cudf_val._column_labels_and_values
        ]

        if ioutils.is_fsspec_open_file(path_or_buf):
            with path_or_buf as file_obj:
                file_obj = ioutils.get_IOBase_writer(file_obj)
                _plc_write_json(
                    cudf_val, colnames, path_or_buf, *args, **kwargs
                )
        else:
            _plc_write_json(cudf_val, colnames, path_or_buf, *args, **kwargs)

        if return_as_string:
            path_or_buf.seek(0)
            return path_or_buf.read()
    elif engine == "pandas":
        warnings.warn("Using CPU via Pandas to write JSON dataset")
        if isinstance(cudf_val, cudf.DataFrame):
            pd_data = {
                col: _maybe_return_nullable_pd_obj(series)
                for col, series in cudf_val.items()
            }
            pd_value = pd.DataFrame(pd_data)
        else:
            pd_value = _maybe_return_nullable_pd_obj(cudf_val)
        return pd_value.to_json(
            path_or_buf,
            orient=orient,
            storage_options=storage_options,
            *args,
            **kwargs,
        )
    else:
        raise ValueError(
            f"`engine` only support {{'auto', 'cudf', 'pandas'}}, "
            f"got: {engine}"
        )
