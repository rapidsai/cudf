# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import warnings
from collections import abc
from io import BytesIO, StringIO

import numpy as np
import pandas as pd

import cudf
from cudf._lib import json as libjson
from cudf.api.types import is_list_like
from cudf.utils import ioutils
from cudf.utils.dtypes import _maybe_convert_to_default_type


@ioutils.doc_read_json()
def read_json(
    path_or_buf,
    engine="auto",
    orient=None,
    dtype=None,
    lines=False,
    compression="infer",
    byte_range=None,
    keep_quotes=False,
    storage_options=None,
    mixed_types_as_string=False,
    prune_columns=False,
    on_bad_lines="error",
    *args,
    **kwargs,
):
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

        if kwargs:
            raise ValueError(
                "cudf engine doesn't support the "
                f"following keyword arguments: {list(kwargs.keys())}"
            )
        if args:
            raise ValueError(
                "cudf engine doesn't support the "
                f"following positional arguments: {list(args)}"
            )

        # Multiple sources are passed as a list. If a single source is passed,
        # wrap it in a list for unified processing downstream.
        if not is_list_like(path_or_buf):
            path_or_buf = [path_or_buf]

        filepaths_or_buffers, compression = (
            ioutils.get_reader_filepath_or_buffer(
                path_or_buf,
                compression=compression,
                iotypes=(BytesIO, StringIO),
                allow_raw_text_input=True,
                storage_options=storage_options,
                warn_on_raw_text_input=True,
                warn_meta=("json", "read_json"),
                expand_dir_pattern="*.json",
            )
        )

        df = libjson.read_json(
            filepaths_or_buffers=filepaths_or_buffers,
            dtype=dtype,
            lines=lines,
            compression=compression,
            byte_range=byte_range,
            keep_quotes=keep_quotes,
            mixed_types_as_string=mixed_types_as_string,
            prune_columns=prune_columns,
            on_bad_lines=on_bad_lines,
        )
    else:
        warnings.warn(
            "Using CPU via Pandas to read JSON dataset, this may "
            "be GPU accelerated in the future"
        )

        if not ioutils.ensure_single_filepath_or_buffer(
            path_or_data=path_or_buf,
            storage_options=storage_options,
        ):
            raise NotImplementedError(
                "`read_json` does not yet support reading "
                "multiple files via pandas"
            )

        path_or_buf, compression = ioutils.get_reader_filepath_or_buffer(
            path_or_data=path_or_buf,
            compression=compression,
            iotypes=(BytesIO, StringIO),
            allow_raw_text_input=True,
            storage_options=storage_options,
        )

        pd_value = pd.read_json(
            path_or_buf,
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


def maybe_return_nullable_pd_obj(cudf_obj):
    try:
        return cudf_obj.to_pandas(nullable=True)
    except NotImplementedError:
        return cudf_obj.to_pandas(nullable=False)


@ioutils.doc_to_json()
def to_json(
    cudf_val,
    path_or_buf=None,
    engine="auto",
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

        if ioutils.is_fsspec_open_file(path_or_buf):
            with path_or_buf as file_obj:
                file_obj = ioutils.get_IOBase_writer(file_obj)
                libjson.write_json(
                    cudf_val, path_or_buf=file_obj, *args, **kwargs
                )
        else:
            libjson.write_json(
                cudf_val, path_or_buf=path_or_buf, *args, **kwargs
            )

        if return_as_string:
            path_or_buf.seek(0)
            return path_or_buf.read()
    elif engine == "pandas":
        warnings.warn("Using CPU via Pandas to write JSON dataset")
        if isinstance(cudf_val, cudf.DataFrame):
            pd_data = {
                col: maybe_return_nullable_pd_obj(series)
                for col, series in cudf_val.items()
            }
            pd_value = pd.DataFrame(pd_data)
        else:
            pd_value = maybe_return_nullable_pd_obj(cudf_val)
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
