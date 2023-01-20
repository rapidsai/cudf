# Copyright (c) 2019-2023, NVIDIA CORPORATION.

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

    if engine == "cudf_experimental":
        raise ValueError(
            "engine='cudf_experimental' support has been removed, "
            "use `engine='cudf'`"
        )

    if engine == "cudf_legacy":
        # TODO: Deprecated in 23.02, please
        # give some time until(more than couple of
        # releases from now) `cudf_legacy`
        # support can be removed completely.
        warnings.warn(
            "engine='cudf_legacy' is a deprecated engine."
            "This will be removed in a future release."
            "Please switch to using engine='cudf'.",
            FutureWarning,
        )
    if engine == "cudf_legacy" and not lines:
        raise ValueError(f"{engine} engine only supports JSON Lines format")
    if engine == "auto":
        engine = "cudf" if lines else "pandas"
    if engine != "cudf" and keep_quotes:
        raise ValueError(
            "keep_quotes='True' is supported only with engine='cudf'"
        )

    if engine == "cudf_legacy" or engine == "cudf":
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

        filepaths_or_buffers = []
        for source in path_or_buf:
            if ioutils.is_directory(
                path_or_data=source, storage_options=storage_options
            ):
                fs = ioutils._ensure_filesystem(
                    passed_filesystem=None,
                    path=source,
                    storage_options=storage_options,
                )
                source = ioutils.stringify_pathlike(source)
                source = fs.sep.join([source, "*.json"])

            tmp_source, compression = ioutils.get_reader_filepath_or_buffer(
                path_or_data=source,
                compression=compression,
                iotypes=(BytesIO, StringIO),
                allow_raw_text_input=True,
                storage_options=storage_options,
            )
            if isinstance(tmp_source, list):
                filepaths_or_buffers.extend(tmp_source)
            else:
                filepaths_or_buffers.append(tmp_source)

        df = libjson.read_json(
            filepaths_or_buffers,
            dtype,
            lines,
            compression,
            byte_range,
            engine == "cudf_legacy",
            keep_quotes,
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
        dtype = {} if dtype is True else dtype
        unspecified_dtypes = {
            name: df._dtypes[name]
            for name in df._column_names
            if name not in dtype
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


@ioutils.doc_to_json()
def to_json(cudf_val, path_or_buf=None, *args, **kwargs):
    """{docstring}"""

    warnings.warn(
        "Using CPU via Pandas to write JSON dataset, this may "
        "be GPU accelerated in the future"
    )
    pd_value = cudf_val.to_pandas(nullable=True)
    return pd.io.json.to_json(path_or_buf, pd_value, *args, **kwargs)
