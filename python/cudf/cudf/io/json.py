# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
    dtype=True,
    lines=False,
    compression="infer",
    byte_range=None,
    *args,
    **kwargs,
):
    """{docstring}"""

    if engine == "cudf" and not lines:
        raise ValueError("cudf engine only supports JSON Lines format")
    if engine == "auto":
        engine = "cudf" if lines else "pandas"
    if engine == "cudf":
        # Multiple sources are passed as a list. If a single source is passed,
        # wrap it in a list for unified processing downstream.
        if not is_list_like(path_or_buf):
            path_or_buf = [path_or_buf]

        filepaths_or_buffers = []
        for source in path_or_buf:
            if ioutils.is_directory(source, **kwargs):
                fs = ioutils._ensure_filesystem(
                    passed_filesystem=None, path=source, **kwargs
                )
                source = ioutils.stringify_pathlike(source)
                source = fs.sep.join([source, "*.json"])

            tmp_source, compression = ioutils.get_reader_filepath_or_buffer(
                path_or_data=source,
                compression=compression,
                iotypes=(BytesIO, StringIO),
                allow_raw_text_input=True,
                **kwargs,
            )
            if isinstance(tmp_source, list):
                filepaths_or_buffers.extend(tmp_source)
            else:
                filepaths_or_buffers.append(tmp_source)

        df = libjson.read_json(
            filepaths_or_buffers, dtype, lines, compression, byte_range
        )
    else:
        warnings.warn(
            "Using CPU via Pandas to read JSON dataset, this may "
            "be GPU accelerated in the future"
        )

        if not ioutils.ensure_single_filepath_or_buffer(
            path_or_data=path_or_buf,
            **kwargs,
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
            **kwargs,
        )

        if kwargs.get("orient") == "table":
            pd_value = pd.read_json(
                path_or_buf,
                lines=lines,
                compression=compression,
                *args,
                **kwargs,
            )
        else:
            pd_value = pd.read_json(
                path_or_buf,
                lines=lines,
                dtype=dtype,
                compression=compression,
                *args,
                **kwargs,
            )
        df = cudf.from_pandas(pd_value)

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
