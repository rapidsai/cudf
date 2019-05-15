# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.bindings.json import cpp_read_json

import cudf
from cudf.utils import ioutils

import pandas as pd
import warnings


@ioutils.doc_read_json()
def read_json(path_or_buf, engine='auto', dtype=True, lines=False,
              compression='infer', byte_range=None, *args, **kwargs):
    """{docstring}"""

    if engine == 'cudf' and not lines:
        raise ValueError("cudf engine only supports JSON Lines format")
    if engine == 'auto':
        engine = 'cudf' if lines else 'pandas'

    if engine == 'cudf':
        df = cpp_read_json(path_or_buf, dtype, lines, compression, byte_range)
    else:
        warnings.warn("Using CPU via Pandas to read JSON dataset, this may "
                      "be GPU accelerated in the future")
        pd_value = pd.read_json(path_or_buf, lines=lines, dtype=dtype,
                                compression=compression, *args, **kwargs)
        df = cudf.from_pandas(pd_value)

    return df


@ioutils.doc_to_json()
def to_json(cudf_val, path_or_buf=None, *args, **kwargs):
    """{docstring}"""

    warnings.warn("Using CPU via Pandas to write JSON dataset, this may "
                  "be GPU accelerated in the future")
    pd_value = cudf_val.to_pandas()
    pd.io.json.to_json(
        path_or_buf,
        pd_value,
        *args,
        **kwargs
    )
