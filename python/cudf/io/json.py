# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.bindings.json import cpp_read_json

import cudf
from cudf.utils import ioutils

import pandas as pd
import warnings


@ioutils.doc_read_json()
def read_json(path_or_buf, engine='cudf', dtype=None, lines=False,
              compression='infer', byte_range=None, *args, **kwargs):

    """
    Load and parse a JSON file into a DataFrame

    Parameters
    ----------
    path_or_buf : str
        Path of file to be read or a str containing the file.
    engine : {{ 'cudf', 'pyarrow' }}, default 'cudf'
        Parser engine to use.
    dtype : list of str or dict of {{col: dtype}}, default None
        List of data types in the same order of the column names
        or a dictionary with column_name:dtype (Pandas style).
    lines : bool, default False
        Read the file as a json object per line
    compression : {{'infer', 'gzip', 'zip', None}}, default 'infer'
        For on-the-fly decompression of on-disk data. If ‘infer’, then detect
        compression from the following extensions: ‘.gz’,‘.zip’ (otherwise no
        decompression). If using ‘zip’, the ZIP file must contain only one
        data file to be read in, otherwise the first non-zero-sized file will
        be used. Set to None for no decompression.
    byte_range : list or tuple, default None
        Byte range within the input file to be read. The first number is the
        offset in bytes, the second number is the range size in bytes. Set the
        size to zero to read all data after the offset location. Reads the row
        that starts before or at the end of the range, even if it ends after
        the end of the range.

    Returns
    -------
    GPU ``DataFrame`` object.
    """

    if lines and engine == 'cudf':
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
