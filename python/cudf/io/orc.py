# Copyright (c) 2019, NVIDIA CORPORATION.

from libgdf_cffi import libgdf, ffi
import nvstrings

from cudf.dataframe.column import Column
from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.datetime import DatetimeColumn
from cudf.dataframe.numerical import NumericalColumn
from cudf.utils import ioutils

import pyarrow.orc as orc
import numpy as np

import warnings
import os
import errno


def _wrap_string(text):
    if text is None:
        return ffi.NULL
    else:
        return ffi.new("char[]", text.encode())


@ioutils.doc_read_orc()
def read_orc(path, engine='pyarrow', columns=None, **kwargs):
    """{docstring}"""

    if engine == 'cudf':
        # Setup arguments
        orc_reader = ffi.new('orc_read_arg*')

        if not os.path.isfile(path) and not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT), path)
        source_ptr = _wrap_string(str(path))
        orc_reader.source_type = libgdf.FILE_PATH
        orc_reader.source = source_ptr

        if columns is not None:
            arr_cols = []
            for col in columns:
                arr_cols.append(_wrap_string(col))
            use_cols_ptr = ffi.new('char*[]', arr_cols)
            orc_reader.use_cols = use_cols_ptr
            orc_reader.use_cols_len = len(usecols)

        # Call to libcudf
        libgdf.read_orc(orc_reader)
        out = orc_reader.data
        if out == ffi.NULL:
            raise ValueError("Failed to parse data")

        # Extract parsed columns
        outcols = []
        new_names = []
        for i in range(orc_reader.num_cols_out):
            if out[i].dtype == libgdf.GDF_STRING:
                ptr = int(ffi.cast("uintptr_t", out[i].data))
                new_names.append(ffi.string(out[i].col_name).decode())
                outcols.append(nvstrings.bind_cpointer(ptr))
            else:
                newcol = Column.from_cffi_view(out[i])
                new_names.append(ffi.string(out[i].col_name).decode())
                if newcol.dtype.type == np.datetime64:
                    outcols.append(
                        newcol.view(DatetimeColumn, dtype='datetime64[ms]')
                    )
                else:
                    outcols.append(
                        newcol.view(NumericalColumn, dtype=newcol.dtype)
                    )

        # Construct dataframe from columns
        df = DataFrame()
        for k, v in zip(new_names, outcols):
            df[k] = v

        # Set column to use as row indexes if available
        if orc_reader.index_col != ffi.NULL:
            df = df.set_index(df.columns[orc_reader.index_col[0]])
    else:
        warnings.warn("Using CPU via PyArrow to read ORC dataset.")
        orc_file = orc.ORCFile(path)
        pa_table = orc_file.read(columns=columns)
        df = DataFrame.from_arrow(pa_table)

    return df
