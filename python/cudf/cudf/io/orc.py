# Copyright (c) 2019, NVIDIA CORPORATION.
import warnings

import pyarrow as pa
from pyarrow import orc as orc

import cudf
from cudf import _lib as libcudf
from cudf.utils import ioutils


@ioutils.doc_read_orc_metadata()
def read_orc_metadata(path):
    """{docstring}"""

    orc_file = orc.ORCFile(path)

    num_rows = orc_file.nrows
    num_stripes = orc_file.nstripes
    col_names = orc_file.schema.names

    return num_rows, num_stripes, col_names


@ioutils.doc_read_orc()
def read_orc(
    filepath_or_buffer,
    engine="cudf",
    columns=None,
    stripe=None,
    stripe_count=None,
    skip_rows=None,
    num_rows=None,
    use_index=True,
    decimals_as_float=True,
    force_decimal_scale=None,
    timestamp_type=None,
    **kwargs,
):
    """{docstring}"""

    from cudf import DataFrame

    filepath_or_buffer, compression = ioutils.get_filepath_or_buffer(
        filepath_or_buffer, None, **kwargs
    )
    if compression is not None:
        ValueError("URL content-encoding decompression is not supported")

    if engine == "cudf":
        df = DataFrame._from_table(
            libcudf.orc.read_orc(
                filepath_or_buffer,
                columns,
                stripe,
                stripe_count,
                skip_rows,
                num_rows,
                use_index,
                decimals_as_float,
                force_decimal_scale,
                timestamp_type,
            )
        )
    else:
        warnings.warn("Using CPU via PyArrow to read ORC dataset.")
        orc_file = orc.ORCFile(filepath_or_buffer)
        if stripe is not None:
            pa_table = orc_file.read_stripe(stripe, columns)
            if isinstance(pa_table, pa.RecordBatch):
                pa_table = pa.Table.from_batches([pa_table])
        else:
            pa_table = orc_file.read(columns=columns)
        df = cudf.DataFrame.from_arrow(pa_table)

    return df


@ioutils.doc_to_orc()
def to_orc(df, fname, compression=None, enable_statistics=False):
    """{docstring}"""

    libcudf.orc.write_orc(df, fname, compression, enable_statistics)
