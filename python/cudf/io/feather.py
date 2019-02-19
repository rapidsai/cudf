# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf.dataframe.dataframe import DataFrame
from cudf.utils import ioutils

from pyarrow import feather
import warnings


@ioutils.doc_read_feather()
def read_feather(path, *args, **kwargs):
    """{docstring}"""

    warnings.warn("Using CPU via PyArrow to read feather dataset, this may "
                  "be GPU accelerated in the future")
    pa_table = feather.read_table(path, *args, **kwargs)
    return DataFrame.from_arrow(pa_table)


@ioutils.doc_to_feather()
def to_feather(df, path, *args, **kwargs):
    """{docstring}"""
    warnings.warn("Using CPU via PyArrow to write Feather dataset, this may "
                  "be GPU accelerated in the future")
    # Feather doesn't support using an index
    pa_table = df.to_arrow(preserve_index=False)
    feather_writer = feather.FeatherWriter(path)
    for i, name in enumerate(pa_table.schema.names):
        col = pa_table[i]
        feather.check_chunked_overflow(col)
        feather_writer.writer.write_array(name, col.data.chunk(0))
    feather_writer.writer.close()
