# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from io import BytesIO, StringIO, TextIOBase

import pylibcudf as plc

import cudf
from cudf.utils import ioutils
from cudf.utils.performance_tracking import _performance_tracking


@_performance_tracking
@ioutils.doc_read_text()
def read_text(
    filepath_or_buffer,
    delimiter=None,
    byte_range=None,
    strip_delimiters=False,
    compression=None,
    compression_offsets=None,
    storage_options=None,
):
    """{docstring}"""

    if delimiter is None:
        raise ValueError("delimiter needs to be provided")

    filepath_or_buffer = ioutils.get_reader_filepath_or_buffer(
        path_or_data=filepath_or_buffer,
        iotypes=(BytesIO, StringIO),
        storage_options=storage_options,
    )
    filepath_or_buffer = ioutils._select_single_source(
        filepath_or_buffer, "read_text"
    )

    if compression is None:
        if isinstance(filepath_or_buffer, TextIOBase):
            datasource = plc.io.text.make_source(filepath_or_buffer.read())
        else:
            datasource = plc.io.text.make_source_from_file(filepath_or_buffer)
    elif compression == "bgzip":
        if isinstance(filepath_or_buffer, TextIOBase):
            raise ValueError("bgzip compression requires a file path")
        if compression_offsets is not None:
            if len(compression_offsets) != 2:
                raise ValueError(
                    "Compression offsets need to consist of two elements"
                )
            datasource = plc.io.text.make_source_from_bgzip_file(
                filepath_or_buffer,
                compression_offsets[0],
                compression_offsets[1],
            )
        else:
            datasource = plc.io.text.make_source_from_bgzip_file(
                filepath_or_buffer,
            )
    else:
        raise ValueError("Only bgzip compression is supported at the moment")

    options = plc.io.text.ParseOptions(
        byte_range=byte_range, strip_delimiters=strip_delimiters
    )
    plc_column = plc.io.text.multibyte_split(datasource, delimiter, options)
    result = cudf._lib.column.Column.from_pylibcudf(plc_column)

    return cudf.Series._from_column(result)
