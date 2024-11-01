# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool

from io import TextIOBase

import pylibcudf as plc

from cudf._lib.column cimport Column


def read_text(object filepaths_or_buffers,
              str delimiter,
              object byte_range,
              bool strip_delimiters,
              object compression,
              object compression_offsets):
    """
    Cython function to call into libcudf API, see `multibyte_split`.

    See Also
    --------
    cudf.io.text.read_text
    """
    if compression is None:
        if isinstance(filepaths_or_buffers, TextIOBase):
            datasource = plc.io.text.make_source(filepaths_or_buffers.read())
        else:
            datasource = plc.io.text.make_source_from_file(filepaths_or_buffers)
    elif compression == "bgzip":
        if isinstance(filepaths_or_buffers, TextIOBase):
            raise ValueError("bgzip compression requires a file path")
        if compression_offsets is not None:
            if len(compression_offsets) != 2:
                raise ValueError(
                    "compression offsets need to consist of two elements")
            datasource = plc.io.text.make_source_from_bgzip_file(
                filepaths_or_buffers,
                compression_offsets[0],
                compression_offsets[1]
            )
        else:
            datasource = plc.io.text.make_source_from_bgzip_file(
                filepaths_or_buffers,
            )
    else:
        raise ValueError("Only bgzip compression is supported at the moment")

    options = plc.io.text.ParseOptions(
        byte_range=byte_range, strip_delimiters=strip_delimiters
    )
    plc_column = plc.io.text.multibyte_split(datasource, delimiter, options)
    return Column.from_pylibcudf(plc_column)
