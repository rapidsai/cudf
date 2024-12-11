# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import pylibcudf as plc

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column


@acquire_spill_lock()
def url_decode(Column source_strings):
    """
    Decode each string in column. No format checking is performed.

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    URL decoded string column
    """
    plc_column = plc.strings.convert.convert_urls.url_decode(
        source_strings.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def url_encode(Column source_strings):
    """
    Encode each string in column. No format checking is performed.
    All characters are encoded except for ASCII letters, digits,
    and these characters: '.','_','-','~'. Encoding converts to
    hex using UTF-8 encoded bytes.

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    URL encoded string column
    """
    plc_column = plc.strings.convert.convert_urls.url_encode(
        source_strings.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_column)
