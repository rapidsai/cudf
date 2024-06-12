# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.strings.convert.convert_urls cimport (
    url_decode as cpp_url_decode,
    url_encode as cpp_url_encode,
)


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
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_url_decode(
            source_view
        ))

    return Column.from_unique_ptr(
        move(c_result)
    )


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
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_url_encode(
            source_view
        ))

    return Column.from_unique_ptr(
        move(c_result)
    )
