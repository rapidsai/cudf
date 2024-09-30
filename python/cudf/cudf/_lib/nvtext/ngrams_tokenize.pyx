# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.ngrams_tokenize cimport (
    ngrams_tokenize as cpp_ngrams_tokenize,
)
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type

from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def ngrams_tokenize(
    Column strings,
    int ngrams,
    object py_delimiter,
    object py_separator
):

    cdef DeviceScalar delimiter = py_delimiter.device_value
    cdef DeviceScalar separator = py_separator.device_value

    cdef column_view c_strings = strings.view()
    cdef size_type c_ngrams = ngrams
    cdef const string_scalar* c_separator = <const string_scalar*>separator\
        .get_raw_ptr()
    cdef const string_scalar* c_delimiter = <const string_scalar*>delimiter\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_ngrams_tokenize(
                c_strings,
                c_ngrams,
                c_delimiter[0],
                c_separator[0]
            )
        )

    return Column.from_unique_ptr(move(c_result))
