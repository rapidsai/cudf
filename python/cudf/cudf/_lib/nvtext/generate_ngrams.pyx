# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.nvtext.generate_ngrams cimport (
    generate_character_ngrams as cpp_generate_character_ngrams,
    generate_ngrams as cpp_generate_ngrams,
    hash_character_ngrams as cpp_hash_character_ngrams,
)
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def generate_ngrams(Column strings, int ngrams, object py_separator):

    cdef DeviceScalar separator = py_separator.device_value

    cdef column_view c_strings = strings.view()
    cdef size_type c_ngrams = ngrams
    cdef const string_scalar* c_separator = <const string_scalar*>separator\
        .get_raw_ptr()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_generate_ngrams(
                c_strings,
                c_ngrams,
                c_separator[0]
            )
        )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def generate_character_ngrams(Column strings, int ngrams):
    cdef column_view c_strings = strings.view()
    cdef size_type c_ngrams = ngrams
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_generate_character_ngrams(
                c_strings,
                c_ngrams
            )
        )

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def hash_character_ngrams(Column strings, int ngrams):
    cdef column_view c_strings = strings.view()
    cdef size_type c_ngrams = ngrams
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_hash_character_ngrams(
                c_strings,
                c_ngrams
            )
        )

    return Column.from_unique_ptr(move(c_result))
