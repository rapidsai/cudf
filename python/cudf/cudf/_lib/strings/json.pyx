# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.json cimport (
    get_json_object as cpp_get_json_object,
    get_json_object_options,
)
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def get_json_object(
        Column col, object py_json_path, GetJsonObjectOptions options):
    """
    Apply a JSONPath string to all rows in an input column
    of json strings.
    """
    cdef unique_ptr[column] c_result

    cdef column_view col_view = col.view()
    cdef DeviceScalar json_path = py_json_path.device_value

    cdef const string_scalar* scalar_json_path = <const string_scalar*>(
        json_path.get_raw_ptr()
    )

    with nogil:
        c_result = move(cpp_get_json_object(
            col_view,
            scalar_json_path[0],
            options.options,
        ))

    return Column.from_unique_ptr(move(c_result))


cdef class GetJsonObjectOptions:
    cdef get_json_object_options options

    def __init__(
        self,
        *,
        allow_single_quotes=False,
        strip_quotes_from_single_strings=True,
        missing_fields_as_nulls=False
    ):
        self.options.set_allow_single_quotes(allow_single_quotes)
        self.options.set_strip_quotes_from_single_strings(
            strip_quotes_from_single_strings
        )
        self.options.set_missing_fields_as_nulls(missing_fields_as_nulls)

    @property
    def allow_single_quotes(self):
        return self.options.get_allow_single_quotes()

    @property
    def strip_quotes_from_single_strings(self):
        return self.options.get_strip_quotes_from_single_strings()

    @property
    def missing_fields_as_nulls(self):
        return self.options.get_missing_fields_as_nulls()

    @allow_single_quotes.setter
    def allow_single_quotes(self, val):
        self.options.set_allow_single_quotes(val)

    @strip_quotes_from_single_strings.setter
    def strip_quotes_from_single_strings(self, val):
        self.options.set_strip_quotes_from_single_strings(val)

    @missing_fields_as_nulls.setter
    def missing_fields_as_nulls(self, val):
        self.options.set_missing_fields_as_nulls(val)
