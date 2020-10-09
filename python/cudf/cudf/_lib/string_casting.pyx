# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np

from cudf._lib.column cimport Column
from cudf._lib.scalar import as_scalar
from cudf._lib.scalar cimport Scalar
from cudf._lib.types import np_to_cudf_types
from cudf._lib.types cimport underlying_type_t_type_id

from cudf.core.column.column import as_column

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.strings.convert.convert_booleans cimport (
    to_booleans as cpp_to_booleans,
    from_booleans as cpp_from_booleans
)
from cudf._lib.cpp.strings.convert.convert_datetime cimport (
    to_timestamps as cpp_to_timestamps,
    from_timestamps as cpp_from_timestamps
)
from cudf._lib.cpp.strings.convert.convert_floats cimport (
    to_floats as cpp_to_floats,
    from_floats as cpp_from_floats
)
from cudf._lib.cpp.strings.convert.convert_integers cimport (
    to_integers as cpp_to_integers,
    from_integers as cpp_from_integers,
    hex_to_integers as cpp_hex_to_integers,
    is_hex as cpp_is_hex
)
from cudf._lib.cpp.strings.convert.convert_ipv4 cimport (
    ipv4_to_integers as cpp_ipv4_to_integers,
    integers_to_ipv4 as cpp_integers_to_ipv4,
    is_ipv4 as cpp_is_ipv4
)
from cudf._lib.cpp.strings.convert.convert_urls cimport (
    url_encode as cpp_url_encode,
    url_decode as cpp_url_decode
)
from cudf._lib.cpp.strings.convert.convert_durations cimport (
    to_durations as cpp_to_durations,
    from_durations as cpp_from_durations
)
from cudf._lib.cpp.types cimport (
    type_id,
    data_type,
)

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.string cimport string


def floating_to_string(Column input_col):
    cdef column_view input_column_view = input_col.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_from_floats(
                input_column_view))

    return Column.from_unique_ptr(move(c_result))


def string_to_floating(Column input_col, object out_type):
    cdef column_view input_column_view = input_col.view()
    cdef unique_ptr[column] c_result
    cdef type_id tid = <type_id> (
        <underlying_type_t_type_id> (
            np_to_cudf_types[out_type]
        )
    )
    cdef data_type c_out_type = data_type(tid)
    with nogil:
        c_result = move(
            cpp_to_floats(
                input_column_view,
                c_out_type))

    return Column.from_unique_ptr(move(c_result))


def dtos(Column input_col, **kwargs):
    """
    Converting/Casting input column of type double to string column

    Parameters
    ----------
    input_col : input column of type double

    Returns
    -------
    A Column with double values cast to string
    """

    return floating_to_string(input_col)


def stod(Column input_col, **kwargs):
    """
    Converting/Casting input column of type string to double

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to double
    """

    return string_to_floating(input_col, np.dtype("float64"))


def ftos(Column input_col, **kwargs):
    """
    Converting/Casting input column of type float to string column

    Parameters
    ----------
    input_col : input column of type double

    Returns
    -------
    A Column with float values cast to string
    """

    return floating_to_string(input_col)


def stof(Column input_col, **kwargs):
    """
    Converting/Casting input column of type string to float

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to float
    """

    return string_to_floating(input_col, np.dtype("float32"))


def integer_to_string(Column input_col):
    cdef column_view input_column_view = input_col.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_from_integers(
                input_column_view))

    return Column.from_unique_ptr(move(c_result))


def string_to_integer(Column input_col, object out_type):
    cdef column_view input_column_view = input_col.view()
    cdef unique_ptr[column] c_result
    cdef type_id tid = <type_id> (
        <underlying_type_t_type_id> (
            np_to_cudf_types[out_type]
        )
    )
    cdef data_type c_out_type = data_type(tid)
    with nogil:
        c_result = move(
            cpp_to_integers(
                input_column_view,
                c_out_type))

    return Column.from_unique_ptr(move(c_result))


def i8tos(Column input_col, **kwargs):
    """
    Converting/Casting input column of type int8 to string column

    Parameters
    ----------
    input_col : input column of type int8

    Returns
    -------
    A Column with int8 values cast to string
    """

    return integer_to_string(input_col)


def stoi8(Column input_col, **kwargs):
    """
    Converting/Casting input column of type string to int8

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to int8
    """

    return string_to_integer(input_col, np.dtype("int8"))


def i16tos(Column input_col, **kwargs):
    """
    Converting/Casting input column of type int16 to string column

    Parameters
    ----------
    input_col : input column of type int16

    Returns
    -------
    A Column with int16 values cast to string
    """

    return integer_to_string(input_col)


def stoi16(Column input_col, **kwargs):
    """
    Converting/Casting input column of type string to int16

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to int16
    """

    return string_to_integer(input_col, np.dtype("int16"))


def itos(Column input_col, **kwargs):
    """
    Converting/Casting input column of type int32 to string column

    Parameters
    ----------
    input_col : input column of type int32

    Returns
    -------
    A Column with int32 values cast to string
    """

    return integer_to_string(input_col)


def stoi(Column input_col, **kwargs):
    """
    Converting/Casting input column of type string to int32

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to int32
    """

    return string_to_integer(input_col, np.dtype("int32"))


def ltos(Column input_col, **kwargs):
    """
    Converting/Casting input column of type int64 to string column

    Parameters
    ----------
    input_col : input column of type int64

    Returns
    -------
    A Column with int64 values cast to string
    """

    return integer_to_string(input_col)


def stol(Column input_col, **kwargs):
    """
    Converting/Casting input column of type string to int64

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to int64
    """

    return string_to_integer(input_col, np.dtype("int64"))


def ui8tos(Column input_col, **kwargs):
    """
    Converting/Casting input column of type uint8 to string column

    Parameters
    ----------
    input_col : input column of type uint8

    Returns
    -------
    A Column with uint8 values cast to string
    """

    return integer_to_string(input_col)


def stoui8(Column input_col, **kwargs):
    """
    Converting/Casting input column of type string to uint8

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to uint8
    """

    return string_to_integer(input_col, np.dtype("uint8"))


def ui16tos(Column input_col, **kwargs):
    """
    Converting/Casting input column of type uint16 to string column

    Parameters
    ----------
    input_col : input column of type uint16

    Returns
    -------
    A Column with uint16 values cast to string
    """

    return integer_to_string(input_col)


def stoui16(Column input_col, **kwargs):
    """
    Converting/Casting input column of type string to uint16

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to uint16
    """

    return string_to_integer(input_col, np.dtype("uint16"))


def uitos(Column input_col, **kwargs):
    """
    Converting/Casting input column of type uint32 to string column

    Parameters
    ----------
    input_col : input column of type uint32

    Returns
    -------
    A Column with uint32 values cast to string
    """

    return integer_to_string(input_col)


def stoui(Column input_col, **kwargs):
    """
    Converting/Casting input column of type string to uint32

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to uint32
    """

    return string_to_integer(input_col, np.dtype("uint32"))


def ultos(Column input_col, **kwargs):
    """
    Converting/Casting input column of type uint64 to string column

    Parameters
    ----------
    input_col : input column of type uint64

    Returns
    -------
    A Column with uint64 values cast to string
    """

    return integer_to_string(input_col)


def stoul(Column input_col, **kwargs):
    """
    Converting/Casting input column of type string to uint64

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to uint64
    """

    return string_to_integer(input_col, np.dtype("uint64"))


def _to_booleans(Column input_col, object string_true="True"):
    """
    Converting/Casting input column of type string to boolean column

    Parameters
    ----------
    input_col : input column of type string
    string_true : string that represents True

    Returns
    -------
    A Column with string values cast to boolean
    """

    cdef Scalar str_true = as_scalar(string_true)
    cdef column_view input_column_view = input_col.view()
    cdef string_scalar* string_scalar_true = <string_scalar*>(
        str_true.c_value.get())
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_to_booleans(
                input_column_view,
                string_scalar_true[0]))

    return Column.from_unique_ptr(move(c_result))


def to_booleans(Column input_col, **kwargs):

    return _to_booleans(input_col)


def _from_booleans(
        Column input_col,
        object string_true="True",
        object string_false="False"):
    """
    Converting/Casting input column of type boolean to string column

    Parameters
    ----------
    input_col : input column of type boolean
    string_true : string that represents True
    string_false : string that represents False

    Returns
    -------
    A Column with boolean values cast to string
    """

    cdef Scalar str_true = as_scalar(string_true)
    cdef Scalar str_false = as_scalar(string_false)
    cdef column_view input_column_view = input_col.view()
    cdef string_scalar* string_scalar_true = <string_scalar*>(
        str_true.c_value.get())
    cdef string_scalar* string_scalar_false = <string_scalar*>(
        str_false.c_value.get())
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_from_booleans(
                input_column_view,
                string_scalar_true[0],
                string_scalar_false[0]))

    return Column.from_unique_ptr(move(c_result))


def from_booleans(Column input_col, **kwargs):

    return _from_booleans(input_col)


def int2timestamp(
        Column input_col,
        **kwargs):
    """
    Converting/Casting input date-time column to string
    column with specified format

    Parameters
    ----------
    input_col : input column of type timestamp in integer format

    Returns
    -------
    A Column with date-time represented in string format

    """

    cdef column_view input_column_view = input_col.view()
    cdef string c_timestamp_format = kwargs.get(
        'format', "%Y-%m-%dT%H:%M:%SZ").encode('UTF-8')
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_from_timestamps(
                input_column_view,
                c_timestamp_format))

    return Column.from_unique_ptr(move(c_result))


def timestamp2int(
        Column input_col,
        **kwargs):
    """
    Converting/Casting input string column to date-time column with specified
    timestamp_format

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with string represented in date-time format

    """
    if input_col.size == 0:
        return as_column([], dtype=kwargs.get('dtype'))
    cdef column_view input_column_view = input_col.view()
    cdef type_id tid = <type_id> (
        <underlying_type_t_type_id> (
            np_to_cudf_types[kwargs.get('dtype')]
        )
    )
    cdef data_type out_type = data_type(tid)
    cdef string c_timestamp_format = kwargs.get('format').encode('UTF-8')
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_to_timestamps(
                input_column_view,
                out_type,
                c_timestamp_format))

    return Column.from_unique_ptr(move(c_result))


def timedelta2int(
        Column input_col,
        **kwargs):
    """
    Converting/Casting input string column to TimeDelta column with specified
    format

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with string represented in TimeDelta format

    """
    if input_col.size == 0:
        return as_column([], dtype=kwargs.get('dtype'))
    cdef column_view input_column_view = input_col.view()
    cdef type_id tid = <type_id> (
        <underlying_type_t_type_id> (
            np_to_cudf_types[kwargs.get('dtype')]
        )
    )
    cdef data_type out_type = data_type(tid)
    cdef string c_duration_format = kwargs.get('format').encode('UTF-8')
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_to_durations(
                input_column_view,
                out_type,
                c_duration_format))

    return Column.from_unique_ptr(move(c_result))


def int2timedelta(
        Column input_col,
        **kwargs):
    """
    Converting/Casting input Timedelta column to string
    column with specified format

    Parameters
    ----------
    input_col : input column of type Timedelta in integer format

    Returns
    -------
    A Column with Timedelta represented in string format

    """

    cdef column_view input_column_view = input_col.view()
    cdef string c_duration_format = kwargs.get(
        'format', "%D days %H:%M:%S").encode('UTF-8')
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_from_durations(
                input_column_view,
                c_duration_format))

    return Column.from_unique_ptr(move(c_result))


def int2ip(Column input_col, **kwargs):
    """
    Converting/Casting integer column to string column in ipv4 format

    Parameters
    ----------
    input_col : input integer column

    Returns
    -------
    A Column with integer represented in string ipv4 format

    """

    cdef column_view input_column_view = input_col.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_integers_to_ipv4(input_column_view))

    return Column.from_unique_ptr(move(c_result))


def ip2int(Column input_col, **kwargs):
    """
    Converting string ipv4 column to integer column

    Parameters
    ----------
    input_col : input string column

    Returns
    -------
    A Column with ipv4 represented as integer

    """

    cdef column_view input_column_view = input_col.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_ipv4_to_integers(input_column_view))

    return Column.from_unique_ptr(move(c_result))


def is_ipv4(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that have strings in IPv4 format. This format is nnn.nnn.nnn.nnn
    where nnn is integer digits in [0,255].
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_is_ipv4(
            source_view
        ))

    return Column.from_unique_ptr(move(c_result))


def htoi(Column input_col, **kwargs):
    """
    Converting input column of type string having hex values
    to integer of out_type

    Parameters
    ----------
    input_col : input column of type string
    out_type : The type of integer column expected

    Returns
    -------
    A Column of integers parsed from hexadecimal string values.
    """

    cdef column_view input_column_view = input_col.view()
    cdef type_id tid = <type_id> (
        <underlying_type_t_type_id> (
            np_to_cudf_types[kwargs.get('dtype', np.dtype("int64"))]
        )
    )
    cdef data_type c_out_type = data_type(tid)

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_hex_to_integers(input_column_view,
                                c_out_type))

    return Column.from_unique_ptr(move(c_result))


def is_hex(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that have hex characters.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_is_hex(
            source_view
        ))

    return Column.from_unique_ptr(move(c_result))
