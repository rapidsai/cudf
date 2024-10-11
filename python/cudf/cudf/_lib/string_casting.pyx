# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf._lib.column cimport Column

from cudf._lib.scalar import as_device_scalar
from cudf._lib.types import SUPPORTED_NUMPY_TO_LIBCUDF_TYPES

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings.convert.convert_integers cimport (
    from_integers as cpp_from_integers,
    hex_to_integers as cpp_hex_to_integers,
    integers_to_hex as cpp_integers_to_hex,
    is_hex as cpp_is_hex,
    to_integers as cpp_to_integers,
)
from pylibcudf.libcudf.types cimport data_type, type_id

from cudf._lib.types cimport underlying_type_t_type_id

import pylibcudf as plc

import cudf

from cudf._lib.types cimport dtype_to_pylibcudf_type


def floating_to_string(Column input_col):
    plc_column = plc.strings.convert.convert_floats.from_floats(
        input_col.to_pylibcudf(mode="read"),
    )
    return Column.from_pylibcudf(plc_column)


def string_to_floating(Column input_col, object out_type):
    plc_column = plc.strings.convert.convert_floats.to_floats(
        input_col.to_pylibcudf(mode="read"),
        dtype_to_pylibcudf_type(out_type)
    )
    return Column.from_pylibcudf(plc_column)


def dtos(Column input_col):
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


def stod(Column input_col):
    """
    Converting/Casting input column of type string to double

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to double
    """

    return string_to_floating(input_col, cudf.dtype("float64"))


def ftos(Column input_col):
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


def stof(Column input_col):
    """
    Converting/Casting input column of type string to float

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to float
    """

    return string_to_floating(input_col, cudf.dtype("float32"))


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
            SUPPORTED_NUMPY_TO_LIBCUDF_TYPES[out_type]
        )
    )
    cdef data_type c_out_type = data_type(tid)
    with nogil:
        c_result = move(
            cpp_to_integers(
                input_column_view,
                c_out_type))

    return Column.from_unique_ptr(move(c_result))


def i8tos(Column input_col):
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


def stoi8(Column input_col):
    """
    Converting/Casting input column of type string to int8

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to int8
    """

    return string_to_integer(input_col, cudf.dtype("int8"))


def i16tos(Column input_col):
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


def stoi16(Column input_col):
    """
    Converting/Casting input column of type string to int16

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to int16
    """

    return string_to_integer(input_col, cudf.dtype("int16"))


def itos(Column input_col):
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


def stoi(Column input_col):
    """
    Converting/Casting input column of type string to int32

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to int32
    """

    return string_to_integer(input_col, cudf.dtype("int32"))


def ltos(Column input_col):
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


def stol(Column input_col):
    """
    Converting/Casting input column of type string to int64

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to int64
    """

    return string_to_integer(input_col, cudf.dtype("int64"))


def ui8tos(Column input_col):
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


def stoui8(Column input_col):
    """
    Converting/Casting input column of type string to uint8

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to uint8
    """

    return string_to_integer(input_col, cudf.dtype("uint8"))


def ui16tos(Column input_col):
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


def stoui16(Column input_col):
    """
    Converting/Casting input column of type string to uint16

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to uint16
    """

    return string_to_integer(input_col, cudf.dtype("uint16"))


def uitos(Column input_col):
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


def stoui(Column input_col):
    """
    Converting/Casting input column of type string to uint32

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to uint32
    """

    return string_to_integer(input_col, cudf.dtype("uint32"))


def ultos(Column input_col):
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


def stoul(Column input_col):
    """
    Converting/Casting input column of type string to uint64

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column with strings cast to uint64
    """

    return string_to_integer(input_col, cudf.dtype("uint64"))


def to_booleans(Column input_col):
    plc_column = plc.strings.convert.convert_booleans.to_booleans(
        input_col.to_pylibcudf(mode="read"),
        as_device_scalar("True").c_value,
    )
    return Column.from_pylibcudf(plc_column)


def from_booleans(Column input_col):
    plc_column = plc.strings.convert.convert_booleans.from_booleans(
        input_col.to_pylibcudf(mode="read"),
        as_device_scalar("True").c_value,
        as_device_scalar("False").c_value,
    )
    return Column.from_pylibcudf(plc_column)


def int2timestamp(
        Column input_col,
        str format,
        Column names):
    """
    Converting/Casting input date-time column to string
    column with specified format

    Parameters
    ----------
    input_col : input column of type timestamp in integer format
    format : The string specifying output format
    names : The string names to use for weekdays ("%a", "%A") and
    months ("%b", "%B")

    Returns
    -------
    A Column with date-time represented in string format

    """
    return Column.from_pylibcudf(
        plc.strings.convert.convert_datetime.from_timestamps(
            input_col.to_pylibcudf(mode="read"),
            format,
            names.to_pylibcudf(mode="read")
        )
    )


def timestamp2int(Column input_col, dtype, format):
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
    dtype = dtype_to_pylibcudf_type(dtype)
    return Column.from_pylibcudf(
        plc.strings.convert.convert_datetime.to_timestamps(
            input_col.to_pylibcudf(mode="read"),
            dtype,
            format
        )
    )


def istimestamp(Column input_col, str format):
    """
    Check input string column matches the specified timestamp format

    Parameters
    ----------
    input_col : input column of type string

    format : format string of timestamp specifiers

    Returns
    -------
    A Column of boolean values identifying strings that matched the format.

    """
    if input_col.size == 0:
        return cudf.core.column.column_empty(0, dtype=cudf.dtype("bool"))
    plc_column = plc.strings.convert.convert_datetime.is_timestamp(
        input_col.to_pylibcudf(mode="read"),
        format
    )
    return Column.from_pylibcudf(plc_column)


def timedelta2int(Column input_col, dtype, format):
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
    dtype = dtype_to_pylibcudf_type(dtype)
    return Column.from_pylibcudf(
        plc.strings.convert.convert_durations.to_durations(
            input_col.to_pylibcudf(mode="read"),
            dtype,
            format
        )
    )


def int2timedelta(Column input_col, str format):
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
    return Column.from_pylibcudf(
        plc.strings.convert.convert_durations.from_durations(
            input_col.to_pylibcudf(mode="read"),
            format
        )
    )


def int2ip(Column input_col):
    """
    Converting/Casting integer column to string column in ipv4 format

    Parameters
    ----------
    input_col : input integer column

    Returns
    -------
    A Column with integer represented in string ipv4 format

    """
    plc_column = plc.strings.convert.convert_ipv4.integers_to_ipv4(
        input_col.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_column)


def ip2int(Column input_col):
    """
    Converting string ipv4 column to integer column

    Parameters
    ----------
    input_col : input string column

    Returns
    -------
    A Column with ipv4 represented as integer

    """
    plc_column = plc.strings.convert.convert_ipv4.ipv4_to_integers(
        input_col.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_column)


def is_ipv4(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that have strings in IPv4 format. This format is nnn.nnn.nnn.nnn
    where nnn is integer digits in [0,255].
    """
    plc_column = plc.strings.convert.convert_ipv4.is_ipv4(
        source_strings.to_pylibcudf(mode="read")
    )
    return Column.from_pylibcudf(plc_column)


def htoi(Column input_col, **kwargs):
    """
    Converting input column of type string having hex values
    to integer of out_type

    Parameters
    ----------
    input_col : input column of type string

    Returns
    -------
    A Column of integers parsed from hexadecimal string values.
    """

    cdef column_view input_column_view = input_col.view()
    cdef type_id tid = <type_id> (
        <underlying_type_t_type_id> (
            SUPPORTED_NUMPY_TO_LIBCUDF_TYPES[cudf.dtype("int64")]
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


def itoh(Column input_col):
    """
    Converting input column of type integer to a string
    column with hexadecimal character digits.

    Parameters
    ----------
    input_col : input column of type integer

    Returns
    -------
    A Column of strings with hexadecimal characters.
    """

    cdef column_view input_column_view = input_col.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_integers_to_hex(input_column_view))

    return Column.from_unique_ptr(move(c_result))
