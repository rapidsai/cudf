# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from pylibcudf.libcudf.utilities cimport traits

from .types cimport DataType

__all__ = [
    "is_bit_castable",
    "is_boolean",
    "is_chrono",
    "is_compound",
    "is_dictionary",
    "is_duration",
    "is_equality_comparable",
    "is_fixed_point",
    "is_fixed_width",
    "is_floating_point",
    "is_index_type",
    "is_integral",
    "is_integral_not_bool",
    "is_nested",
    "is_numeric",
    "is_numeric_not_bool",
    "is_relationally_comparable",
    "is_timestamp",
    "is_unsigned",
]

cpdef bool is_relationally_comparable(DataType typ):
    """Checks if the given data type supports relational comparisons.

    For details, see :cpp:func:`is_relationally_comparable`.
    """
    return traits.is_relationally_comparable(typ.c_obj)


cpdef bool is_equality_comparable(DataType typ):
    """Checks if the given data type supports equality comparisons.

    For details, see :cpp:func:`is_equality_comparable`.
    """
    return traits.is_equality_comparable(typ.c_obj)


cpdef bool is_numeric(DataType typ):
    """Checks if the given data type is numeric.

    For details, see :cpp:func:`is_numeric`.
    """
    return traits.is_numeric(typ.c_obj)

cpdef bool is_numeric_not_bool(DataType typ):
    """Checks if the given data type is numeric excluding booleans.

    For details, see :cpp:func:`is_numeric_not_bool`.
    """
    return traits.is_numeric_not_bool(typ.c_obj)

cpdef bool is_index_type(DataType typ):
    """Checks if the given data type is an index type.

    For details, see :cpp:func:`is_index_type`.
    """
    return traits.is_index_type(typ.c_obj)


cpdef bool is_unsigned(DataType typ):
    """Checks if the given data type is an unsigned type.

    For details, see :cpp:func:`is_unsigned`.
    """
    return traits.is_unsigned(typ.c_obj)


cpdef bool is_integral(DataType typ):
    """Checks if the given data type is an integral type.

    For details, see :cpp:func:`is_integral`.
    """
    return traits.is_integral(typ.c_obj)


cpdef bool is_integral_not_bool(DataType typ):
    """Checks if the given data type is an integral type excluding booleans.

    For details, see :cpp:func:`is_integral_not_bool`.
    """
    return traits.is_integral_not_bool(typ.c_obj)


cpdef bool is_floating_point(DataType typ):
    """Checks if the given data type is a floating point type.

    For details, see :cpp:func:`is_floating_point`.
    """
    return traits.is_floating_point(typ.c_obj)


cpdef bool is_boolean(DataType typ):
    """Checks if the given data type is a boolean type.

    For details, see :cpp:func:`is_boolean`.
    """
    return traits.is_boolean(typ.c_obj)


cpdef bool is_timestamp(DataType typ):
    """Checks if the given data type is a timestamp type.

    For details, see :cpp:func:`is_timestamp`.
    """
    return traits.is_timestamp(typ.c_obj)


cpdef bool is_fixed_point(DataType typ):
    """Checks if the given data type is a fixed point type.

    For details, see :cpp:func:`is_fixed_point`.
    """
    return traits.is_fixed_point(typ.c_obj)


cpdef bool is_duration(DataType typ):
    """Checks if the given data type is a duration type.

    For details, see :cpp:func:`is_duration`.
    """
    return traits.is_duration(typ.c_obj)


cpdef bool is_chrono(DataType typ):
    """Checks if the given data type is a chrono type.

    For details, see :cpp:func:`is_chrono`.
    """
    return traits.is_chrono(typ.c_obj)


cpdef bool is_dictionary(DataType typ):
    """Checks if the given data type is a dictionary type.

    For details, see :cpp:func:`is_dictionary`.
    """
    return traits.is_dictionary(typ.c_obj)


cpdef bool is_fixed_width(DataType typ):
    """Checks if the given data type is a fixed width type.

    For details, see :cpp:func:`is_fixed_width`.
    """
    return traits.is_fixed_width(typ.c_obj)


cpdef bool is_compound(DataType typ):
    """Checks if the given data type is a compound type.

    For details, see :cpp:func:`is_compound`.
    """
    return traits.is_compound(typ.c_obj)


cpdef bool is_nested(DataType typ):
    """Checks if the given data type is a nested type.

    For details, see :cpp:func:`is_nested`.
    """
    return traits.is_nested(typ.c_obj)


cpdef bool is_bit_castable(DataType source, DataType target):
    """Checks if the source type is bit-castable to the target type.

    For details, see :cpp:func:`is_bit_castable`.
    """
    return traits.is_bit_castable(source.c_obj, target.c_obj)
