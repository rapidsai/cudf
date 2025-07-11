# Copyright (c) 2024, NVIDIA CORPORATION.

import pylibcudf as plc


def test_is_relationally_comparable():
    assert plc.traits.is_relationally_comparable(plc.DataType(plc.TypeId.INT8))
    assert not plc.traits.is_relationally_comparable(
        plc.DataType(plc.TypeId.LIST)
    )


def test_is_equality_comparable():
    assert plc.traits.is_equality_comparable(plc.DataType(plc.TypeId.INT8))
    assert not plc.traits.is_equality_comparable(plc.DataType(plc.TypeId.LIST))


def test_is_numeric():
    assert plc.traits.is_numeric(plc.DataType(plc.TypeId.FLOAT64))
    assert not plc.traits.is_numeric(plc.DataType(plc.TypeId.LIST))


def test_is_numeric_not_bool():
    assert plc.traits.is_numeric_not_bool(plc.DataType(plc.TypeId.FLOAT64))
    assert not plc.traits.is_numeric_not_bool(plc.DataType(plc.TypeId.BOOL8))


def test_is_index_type():
    assert plc.traits.is_index_type(plc.DataType(plc.TypeId.INT8))
    assert not plc.traits.is_index_type(plc.DataType(plc.TypeId.BOOL8))


def test_is_unsigned():
    assert plc.traits.is_unsigned(plc.DataType(plc.TypeId.UINT8))
    assert not plc.traits.is_unsigned(plc.DataType(plc.TypeId.INT8))


def test_is_integral():
    assert plc.traits.is_integral(plc.DataType(plc.TypeId.BOOL8))
    assert not plc.traits.is_integral(plc.DataType(plc.TypeId.DECIMAL32))


def test_is_integral_not_bool():
    assert plc.traits.is_integral_not_bool(plc.DataType(plc.TypeId.INT8))
    assert not plc.traits.is_integral_not_bool(plc.DataType(plc.TypeId.BOOL8))


def test_is_floating_point():
    assert plc.traits.is_floating_point(plc.DataType(plc.TypeId.FLOAT64))
    assert not plc.traits.is_floating_point(plc.DataType(plc.TypeId.UINT8))


def test_is_boolean():
    assert plc.traits.is_boolean(plc.DataType(plc.TypeId.BOOL8))
    assert not plc.traits.is_boolean(plc.DataType(plc.TypeId.UINT8))


def test_is_timestamp():
    assert plc.traits.is_timestamp(
        plc.DataType(plc.TypeId.TIMESTAMP_MICROSECONDS)
    )
    assert not plc.traits.is_timestamp(
        plc.DataType(plc.TypeId.DURATION_MICROSECONDS)
    )


def test_is_fixed_point():
    assert plc.traits.is_fixed_point(plc.DataType(plc.TypeId.DECIMAL128))
    assert not plc.traits.is_fixed_point(plc.DataType(plc.TypeId.FLOAT32))


def test_is_duration():
    assert plc.traits.is_duration(
        plc.DataType(plc.TypeId.DURATION_MICROSECONDS)
    )
    assert not plc.traits.is_duration(
        plc.DataType(plc.TypeId.TIMESTAMP_MICROSECONDS)
    )


def test_is_chrono():
    assert plc.traits.is_chrono(plc.DataType(plc.TypeId.DURATION_MICROSECONDS))
    assert plc.traits.is_chrono(
        plc.DataType(plc.TypeId.TIMESTAMP_MICROSECONDS)
    )
    assert not plc.traits.is_chrono(plc.DataType(plc.TypeId.UINT8))


def test_is_dictionary():
    assert plc.traits.is_dictionary(plc.DataType(plc.TypeId.DICTIONARY32))
    assert not plc.traits.is_dictionary(plc.DataType(plc.TypeId.UINT8))


def test_is_fixed_width():
    assert plc.traits.is_fixed_width(plc.DataType(plc.TypeId.INT8))
    assert not plc.traits.is_fixed_width(plc.DataType(plc.TypeId.STRING))


def test_is_compound():
    assert plc.traits.is_compound(plc.DataType(plc.TypeId.STRUCT))
    assert not plc.traits.is_compound(plc.DataType(plc.TypeId.UINT8))


def test_is_nested():
    assert plc.traits.is_nested(plc.DataType(plc.TypeId.STRUCT))
    assert not plc.traits.is_nested(plc.DataType(plc.TypeId.STRING))


def test_is_bit_castable():
    assert plc.traits.is_bit_castable(
        plc.DataType(plc.TypeId.INT8), plc.DataType(plc.TypeId.UINT8)
    )
    assert not plc.traits.is_bit_castable(
        plc.DataType(plc.TypeId.UINT8), plc.DataType(plc.TypeId.UINT16)
    )
