# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc
import pytest


def test_list_dtype_roundtrip():
    list_type = pa.list_(pa.int32())
    plc_type = plc.interop.from_arrow(list_type)

    assert plc_type == plc.types.DataType(plc.types.TypeId.LIST)

    with pytest.raises(ValueError):
        plc.interop.to_arrow(plc_type)

    arrow_type = plc.interop.to_arrow(
        plc_type, value_type=list_type.value_type
    )
    assert arrow_type == list_type


def test_struct_dtype_roundtrip():
    struct_type = pa.struct([("a", pa.int32()), ("b", pa.string())])
    plc_type = plc.interop.from_arrow(struct_type)

    assert plc_type == plc.types.DataType(plc.types.TypeId.STRUCT)

    with pytest.raises(ValueError):
        plc.interop.to_arrow(plc_type)

    arrow_type = plc.interop.to_arrow(
        plc_type,
        fields=[struct_type.field(i) for i in range(struct_type.num_fields)],
    )
    assert arrow_type == struct_type


def test_decimal128_roundtrip():
    decimal_type = pa.decimal128(10, 2)
    plc_type = plc.interop.from_arrow(decimal_type)

    assert plc_type.id() == plc.types.TypeId.DECIMAL128

    with pytest.raises(ValueError):
        plc.interop.to_arrow(plc_type)

    arrow_type = plc.interop.to_arrow(
        plc_type, precision=decimal_type.precision
    )
    assert arrow_type == decimal_type


@pytest.mark.parametrize(
    "data_type",
    [
        plc.types.DataType(plc.types.TypeId.DECIMAL32),
        plc.types.DataType(plc.types.TypeId.DECIMAL64),
    ],
)
def test_decimal_other(data_type):
    precision = 3

    with pytest.raises(ValueError):
        plc.interop.to_arrow(data_type)

    arrow_type = plc.interop.to_arrow(data_type, precision=precision)
    assert arrow_type == pa.decimal128(precision, 0)
