# Copyright (c) 2024, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pyarrow as pa
import pytest
from utils import assert_table_eq

import pylibcudf as plc


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


def test_table_with_nested_dtype_to_arrow():
    pa_array = pa.array([[{"": 1}]])
    plc_table = plc.Table([plc.interop.from_arrow(pa_array)])
    result = plc.interop.to_arrow(plc_table)
    expected_schema = pa.schema(
        [
            pa.field(
                "",
                pa.list_(
                    pa.field(
                        "",
                        pa.struct([pa.field("", pa.int64(), nullable=False)]),
                        nullable=False,
                    )
                ),
                nullable=False,
            )
        ]
    )
    assert result.schema == expected_schema


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


def test_round_trip_dlpack_plc_table():
    expected = pa.table({"a": [1, 2, 3], "b": [5, 6, 7]})
    plc_table = plc.interop.from_arrow(expected)
    result = plc.interop.from_dlpack(plc.interop.to_dlpack(plc_table))
    assert_table_eq(expected, result)


@pytest.mark.parametrize("array", [np.array, cp.array])
def test_round_trip_dlpack_array(array):
    arr = array([1, 2, 3])
    result = plc.interop.from_dlpack(arr.__dlpack__())
    expected = pa.table({"a": [1, 2, 3]})
    assert_table_eq(expected, result)


def test_to_dlpack_error():
    plc_table = plc.interop.from_arrow(
        pa.table({"a": [1, None, 3], "b": [5, 6, 7]})
    )
    with pytest.raises(ValueError, match="Cannot create a DLPack tensor"):
        plc.interop.from_dlpack(plc.interop.to_dlpack(plc_table))


def test_from_dlpack_error():
    with pytest.raises(ValueError, match="Invalid PyCapsule object"):
        plc.interop.from_dlpack(1)
