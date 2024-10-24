# Copyright (c) 2024, NVIDIA CORPORATION.
import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_ipv4_to_integers():
    arr = pa.array(["123.45.67.890", None])
    result = plc.strings.convert.convert_ipv4.ipv4_to_integers(
        plc.interop.from_arrow(arr)
    )
    expected = pa.array([2066564730, None], type=pa.uint32())
    assert_column_eq(result, expected)


def test_integers_to_ipv4():
    arr = pa.array([1, 0, None], type=pa.uint32())
    result = plc.strings.convert.convert_ipv4.integers_to_ipv4(
        plc.interop.from_arrow(arr)
    )
    expected = pa.array(["0.0.0.1", "0.0.0.0", None])
    assert_column_eq(result, expected)


def test_is_ipv4():
    arr = pa.array(["0.0.0.1", "1.2.34", "A", None])
    result = plc.strings.convert.convert_ipv4.is_ipv4(
        plc.interop.from_arrow(arr)
    )
    expected = pa.array([True, False, False, None])
    assert_column_eq(result, expected)
