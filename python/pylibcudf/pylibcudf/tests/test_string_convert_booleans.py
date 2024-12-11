# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_to_booleans():
    pa_array = pa.array(["true", None, "True"])
    result = plc.strings.convert.convert_booleans.to_booleans(
        plc.interop.from_arrow(pa_array),
        plc.interop.from_arrow(pa.scalar("True")),
    )
    expected = pa.array([False, None, True])
    assert_column_eq(result, expected)


def test_from_booleans():
    pa_array = pa.array([True, None, False])
    result = plc.strings.convert.convert_booleans.from_booleans(
        plc.interop.from_arrow(pa_array),
        plc.interop.from_arrow(pa.scalar("A")),
        plc.interop.from_arrow(pa.scalar("B")),
    )
    expected = pa.array(["A", None, "B"])
    assert_column_eq(result, expected)
