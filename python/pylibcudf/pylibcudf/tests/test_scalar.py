# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import pyarrow as pa
import pytest

import pylibcudf as plc


@pytest.mark.parametrize(
    "val", [True, False, -1, 0, 1 - 1.0, 0.0, 1.52, "", "a1!"]
)
def test_from_py(val):
    result = plc.Scalar.from_py(val)
    expected = pa.scalar(val)
    assert plc.interop.to_arrow(result).equals(expected)
