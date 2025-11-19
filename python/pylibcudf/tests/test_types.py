# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from packaging.version import parse

import pylibcudf as plc


def test_dtype_from_arrow():
    assert (
        plc.DataType.from_arrow(pa.decimal128(10, 2)).id()
        == plc.TypeId.DECIMAL128
    )

    assert plc.DataType.from_arrow(pa.struct([])).id() == plc.TypeId.STRUCT

    assert (
        plc.DataType.from_arrow(pa.list_(pa.int32())).id() == plc.TypeId.LIST
    )

    assert (
        plc.DataType.from_arrow(pa.list_(pa.list_(pa.int32()))).id()
        == plc.TypeId.LIST
    )

    if parse(pa.__version__) > parse("19.0.0"):
        assert (
            plc.DataType.from_arrow(pa.decimal32(3)).id()
            == plc.TypeId.DECIMAL32
        )

        assert (
            plc.DataType.from_arrow(pa.decimal64(5)).id()
            == plc.TypeId.DECIMAL64
        )

    assert plc.DataType.from_arrow(pa.int32()).id() == plc.TypeId.INT32


def test_dtype_from_arrow_unsupported():
    class Foo:
        pass

    with pytest.raises(TypeError, match="Unable to convert"):
        plc.DataType.from_arrow(Foo())

    with pytest.raises(TypeError, match="Unable to convert"):
        plc.DataType.from_arrow(pa.list_(pa.list_(pa.binary())))

    with pytest.raises(TypeError, match="Unable to convert"):
        plc.DataType.from_arrow(pa.struct([pa.field("foo", pa.binary())]))
