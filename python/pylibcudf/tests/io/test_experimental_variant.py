# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the experimental Parquet VARIANT pylibcudf bindings.

These tests build VARIANT struct columns directly from hand-crafted Apache
Variant binary blobs and exercise the GPU extraction API. End-to-end
parquet-to-variant round-trip coverage lives in cudf-python tests, which
have access to ``cudf.read_parquet`` for materializing struct columns from
on-disk fixtures.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import pylibcudf as plc


def _variant_struct_column(
    metadata_blobs: list[bytes], value_blobs: list[bytes]
) -> plc.Column:
    """Build a ``struct<list<uint8> metadata, list<uint8> value>`` column."""
    arr = pa.StructArray.from_arrays(
        [
            pa.array(
                [list(b) for b in metadata_blobs], type=pa.list_(pa.uint8())
            ),
            pa.array(
                [list(b) for b in value_blobs], type=pa.list_(pa.uint8())
            ),
        ],
        names=["metadata", "value"],
    )
    return plc.Column.from_arrow(arr)


def _column_to_pylist(col: plc.Column) -> list:
    return col.to_arrow().to_pylist()


# Single-row VARIANT object {"x": 7} (INT32 short value).
_META_X = bytes([0x01, 0x01, 0x00, 0x01, ord("x")])
_VAL_X7 = bytes(
    [
        0x02,  # object header (basic_type=2, no large fields)
        0x01,  # n=1 field
        0x00,  # field id 0 -> "x"
        0x00,  # field offset 0
        0x05,  # values-blob length / next-byte position
        0x14,  # primitive header: INT32 (type tag 0x14)
        0x07,
        0x00,
        0x00,
        0x00,  # int32 7 little-endian
    ]
)


def test_extract_int32_field_present():
    col = _variant_struct_column([_META_X], [_VAL_X7])
    result = plc.io.experimental.extract_variant_field(
        col, "x", plc.DataType(plc.TypeId.INT32)
    )
    assert isinstance(result, plc.Column)
    assert result.type().id() == plc.TypeId.INT32
    assert _column_to_pylist(result) == [7]


def test_extract_missing_field_yields_null():
    col = _variant_struct_column([_META_X], [_VAL_X7])
    result = plc.io.experimental.extract_variant_field(
        col, "missing", plc.DataType(plc.TypeId.INT32)
    )
    assert result.type().id() == plc.TypeId.INT32
    assert _column_to_pylist(result) == [None]


def test_get_variant_field_returns_variant_struct():
    col = _variant_struct_column([_META_X], [_VAL_X7])
    sub = plc.io.experimental.get_variant_field(col, "x")
    assert sub.type().id() == plc.TypeId.STRUCT
    assert sub.num_children() == 2


def test_cast_variant_int32():
    # Build a VARIANT struct whose value blob is a bare INT32 (tag 0x14, 7).
    bare_int = bytes([0x14, 0x07, 0x00, 0x00, 0x00])
    empty_meta = bytes([0x02])  # minimal valid metadata
    col = _variant_struct_column([empty_meta], [bare_int])
    result = plc.io.experimental.cast_variant(
        col, plc.DataType(plc.TypeId.INT32)
    )
    assert _column_to_pylist(result) == [7]


def test_extract_invalid_dtype_raises():
    col = _variant_struct_column([_META_X], [_VAL_X7])
    with pytest.raises(ValueError, match="STRING and INT32 only"):
        plc.io.experimental.extract_variant_field(
            col, "x", plc.DataType(plc.TypeId.FLOAT64)
        )
