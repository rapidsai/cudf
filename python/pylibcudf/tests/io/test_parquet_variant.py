# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the libcudf Parquet reader's handling of VARIANT logical type.

PyArrow does not annotate group nodes with the Parquet VARIANT logical type, so
each fixture is built as a normal ``struct<metadata:binary,value:binary>``
parquet file in memory and the Thrift footer is patched to set logical type
VARIANT (union field 16) on the parent group ``v``. The libcudf reader is then
expected to surface the column as ``struct<list<uint8>, list<uint8>>``.
"""

from __future__ import annotations

import io
import struct

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pylibcudf as plc


def _metadata_v1(keys: list[str]) -> bytes:
    """VARIANT metadata v1 with 1-byte dictionary offsets."""
    header = (0 << 6) | 1  # offset_size code 0 -> 1 byte; version 1
    out = bytearray([header, len(keys)])
    pos = 0
    for k in keys:
        out.append(pos & 0xFF)
        pos += len(k)
    out.append(pos & 0xFF)
    for k in keys:
        out.extend(k.encode("utf-8"))
    return bytes(out)


def _enc_int32(v: int) -> bytes:
    """VARIANT primitive INT32 (basic_type 0, primitive header = 5)."""
    u = v & 0xFFFFFFFF
    return bytes(
        [
            (5 << 2) | 0,
            u & 0xFF,
            (u >> 8) & 0xFF,
            (u >> 16) & 0xFF,
            (u >> 24) & 0xFF,
        ]
    )


def _enc_short_str(s: str) -> bytes:
    """VARIANT short string (basic_type 1, length in upper 6 bits, len <= 63)."""
    b = s.encode("utf-8")
    if len(b) > 63:
        raise ValueError("short string only")
    return bytes([0x01 | (len(b) << 2)]) + b


def _enc_object(field_values: list[tuple[int, bytes]]) -> bytes:
    """VARIANT object value blob with 1-byte field ids and offsets."""
    n = len(field_values)
    if n > 255:
        raise ValueError("too many fields")
    out = bytearray([0x02, n])
    for fid, _ in field_values:
        out.append(fid)
    off = 0
    for _, payload in field_values:
        out.append(off)
        off += len(payload)
    out.append(off)
    for _, payload in field_values:
        out.extend(payload)
    return bytes(out)


def _patch_variant_logical_type(parquet_bytes: bytes) -> bytes:
    """Insert LOGICAL_TYPE VARIANT (union field 16) on schema element ``v``."""
    if parquet_bytes[-4:] != b"PAR1":
        raise ValueError("not parquet")
    footer_len = struct.unpack("<I", parquet_bytes[-8:-4])[0]
    footer = bytearray(parquet_bytes[-8 - footer_len : -8])
    # 0x18 0x01 0x76 = string-type, length 1, "v"; 0x15 0x04 = num_children 2
    anchor = footer.find(bytes.fromhex("1801761504"))
    if anchor < 0:
        raise RuntimeError("could not find schema element name 'v' in footer")
    if footer[anchor + 4] != 0x04 or footer[anchor + 5] != 0x00:
        raise RuntimeError("unexpected bytes after num_children for element v")
    ins = anchor + 5
    # struct-begin field 12 (logical type) + union field 16 (VARIANT) empty struct + struct-end
    patch = bytes.fromhex("5c0c200000")
    new_footer = footer[:ins] + patch + footer[ins:]
    body = parquet_bytes[: -8 - footer_len]
    return (
        body + bytes(new_footer) + struct.pack("<I", len(new_footer)) + b"PAR1"
    )


def _make_variant_parquet(
    meta_list: list[bytes], val_list: list[bytes]
) -> bytes:
    """Build an in-memory unshredded VARIANT parquet file."""
    if len(meta_list) != len(val_list):
        raise ValueError("meta/val length mismatch")
    struct_arr = pa.StructArray.from_arrays(
        [
            pa.array(meta_list, type=pa.binary()),
            pa.array(val_list, type=pa.binary()),
        ],
        names=["metadata", "value"],
    )
    table = pa.table({"v": struct_arr})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, version="2.6", compression="NONE")
    return _patch_variant_logical_type(sink.getvalue().to_pybytes())


def _read_first_column(parquet_bytes: bytes) -> plc.Column:
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([io.BytesIO(parquet_bytes)])
    ).build()
    table_w_meta = plc.io.parquet.read_parquet(options)
    return table_w_meta.tbl.columns()[0]


def _expect_variant_struct_shape(col: plc.Column, expected_size: int) -> None:
    assert col.type().id() == plc.TypeId.STRUCT
    assert col.size() == expected_size
    assert col.num_children() == 2
    metadata_child, value_child = col.child(0), col.child(1)
    assert metadata_child.type().id() == plc.TypeId.LIST
    assert value_child.type().id() == plc.TypeId.LIST
    assert metadata_child.list_view().child().type().id() == plc.TypeId.UINT8
    assert value_child.list_view().child().type().id() == plc.TypeId.UINT8


@pytest.fixture(scope="module")
def variant_minimal_parquet() -> bytes:
    """Single row, object ``{ "x": 7 }``."""
    return _make_variant_parquet(
        [_metadata_v1(["x"])], [_enc_object([(0, _enc_int32(7))])]
    )


@pytest.fixture(scope="module")
def variant_multirow_parquet() -> bytes:
    """Three rows with distinct dictionaries and object shapes."""
    return _make_variant_parquet(
        [
            _metadata_v1(["x", "k"]),
            _metadata_v1(["x", "y"]),
            _metadata_v1(["k"]),
        ],
        [
            _enc_object([(0, _enc_int32(7)), (1, _enc_short_str("hi"))]),
            _enc_object([(0, _enc_int32(42)), (1, _enc_int32(99))]),
            _enc_object([(0, _enc_short_str("zzz"))]),
        ],
    )


def test_read_variant_minimal_parquet(variant_minimal_parquet):
    col = _read_first_column(variant_minimal_parquet)
    _expect_variant_struct_shape(col, expected_size=1)


def test_read_variant_multirow_parquet(variant_multirow_parquet):
    col = _read_first_column(variant_multirow_parquet)
    _expect_variant_struct_shape(col, expected_size=3)
