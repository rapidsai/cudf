#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
Generate Parquet VARIANT fixtures for cudf tests.

PyArrow writes struct<metadata:binary,value:binary> without Parquet logical type VARIANT on
the parent group. This script patches the Thrift footer (compact protocol) to annotate the
``v`` group with LOGICAL_TYPE VARIANT (union field 16, empty struct), matching
``reader_impl_helpers.cpp`` expectations.

Requires: pyarrow, Python 3.9+

Usage (from repo root or this directory)::

    python3 python/cudf/cudf/tests/scripts/parquet_variant_fixture_gen.py

Outputs (default)::

    python/cudf/cudf/tests/data/parquet/variant_minimal.parquet
    python/cudf/cudf/tests/data/parquet/variant_multirow.parquet
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def metadata_v1(keys: list[str]) -> bytes:
    """VARIANT metadata v1: 1-byte dictionary offsets, keys in order (ids 0..n-1)."""
    header = (0 << 6) | 1  # offset_size code 0 -> 1 byte; version 1
    out = bytearray([header, len(keys)])
    # offsets: n+1 entries into following string blob
    pos = 0
    for k in keys:
        out.append(pos & 0xFF)
        pos += len(k)
    out.append(pos & 0xFF)
    for k in keys:
        out.extend(k.encode("utf-8"))
    return bytes(out)


def enc_int32(v: int) -> bytes:
    """Primitive INT32: basic_type 0, upper 6 bits of first byte = 5 → first byte (5<<2)|0 = 0x14."""
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


def enc_short_str(s: str) -> bytes:
    """Short string: basic_type 1, length in upper 6 bits of first byte (len <= 63)."""
    b = s.encode("utf-8")
    if len(b) > 63:
        raise ValueError("short string only")
    return bytes([0x01 | (len(b) << 2)]) + b


def enc_object(field_values: list[tuple[int, bytes]]) -> bytes:
    """
    Object value blob: field_id order matches field_values order (ids must match metadata).
    Small header: 1-byte counts/offsets, non-large.
    """
    n = len(field_values)
    if n > 255:
        raise ValueError("too many fields")
    # value_metadata: basic_type=2, value_header=0 -> 1-byte field ids and offsets
    out = bytearray([0x02, n])
    for fid, _ in field_values:
        out.append(fid)
    # offsets into concatenated payloads (1-byte)
    off = 0
    for _, payload in field_values:
        out.append(off)
        off += len(payload)
    out.append(off)
    for _, payload in field_values:
        out.extend(payload)
    return bytes(out)


def patch_variant_logical_type(parquet_bytes: bytes) -> bytes:
    """Insert LOGICAL_TYPE VARIANT on schema element ``v`` (group with metadata/value)."""
    if parquet_bytes[-4:] != b"PAR1":
        raise ValueError("not parquet")
    footer_len = struct.unpack("<I", parquet_bytes[-8:-4])[0]
    footer = bytearray(parquet_bytes[-8 - footer_len : -8])
    anchor = footer.find(bytes.fromhex("1801761504"))
    if anchor < 0:
        raise RuntimeError("could not find schema element name 'v' in footer")
    if footer[anchor + 4] != 0x04 or footer[anchor + 5] != 0x00:
        raise RuntimeError("unexpected bytes after num_children for element v")
    ins = anchor + 5
    patch = bytes.fromhex("5c0c200000")
    new_footer = footer[:ins] + patch + footer[ins:]
    new_len = len(new_footer)
    body = parquet_bytes[: -8 - footer_len]
    return body + bytes(new_footer) + struct.pack("<I", new_len) + b"PAR1"


def write_fixture(
    path: Path, meta_list: list[bytes], val_list: list[bytes]
) -> None:
    if len(meta_list) != len(val_list):
        raise ValueError("meta/val length mismatch")
    meta_col = pa.array(meta_list, type=pa.binary())
    val_col = pa.array(val_list, type=pa.binary())
    struct_arr = pa.StructArray.from_arrays(
        [meta_col, val_col], names=["metadata", "value"]
    )
    table = pa.table({"v": struct_arr})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, version="2.6", compression="NONE")
    raw = sink.getvalue().to_pybytes()
    patched = patch_variant_logical_type(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(patched)


def main() -> int:
    # python/cudf/cudf/tests/scripts -> python/cudf/cudf/tests
    tests_dir = Path(__file__).resolve().parents[1]
    out_dir = tests_dir / "data" / "parquet"

    # Minimal: single row, object { "x": 7 }
    m0 = metadata_v1(["x"])
    v0 = enc_object([(0, enc_int32(7))])
    write_fixture(out_dir / "variant_minimal.parquet", [m0], [v0])

    # Multi-row: distinct metadata dictionaries and objects per row (STRING + INT32 fields).
    mr1 = metadata_v1(["x", "k"])
    vr1 = enc_object([(0, enc_int32(7)), (1, enc_short_str("hi"))])
    mr2 = metadata_v1(["x", "y"])
    vr2 = enc_object([(0, enc_int32(42)), (1, enc_int32(99))])
    mr3 = metadata_v1(["k"])
    vr3 = enc_object([(0, enc_short_str("zzz"))])
    write_fixture(
        out_dir / "variant_multirow.parquet", [mr1, mr2, mr3], [vr1, vr2, vr3]
    )

    print(f"Wrote {out_dir / 'variant_minimal.parquet'}")  # noqa: T201
    print(f"Wrote {out_dir / 'variant_multirow.parquet'}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
