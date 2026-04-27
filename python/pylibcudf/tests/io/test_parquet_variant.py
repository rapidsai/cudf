# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the libcudf Parquet reader's handling of VARIANT logical type.

The committed parquet fixtures under ``data/parquet/`` are unshredded VARIANT
columns (parent group ``v`` annotated with Parquet logical type VARIANT, union
field 16). The libcudf reader is expected to surface them as
``struct<list<uint8>, list<uint8>>``.
"""

from __future__ import annotations

import os

import pytest

import pylibcudf as plc

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "data", "parquet")


def _read_first_column(filename):
    path = os.path.join(_FIXTURE_DIR, filename)
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([path])
    ).build()
    return plc.io.parquet.read_parquet(options).tbl.columns()[0]


def _expect_variant_struct_shape(col, expected_size):
    assert col.type().id() == plc.TypeId.STRUCT
    assert col.size() == expected_size
    assert col.num_children() == 2
    metadata_child, value_child = col.child(0), col.child(1)
    assert metadata_child.type().id() == plc.TypeId.LIST
    assert value_child.type().id() == plc.TypeId.LIST
    assert metadata_child.list_view().child().type().id() == plc.TypeId.UINT8
    assert value_child.list_view().child().type().id() == plc.TypeId.UINT8


@pytest.mark.parametrize(
    "filename,expected_size",
    [
        ("variant_minimal.parquet", 1),
        ("variant_multirow.parquet", 3),
    ],
)
def test_read_variant_parquet_shape(filename, expected_size):
    col = _read_first_column(filename)
    _expect_variant_struct_shape(col, expected_size)
