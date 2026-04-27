# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest

import pylibcudf as plc

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "data", "parquet")


def _read_table(filename):
    path = os.path.join(_FIXTURE_DIR, filename)
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([path])
    ).build()
    return plc.io.parquet.read_parquet(options).tbl


def _read_first_column(filename):
    return _read_table(filename).columns()[0]


def _expect_variant_metadata_value_pair(col):
    assert col.type().id() == plc.TypeId.STRUCT
    assert col.num_children() >= 2
    metadata_child, value_child = col.child(0), col.child(1)
    assert metadata_child.type().id() == plc.TypeId.LIST
    assert value_child.type().id() == plc.TypeId.LIST
    assert metadata_child.list_view().child().type().id() == plc.TypeId.UINT8
    assert value_child.list_view().child().type().id() == plc.TypeId.UINT8


@pytest.mark.parametrize(
    "filename",
    ["variant_minimal.parquet", "variant_multirow.parquet"],
)
def test_read_unshredded_variant_parquet_shape(filename):
    col = _read_first_column(filename)
    _expect_variant_metadata_value_pair(col)
    assert col.num_children() == 2


def test_read_shredded_variant_parquet_shape():
    tbl = _read_table("duckdb_variant_sample.parquet")
    columns = tbl.columns()
    assert tbl.num_rows() == 2
    assert len(columns) == 2

    id_col, variant_col = columns
    assert id_col.type().id() == plc.TypeId.INT32

    _expect_variant_metadata_value_pair(variant_col)
    assert variant_col.size() == 2
    assert variant_col.num_children() > 2, (
        "shredded VARIANT must surface typed children alongside metadata/value"
    )
