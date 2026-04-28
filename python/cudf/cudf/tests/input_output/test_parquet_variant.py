# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

import cudf


@pytest.fixture(scope="module")
def datadir(datadir):
    return datadir / "parquet"


def _expect_variant_metadata_value_pair(dtype):
    assert isinstance(dtype, cudf.StructDtype)
    fields = dtype.fields
    for name in ("metadata", "value"):
        assert name in fields, f"missing VARIANT child {name!r}"
        child = fields[name]
        assert isinstance(child, cudf.ListDtype)
        assert child.element_type == np.dtype("uint8")


@pytest.mark.parametrize(
    "filename",
    ["variant_minimal.parquet", "variant_multirow.parquet"],
)
def test_read_unshredded_variant_parquet_shape(datadir, filename):
    df = cudf.read_parquet(datadir / filename)
    assert df.shape[1] == 1
    variant_dtype = df.dtypes.iloc[0]
    _expect_variant_metadata_value_pair(variant_dtype)
    assert set(variant_dtype.fields) == {"metadata", "value"}


def test_read_variant_parquet_zero_rows(datadir):
    df = cudf.read_parquet(datadir / "variant_multirow.parquet", nrows=0)
    assert len(df) == 0
    _expect_variant_metadata_value_pair(df.dtypes.iloc[0])


def test_read_shredded_variant_parquet_shape(datadir):
    df = cudf.read_parquet(datadir / "duckdb_variant_sample.parquet")
    assert df.shape == (2, 2)
    assert df["id"].dtype == np.dtype("int32")

    variant_dtype = df["v"].dtype
    _expect_variant_metadata_value_pair(variant_dtype)
    assert "typed_value" in variant_dtype.fields, (
        "shredded VARIANT must surface typed children alongside metadata/value"
    )
