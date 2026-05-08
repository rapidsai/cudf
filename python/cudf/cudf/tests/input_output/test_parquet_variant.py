# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from packaging import version

import cudf

pytestmark = pytest.mark.skipif(
    version.parse(pa.__version__) < version.parse("21"),
    reason="pyarrow < 21 does not recognize the Parquet VARIANT logical type",
)


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


def _expect_variant_struct_shape(dtype):
    """Looser check used for ``get_variant_field`` results, whose children are
    unnamed (``"0"`` and ``"1"``) because libcudf columns carry no names.
    """
    assert isinstance(dtype, cudf.StructDtype)
    children = list(dtype.fields.values())
    assert len(children) == 2
    for child in children:
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


# ---------------------------------------------------------------------------
# extract_variant_field round-trip tests (formerly the libcudf gtest cases in
# cpp/tests/io/parquet_variant_roundtrip_test.cpp).
# ---------------------------------------------------------------------------


def test_extract_variant_field_minimal(datadir):
    """``variant_minimal.parquet`` is a single-row VARIANT object ``{"x": 7}``."""
    from cudf.io.parquet import extract_variant_field

    df = cudf.read_parquet(datadir / "variant_minimal.parquet")
    assert df.shape == (1, 1)
    variant_col = df.iloc[:, 0]
    _expect_variant_metadata_value_pair(variant_col.dtype)

    extracted = extract_variant_field(variant_col, "x", np.int32)
    assert extracted.dtype == np.int32
    assert extracted.to_arrow().to_pylist() == [7]


def test_extract_variant_field_multirow(datadir):
    """``variant_multirow.parquet`` has three rows with mixed dictionaries:
    ``{"x", "k"}`` (INT32+STRING), ``{"x", "y"}`` (two INT32),
    and ``{"k"}`` (STRING).
    """
    from cudf.io.parquet import extract_variant_field

    df = cudf.read_parquet(datadir / "variant_multirow.parquet")
    assert df.shape == (3, 1)
    variant_col = df.iloc[:, 0]

    x = extract_variant_field(variant_col, "x", np.int32)
    assert x.to_arrow().to_pylist() == [7, 42, None]

    k = extract_variant_field(variant_col, "k", str)
    assert k.to_arrow().to_pylist() == ["hi", None, "zzz"]

    y = extract_variant_field(variant_col, "y", np.int32)
    assert y.to_arrow().to_pylist() == [None, 99, None]


def test_get_then_cast_matches_extract(datadir):
    """``get_variant_field`` followed by ``cast_variant`` must equal
    ``extract_variant_field`` (the convenience wrapper).
    """
    from cudf.io.parquet import (
        cast_variant,
        extract_variant_field,
        get_variant_field,
    )

    df = cudf.read_parquet(datadir / "variant_multirow.parquet")
    variant_col = df.iloc[:, 0]

    intermediate = get_variant_field(variant_col, "x")
    _expect_variant_struct_shape(intermediate.dtype)

    two_step = cast_variant(intermediate, np.int32)
    one_step = extract_variant_field(variant_col, "x", np.int32)
    assert two_step.to_arrow() == one_step.to_arrow()


def test_extract_unsupported_dtype_raises(datadir):
    from cudf.io.parquet import extract_variant_field

    df = cudf.read_parquet(datadir / "variant_minimal.parquet")
    variant_col = df.iloc[:, 0]
    with pytest.raises(ValueError, match="STRING and INT32 only"):
        extract_variant_field(variant_col, "x", np.float64)
