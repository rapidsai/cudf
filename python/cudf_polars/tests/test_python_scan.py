# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_ir_translation_raises
from cudf_polars.utils.versions import POLARS_VERSION_LT_125


@pytest.mark.skipif(
    POLARS_VERSION_LT_125, reason="This test is written for polars>=1.25"
)
def test_python_scan():
    df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int8())})

    def generator(with_columns, predicate, nrows, batch_size):
        yield df

    def source(with_columns, predicate, nrows, batch_size):
        return generator(with_columns, predicate, nrows, batch_size), False

    # Source is expected to return a generator of batches of the
    # dataframe and a boolean indicating whether the predicate was
    # applied.
    q = pl.LazyFrame._scan_python_function({"a": pl.Int8}, source, pyarrow=False)
    assert_ir_translation_raises(q, NotImplementedError)

    assert q.collect().equals(df)
