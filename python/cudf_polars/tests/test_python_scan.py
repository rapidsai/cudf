# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_ir_translation_raises


def test_python_scan():
    def source(with_columns, predicate, nrows):
        return pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int8())})

    q = pl.LazyFrame._scan_python_function({"a": pl.Int8}, source, pyarrow=False)
    assert_ir_translation_raises(q, NotImplementedError)

    assert q.collect().equals(source(None, None, None))
