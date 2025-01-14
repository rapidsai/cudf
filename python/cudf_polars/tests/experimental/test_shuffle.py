# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.experimental.parallel import evaluate_dask
from cudf_polars.experimental.shuffle import ShuffleByHash


@pytest.fixture(scope="module")
def engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
        executor_options={"max_rows_per_partition": 30000},
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    )


def test_hash_shuffle(df, engine):
    # Extract translated IR
    qir = Translator(df._ldf.visit(), engine).translate_ir()

    # Add ShuffleByHash node
    qir_shuffled = ShuffleByHash(
        qir.schema,
        (NamedExpr("a", Col(qir.schema["a"], "a")),),
        {},
        qir,
    )

    # Check that Dask evaluation works
    evaluate_dask(qir_shuffled)
