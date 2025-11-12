# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal, cast

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars import Translator
from cudf_polars.dsl.expr import Col, NamedExpr
from cudf_polars.experimental.parallel import evaluate_streaming, lower_ir_graph
from cudf_polars.experimental.shuffle import Shuffle
from cudf_polars.testing.asserts import DEFAULT_CLUSTER, DEFAULT_RUNTIME
from cudf_polars.utils.config import ConfigOptions

SHUFFLE_METHODS = ["tasks", None] if DEFAULT_RUNTIME == "tasks" else [None]


@pytest.fixture(scope="module", params=SHUFFLE_METHODS)
def engine(request):
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 4,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "shuffle_method": request.param,
        },
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7],
            "y": [1, 1, 1, 1, 1, 1, 1],
            "z": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )


def test_hash_shuffle(df: pl.LazyFrame, engine: pl.GPUEngine) -> None:
    # Extract translated IR
    qir = Translator(df._ldf.visit(), engine).translate_ir()

    # Add first Shuffle node
    keys = (NamedExpr("x", Col(qir.schema["x"], "x")),)
    options = ConfigOptions.from_polars_engine(engine)
    assert options.executor.name == "streaming"
    qir1 = Shuffle(qir.schema, keys, options.executor.shuffle_method, qir)

    # Add second Shuffle node (on the same keys)
    qir2 = Shuffle(qir.schema, keys, options.executor.shuffle_method, qir1)

    # Check that sequential shuffles on the same keys
    # are replaced with a single shuffle node
    partition_info = lower_ir_graph(qir2, options)[1]
    assert len([node for node in partition_info if isinstance(node, Shuffle)]) == 1

    # Add second Shuffle node (on different keys)
    keys2 = (NamedExpr("z", Col(qir.schema["z"], "z")),)
    qir3 = Shuffle(qir2.schema, keys2, options.executor.shuffle_method, qir2)

    # Check that we have an additional shuffle
    # node after shuffling on different keys
    partition_info = lower_ir_graph(qir3, options)[1]
    assert len([node for node in partition_info if isinstance(node, Shuffle)]) == 2

    # Check that streaming evaluation works
    result = evaluate_streaming(
        qir3,
        options,
    )
    # Cast needed because polars' EngineType "cpu" isn't publicly exported.
    # https://github.com/pola-rs/polars/issues/17420
    expect = df.collect(
        engine=cast(Literal["auto", "in-memory", "streaming", "gpu"], "cpu")
    )
    assert_frame_equal(result, expect, check_row_order=False)
