# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl import ir
from cudf_polars.dsl.traversal import traversal
from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_cache():
    df1 = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    )
    df2 = pl.LazyFrame({"a": [7, 8], "b": [12, 13]})

    q = pl.concat([df1, df2, df1, df2, df1])
    assert_gpu_result_equal(q)

    t = Translator(q._ldf.visit(), pl.GPUEngine())
    qir = t.translate_ir()
    # There should be two unique cache nodes.
    cache_df2, cache_df1 = (
        node for node in traversal([qir]) if isinstance(node, ir.Cache)
    )
    assert cache_df1 != cache_df2
    assert not cache_df1.is_equal(cache_df2)
    assert cache_df1 == qir.children[0]
    assert cache_df2 == qir.children[1]
    assert cache_df1 == qir.children[2]
    assert cache_df2 == qir.children[3]
    assert cache_df1 == qir.children[4]

    class HitCounter(dict):
        def __init__(self, *args, **kwargs):
            self.hits = 0
            super().__init__(*args, **kwargs)

        def __getitem__(self, key):
            result = super().__getitem__(key)
            self.hits += 1
            return result

    node_cache = HitCounter()
    qir.evaluate(cache=node_cache, timer=None)
    assert len(node_cache) == 0
    assert node_cache.hits == 3
