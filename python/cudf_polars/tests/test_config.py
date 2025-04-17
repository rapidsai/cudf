# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars.testing.asserts import assert_frame_equal

import rmm

from cudf_polars.callback import default_memory_resource
from cudf_polars.dsl.ir import DataFrameScan
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.config import ConfigOptions


def test_polars_verbose_warns(monkeypatch):
    def raise_unimplemented(self, *args):
        raise NotImplementedError("We don't support this")

    monkeypatch.setattr(DataFrameScan, "__init__", raise_unimplemented)
    q = pl.LazyFrame({})
    # Ensure that things raise
    assert_ir_translation_raises(q, NotImplementedError)
    with (
        pl.Config(verbose=True),
        pytest.raises(pl.exceptions.ComputeError),
        pytest.warns(
            pl.exceptions.PerformanceWarning,
            match="Query execution with GPU not possible",
        ),
    ):
        # And ensure that collecting issues the correct warning.
        assert_gpu_result_equal(q)


def test_unsupported_config_raises():
    q = pl.LazyFrame({})

    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine=pl.GPUEngine(unknown_key=True))


@pytest.mark.parametrize("device", [-1, "foo"])
def test_invalid_device_raises(device):
    q = pl.LazyFrame({})
    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine=pl.GPUEngine(device=device))


@pytest.mark.parametrize("mr", [1, object()])
def test_invalid_memory_resource_raises(mr):
    q = pl.LazyFrame({})
    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine=pl.GPUEngine(memory_resource=mr))


@pytest.mark.parametrize("disable_managed_memory", ["1", "0"])
def test_cudf_polars_enable_disable_managed_memory(monkeypatch, disable_managed_memory):
    q = pl.LazyFrame({"a": [1, 2, 3]})

    with monkeypatch.context() as monkeycontext:
        monkeycontext.setenv(
            "POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY", disable_managed_memory
        )
        result = q.collect(engine=pl.GPUEngine())
        mr = default_memory_resource(0, bool(disable_managed_memory == "1"))
        if disable_managed_memory == "1":
            assert isinstance(mr, rmm.mr.PrefetchResourceAdaptor)
            assert isinstance(mr.upstream_mr, rmm.mr.PoolMemoryResource)
        else:
            assert isinstance(mr, rmm.mr.CudaAsyncMemoryResource)
        monkeycontext.delenv("POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY")
    assert_frame_equal(q.collect(), result)


def test_explicit_device_zero():
    q = pl.LazyFrame({"a": [1, 2, 3]})

    result = q.collect(engine=pl.GPUEngine(device=0))
    assert_frame_equal(q.collect(), result)


def test_explicit_memory_resource():
    upstream = rmm.mr.CudaMemoryResource()
    n_allocations = 0

    def allocate(bytes, stream):
        nonlocal n_allocations
        n_allocations += 1
        return upstream.allocate(bytes, stream)

    mr = rmm.mr.CallbackMemoryResource(allocate, upstream.deallocate)

    q = pl.LazyFrame({"a": [1, 2, 3]})
    result = q.collect(engine=pl.GPUEngine(memory_resource=mr))
    assert_frame_equal(q.collect(), result)
    assert n_allocations > 0


def test_user_streaming_options():
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
        )
    )
    assert config.parquet_options.chunked is False

    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            parquet_options={"chunked": True},
        )
    )
    assert config.parquet_options.chunked is True

    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="in-memory",
        )
    )
    assert config.parquet_options.chunked is True
