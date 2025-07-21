# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest

import polars as pl
from polars.testing.asserts import assert_frame_equal

import rmm

import cudf_polars.utils.config
from cudf_polars.callback import default_memory_resource
from cudf_polars.dsl.ir import DataFrameScan
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.config import ConfigOptions
from cudf_polars.utils.versions import POLARS_VERSION_LT_130


@pytest.fixture(params=[False, True], ids=["norapidsmpf", "rapidsmpf"])
def rapidsmpf_available(request, monkeypatch):
    monkeypatch.setattr(
        cudf_polars.utils.config, "rapidsmpf_available", lambda: request.param
    )
    return request.param


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
    if POLARS_VERSION_LT_130:
        with pytest.raises(pl.exceptions.ComputeError):
            q.collect(engine=pl.GPUEngine(device=device))
    elif isinstance(device, int):
        with pytest.raises(rmm._cuda.gpu.CUDARuntimeError):
            q.collect(engine=pl.GPUEngine(device=device))
    elif isinstance(device, str):
        with pytest.raises(TypeError):
            q.collect(engine=pl.GPUEngine(device=device))


@pytest.mark.parametrize("mr", [1, object()])
def test_invalid_memory_resource_raises(mr):
    q = pl.LazyFrame({})
    if POLARS_VERSION_LT_130:
        with pytest.raises(pl.exceptions.ComputeError):
            q.collect(engine=pl.GPUEngine(memory_resource=mr))
    else:
        with pytest.raises(TypeError):
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


@pytest.mark.parametrize("executor", ["streaming", "in-memory"])
def test_parquet_options(executor: str) -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor=executor,
        )
    )
    assert config.parquet_options.chunked is True
    assert config.parquet_options.n_output_chunks == 1

    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor=executor,
            parquet_options={"chunked": False, "n_output_chunks": 16},
        )
    )
    assert config.parquet_options.chunked is False
    assert config.parquet_options.n_output_chunks == 16


def test_validate_streaming_executor_shuffle_method(rapidsmpf_available) -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={"shuffle_method": "tasks"},
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.shuffle_method == "tasks"

    engine = pl.GPUEngine(
        executor="streaming",
        executor_options={"shuffle_method": "rapidsmpf", "scheduler": "distributed"},
    )
    if rapidsmpf_available:
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.name == "streaming"
        assert config.executor.shuffle_method == "rapidsmpf"
    else:
        with pytest.raises(ValueError, match="rapidsmpf is not installed"):
            ConfigOptions.from_polars_engine(engine)

    # rapidsmpf with sync is not allowed

    with pytest.raises(ValueError, match="rapidsmpf shuffle method"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={
                    "shuffle_method": "rapidsmpf",
                    "scheduler": "synchronous",
                },
            )
        )


@pytest.mark.parametrize("executor", ["in-memory", "streaming"])
def test_hashable(executor: str) -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor=executor,
        )
    )
    assert hash(config) == hash(config)


def test_validate_fallback_mode() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.fallback_mode == "warn"

    with pytest.raises(ValueError, match="'foo' is not a valid StreamingFallbackMode"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"fallback_mode": "foo"},
            )
        )


def test_validate_scheduler() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.scheduler == "synchronous"

    with pytest.raises(ValueError, match="'foo' is not a valid Scheduler"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"scheduler": "foo"},
            )
        )


def test_validate_shuffle_method_defaults(rapidsmpf_available) -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
        )
    )
    assert config.executor.name == "streaming"
    assert (
        config.executor.shuffle_method == "tasks"
    )  # Default for synchronous scheduler

    # Test default for distributed scheduler depends on rapidsmpf availability
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={"scheduler": "distributed"},
        )
    )
    assert config.executor.name == "streaming"
    if rapidsmpf_available:
        # Should be "rapidsmpf" if available, otherwise "tasks"
        assert config.executor.shuffle_method == "rapidsmpf"
    else:
        assert config.executor.shuffle_method == "tasks"

    with pytest.raises(ValueError, match="'foo' is not a valid ShuffleMethod"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"shuffle_method": "foo"},
            )
        )


@pytest.mark.parametrize(
    "option",
    [
        "max_rows_per_partition",
        "unique_fraction",
        "target_partition_size",
        "groupby_n_ary",
        "broadcast_join_limit",
        "rapidsmpf_spill",
        "sink_to_directory",
    ],
)
def test_validate_max_rows_per_partition(option: str) -> None:
    with pytest.raises(TypeError, match=f"{option} must be"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={option: object()},
            )
        )


def test_executor_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR", "in-memory")
        engine = pl.GPUEngine()
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.name == "in-memory"

    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR", "invalid")
        engine = pl.GPUEngine()
        with pytest.raises(ValueError, match="Unknown executor 'invalid'"):
            ConfigOptions.from_polars_engine(engine)


def test_parquet_options_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__CHUNKED", "0")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__N_OUTPUT_CHUNKS", "2")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__CHUNK_READ_LIMIT", "100")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__PASS_READ_LIMIT", "200")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__MAX_FOOTER_SAMPLES", "0")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__MAX_ROW_GROUP_SAMPLES", "0")

        # Test default
        engine = pl.GPUEngine()
        config = ConfigOptions.from_polars_engine(engine)
        assert config.parquet_options.chunked is False
        assert config.parquet_options.n_output_chunks == 2
        assert config.parquet_options.chunk_read_limit == 100
        assert config.parquet_options.pass_read_limit == 200
        assert config.parquet_options.max_footer_samples == 0
        assert config.parquet_options.max_row_group_samples == 0

    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__CHUNKED", "foo")
        engine = pl.GPUEngine()
        with pytest.raises(ValueError, match="Invalid boolean value: 'foo'"):
            ConfigOptions.from_polars_engine(engine)


def test_config_option_from_env(
    monkeypatch: pytest.MonkeyPatch, *, rapidsmpf_available: bool
) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__SCHEDULER", "distributed")
        m.setenv("CUDF_POLARS__EXECUTOR__FALLBACK_MODE", "silent")
        m.setenv("CUDF_POLARS__EXECUTOR__MAX_ROWS_PER_PARTITION", "42")
        m.setenv("CUDF_POLARS__EXECUTOR__UNIQUE_FRACTION", '{"a": 0.5}')
        m.setenv("CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE", "100")
        m.setenv("CUDF_POLARS__EXECUTOR__GROUPBY_N_ARY", "43")
        m.setenv("CUDF_POLARS__EXECUTOR__BROADCAST_JOIN_LIMIT", "44")
        m.setenv("CUDF_POLARS__EXECUTOR__RAPIDSMPF_SPILL", "1")
        m.setenv("CUDF_POLARS__EXECUTOR__SINK_TO_DIRECTORY", "1")

        if rapidsmpf_available:
            m.setenv("CUDF_POLARS__EXECUTOR__SHUFFLE_METHOD", "rapidsmpf")
        else:
            m.setenv("CUDF_POLARS__EXECUTOR__SHUFFLE_METHOD", "tasks")

        engine = pl.GPUEngine()
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.name == "streaming"
        assert config.executor.scheduler == "distributed"
        assert config.executor.fallback_mode == "silent"
        assert config.executor.max_rows_per_partition == 42
        assert config.executor.unique_fraction == {"a": 0.5}
        assert config.executor.target_partition_size == 100
        assert config.executor.groupby_n_ary == 43
        assert config.executor.broadcast_join_limit == 44
        assert config.executor.rapidsmpf_spill is True
        assert config.executor.sink_to_directory is True

        if rapidsmpf_available:
            assert config.executor.shuffle_method == "rapidsmpf"
        else:
            assert config.executor.shuffle_method == "tasks"


def test_target_partition_from_env(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "pynvml", None)
        m.setenv("CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE", "100")

        engine = pl.GPUEngine(executor="streaming")
        ConfigOptions.from_polars_engine(engine)  # no warning
        assert len(recwarn) == 0


def test_fallback_mode_default(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__FALLBACK_MODE", "silent")
        engine = pl.GPUEngine(executor="streaming")
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.name == "streaming"
        assert config.executor.fallback_mode == "silent"

    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__FALLBACK_MODE", "foo")
        engine = pl.GPUEngine(executor="streaming")
        with pytest.raises(
            ValueError, match="'foo' is not a valid StreamingFallbackMode"
        ):
            ConfigOptions.from_polars_engine(engine)


def test_cardinality_factor_compat() -> None:
    with pytest.warns(FutureWarning, match="configuration is deprecated"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"cardinality_factor": {}},
            )
        )


@pytest.mark.parametrize(
    "option",
    [
        "chunked",
        "n_output_chunks",
        "chunk_read_limit",
        "pass_read_limit",
        "max_footer_samples",
        "max_row_group_samples",
    ],
)
def test_validate_parquet_options(option: str) -> None:
    with pytest.raises(TypeError, match=f"{option} must be"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                parquet_options={option: object()},
            )
        )


def test_validate_raise_on_fail() -> None:
    with pytest.raises(TypeError, match="'raise_on_fail' must be"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(executor="streaming", raise_on_fail=object())  # type: ignore[arg-type]
        )


def test_validate_executor() -> None:
    with pytest.raises(ValueError, match="Unknown executor 'foo'"):
        ConfigOptions.from_polars_engine(pl.GPUEngine(executor="foo"))


def test_default_executor() -> None:
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.name == "streaming"
