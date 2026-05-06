# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import Any, cast

import pytest

import polars as pl
from polars.testing.asserts import assert_frame_equal

import rmm
from rmm._cuda import gpu
from rmm.pylibrmm import CudaStreamFlags

import cudf_polars.callback
import cudf_polars.utils.config
from cudf_polars.callback import (
    _is_concurrent_managed_access_supported,
    default_memory_resource,
    set_memory_resource,
)
from cudf_polars.dsl.ir import DataFrameScan, IRExecutionContext
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.config import (
    CUDAStreamPoolConfig,
    Cluster,
    ConfigOptions,
    MemoryResourceConfig,
    StreamingExecutor,
    _default_cuda_stream_policy,
)
from cudf_polars.utils.cuda_stream import (
    get_cuda_stream,
    get_dask_cuda_stream,
)


@pytest.fixture(params=[False, True], ids=["norapidsmpf.single", "rapidsmpf.single"])
def rapidsmpf_single_available(request, monkeypatch):
    monkeypatch.setattr(
        cudf_polars.utils.config,
        "rapidsmpf_single_available",
        lambda: request.param,
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


def test_use_device_not_current(monkeypatch):
    # This is testing the set/restore device functionality in callback
    # for the case where the device to use is not the current device and no
    # previous query has used a device.
    monkeypatch.setattr(cudf_polars.callback, "SEEN_DEVICE", None)
    monkeypatch.setattr(gpu, "setDevice", lambda arg: None)
    # Fake that the current device is 1.
    monkeypatch.setattr(gpu, "getDevice", lambda: 1)
    q = pl.LazyFrame({})
    assert_gpu_result_equal(q, engine=pl.GPUEngine(device=0))


@pytest.mark.parametrize("device", [-1, "foo"])
def test_invalid_device_raises(device, monkeypatch):
    monkeypatch.setattr(cudf_polars.callback, "SEEN_DEVICE", None)
    q = pl.LazyFrame({})
    if isinstance(device, int):
        with pytest.raises(rmm._cuda.gpu.CUDARuntimeError):
            q.collect(engine=pl.GPUEngine(device=device))
    elif isinstance(device, str):
        with pytest.raises(TypeError):
            q.collect(engine=pl.GPUEngine(device=device))


def test_multiple_devices_in_same_process_raise(monkeypatch):
    # A device we haven't already seen
    monkeypatch.setattr(cudf_polars.callback, "SEEN_DEVICE", 4)
    q = pl.LazyFrame({})
    with pytest.raises(RuntimeError):
        q.collect(engine=pl.GPUEngine())


@pytest.mark.parametrize("mr", [1, object()])
def test_invalid_memory_resource_raises(mr, monkeypatch):
    monkeypatch.setattr(cudf_polars.callback, "SEEN_DEVICE", None)
    q = pl.LazyFrame({})
    with pytest.raises(TypeError):
        q.collect(engine=pl.GPUEngine(memory_resource=mr))


@pytest.mark.skipif(
    not _is_concurrent_managed_access_supported(),
    reason="managed memory not supported",
)
@pytest.mark.parametrize("enable_managed_memory", ["1", "0"])
@pytest.mark.usefixtures("clear_memory_resource_cache")
def test_cudf_polars_enable_disable_managed_memory(monkeypatch, enable_managed_memory):
    q = pl.LazyFrame({"a": [1, 2, 3]})

    with monkeypatch.context() as monkeycontext:
        monkeycontext.setenv(
            "POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY", enable_managed_memory
        )
        result = q.collect(engine=pl.GPUEngine())
        mr = default_memory_resource(
            0,
            cuda_managed_memory=bool(enable_managed_memory == "1"),
            memory_resource_config=None,
        )
        if enable_managed_memory == "1":
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


def test_nested_memory_resource_config():
    spec = {
        "qualname": "rmm.mr.PrefetchResourceAdaptor",
        "options": {
            "upstream_mr": {
                "qualname": "rmm.mr.PoolMemoryResource",
                "options": {
                    "upstream_mr": {
                        "qualname": "rmm.mr.ManagedMemoryResource",
                    },
                    "initial_pool_size": 256,
                },
            }
        },
    }

    engine = pl.GPUEngine(
        executor="streaming",
        memory_resource_config=MemoryResourceConfig(
            **spec,
        ),
    )
    config = ConfigOptions.from_polars_engine(engine)
    mr = config.memory_resource_config.create_memory_resource()
    assert isinstance(mr, rmm.mr.PrefetchResourceAdaptor)
    assert isinstance(mr.upstream_mr, rmm.mr.PoolMemoryResource)
    assert mr.upstream_mr.pool_size() == 256
    assert isinstance(mr.upstream_mr.upstream_mr, rmm.mr.ManagedMemoryResource)

    assert hash(config.memory_resource_config) == hash(config.memory_resource_config)


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


def test_parquet_options_from_none() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            parquet_options=None,
        )
    )
    assert config.parquet_options.chunked is True


def test_validate_streaming_executor_shuffle_method(
    *, rapidsmpf_single_available: bool
) -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={"shuffle_method": "tasks"},
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.shuffle_method == "tasks"

    # rapidsmpf with single cluster
    engine = pl.GPUEngine(
        executor="streaming",
        executor_options={"shuffle_method": "rapidsmpf", "cluster": "single"},
    )

    if rapidsmpf_single_available:
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.name == "streaming"
        assert config.executor.shuffle_method == "rapidsmpf-single"
    else:
        with pytest.raises(ValueError, match="rapidsmpf is not installed"):
            ConfigOptions.from_polars_engine(engine)


def test_join_rapidsmpf_single_private_config() -> None:
    # The user may not specify "rapidsmpf-single" directly
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "shuffle_method": "rapidsmpf-single",
            "runtime": "tasks",
        },
    )
    with pytest.raises(ValueError, match="not a supported shuffle method"):
        ConfigOptions.from_polars_engine(engine)


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


def test_validate_cluster() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.cluster == "single"

    with pytest.raises(ValueError, match="'foo' is not a valid Cluster"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"cluster": "foo"},
            )
        )


def test_validate_shuffle_method_defaults() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.shuffle_method == "tasks"  # Default for single cluster

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
        "sink_to_directory",
        "client_device_threshold",
        "max_io_threads",
        "spill_to_pinned_memory",
        "num_py_executors",
    ],
)
def test_validate_streaming_executor_options(option: str) -> None:
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
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__USE_RAPIDSMPF_NATIVE", "0")

        # Test default
        engine = pl.GPUEngine()
        config = ConfigOptions.from_polars_engine(engine)
        assert config.parquet_options.chunked is False
        assert config.parquet_options.n_output_chunks == 2
        assert config.parquet_options.chunk_read_limit == 100
        assert config.parquet_options.pass_read_limit == 200
        assert config.parquet_options.max_footer_samples == 0
        assert config.parquet_options.max_row_group_samples == 0
        assert config.parquet_options.use_rapidsmpf_native is False

    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__CHUNKED", "foo")
        engine = pl.GPUEngine()
        with pytest.raises(ValueError, match="Invalid boolean value: 'foo'"):
            ConfigOptions.from_polars_engine(engine)


def test_config_option_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__CLUSTER", "single")
        m.setenv("CUDF_POLARS__EXECUTOR__FALLBACK_MODE", "silent")
        m.setenv("CUDF_POLARS__EXECUTOR__MAX_ROWS_PER_PARTITION", "42")
        m.setenv("CUDF_POLARS__EXECUTOR__UNIQUE_FRACTION", '{"a": 0.5}')
        m.setenv("CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE", "100")
        m.setenv("CUDF_POLARS__EXECUTOR__GROUPBY_N_ARY", "43")
        m.setenv("CUDF_POLARS__EXECUTOR__BROADCAST_JOIN_LIMIT", "44")
        m.setenv("CUDF_POLARS__EXECUTOR__SHUFFLE_METHOD", "tasks")
        m.setenv("CUDF_POLARS__CUDA_STREAM_POLICY", "default")

        engine = pl.GPUEngine()
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.name == "streaming"
        assert config.executor.cluster == "single"
        assert config.executor.fallback_mode == "silent"
        assert config.executor.max_rows_per_partition == 42
        assert config.executor.unique_fraction == {"a": 0.5}
        assert config.executor.target_partition_size == 100
        assert config.executor.groupby_n_ary == 43
        assert config.executor.broadcast_join_limit == 44
        assert config.executor.shuffle_method == "tasks"
        assert config.cuda_stream_policy is None


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


@pytest.mark.parametrize(
    "option",
    [
        "chunked",
        "n_output_chunks",
        "chunk_read_limit",
        "pass_read_limit",
        "max_footer_samples",
        "max_row_group_samples",
        "use_rapidsmpf_native",
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
            pl.GPUEngine(executor="streaming", raise_on_fail=cast(bool, object()))
        )


def test_validate_executor() -> None:
    with pytest.raises(ValueError, match="Unknown executor 'foo'"):
        ConfigOptions.from_polars_engine(pl.GPUEngine(executor="foo"))


def test_default_executor() -> None:
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.name == "streaming"


def test_default_runtime() -> None:
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.name == "streaming"
    assert config.executor.runtime == "tasks"


@pytest.mark.parametrize(
    "memory_resource, memory_resource_config",
    [
        (None, None),
        (
            None,
            MemoryResourceConfig(
                qualname="rmm.mr.CudaAsyncMemoryResource",
                options={"initial_pool_size": 123, "release_threshold": 456},
            ),
        ),
        (rmm.mr.CudaAsyncMemoryResource(initial_pool_size=100), None),
        # prioritize the concrete MR
        (
            rmm.mr.CudaAsyncMemoryResource(initial_pool_size=100),
            MemoryResourceConfig(qualname="rmm.mr.CudaMemoryResource"),
        ),
    ],
)
def test_memory_resource(memory_resource, memory_resource_config) -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            memory_resource=memory_resource,
            memory_resource_config=memory_resource_config,
        )
    )

    with set_memory_resource(
        memory_resource, memory_resource_config, config.executor
    ) as result:
        if memory_resource is None and memory_resource_config is None:
            # The default case: We make a new RMM MR, whose type depends on the GPU's features.

            if _is_concurrent_managed_access_supported():
                assert isinstance(result, rmm.mr.PrefetchResourceAdaptor)
            else:
                assert isinstance(result, rmm.mr.CudaAsyncMemoryResource)

        elif memory_resource is None:
            # Configured through memory_resource_config
            assert isinstance(result, rmm.mr.CudaAsyncMemoryResource)
            assert config.memory_resource_config is not None
            assert (
                config.memory_resource_config.qualname
                == "rmm.mr.CudaAsyncMemoryResource"
            )
            assert config.memory_resource_config.options == {
                "initial_pool_size": 123,
                "release_threshold": 456,
            }
            assert isinstance(
                config.memory_resource_config.create_memory_resource(),
                rmm.mr.CudaAsyncMemoryResource,
            )

        elif memory_resource is not None:
            assert result is memory_resource
        else:  # pragma: no cover; Unreachable
            raise ValueError("Unreachable")


def test_memory_resource_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv(
            "CUDF_POLARS__MEMORY_RESOURCE_CONFIG__QUALNAME",
            "rmm.mr.CudaAsyncMemoryResource",
        )
        m.setenv(
            "CUDF_POLARS__MEMORY_RESOURCE_CONFIG__OPTIONS",
            '{"initial_pool_size": 123, "release_threshold": 456}',
        )
        engine = pl.GPUEngine()
        config = ConfigOptions.from_polars_engine(engine)
        assert config.memory_resource_config is not None
        assert (
            config.memory_resource_config.qualname == "rmm.mr.CudaAsyncMemoryResource"
        )
        assert config.memory_resource_config.options == {
            "initial_pool_size": 123,
            "release_threshold": 456,
        }


def test_ir_execution_context() -> None:
    context = IRExecutionContext()
    assert context.get_cuda_stream is get_cuda_stream
    context.get_cuda_stream()  # no exception


def test_cuda_stream_pool():
    pool_config = CUDAStreamPoolConfig()
    pool = pool_config.build()

    assert pool.get_pool_size() == 16

    # override the defaults
    pool_config = CUDAStreamPoolConfig(pool_size=32, flags=CudaStreamFlags.NON_BLOCKING)
    pool = pool_config.build()
    assert pool.get_pool_size() == 32


def test_cuda_stream_policy_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default from engine
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.cuda_stream_policy is None

    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(executor_options={"runtime": "tasks"})
    )
    assert config.cuda_stream_policy is None

    # Default from env
    monkeypatch.setenv("CUDF_POLARS__CUDA_STREAM_POLICY", "default")
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.cuda_stream_policy is None

    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(executor_options={"runtime": "tasks"})
    )
    assert config.cuda_stream_policy is None


def test_default_cuda_stream_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDF_POLARS__CUDA_STREAM_POLICY", raising=False)
    assert _default_cuda_stream_policy() is None

    monkeypatch.setenv("CUDF_POLARS__CUDA_STREAM_POLICY", "pool")
    result = _default_cuda_stream_policy()
    assert isinstance(result, CUDAStreamPoolConfig)


def test_cuda_stream_policy_from_config(*, rapidsmpf_single_available: bool) -> None:
    engine = pl.GPUEngine(
        executor="streaming",
        executor_options={"runtime": "rapidsmpf"},
        cuda_stream_policy={
            "pool_size": 32,
            "flags": rmm.pylibrmm.CudaStreamFlags.NON_BLOCKING,
        },
    )
    if rapidsmpf_single_available:
        config = ConfigOptions.from_polars_engine(engine)
        assert isinstance(config.cuda_stream_policy, CUDAStreamPoolConfig)
        assert config.cuda_stream_policy.pool_size == 32
        assert (
            config.cuda_stream_policy.flags == rmm.pylibrmm.CudaStreamFlags.NON_BLOCKING
        )
        config.cuda_stream_policy.build().get_stream()  # no exception
    else:
        with pytest.raises(ValueError, match="The rapidsmpf streaming engine"):
            ConfigOptions.from_polars_engine(engine)


@pytest.mark.parametrize(
    "env",
    [
        "default",
        "pool",
        '{"pool_size": 32, "flags": "SYNC_DEFAULT"}',
        '{"pool_size": 32, "flags": 0}',
        '{"pool_size": 32}',
    ],
)
def test_cuda_stream_policy_from_env(
    monkeypatch: pytest.MonkeyPatch, env: str, *, rapidsmpf_single_available: bool
) -> None:
    monkeypatch.setenv("CUDF_POLARS__CUDA_STREAM_POLICY", env)
    runtime = "tasks" if env == "default" else "rapidsmpf"
    engine = pl.GPUEngine(executor="streaming", executor_options={"runtime": runtime})
    if runtime == "rapidsmpf" and rapidsmpf_single_available:
        config = ConfigOptions.from_polars_engine(engine)
        assert isinstance(config.cuda_stream_policy, CUDAStreamPoolConfig)
        if env == "pool":
            assert config.cuda_stream_policy.pool_size == 16
            assert config.cuda_stream_policy.flags == CudaStreamFlags.NON_BLOCKING
        else:
            assert config.cuda_stream_policy.pool_size == 32
    elif runtime == "rapidsmpf":
        with pytest.raises(ValueError, match="The rapidsmpf streaming engine"):
            ConfigOptions.from_polars_engine(engine)
    else:
        config = ConfigOptions.from_polars_engine(engine)
        assert config.cuda_stream_policy is None


def test_cuda_stream_policy_from_env_invalid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUDF_POLARS__CUDA_STREAM_POLICY", '{"foo": "bar"}')
    with pytest.raises(ValueError, match="Invalid CUDA stream policy"):
        ConfigOptions.from_polars_engine(pl.GPUEngine())


def test_cuda_stream_policy_default_rapidsmpf(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rapidsmpf")

    # Default from engine
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(executor_options={"runtime": "rapidsmpf"})
    )
    assert isinstance(config.cuda_stream_policy, CUDAStreamPoolConfig)
    assert config.cuda_stream_policy.pool_size == 16
    assert config.cuda_stream_policy.flags == rmm.pylibrmm.CudaStreamFlags.NON_BLOCKING

    # "default" user argument overrides pool default
    monkeypatch.setenv("CUDF_POLARS__CUDA_STREAM_POLICY", "default")
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(executor_options={"runtime": "rapidsmpf"})
    )
    assert config.cuda_stream_policy is None


@pytest.mark.parametrize(
    "polars_kwargs",
    [
        {"executor": "in-memory"},
        {"executor": "streaming", "executor_options": {"runtime": "tasks"}},
    ],
)
def test_cuda_stream_policy_pool_only_supported_by_rapidsmpf(
    polars_kwargs: dict[str, Any],
) -> None:
    with pytest.raises(
        ValueError,
        match="A stream pool is only supported by the rapidsmpf runtime.",
    ):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                **polars_kwargs,
                cuda_stream_policy={"pool_size": 32, "flags": "NON_BLOCKING"},
            )
        )


def test_validate_cuda_stream_policy() -> None:
    with pytest.raises(ValueError, match="Invalid CUDA stream policy: 'foo'"):
        ConfigOptions.from_polars_engine(pl.GPUEngine(cuda_stream_policy="foo"))


def test_validate_dynamic_planning() -> None:
    with pytest.raises(TypeError, match="sample_chunk_count must be"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"dynamic_planning": {"sample_chunk_count": object()}},
            )
        )


def test_dynamic_planning_sample_chunk_count_min() -> None:
    with pytest.raises(ValueError, match="sample_chunk_count must be at least 1"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"dynamic_planning": {"sample_chunk_count": 0}},
            )
        )


def test_dynamic_planning_defaults() -> None:
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.name == "streaming"
    # Dynamic planning is enabled by default
    assert config.executor.dynamic_planning is not None
    assert config.executor.dynamic_planning.sample_chunk_count == 2
    assert config.executor.dynamic_planning.bloom_filter_threshold == 0.5


def test_dynamic_planning_disabled_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test that env var can disable dynamic planning
    monkeypatch.setenv("CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING", "0")
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.name == "streaming"
    assert config.executor.dynamic_planning is None


def test_dynamic_planning_sample_chunk_count_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Test that sample_chunk_count_reduce can be configured via env var
    monkeypatch.setenv(
        "CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING__SAMPLE_CHUNK_COUNT", "3"
    )
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.name == "streaming"
    assert config.executor.dynamic_planning is not None
    assert config.executor.dynamic_planning.sample_chunk_count == 3


def test_validate_bloom_filter_threshold_type() -> None:
    with pytest.raises(TypeError, match="bloom_filter_threshold must be a float"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={
                    "dynamic_planning": {"bloom_filter_threshold": "bad"}
                },
            )
        )


def test_validate_bloom_filter_threshold_range() -> None:
    with pytest.raises(ValueError, match="bloom_filter_threshold must be between"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"dynamic_planning": {"bloom_filter_threshold": 1.5}},
            )
        )


def test_bloom_filter_threshold_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING__BLOOM_FILTER_THRESHOLD", "0.3"
    )
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.dynamic_planning is not None
    assert config.executor.dynamic_planning.bloom_filter_threshold == 0.3


def test_dynamic_planning_from_instance() -> None:
    from cudf_polars.utils.config import DynamicPlanningOptions

    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={"dynamic_planning": DynamicPlanningOptions()},
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.dynamic_planning is not None
    assert config.executor.dynamic_planning.sample_chunk_count == 2  # default


def test_parse_memory_resource_config() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            memory_resource_config={
                "qualname": "rmm.mr.CudaAsyncMemoryResource",
                "options": {
                    "initial_pool_size": 123,
                    "release_threshold": 456,
                },
            }
        )
    )
    assert isinstance(config.memory_resource_config, MemoryResourceConfig)
    assert config.memory_resource_config.qualname == "rmm.mr.CudaAsyncMemoryResource"


def test_memory_resource_config_raises() -> None:
    with pytest.raises(
        ValueError,
        match="MemoryResourceConfig.qualname 'foo' must be a fully qualified name to a class",
    ):
        MemoryResourceConfig(qualname="foo")


@pytest.mark.parametrize("options", [None, {}])
def test_memory_resource_config_hash(options) -> None:
    config = MemoryResourceConfig(qualname="rmm.mr.CudaMemoryResource", options=options)
    assert hash(config) == hash(config)


def test_num_py_executors_default() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.num_py_executors == 8


def test_num_py_executors_from_executor_options() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={"num_py_executors": 4},
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.num_py_executors == 4


def test_num_py_executors_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__NUM_PY_EXECUTORS", "8")
        engine = pl.GPUEngine(executor="streaming")
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.name == "streaming"
        assert config.executor.num_py_executors == 8


def test_dask_sink_to_directory_false_raises() -> None:
    with pytest.raises(
        ValueError, match="The dask cluster requires sink_to_directory=True"
    ):
        StreamingExecutor(cluster=Cluster.DASK, sink_to_directory=False)


def test_get_dask_cuda_stream() -> None:
    stream = get_dask_cuda_stream()
    assert stream is not None
