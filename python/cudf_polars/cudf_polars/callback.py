# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Callback for the polars collect function to execute on device."""

from __future__ import annotations

import contextlib
import os
import textwrap
import time
import warnings
from functools import cache, partial
from typing import TYPE_CHECKING, Literal, overload

import nvtx
from typing_extensions import assert_never

from polars.exceptions import ComputeError, PerformanceWarning

import pylibcudf
import rmm
from rmm._cuda import gpu

from cudf_polars.dsl.tracing import CUDF_POLARS_NVTX_DOMAIN
from cudf_polars.dsl.translate import Translator
from cudf_polars.utils.config import _env_get_int, get_total_device_memory
from cudf_polars.utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import Generator

    import polars as pl
    from polars import GPUEngine

    from cudf_polars.dsl.ir import IR
    from cudf_polars.typing import NodeTraverser
    from cudf_polars.utils.config import ConfigOptions

__all__: list[str] = ["execute_with_cudf"]


_SUPPORTED_PREFETCHES = {
    "column_view::get_data",
    "mutable_column_view::get_data",
    "gather",
    "hash_join",
}


@cache
def default_memory_resource(
    device: int,
    cuda_managed_memory: bool,  # noqa: FBT001
) -> rmm.mr.DeviceMemoryResource:
    """
    Return the default memory resource for cudf-polars.

    Parameters
    ----------
    device
        Disambiguating device id when selecting the device. Must be
        the active device when this function is called.
    cuda_managed_memory
        Whether to use managed memory or not.

    Returns
    -------
    rmm.mr.DeviceMemoryResource
        The default memory resource that cudf-polars uses. Currently
        a managed memory resource, if `cuda_managed_memory` is `True`.
        else, an async pool resource is returned.
    """
    try:
        if (
            cuda_managed_memory
            and pylibcudf.utils._is_concurrent_managed_access_supported()
        ):
            # Allocating 80% of the available memory for the pool.
            # Leaving a 20% headroom to avoid OOM errors.
            free_memory, _ = rmm.mr.available_device_memory()
            free_memory = int(round(float(free_memory) * 0.80 / 256) * 256)
            for key in _SUPPORTED_PREFETCHES:
                pylibcudf.experimental.enable_prefetching(key)
            mr = rmm.mr.PrefetchResourceAdaptor(
                rmm.mr.PoolMemoryResource(
                    rmm.mr.ManagedMemoryResource(),
                    initial_pool_size=free_memory,
                )
            )
        else:
            mr = rmm.mr.CudaAsyncMemoryResource()
    except RuntimeError as e:  # pragma: no cover
        msg, *_ = e.args
        if (
            msg.startswith("RMM failure")
            and msg.find("not supported with this CUDA driver/runtime version") > -1
        ):
            raise ComputeError(
                "GPU engine requested, but incorrect cudf-polars package installed. "
                "cudf-polars requires CUDA 12.0+ to installed."
            ) from None
        else:
            raise
    else:
        return mr


@contextlib.contextmanager
def set_memory_resource(
    mr: rmm.mr.DeviceMemoryResource | None,
) -> Generator[rmm.mr.DeviceMemoryResource, None, None]:
    """
    Set the current memory resource for an execution block.

    Parameters
    ----------
    mr
        Memory resource to use. If `None`, calls :func:`default_memory_resource`
        to obtain an mr on the currently active device.

    Returns
    -------
    Memory resource used.

    Notes
    -----
    At exit, the memory resource is restored to whatever was current
    at entry. If a memory resource is provided, it must be valid to
    use with the currently active device.
    """
    previous = rmm.mr.get_current_device_resource()
    if mr is None:
        device: int = gpu.getDevice()
        mr = default_memory_resource(
            device=device,
            cuda_managed_memory=bool(
                _env_get_int(
                    "POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY",
                    default=1 if get_total_device_memory() is not None else 0,
                )
                != 0
            ),
        )
    rmm.mr.set_current_device_resource(mr)
    try:
        yield mr
    finally:
        rmm.mr.set_current_device_resource(previous)


@contextlib.contextmanager
def set_device(device: int | None) -> Generator[int, None, None]:
    """
    Set the device the query is executed on.

    Parameters
    ----------
    device
        Device to use. If `None`, uses the current device.

    Returns
    -------
    Device active for the execution of the block.

    Notes
    -----
    At exit, the device is restored to whatever was current at entry.
    """
    previous: int = gpu.getDevice()
    if device is not None:
        gpu.setDevice(device)
    try:
        yield previous
    finally:
        gpu.setDevice(previous)


@overload
def _callback(
    ir: IR,
    with_columns: list[str] | None,
    pyarrow_predicate: str | None,
    n_rows: int | None,
    should_time: Literal[False],
    *,
    memory_resource: rmm.mr.DeviceMemoryResource | None,
    config_options: ConfigOptions,
    timer: Timer | None,
) -> pl.DataFrame: ...


@overload
def _callback(
    ir: IR,
    with_columns: list[str] | None,
    pyarrow_predicate: str | None,
    n_rows: int | None,
    should_time: Literal[True],
    *,
    memory_resource: rmm.mr.DeviceMemoryResource | None,
    config_options: ConfigOptions,
    timer: Timer | None,
) -> tuple[pl.DataFrame, list[tuple[int, int, str]]]: ...


def _callback(
    ir: IR,
    with_columns: list[str] | None,
    pyarrow_predicate: str | None,
    n_rows: int | None,
    should_time: bool,  # noqa: FBT001
    *,
    memory_resource: rmm.mr.DeviceMemoryResource | None,
    config_options: ConfigOptions,
    timer: Timer | None,
) -> pl.DataFrame | tuple[pl.DataFrame, list[tuple[int, int, str]]]:
    assert with_columns is None
    assert pyarrow_predicate is None
    assert n_rows is None
    if timer is not None:
        assert should_time
    with (
        nvtx.annotate(message="ExecuteIR", domain=CUDF_POLARS_NVTX_DOMAIN),
        # Device must be set before memory resource is obtained.
        set_device(config_options.device),
        set_memory_resource(memory_resource),
    ):
        if config_options.executor.name == "in-memory":
            df = ir.evaluate(cache={}, timer=timer).to_polars()
            if timer is None:
                return df
            else:
                return df, timer.timings
        elif config_options.executor.name == "streaming":
            from cudf_polars.experimental.parallel import evaluate_streaming

            if timer is not None:
                msg = textwrap.dedent("""\
                    LazyFrame.profile() is not supported with the streaming executor.
                    To profile execution with the streaming executor, use:

                    - NVIDIA NSight Systems with the 'streaming' scheduler.
                    - Dask's built-in profiling tools with the 'distributed' scheduler.
                    """)
                raise NotImplementedError(msg)

            return evaluate_streaming(ir, config_options).to_polars()
        assert_never(f"Unknown executor '{config_options.executor}'")


def execute_with_cudf(
    nt: NodeTraverser, duration_since_start: int | None, *, config: GPUEngine
) -> None:
    """
    A post optimization callback that attempts to execute the plan with cudf.

    Parameters
    ----------
    nt
        NodeTraverser

    duration_since_start
        Time since the user started executing the query (or None if no
        profiling should occur).

    config
        GPUEngine object. Configuration is available as ``engine.config``.

    Raises
    ------
    ValueError
        If the config contains unsupported keys.
    NotImplementedError
        If translation of the plan is unsupported.

    Notes
    -----
    The NodeTraverser is mutated if the libcudf executor can handle the plan.
    """
    if duration_since_start is None:
        timer = None
    else:
        start = time.monotonic_ns()
        timer = Timer(start - duration_since_start)

    memory_resource = config.memory_resource

    with nvtx.annotate(message="ConvertIR", domain=CUDF_POLARS_NVTX_DOMAIN):
        translator = Translator(nt, config)
        ir = translator.translate_ir()
        ir_translation_errors = translator.errors
        if timer is not None:
            timer.store(start, time.monotonic_ns(), "gpu-ir-translation")

        if (
            memory_resource is None
            and translator.config_options.executor.name == "streaming"
            and translator.config_options.executor.scheduler == "distributed"
        ):  # pragma: no cover; Requires distributed cluster
            memory_resource = rmm.mr.get_current_device_resource()
        if len(ir_translation_errors):
            # TODO: Display these errors in user-friendly way.
            # tracked in https://github.com/rapidsai/cudf/issues/17051
            unique_errors = sorted(set(ir_translation_errors), key=str)
            formatted_errors = "\n".join(
                f"- {e.__class__.__name__}: {e}" for e in unique_errors
            )
            error_message = (
                "Query execution with GPU not possible: unsupported operations."
                f"\nThe errors were:\n{formatted_errors}"
            )
            exception = NotImplementedError(error_message, unique_errors)
            if bool(int(os.environ.get("POLARS_VERBOSE", 0))):
                warnings.warn(error_message, PerformanceWarning, stacklevel=2)
            if translator.config_options.raise_on_fail:
                raise exception
        else:
            nt.set_udf(
                partial(
                    _callback,
                    ir,
                    memory_resource=memory_resource,
                    config_options=translator.config_options,
                    timer=timer,
                )
            )
