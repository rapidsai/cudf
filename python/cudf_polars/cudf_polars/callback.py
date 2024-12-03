# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Callback for the polars collect function to execute on device."""

from __future__ import annotations

import contextlib
import os
import warnings
from functools import cache, partial
from typing import TYPE_CHECKING, Literal

import nvtx

from polars.exceptions import ComputeError, PerformanceWarning

import pylibcudf
import rmm
from rmm._cuda import gpu

from cudf_polars.dsl.translate import Translator

if TYPE_CHECKING:
    from collections.abc import Generator

    import polars as pl
    from polars import GPUEngine

    from cudf_polars.dsl.ir import IR
    from cudf_polars.typing import NodeTraverser

__all__: list[str] = ["execute_with_cudf"]


_SUPPORTED_PREFETCHES = {
    "column_view::get_data",
    "mutable_column_view::get_data",
    "gather",
    "hash_join",
}


def _env_get_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (ValueError, TypeError):  # pragma: no cover
        return default  # pragma: no cover


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
                "If your system has a CUDA 11 driver, please uninstall `cudf-polars-cu12` "
                "and install `cudf-polars-cu11`"
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
                _env_get_int("POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY", default=1) != 0
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


def _callback(
    ir: IR,
    with_columns: list[str] | None,
    pyarrow_predicate: str | None,
    n_rows: int | None,
    *,
    device: int | None,
    memory_resource: int | None,
    executor: Literal["pylibcudf", "dask-experimental"] | None,
) -> pl.DataFrame:
    assert with_columns is None
    assert pyarrow_predicate is None
    assert n_rows is None
    with (
        nvtx.annotate(message="ExecuteIR", domain="cudf_polars"),
        # Device must be set before memory resource is obtained.
        set_device(device),
        set_memory_resource(memory_resource),
    ):
        if executor is None or executor == "pylibcudf":
            return ir.evaluate(cache={}).to_polars()
        elif executor == "dask-experimental":
            from cudf_polars.experimental.parallel import evaluate_dask

            return evaluate_dask(ir).to_polars()
        else:
            raise ValueError(f"Unknown executor '{executor}'")


def validate_config_options(config: dict) -> None:
    """
    Validate the configuration options for the GPU engine.

    Parameters
    ----------
    config
        Configuration options to validate.

    Raises
    ------
    ValueError
        If the configuration contains unsupported options.
    """
    if unsupported := (
        config.keys()
        - {"raise_on_fail", "parquet_options", "executor", "executor_options"}
    ):
        raise ValueError(
            f"Engine configuration contains unsupported settings: {unsupported}"
        )
    assert {"chunked", "chunk_read_limit", "pass_read_limit"}.issuperset(
        config.get("parquet_options", {})
    )

    # Validate executor_options
    executor = config.get("executor", "pylibcudf")
    if executor == "dask-experimental":
        unsupported = config.get("executor_options", {}).keys() - {
            "max_rows_per_partition"
        }
    else:
        unsupported = config.get("executor_options", {}).keys()
    if unsupported:
        raise ValueError(f"Unsupported executor_options for {executor}: {unsupported}")


def execute_with_cudf(nt: NodeTraverser, *, config: GPUEngine) -> None:
    """
    A post optimization callback that attempts to execute the plan with cudf.

    Parameters
    ----------
    nt
        NodeTraverser

    config
        GPUEngine configuration object

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
    device = config.device
    memory_resource = config.memory_resource
    raise_on_fail = config.config.get("raise_on_fail", False)
    executor = config.config.get("executor", None)
    validate_config_options(config.config)

    with nvtx.annotate(message="ConvertIR", domain="cudf_polars"):
        translator = Translator(nt, config)
        ir = translator.translate_ir()
        ir_translation_errors = translator.errors
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
            if raise_on_fail:
                raise exception
        else:
            nt.set_udf(
                partial(
                    _callback,
                    ir,
                    device=device,
                    memory_resource=memory_resource,
                    executor=executor,
                )
            )
