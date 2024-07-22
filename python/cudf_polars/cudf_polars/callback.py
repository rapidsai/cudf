# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Callback for the polars collect function to execute on device."""

from __future__ import annotations

import contextlib
import os
import warnings
from functools import cache, partial
from typing import TYPE_CHECKING

import nvtx

from polars.exceptions import PerformanceWarning

import rmm
from rmm._cuda import gpu

from cudf_polars.dsl.translate import translate_ir

if TYPE_CHECKING:
    from collections.abc import Generator

    import polars as pl
    from polars import GPUEngine

    from cudf_polars.dsl.ir import IR
    from cudf_polars.typing import NodeTraverser

__all__: list[str] = ["execute_with_cudf"]


@cache
def default_memory_resource(device: int) -> rmm.mr.DeviceMemoryResource:
    """
    Return the default memory resource for cudf-polars.

    Parameters
    ----------
    device
        Disambiguating device id when selecting the device. Must be
        the active device when this function is called.

    Returns
    -------
    rmm.mr.DeviceMemoryResource
        The default memory resource that cudf-polars uses. Currently
        an async pool resource.
    """
    return rmm.mr.CudaAsyncMemoryResource()


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
    if mr is None:
        device: int = gpu.getDevice()
        mr = default_memory_resource(device)
    previous = rmm.mr.get_current_device_resource()
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
        return ir.evaluate(cache={}).to_polars()


def execute_with_cudf(
    nt: NodeTraverser,
    *,
    config: GPUEngine,
    exception: type[Exception] | tuple[type[Exception], ...] = Exception,
) -> None:
    """
    A post optimization callback that attempts to execute the plan with cudf.

    Parameters
    ----------
    nt
        NodeTraverser

    config
        GPUEngine configuration object

    exception
        Optional exception, or tuple of exceptions, to catch during
        translation. Defaults to ``Exception``.

    The NodeTraverser is mutated if the libcudf executor can handle the plan.
    """
    device = config.device
    memory_resource = config.memory_resource
    raise_on_fail = config.config.get("raise_on_fail", False)
    if unsupported := (config.config.keys() - {"raise_on_fail"}):
        raise ValueError(
            f"Engine configuration contains unsupported settings {unsupported}"
        )
    try:
        with nvtx.annotate(message="ConvertIR", domain="cudf_polars"):
            nt.set_udf(
                partial(
                    _callback,
                    translate_ir(nt),
                    device=device,
                    memory_resource=memory_resource,
                )
            )
    except exception as e:
        if bool(int(os.environ.get("POLARS_VERBOSE", 0))):
            warnings.warn(
                f"Query execution with GPU not supported, reason: {type(e)}: {e}",
                PerformanceWarning,
                stacklevel=2,
            )
        if raise_on_fail:
            raise
