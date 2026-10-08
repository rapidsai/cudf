# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracing and monitoring IR execution."""

from __future__ import annotations

import contextlib
import enum
import functools
import importlib.util
import os
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

import nvtx

from cudf_polars.utils.config import _bool_converter

try:  # pragma: no cover; requires structlog and cudf_polars_quent
    import structlog
except ImportError:  # pragma: no cover; requires no structlog
    _HAS_STRUCTLOG = False
else:  # pragma: no cover; requires structlog
    _HAS_STRUCTLOG = True
_HAS_QUENT = importlib.util.find_spec("cudf_polars_quent") is not None


LOG_TRACES = _HAS_STRUCTLOG and _bool_converter(
    os.environ.get("CUDF_POLARS_LOG_TRACES", "0")
)

CUDF_POLARS_NVTX_DOMAIN = "cudf_polars"

nvtx_annotate_cudf_polars = functools.partial(
    nvtx.annotate, domain=CUDF_POLARS_NVTX_DOMAIN
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import cudf_polars.containers
    from cudf_polars.dsl import ir
    from cudf_polars.dsl.ir import IRExecutionContext


class Scope(enum.StrEnum):
    """Scope values for structured logging."""

    IO_TASK = "io_task"


IRType = TypeVar("IRType", bound="ir.IR")
P = ParamSpec("P")


def log_do_evaluate(
    func: Callable[Concatenate[type[IRType], P], cudf_polars.containers.DataFrame],
) -> Callable[Concatenate[type[IRType], P], cudf_polars.containers.DataFrame]:
    """Emit a Quent lifecycle around an ``IR.do_evaluate`` method."""

    @functools.wraps(func)
    def wrapper(
        cls: type[IRType],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> cudf_polars.containers.DataFrame:
        ir_execution_context: IRExecutionContext = kwargs["context"]  # type: ignore[assignment]
        if (quent_state := ir_execution_context.quent_ir_execution_state) is None:
            return func(cls, *args, **kwargs)

        import cudf_polars_quent as _quent

        # By convention, all non-dataframe arguments (non-child) come first.
        # Anything remaining is a dataframe, except for 'context' kwarg.
        frames: list[cudf_polars.containers.DataFrame] = (
            list(args) + [value for key, value in kwargs.items() if key != "context"]
        )[cls._n_non_child_args :]  # type: ignore[assignment]
        quent_evaluate_id = _quent.now_v7()
        runtime = quent_state.query_worker_state.runtime
        runtime.emit_evaluate_begin(
            cls,
            quent_evaluate_id,
            (
                f"{cls.__name__}-{quent_state.operator_id.hex[:8]}-"
                f"{quent_evaluate_id.hex[:8]}"
            ),
            quent_state,
            frames,
        )
        try:
            result = func(cls, *args, **kwargs)
        except BaseException as error:
            runtime.emit_evaluate_end(quent_evaluate_id, None, error)
            raise
        else:
            runtime.emit_evaluate_end(quent_evaluate_id, result, None)
            return result

    return wrapper


@contextlib.contextmanager
def bound_contextvars(**kwargs: Any) -> Generator[None, None, None]:
    """Wrapper around structlog.contextvars.bound_contextvars."""
    if LOG_TRACES:  # pragma: no cover; requires CUDF_POLARS_LOG_TRACES=1
        with structlog.contextvars.bound_contextvars(**kwargs):
            yield
    else:
        yield


def log(message: str, **kwargs: Any) -> None:
    """Wrapper around structlog.get_logger().info."""
    if LOG_TRACES:  # pragma: no cover; requires CUDF_POLARS_LOG_TRACES=1
        log = structlog.get_logger()
        log.info(message, **kwargs)
