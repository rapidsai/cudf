# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Config utilities."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from typing_extensions import Self

    import polars as pl


__all__ = ["ConfigOptions"]


@dataclasses.dataclass(frozen=True)
class ParquetOptions:
    """Configuration for the cudf-polars Parquet engine."""

    chunked: bool = True
    chunk_read_limit: int = 0
    pass_read_limit: int = 0


FALLBACK_MODE = Literal["warn", "raise"]


STREAMING_EXECUTOR_PARQUET_DEFAULTS = {
    "chunked": False,
}


@dataclasses.dataclass(frozen=True, eq=True)
class StreamingExecutor:
    """
    Configuration for the cudf-polars streaming executor.

    Parameters
    ----------
    scheduler
        The scheduler to use for the streaming executor.

        "distributed" requires a Dask cluster to be running.
    """

    name: Literal["streaming"] = "streaming"
    scheduler: Literal["synchronous", "distributed"] = "synchronous"
    fallback_mode: FALLBACK_MODE = "warn"
    max_rows_per_partition: int = 1_000_000
    cardinality_factor: dict[str, float] = dataclasses.field(default_factory=dict)
    parquet_blocksize: int = 1_000_000_000  # why isn't this a ParquetOption?
    groupby_n_ary: int = 32
    broadcast_join_limit: int = 4
    shuffle_method: Literal["tasks"] | None = None


@dataclasses.dataclass(frozen=True, eq=True)
class InMemoryExecutor:
    """Configuration for the cudf-polars in-memory executor."""

    name: Literal["in-memory"] = "in-memory"
    scheduler: Literal["synchronous"] = "synchronous"
    shuffle_method: Literal["tasks", "rapidsmpf"] | None = None
    broadcast_join_limit: int = 32


@dataclasses.dataclass(frozen=True, eq=True)
class ConfigOptions:
    """Configuration for the polars GPUEngine."""

    raise_on_fail: bool = False
    # device?
    # memory resource?
    parquet_options: ParquetOptions = dataclasses.field(default_factory=ParquetOptions)
    executor: StreamingExecutor | InMemoryExecutor = dataclasses.field(
        default_factory=InMemoryExecutor
    )

    @classmethod
    def from_polars_engine(cls, engine: pl.GPUEngine) -> Self:
        """
        Create a `ConfigOptions` object from a `pl.GPUEngine` object.

        This creates our internal, typed, configuration object from the
        user-provided `polars.GPUEngine` object.
        """
        # these are the valid top-level keys in the engine.config that
        # the user passes as **kwargs to GPUEngine.
        valid_options = {
            "executor",
            "executor_options",
            "parquet_options",
        }

        extra_options = set(engine.config.keys()) - valid_options
        if extra_options:
            raise ValueError(f"Unsupported options: {extra_options}")

        user_executor = engine.config.get("executor", "in-memory")
        user_executor_options = engine.config.get("executor_options", {})
        user_parquet_options = engine.config.get("parquet_options", {})

        executor: InMemoryExecutor | StreamingExecutor
        match user_executor:
            case "in-memory":
                executor = InMemoryExecutor(**user_executor_options)
            case "streaming":
                executor = StreamingExecutor(**user_executor_options)
                # Update with the streaming defaults, but user options take precedence.
                # TODO: test this!
                user_parquet_options = {
                    **STREAMING_EXECUTOR_PARQUET_DEFAULTS,
                    **user_parquet_options,
                }
            case _:
                raise ValueError(f"Unsupported executor: {user_executor}")

        return cls(
            raise_on_fail=engine.raise_on_fail,
            parquet_options=ParquetOptions(**user_parquet_options),
            executor=executor,
        )
