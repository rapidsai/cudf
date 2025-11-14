# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Device-aware assertions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from polars import GPUEngine
from polars.testing.asserts import assert_frame_equal

from cudf_polars.dsl.translate import Translator
from cudf_polars.utils.config import ConfigOptions, StreamingFallbackMode
from cudf_polars.utils.versions import POLARS_VERSION_LT_1323

if TYPE_CHECKING:
    from cudf_polars.typing import OptimizationArgs


__all__: list[str] = [
    "assert_gpu_result_equal",
    "assert_ir_translation_raises",
    "assert_sink_ir_translation_raises",
    "assert_sink_result_equal",
]

# Will be overriden by `conftest.py` with the value from the `--executor`
# and `--cluster` command-line arguments
DEFAULT_EXECUTOR = "in-memory"
DEFAULT_RUNTIME = "tasks"
DEFAULT_CLUSTER = "single"
DEFAULT_BLOCKSIZE_MODE: Literal["small", "default"] = "default"


def assert_gpu_result_equal(
    lazydf: pl.LazyFrame,
    *,
    engine: GPUEngine | None = None,
    collect_kwargs: dict[OptimizationArgs, bool] | None = None,
    polars_collect_kwargs: dict[OptimizationArgs, bool] | None = None,
    cudf_collect_kwargs: dict[OptimizationArgs, bool] | None = None,
    check_row_order: bool = True,
    check_column_order: bool = True,
    check_dtypes: bool = True,
    check_exact: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    categorical_as_str: bool = False,
    executor: str | None = None,
    blocksize_mode: Literal["small", "default"] | None = None,
) -> None:
    """
    Assert that collection of a lazyframe on GPU produces correct results.

    Parameters
    ----------
    lazydf
        frame to collect.
    engine
        Custom GPU engine configuration.
    collect_kwargs
        Common keyword arguments to pass to collect for both polars CPU and
        cudf-polars.
        Useful for controlling optimization settings.
    polars_collect_kwargs
        Keyword arguments to pass to collect for execution on polars CPU.
        Overrides kwargs in collect_kwargs.
        Useful for controlling optimization settings.
    cudf_collect_kwargs
        Keyword arguments to pass to collect for execution on cudf-polars.
        Overrides kwargs in collect_kwargs.
        Useful for controlling optimization settings.
    check_row_order
        Expect rows to be in same order
    check_column_order
        Expect columns to be in same order
    check_dtypes
        Expect dtypes to match
    check_exact
        Require exact equality for floats, if `False` compare using
        rtol and atol.
    rtol
        Relative tolerance for float comparisons
    atol
        Absolute tolerance for float comparisons
    categorical_as_str
        Decat categoricals to strings before comparing
    executor
        The executor configuration to pass to `GPUEngine`. If not specified
        uses the module level `Executor` attribute.
    blocksize_mode
        The "mode" to use for choosing the blocksize for the streaming executor.
        If not specified, uses the module level ``DEFAULT_BLOCKSIZE_MODE`` attribute.
        Set to "small" to configure small values for ``max_rows_per_partition``
        and ``target_partition_size``, which will typically cause many partitions
        to be created while executing the query.

    Raises
    ------
    AssertionError
        If the GPU and CPU collection do not match.
    NotImplementedError
        If GPU collection failed in some way.
    """
    engine = engine or get_default_engine(executor, blocksize_mode)
    final_polars_collect_kwargs, final_cudf_collect_kwargs = _process_kwargs(
        collect_kwargs, polars_collect_kwargs, cudf_collect_kwargs
    )

    # These keywords are correct, but mypy doesn't see that.
    # the 'misc' is for 'error: Keywords must be strings'
    expect = lazydf.collect(**final_polars_collect_kwargs)  # type: ignore[misc, call-overload]
    got = lazydf.collect(**final_cudf_collect_kwargs, engine=engine)  # type: ignore[misc, call-overload]

    assert_kwargs_bool: dict[str, bool] = {
        "check_row_order": check_row_order,
        "check_column_order": check_column_order,
        "check_dtypes": check_dtypes,
        "check_exact": check_exact,
        "categorical_as_str": categorical_as_str,
    }

    tol_kwargs: dict[str, float]
    if POLARS_VERSION_LT_1323:  # pragma: no cover
        tol_kwargs = {"rtol": rtol, "atol": atol}
    else:
        tol_kwargs = {"rel_tol": rtol, "abs_tol": atol}

    # the type checker errors with:
    # Argument 4 to "assert_frame_equal" has incompatible type "**dict[str, float]"; expected "bool"  [arg-type]
    # which seems to be a bug in the type checker / type annotations.
    assert_frame_equal(
        expect,
        got,
        **assert_kwargs_bool,
        **tol_kwargs,  # type: ignore[arg-type]
    )


def assert_ir_translation_raises(q: pl.LazyFrame, *exceptions: type[Exception]) -> None:
    """
    Assert that translation of a query raises an exception.

    Parameters
    ----------
    q
        Query to translate.
    exceptions
        Exceptions that one expects might be raised.

    Returns
    -------
    None
        If translation successfully raised the specified exceptions.

    Raises
    ------
    AssertionError
       If the specified exceptions were not raised.
    """
    translator = Translator(q._ldf.visit(), GPUEngine())
    translator.translate_ir()
    if errors := translator.errors:
        for err in errors:
            assert any(isinstance(err, err_type) for err_type in exceptions), (
                f"Translation DID NOT RAISE {exceptions}. The following "
                f"errors were seen instead: {errors}"
            )
        return
    else:
        raise AssertionError(f"Translation DID NOT RAISE {exceptions}")


def get_default_engine(
    executor: str | None = None,
    blocksize_mode: Literal["small", "default"] | None = None,
) -> GPUEngine:
    """
    Get the default engine used for testing.

    Parameters
    ----------
    executor
        The executor configuration to pass to `GPUEngine`. If not specified
        uses the module level `Executor` attribute.
    blocksize_mode
        The "mode" to use for choosing the blocksize for the streaming executor.
        If not specified, uses the module level ``DEFAULT_BLOCKSIZE_MODE`` attribute.
        Set to "small" to configure small values for ``max_rows_per_partition``
        and ``target_partition_size``, which will typically cause many partitions
        to be created while executing the query.

    Returns
    -------
    engine
        A polars GPUEngine configured with the default settings for tests.

    See Also
    --------
    assert_gpu_result_equal
    assert_sink_result_equal
    """
    executor_options: dict[str, Any] = {}
    executor = executor or DEFAULT_EXECUTOR
    if executor == "streaming":
        executor_options["cluster"] = DEFAULT_CLUSTER
        executor_options["runtime"] = DEFAULT_RUNTIME

        blocksize_mode = blocksize_mode or DEFAULT_BLOCKSIZE_MODE

        if blocksize_mode == "small":  # pragma: no cover
            executor_options["max_rows_per_partition"] = 4
            executor_options["target_partition_size"] = 10
            # We expect many tests to fall back, so silence the warnings
            executor_options["fallback_mode"] = StreamingFallbackMode.SILENT

    return GPUEngine(
        raise_on_fail=True,
        executor=executor,
        executor_options=executor_options,
    )


def _process_kwargs(
    collect_kwargs: dict[OptimizationArgs, bool] | None,
    polars_collect_kwargs: dict[OptimizationArgs, bool] | None,
    cudf_collect_kwargs: dict[OptimizationArgs, bool] | None,
) -> tuple[dict[OptimizationArgs, bool], dict[OptimizationArgs, bool]]:
    if collect_kwargs is None:
        collect_kwargs = {}
    final_polars_collect_kwargs = collect_kwargs.copy()
    final_cudf_collect_kwargs = collect_kwargs.copy()
    if polars_collect_kwargs is not None:  # pragma: no cover; not currently used
        final_polars_collect_kwargs.update(polars_collect_kwargs)
    if cudf_collect_kwargs is not None:  # pragma: no cover; not currently used
        final_cudf_collect_kwargs.update(cudf_collect_kwargs)
    return final_polars_collect_kwargs, final_cudf_collect_kwargs


def assert_collect_raises(
    lazydf: pl.LazyFrame,
    *,
    polars_except: type[Exception] | tuple[type[Exception], ...],
    cudf_except: type[Exception] | tuple[type[Exception], ...],
    collect_kwargs: dict[OptimizationArgs, bool] | None = None,
    polars_collect_kwargs: dict[OptimizationArgs, bool] | None = None,
    cudf_collect_kwargs: dict[OptimizationArgs, bool] | None = None,
) -> None:
    """
    Assert that collecting the result of a query raises the expected exceptions.

    Parameters
    ----------
    lazydf
        frame to collect.
    collect_kwargs
        Common keyword arguments to pass to collect for both polars CPU and
        cudf-polars.
        Useful for controlling optimization settings.
    polars_except
        Exception or exceptions polars CPU is expected to raise. If
        an empty tuple ``()``, CPU is expected to succeed without raising.
    cudf_except
        Exception or exceptions polars GPU is expected to raise. If
        an empty tuple ``()``, GPU is expected to succeed without raising.
    collect_kwargs
        Common keyword arguments to pass to collect for both polars CPU and
        cudf-polars.
        Useful for controlling optimization settings.
    polars_collect_kwargs
        Keyword arguments to pass to collect for execution on polars CPU.
        Overrides kwargs in collect_kwargs.
        Useful for controlling optimization settings.
    cudf_collect_kwargs
        Keyword arguments to pass to collect for execution on cudf-polars.
        Overrides kwargs in collect_kwargs.
        Useful for controlling optimization settings.

    Returns
    -------
    None
        If both sides raise the expected exceptions.

    Raises
    ------
    AssertionError
        If either side did not raise the expected exceptions.
    """
    final_polars_collect_kwargs, final_cudf_collect_kwargs = _process_kwargs(
        collect_kwargs, polars_collect_kwargs, cudf_collect_kwargs
    )

    try:
        lazydf.collect(**final_polars_collect_kwargs)  # type: ignore[misc, call-overload]
    except polars_except:
        pass
    except Exception as e:
        raise AssertionError(
            f"CPU execution RAISED {type(e)}, EXPECTED {polars_except}"
        ) from e
    else:
        if polars_except != ():
            raise AssertionError(f"CPU execution DID NOT RAISE {polars_except}")

    engine = GPUEngine(raise_on_fail=True)
    try:
        lazydf.collect(**final_cudf_collect_kwargs, engine=engine)  # type: ignore[misc, call-overload]
    except cudf_except:
        pass
    except Exception as e:
        raise AssertionError(
            f"GPU execution RAISED {type(e)}, EXPECTED {cudf_except}"
        ) from e
    else:
        if cudf_except != ():
            raise AssertionError(f"GPU execution DID NOT RAISE {cudf_except}")


def _resolve_sink_format(path: Path) -> str:
    """Returns valid sink format for assert utilities."""
    suffix = path.suffix.lower()
    supported_ext = {
        ".csv": "csv",
        ".pq": "parquet",
        ".parquet": "parquet",
        ".json": "ndjson",
        ".ndjson": "ndjson",
    }
    if suffix not in supported_ext:
        raise ValueError(f"Unsupported file format: {suffix}")
    return supported_ext[suffix]


def assert_sink_result_equal(
    lazydf: pl.LazyFrame,
    path: str | Path,
    *,
    engine: str | GPUEngine | None = None,
    read_kwargs: dict | None = None,
    write_kwargs: dict | None = None,
    executor: str | None = None,
    blocksize_mode: Literal["small", "default"] | None = None,
) -> None:
    """
    Assert that writing a LazyFrame via sink produces the same output.

    Parameters
    ----------
    lazydf
        The LazyFrame to sink.
    path
        The file path to use. Suffix must be one of:
        '.csv', '.parquet', '.pq', '.json', '.ndjson'.
    engine
        The GPU engine to use for the sink operation.
    read_kwargs
        Optional keyword arguments to pass to the corresponding `pl.read_*` function.
    write_kwargs
        Optional keyword arguments to pass to the corresponding `sink_*` function.
    executor
        The executor configuration to pass to `GPUEngine`. If not specified
        uses the module level `Executor` attribute.
    blocksize_mode
        The "mode" to use for choosing the blocksize for the streaming executor.
        If not specified, uses the module level ``DEFAULT_BLOCKSIZE_MODE`` attribute.
        Set to "small" to configure small values for ``max_rows_per_partition``
        and ``target_partition_size``, which will typically cause many partitions
        to be created while executing the query.

    Raises
    ------
    AssertionError
        If the outputs from CPU and GPU sink differ.
    ValueError
        If the file extension is not one of the supported formats.
    """
    engine = engine or get_default_engine(executor, blocksize_mode)
    path = Path(path)
    read_kwargs = read_kwargs or {}
    write_kwargs = write_kwargs or {}

    fmt = _resolve_sink_format(path)

    cpu_path = path.with_name(f"{path.stem}_cpu{path.suffix}")
    gpu_path = path.with_name(f"{path.stem}_gpu{path.suffix}")

    sink_fn = getattr(lazydf, f"sink_{fmt}")
    read_fn = getattr(pl, f"read_{fmt}")

    sink_fn(cpu_path, **write_kwargs)
    sink_fn(gpu_path, engine=engine, **write_kwargs)

    expected = read_fn(cpu_path, **read_kwargs)
    # the multi-partition executor might produce multiple files, one per partition.
    if (
        isinstance(engine, GPUEngine)
        and ConfigOptions.from_polars_engine(engine).executor.name == "streaming"
        and gpu_path.is_dir()
    ):  # pragma: no cover
        result = read_fn(gpu_path.joinpath("*"), **read_kwargs)
    else:
        result = read_fn(gpu_path, **read_kwargs)

    assert_frame_equal(expected, result)


def assert_sink_ir_translation_raises(
    lazydf: pl.LazyFrame,
    path: str | Path,
    write_kwargs: dict,
    *exceptions: type[Exception],
) -> None:
    """
    Assert that translation of a sink query raises an exception.

    Parameters
    ----------
    lazydf
        The LazyFrame to sink.
    path
        The file path. Must have one of the supported suffixes.
    write_kwargs
        Keyword arguments to pass to the `sink_*` method.
    *exceptions
        One or more expected exception types that should be raised during translation.

    Raises
    ------
    AssertionError
        If translation does not raise any of the expected exceptions.
        If an exception occurs before translation begins.
    ValueError
        If the file extension is not one of the supported formats.
    """
    path = Path(path)
    fmt = _resolve_sink_format(path)

    try:
        lazy_sink = getattr(lazydf, f"sink_{fmt}")(
            path,
            engine="gpu",
            lazy=True,
            **write_kwargs,
        )
    except Exception as e:
        raise AssertionError(
            f"Sink function raised an exception before translation: {e}"
        ) from e

    assert_ir_translation_raises(lazy_sink, *exceptions)
