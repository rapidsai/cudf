# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Device-aware assertions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars import GPUEngine
from polars.testing.asserts import assert_frame_equal

from cudf_polars.dsl.translate import Translator

if TYPE_CHECKING:
    import polars as pl

    from cudf_polars.typing import OptimizationArgs

__all__: list[str] = ["assert_gpu_result_equal", "assert_ir_translation_raises"]


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

    Raises
    ------
    AssertionError
        If the GPU and CPU collection do not match.
    NotImplementedError
        If GPU collection failed in some way.
    """
    if engine is None:
        engine = GPUEngine(raise_on_fail=True)

    final_polars_collect_kwargs, final_cudf_collect_kwargs = _process_kwargs(
        collect_kwargs, polars_collect_kwargs, cudf_collect_kwargs
    )

    expect = lazydf.collect(**final_polars_collect_kwargs)
    got = lazydf.collect(**final_cudf_collect_kwargs, engine=engine)
    assert_frame_equal(
        expect,
        got,
        check_row_order=check_row_order,
        check_column_order=check_column_order,
        check_dtypes=check_dtypes,
        check_exact=check_exact,
        rtol=rtol,
        atol=atol,
        categorical_as_str=categorical_as_str,
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
            assert any(
                isinstance(err, err_type) for err_type in exceptions
            ), f"Translation DID NOT RAISE {exceptions}"
        return
    else:
        raise AssertionError(f"Translation DID NOT RAISE {exceptions}")


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
        None, CPU is not expected to raise an exception.
    cudf_except
        Exception or exceptions polars GPU is expected to raise. If
        None, GPU is not expected to raise an exception.
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
        lazydf.collect(**final_polars_collect_kwargs)
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
        lazydf.collect(**final_cudf_collect_kwargs, engine=engine)
    except cudf_except:
        pass
    except Exception as e:
        raise AssertionError(
            f"GPU execution RAISED {type(e)}, EXPECTED {cudf_except}"
        ) from e
    else:
        if cudf_except != ():
            raise AssertionError(f"GPU execution DID NOT RAISE {cudf_except}")
