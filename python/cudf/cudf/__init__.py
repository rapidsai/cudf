# Copyright (c) 2018-2025, NVIDIA CORPORATION.

# If libcudf was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libcudf
except ModuleNotFoundError:
    pass
else:
    libcudf.load_library()
    del libcudf

from cudf import api, core, datasets, testing
from cudf._version import __git_commit__, __version__
from cudf.api.extensions import (
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)
from cudf.api.types import dtype
from cudf.core.algorithms import factorize, unique
from cudf.core.cut import cut
from cudf.core.dataframe import DataFrame, from_pandas, merge
from cudf.core.dtypes import (
    CategoricalDtype,
    Decimal32Dtype,
    Decimal64Dtype,
    Decimal128Dtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)
from cudf.core.groupby import Grouper, NamedAgg
from cudf.core.index import (
    CategoricalIndex,
    DatetimeIndex,
    Index,
    IntervalIndex,
    RangeIndex,
    TimedeltaIndex,
    interval_range,
)
from cudf.core.missing import NA, NaT
from cudf.core.multiindex import MultiIndex
from cudf.core.reshape import (
    concat,
    crosstab,
    get_dummies,
    melt,
    pivot,
    pivot_table,
    unstack,
)
from cudf.core.series import Series
from cudf.core.tools.datetimes import DateOffset, date_range, to_datetime
from cudf.core.tools.numeric import to_numeric
from cudf.io import (
    from_dlpack,
    read_avro,
    read_csv,
    read_feather,
    read_hdf,
    read_json,
    read_orc,
    read_parquet,
    read_text,
)
from cudf.options import (
    describe_option,
    get_option,
    option_context,
    set_option,
)


def configure_mr():
    import os
    import warnings

    import cupy
    from numba import cuda

    import pylibcudf
    import rmm.mr
    from rmm.allocators.cupy import rmm_cupy_allocator
    from rmm.allocators.numba import RMMNumbaManager

    if (
        "RAPIDS_NO_INITIALIZE" in os.environ
        or "CUDF_NO_INITIALIZE" in os.environ
    ):
        return

    # Set up cupy and numba to use RMM for allocations
    cuda.set_memory_manager(RMMNumbaManager)
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    # Configure rmm's default allocator to be a managed pool if supported unless the
    # user has specified otherwise.
    try:
        # The default mode is "managed_pool" if UVM is supported, otherwise "pool"
        managed_memory_is_supported = (
            pylibcudf.utils._is_concurrent_managed_access_supported()
        )
    except RuntimeError as e:
        warnings.warn(str(e))
        return

    cudf_rmm_mode = os.getenv("CUDF_RMM_MODE")
    cudf_pandas_rmm_mode = os.getenv("CUDF_PANDAS_RMM_MODE")
    rmm_mode = cudf_pandas_rmm_mode or cudf_rmm_mode
    if rmm_mode is None:
        rmm_mode = "managed_pool" if managed_memory_is_supported else "pool"

    # Check if a non-default memory resource is set
    current_mr = rmm.mr.get_current_device_resource()
    if not isinstance(current_mr, rmm.mr.CudaMemoryResource):
        # Warn only if the user explicitly set CUDF_PANDAS_RMM_MODE or CUDF_RMM_MODE
        if cudf_rmm_mode:
            warnings.warn(
                "cudf detected an already configured memory resource, ignoring "
                f"'CUDF_RMM_MODE={rmm_mode}'",
                UserWarning,
            )
        if cudf_pandas_rmm_mode:
            warnings.warn(
                "cudf.pandas detected an already configured memory resource, "
                f"ignoring 'CUDF_PANDAS_RMM_MODE={rmm_mode}'",
                UserWarning,
            )
        return

    free_memory, _ = rmm.mr.available_device_memory()
    free_memory = int(round(float(free_memory) * 0.80 / 256) * 256)
    new_mr = current_mr

    if rmm_mode == "pool":
        new_mr = rmm.mr.PoolMemoryResource(
            current_mr,
            initial_pool_size=free_memory,
        )
    elif rmm_mode == "async":
        new_mr = rmm.mr.CudaAsyncMemoryResource(initial_pool_size=free_memory)
    elif "managed" in rmm_mode:
        if not managed_memory_is_supported:
            raise ValueError(
                "Managed memory is not supported on this system, so the "
                f"requested {rmm_mode=} is invalid."
            )
        if rmm_mode == "managed":
            new_mr = rmm.mr.PrefetchResourceAdaptor(
                rmm.mr.ManagedMemoryResource()
            )
        elif rmm_mode == "managed_pool":
            new_mr = rmm.mr.PrefetchResourceAdaptor(
                rmm.mr.PoolMemoryResource(
                    rmm.mr.ManagedMemoryResource(),
                    initial_pool_size=free_memory,
                )
            )
        else:
            raise ValueError(f"Unsupported {rmm_mode=}")
        pylibcudf.prefetch.enable()
    elif rmm_mode != "cuda":
        raise ValueError(f"Unsupported {rmm_mode=}")

    rmm.mr.set_current_device_resource(new_mr)


configure_mr()


__all__ = [
    "NA",
    "CategoricalDtype",
    "CategoricalIndex",
    "DataFrame",
    "DateOffset",
    "DatetimeIndex",
    "Decimal32Dtype",
    "Decimal64Dtype",
    "Decimal128Dtype",
    "Grouper",
    "Index",
    "IntervalDtype",
    "IntervalIndex",
    "ListDtype",
    "MultiIndex",
    "NaT",
    "NamedAgg",
    "RangeIndex",
    "Series",
    "StructDtype",
    "TimedeltaIndex",
    "api",
    "concat",
    "core",  # TODO: core should not be publicly exposed
    "crosstab",
    "cut",
    "datasets",
    "date_range",
    "describe_option",
    "dtype",  # TODO: dtype should not be a public function
    "errors",
    "factorize",
    "from_dlpack",
    "from_pandas",
    "get_dummies",
    "get_option",
    "interval_range",
    "io",
    "melt",
    "merge",
    "option_context",
    "options",  # TODO: Move options.py to core, not all objects should be public
    "pivot",
    "pivot_table",
    "read_avro",
    "read_csv",
    "read_feather",
    "read_hdf",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_text",
    "register_dataframe_accessor",
    "register_index_accessor",
    "register_series_accessor",
    "set_option",
    "testing",
    "to_datetime",
    "to_numeric",
    "unique",
    "unstack",
    "utils",
]
