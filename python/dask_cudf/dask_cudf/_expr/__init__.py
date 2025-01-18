# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from packaging.version import Version

import dask

if Version(dask.__version__) > Version("2024.12.1"):
    import dask.dataframe.dask_expr._shuffle as _shuffle_module
    from dask.dataframe.dask_expr import (
        DataFrame as DXDataFrame,
        FrameBase,
        Index as DXIndex,
        Series as DXSeries,
        from_dict,
        get_collection_type,
        new_collection,
    )
    from dask.dataframe.dask_expr._cumulative import (
        CumulativeBlockwise,
    )
    from dask.dataframe.dask_expr._expr import (
        Elemwise,
        Expr,
        RenameAxis,
        VarColumns,
    )
    from dask.dataframe.dask_expr._groupby import (
        DecomposableGroupbyAggregation,
        GroupBy as DXGroupBy,
        GroupbyAggregation,
        SeriesGroupBy as DXSeriesGroupBy,
        SingleAggregation,
    )
    from dask.dataframe.dask_expr._reductions import (
        Reduction,
        Var,
    )
    from dask.dataframe.dask_expr._util import (
        _convert_to_list,
        _raise_if_object_series,
        is_scalar,
    )
    from dask.dataframe.dask_expr.io.io import (
        FusedIO,
        FusedParquetIO,
    )
    from dask.dataframe.dask_expr.io.parquet import (
        FragmentWrapper,
        ReadParquetFSSpec,
        ReadParquetPyarrowFS,
    )
else:
    import dask_expr._shuffle as _shuffle_module  # noqa: F401
    from dask_expr import (  # noqa: F401
        DataFrame as DXDataFrame,
        FrameBase,
        Index as DXIndex,
        Series as DXSeries,
        from_dict,
        get_collection_type,
        new_collection,
    )
    from dask_expr._cumulative import CumulativeBlockwise  # noqa: F401
    from dask_expr._expr import (  # noqa: F401
        Elemwise,
        Expr,
        RenameAxis,
        VarColumns,
    )
    from dask_expr._groupby import (  # noqa: F401
        DecomposableGroupbyAggregation,
        GroupBy as DXGroupBy,
        GroupbyAggregation,
        SeriesGroupBy as DXSeriesGroupBy,
        SingleAggregation,
    )
    from dask_expr._reductions import Reduction, Var  # noqa: F401
    from dask_expr._util import (  # noqa: F401
        _convert_to_list,
        _raise_if_object_series,
        is_scalar,
    )
    from dask_expr.io.io import FusedIO, FusedParquetIO  # noqa: F401
    from dask_expr.io.parquet import (  # noqa: F401
        FragmentWrapper,
        ReadParquetFSSpec,
        ReadParquetPyarrowFS,
    )

    from dask.dataframe import _dask_expr_enabled

    if not _dask_expr_enabled():
        raise ValueError(
            "The legacy DataFrame API is not supported for RAPIDS >24.12. "
            "The 'dataframe.query-planning' config must be True or None."
        )
