# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

from packaging.version import Version

import dask
import dask.dataframe.dask_expr._shuffle as _shuffle_module
from dask.dataframe import get_collection_type
from dask.dataframe.dask_expr import (
    DataFrame as DXDataFrame,
    FrameBase,
    Index as DXIndex,
    Series as DXSeries,
    from_dict,
    new_collection,
)
from dask.dataframe.dask_expr._cumulative import (
    CumulativeBlockwise,
)
from dask.dataframe.dask_expr._expr import (
    Elemwise,
    EnforceRuntimeDivisions,
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

_dask_version = importlib.metadata.version("dask")

# TODO: change ">2025.2.0" to ">={next-version}" when released.
DASK_2025_3_0 = Version(_dask_version) > Version("2025.2.0")


if DASK_2025_3_0:
    from dask.dataframe.utils import is_scalar
else:
    from dask.dataframe.dask_expr._util import is_scalar


__all__ = [
    "CumulativeBlockwise",
    "DXDataFrame",
    "DXGroupBy",
    "DXIndex",
    "DXSeries",
    "DXSeriesGroupBy",
    "DecomposableGroupbyAggregation",
    "Elemwise",
    "EnforceRuntimeDivisions",
    "Expr",
    "FragmentWrapper",
    "FrameBase",
    "FusedIO",
    "FusedParquetIO",
    "GroupbyAggregation",
    "ReadParquetFSSpec",
    "ReadParquetPyarrowFS",
    "Reduction",
    "RenameAxis",
    "SingleAggregation",
    "Var",
    "VarColumns",
    "_convert_to_list",
    "_raise_if_object_series",
    "_shuffle_module",
    "dask",
    "from_dict",
    "get_collection_type",
    "is_scalar",
    "new_collection",
]
