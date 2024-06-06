# Copyright (c) 2024, NVIDIA CORPORATION.


from typing import Optional

import dask_cudf.expr._collection
import dask_cudf.expr._groupby

import cudf
import cudf.core.groupby.groupby
from cudf.options import get_option
from cudf.pandas.fast_slow_proxy import _Unusable, make_final_proxy_type

try:
    import dask_cudf
except ImportError:

    def parse_lazy_argument(arg: Optional[bool]) -> bool:
        if arg is None:
            arg = get_option("lazy")
        if not arg:
            raise ValueError("Using the 'lazy' option requires dask_cudf")
        return arg
else:

    def _slow_attr(name: str):
        def f(x):
            return getattr(x._fsproxy_slow, name)()

        return f

    DataFrame = make_final_proxy_type(
        "DataFrame",
        dask_cudf.expr._collection.DataFrame,
        cudf.DataFrame,
        fast_to_slow=lambda fast: fast.compute(),
        slow_to_fast=_Unusable(),  # da.from_pandas,
        additional_attributes={
            "__str__": _slow_attr("__str__"),
            "__repr__": _slow_attr("__repr__"),
        },
    )

    Series = make_final_proxy_type(
        "Series",
        dask_cudf.expr._collection.Series,
        cudf.Series,
        fast_to_slow=lambda fast: fast.compute(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "__str__": _slow_attr("__str__"),
            "__repr__": _slow_attr("__repr__"),
        },
    )

    GroupBy = make_final_proxy_type(
        "GroupBy",
        dask_cudf.expr._groupby.GroupBy,
        cudf.core.groupby.groupby.DataFrameGroupBy,
        fast_to_slow=lambda fast: fast.compute(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "__str__": _slow_attr("__str__"),
            "__repr__": _slow_attr("__repr__"),
        },
    )

    SeriesGroupBy = make_final_proxy_type(
        "SeriesGroupBy",
        dask_cudf.expr._groupby.SeriesGroupBy,
        cudf.core.groupby.groupby.SeriesGroupBy,
        fast_to_slow=lambda fast: fast.compute(),
        slow_to_fast=_Unusable(),
        additional_attributes={
            "__str__": _slow_attr("__str__"),
            "__repr__": _slow_attr("__repr__"),
        },
    )

    def parse_lazy_argument(arg: Optional[bool]) -> bool:
        if arg is None:
            arg = get_option("lazy")
        return arg
