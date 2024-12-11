# Copyright (c) 2024, NVIDIA CORPORATION.
import functools

import pandas as pd
from dask_expr._collection import new_collection
from dask_expr._groupby import (
    DecomposableGroupbyAggregation,
    GroupBy as DXGroupBy,
    GroupbyAggregation,
    SeriesGroupBy as DXSeriesGroupBy,
    SingleAggregation,
)
from dask_expr._util import is_scalar

from dask.dataframe.core import _concat
from dask.dataframe.groupby import Aggregation

from cudf.core.groupby.groupby import _deprecate_collect

##
## Fused groupby aggregations
##


def _get_spec_info(gb):
    if isinstance(gb.arg, (dict, list)):
        aggs = gb.arg.copy()
    else:
        aggs = gb.arg

    if gb._slice and not isinstance(aggs, dict):
        aggs = {gb._slice: aggs}

    gb_cols = gb._by_columns
    if isinstance(gb_cols, str):
        gb_cols = [gb_cols]
    columns = [c for c in gb.frame.columns if c not in gb_cols]
    if not isinstance(aggs, dict):
        aggs = {col: aggs for col in columns}

    # Assert if our output will have a MultiIndex; this will be the case if
    # any value in the `aggs` dict is not a string (i.e. multiple/named
    # aggregations per column)
    str_cols_out = True
    aggs_renames = {}
    for col in aggs:
        if isinstance(aggs[col], str) or callable(aggs[col]):
            aggs[col] = [aggs[col]]
        elif isinstance(aggs[col], dict):
            str_cols_out = False
            col_aggs = []
            for k, v in aggs[col].items():
                aggs_renames[col, v] = k
                col_aggs.append(v)
            aggs[col] = col_aggs
        else:
            str_cols_out = False
        if col in gb_cols:
            columns.append(col)

    return {
        "aggs": aggs,
        "columns": columns,
        "str_cols_out": str_cols_out,
        "aggs_renames": aggs_renames,
    }


def _get_meta(gb):
    spec_info = gb.spec_info
    gb_cols = gb._by_columns
    aggs = spec_info["aggs"].copy()
    aggs_renames = spec_info["aggs_renames"]
    if spec_info["str_cols_out"]:
        # Metadata should use `str` for dict values if that is
        # what the user originally specified (column names will
        # be str, rather than tuples).
        for col in aggs:
            aggs[col] = aggs[col][0]
    _meta = gb.frame._meta.groupby(gb_cols).agg(aggs)
    if aggs_renames:
        col_array = []
        agg_array = []
        for col, agg in _meta.columns:
            col_array.append(col)
            agg_array.append(aggs_renames.get((col, agg), agg))
        _meta.columns = pd.MultiIndex.from_arrays([col_array, agg_array])
    return _meta


class DecomposableCudfGroupbyAgg(DecomposableGroupbyAggregation):
    sep = "___"

    @functools.cached_property
    def spec_info(self):
        return _get_spec_info(self)

    @functools.cached_property
    def _meta(self):
        return _get_meta(self)

    @property
    def shuffle_by_index(self):
        return False  # We always group by column(s)

    @classmethod
    def chunk(cls, df, *by, **kwargs):
        from dask_cudf._legacy.groupby import _groupby_partition_agg

        return _groupby_partition_agg(df, **kwargs)

    @classmethod
    def combine(cls, inputs, **kwargs):
        from dask_cudf._legacy.groupby import _tree_node_agg

        return _tree_node_agg(_concat(inputs), **kwargs)

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        from dask_cudf._legacy.groupby import _finalize_gb_agg

        return _finalize_gb_agg(_concat(inputs), **kwargs)

    @property
    def chunk_kwargs(self) -> dict:
        dropna = True if self.dropna is None else self.dropna
        return {
            "gb_cols": self._by_columns,
            "aggs": self.spec_info["aggs"],
            "columns": self.spec_info["columns"],
            "dropna": dropna,
            "sort": self.sort,
            "sep": self.sep,
        }

    @property
    def combine_kwargs(self) -> dict:
        dropna = True if self.dropna is None else self.dropna
        return {
            "gb_cols": self._by_columns,
            "dropna": dropna,
            "sort": self.sort,
            "sep": self.sep,
        }

    @property
    def aggregate_kwargs(self) -> dict:
        dropna = True if self.dropna is None else self.dropna
        final_columns = self._slice or self._meta.columns
        return {
            "gb_cols": self._by_columns,
            "aggs": self.spec_info["aggs"],
            "columns": self.spec_info["columns"],
            "final_columns": final_columns,
            "as_index": True,
            "dropna": dropna,
            "sort": self.sort,
            "sep": self.sep,
            "str_cols_out": self.spec_info["str_cols_out"],
            "aggs_renames": self.spec_info["aggs_renames"],
        }


class CudfGroupbyAgg(GroupbyAggregation):
    @functools.cached_property
    def spec_info(self):
        return _get_spec_info(self)

    @functools.cached_property
    def _meta(self):
        return _get_meta(self)

    def _lower(self):
        return DecomposableCudfGroupbyAgg(
            self.frame,
            self.arg,
            self.observed,
            self.dropna,
            self.split_every,
            self.split_out,
            self.sort,
            self.shuffle_method,
            self._slice,
            *self.by,
        )


def _maybe_get_custom_expr(
    gb,
    aggs,
    split_every=None,
    split_out=None,
    shuffle_method=None,
    **kwargs,
):
    from dask_cudf._legacy.groupby import (
        OPTIMIZED_AGGS,
        _aggs_optimized,
        _redirect_aggs,
    )

    if kwargs:
        # Unsupported key-word arguments
        return None

    if not hasattr(gb.obj._meta, "to_pandas"):
        # Not cuDF-backed data
        return None

    _aggs = _redirect_aggs(aggs)
    if not _aggs_optimized(_aggs, OPTIMIZED_AGGS):
        # One or more aggregations are unsupported
        return None

    return CudfGroupbyAgg(
        gb.obj.expr,
        _aggs,
        gb.observed,
        gb.dropna,
        split_every,
        split_out,
        gb.sort,
        shuffle_method,
        gb._slice,
        *gb.by,
    )


##
## Custom groupby classes
##


class ListAgg(SingleAggregation):
    @staticmethod
    def groupby_chunk(arg):
        return arg.agg(list)

    @staticmethod
    def groupby_aggregate(arg):
        gb = arg.agg(list)
        if gb.ndim > 1:
            for col in gb.columns:
                gb[col] = gb[col].list.concat()
            return gb
        else:
            return gb.list.concat()


list_aggregation = Aggregation(
    name="list",
    chunk=ListAgg.groupby_chunk,
    agg=ListAgg.groupby_aggregate,
)


def _translate_arg(arg):
    # Helper function to translate args so that
    # they can be processed correctly by upstream
    # dask & dask-expr. Right now, the only necessary
    # translation is list aggregations.
    if isinstance(arg, dict):
        return {k: _translate_arg(v) for k, v in arg.items()}
    elif isinstance(arg, list):
        return [_translate_arg(x) for x in arg]
    elif arg in ("collect", "list", list):
        return list_aggregation
    else:
        return arg


# We define our own GroupBy classes in Dask cuDF for
# the following reasons:
#  (1) We want to use a custom `aggregate` algorithm
#      that performs multiple aggregations on the
#      same dataframe partition at once. The upstream
#      algorithm breaks distinct aggregations into
#      separate tasks.
#  (2) We need to work around missing `observed=False`
#      support:
#      https://github.com/rapidsai/cudf/issues/15173


class GroupBy(DXGroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)

    def __getitem__(self, key):
        if is_scalar(key):
            return SeriesGroupBy(
                self.obj,
                by=self.by,
                slice=key,
                sort=self.sort,
                dropna=self.dropna,
                observed=self.observed,
            )
        g = GroupBy(
            self.obj,
            by=self.by,
            slice=key,
            sort=self.sort,
            dropna=self.dropna,
            observed=self.observed,
            group_keys=self.group_keys,
        )
        return g

    def collect(self, **kwargs):
        _deprecate_collect()
        return self._single_agg(ListAgg, **kwargs)

    def aggregate(self, arg, fused=True, **kwargs):
        if (
            fused
            and (expr := _maybe_get_custom_expr(self, arg, **kwargs))
            is not None
        ):
            return new_collection(expr)
        else:
            return super().aggregate(_translate_arg(arg), **kwargs)


class SeriesGroupBy(DXSeriesGroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)

    def collect(self, **kwargs):
        _deprecate_collect()
        return self._single_agg(ListAgg, **kwargs)

    def aggregate(self, arg, **kwargs):
        return super().aggregate(_translate_arg(arg), **kwargs)
