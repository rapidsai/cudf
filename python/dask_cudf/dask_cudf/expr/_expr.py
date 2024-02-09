# Copyright (c) 2024, NVIDIA CORPORATION.

import functools

from dask_expr._cumulative import CumulativeBlockwise, TakeLast
from dask_expr._groupby import (
    DecomposableGroupbyAggregation,
    GroupbyAggregation,
)
from dask_expr._reductions import Var
from dask_expr._shuffle import DiskShuffle

from dask.dataframe.core import _concat

from dask_cudf.groupby import (
    _finalize_gb_agg,
    _groupby_partition_agg,
    _tree_node_agg,
)

##
## Custom expression patching
##


class PatchCumulativeBlockwise(CumulativeBlockwise):
    @property
    def _args(self) -> list:
        return self.operands[:1]

    @property
    def _kwargs(self) -> dict:
        # Must pass axis and skipna as kwargs in cudf
        return {"axis": self.axis, "skipna": self.skipna}


CumulativeBlockwise._args = PatchCumulativeBlockwise._args
CumulativeBlockwise._kwargs = PatchCumulativeBlockwise._kwargs


def _takelast(a, skipna=True):
    if not len(a):
        return a
    if skipna:
        a = a.bfill()
    # Cannot use `squeeze` with cudf
    return a.tail(n=1).iloc[0]


TakeLast.operation = staticmethod(_takelast)


_dx_reduction_aggregate = Var.reduction_aggregate


def _reduction_aggregate(*args, **kwargs):
    result = _dx_reduction_aggregate(*args, **kwargs)
    if result.ndim == 0:
        return result.tolist()
    return result


Var.reduction_aggregate = staticmethod(_reduction_aggregate)


def _shuffle_group(df, col, _filter, p):
    from dask.dataframe.shuffle import ensure_cleanup_on_exception

    with ensure_cleanup_on_exception(p):
        g = df.groupby(col)
        if hasattr(g, "_grouped"):
            # Avoid `get_group` for cudf data.
            # See: https://github.com/rapidsai/cudf/issues/14955
            keys, part_offsets, _, grouped_df = df.groupby(col)._grouped()
            d = {
                k: grouped_df.iloc[part_offsets[i] : part_offsets[i + 1]]
                for i, k in enumerate(keys.to_pandas())
                if k in _filter
            }
        else:
            d = {i: g.get_group(i) for i in g.groups if i in _filter}
        p.append(d, fsync=True)


DiskShuffle._shuffle_group = staticmethod(_shuffle_group)


class CudfGroupbyAggregation(GroupbyAggregation):
    def _lower(self):
        return DecomposableCudfGroupbyAggregation(
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


class DecomposableCudfGroupbyAggregation(DecomposableGroupbyAggregation):
    sep = "___"

    @property
    def shuffle_by_index(self):
        return False  # We always group by column(s) in dask-cudf

    @functools.cached_property
    def spec_info(self):
        if isinstance(self.arg, (dict, list)):
            aggs = self.arg.copy()
        else:
            aggs = self.arg

        if self._slice and not isinstance(aggs, dict):
            aggs = {self._slice: aggs}

        gb_cols = self._by_columns
        if isinstance(gb_cols, str):
            gb_cols = [gb_cols]
        columns = [c for c in self.frame.columns if c not in gb_cols]
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

    @classmethod
    def chunk(cls, df, *by, **kwargs):
        # `by` columns are already specified in kwargs
        return _groupby_partition_agg(df, **kwargs)

    @classmethod
    def combine(cls, inputs, **kwargs):
        return _tree_node_agg(_concat(inputs), **kwargs)

    @classmethod
    def aggregate(cls, inputs, **kwargs):
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
            "as_index": True,  # False not supported in dask-expr
            "dropna": dropna,
            "sort": self.sort,
            "sep": self.sep,
            "str_cols_out": self.spec_info["str_cols_out"],
            "aggs_renames": self.spec_info["aggs_renames"],
        }
