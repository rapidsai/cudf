# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_expr._collection import new_collection
from dask_expr._groupby import (
    GroupBy as DXGroupBy,
    SeriesGroupBy as DXSeriesGroupBy,
    SingleAggregation,
)
from dask_expr._util import is_scalar

from dask.dataframe.groupby import Aggregation

from cudf.core.groupby.groupby import _deprecate_collect

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


# def _get_custom_arg(gb, arg):
#     from dask_cudf.groupby import (
#         OPTIMIZED_AGGS,
#         _aggs_optimized,
#         _redirect_aggs,
#     )

#     _arg = _redirect_aggs(arg)
#     if not _aggs_optimized(_arg, OPTIMIZED_AGGS) or not hasattr(
#         gb.obj._meta, "to_pandas"
#     ):
#         # Not supported
#         return None

#     # Convert all agg specs to dict
#     use_list = False
#     if not isinstance(_arg, dict):
#         use_list = True
#         gb_cols = gb._meta.grouping.keys.names
#         columns = [c for c in gb.obj.columns if c not in gb_cols]
#         _arg = {col: _arg for col in columns}

#     # Normalize the dict and count ops
#     naggs = 0
#     str_cols_out = True
#     for col in _arg:
#         if isinstance(_arg[col], str) or callable(_arg[col]):
#             _arg[col] = [_arg[col]]
#             naggs += 1
#         elif isinstance(_arg[col], dict):
#             # TODO: Support named aggs
#             return None
#             str_cols_out = False
#             col_aggs = []
#             for k, v in _arg[col].items():
#                 col_aggs.append(v)
#             _arg[col] = col_aggs
#             naggs += len(col_aggs)
#         else:
#             str_cols_out = False

#     # if str_cols_out:
#     #     # Metadata should use `str` for dict values if that is
#     #     # what the user originally specified (column names will
#     #     # be str, rather than tuples).
#     #     for col in _arg.keys():
#     #         _arg[col] = _arg[col][0]
#     # if use_list:
#     #     _arg = next(_arg.values())

#     if naggs > 1:
#         # Only use the custom code path if we
#         # are performing multiple aggs at once
#         return _arg
#     else:
#         return None


# TODO: These classes are mostly a work-around for missing
# `observed=False` support.
# See: https://github.com/rapidsai/cudf/issues/15173


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

    def aggregate(self, arg, **kwargs):
        from dask_cudf.expr._expr import CudfGroupbyAggregation
        from dask_cudf.groupby import (
            OPTIMIZED_AGGS,
            _aggs_optimized,
            _redirect_aggs,
        )

        # Check if "custom" aggregation is supported/needed
        _arg = _redirect_aggs(arg)
        supported = _aggs_optimized(_arg, OPTIMIZED_AGGS) and hasattr(
            self.obj._meta, "to_pandas"
        )
        n_aggs = 0
        if supported:
            if isinstance(_arg, list):
                # TODO: Support named aggregations
                supported = supported and all(
                    [isinstance(v, str) for v in _arg]
                )
                n_aggs += len(_arg)
            elif isinstance(_arg, dict):
                for val in _arg.values():
                    if isinstance(val, str):
                        n_aggs += 1
                    elif isinstance(val, list):
                        n_aggs += len(val)
                        supported = supported and all(
                            [isinstance(v, str) for v in val]
                        )
                    else:
                        # TODO: Support named aggregations
                        supported = False
                    if not supported:
                        break
            else:
                n_aggs = 1

        if supported and n_aggs > 1:
            # Use custom agg logic from "legacy" dask-cudf.
            # This code path may be more efficient than dask-expr
            # when we are performing multiple aggregations on the
            # same DataFrame at once.
            return new_collection(
                CudfGroupbyAggregation(
                    self.obj.expr,
                    _arg,
                    self.observed,
                    self.dropna,
                    kwargs.get("split_every"),
                    kwargs.get("split_out"),
                    self.sort,
                    kwargs.get("shuffle_method"),
                    self._slice,
                    *self.by,
                )
            )
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
