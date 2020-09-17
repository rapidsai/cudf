# Copyright (c) 2020, NVIDIA CORPORATION.
import math
from operator import getitem

import numpy as np
import pandas as pd

from dask.base import tokenize
from dask.dataframe.core import _concat, new_dd_object
from dask.highlevelgraph import HighLevelGraph

import cudf

# from dask.dataframe.groupby import DataFrameGroupBy
# class CudfDataFrameGroupBy(DataFrameGroupBy):

#     def aggregate(
#         self,
#         arg,
#         split_every=None,
#         split_out=1,
#         dropna=True,
#         out_to_host=False,
#         sep="___",
#         sort=False,
#     ):
#         if arg == "size":
#             return self.size()

#         if isinstance(self.obj, DataFrame) and isinstance(arg, list):
#             _supported = {"count", "mean", "std", "var", "sum", "min", "max"}
#             if set(arg).issubset(_supported):
#                 index = self.index
#                 return groupby_agg(
#                     self,
#                     index,
#                     arg,
#                     split_every=split_every,
#                     split_out=split_out,
#                     dropna=dropna,
#                     out_to_host=out_to_host,
#                     sep=sep,
#                     sort=sort,
#                 )

#         return super().aggregate(
#             arg, split_every=split_every, split_out=split_out)


def groupby_agg(
    ddf,
    gb_cols: list,
    agg_list: list,
    split_every=None,
    split_out=None,
    dropna=True,
    out_to_host=False,
    sep="___",
    sort=False,
):
    # Deal with default split_out and split_every params
    if split_every is False:
        split_every = ddf.npartitions
    split_every = split_every or 8
    split_out = split_out or 1

    columns = [c for c in ddf.columns if c not in gb_cols]

    # Only support basic aggs and mean
    _supported = {"count", "mean", "std", "var", "sum", "min", "max"}
    if not set(agg_list).issubset(_supported):
        raise ValueError(
            f"Supported aggs include {_supported} for groupby_agg API."
        )

    dsk = {}
    token = tokenize(ddf, gb_cols, agg_list)
    partition_agg_name = "groupby_partition_agg-" + token
    tree_reduce_name = "groupby_tree_reduce-" + token
    gb_agg_name = "groupby_agg-" + token
    for p in range(ddf.npartitions):
        # Perform groupby aggregation on each partition.
        # Split each result into `split_out` chunks (by hashing `gb_cols`)
        dsk[(partition_agg_name, p)] = (
            _groupby_partition_agg,
            (ddf._name, p),
            gb_cols,
            agg_list,
            columns,
            split_out,
            dropna,
            out_to_host,
            sort,
            sep,
        )
        # Pick out each chunk using `getitem`
        for s in range(split_out):
            dsk[(tree_reduce_name, p, s, 0)] = (
                getitem,
                (partition_agg_name, p),
                s,
            )

    # Build reduction tree
    parts = ddf.npartitions
    widths = [parts]
    while parts > 1:
        parts = math.ceil(parts / split_every)
        widths.append(parts)
    height = len(widths)
    for s in range(split_out):
        for depth in range(1, height):
            for group in range(widths[depth]):

                p_max = widths[depth - 1]
                lstart = split_every * group
                lstop = min(lstart + split_every, p_max)
                node_list = [
                    (tree_reduce_name, p, s, depth - 1)
                    for p in range(lstart, lstop)
                ]

                dsk[(tree_reduce_name, group, s, depth)] = (
                    _tree_node_agg,
                    node_list,
                    gb_cols,
                    agg_list,
                    split_out,
                    dropna,
                    out_to_host,
                    sort,
                    sep,
                )

    # Final output partitions
    for s in range(split_out):
        dsk[(gb_agg_name, s)] = (
            _finalize_gb_agg,
            (tree_reduce_name, 0, s, height - 1),
            gb_cols,
            agg_list,
            columns,
            sep,
        )

    divisions = [None] * (split_out + 1)
    _meta = ddf._meta.groupby(gb_cols, as_index=False).agg(agg_list)
    graph = HighLevelGraph.from_collections(
        gb_agg_name, dsk, dependencies=[ddf]
    )
    return new_dd_object(graph, gb_agg_name, _meta, divisions)


def _make_name(*args, sep="_"):
    _args = (arg for arg in args if arg != "")
    return sep.join(_args)


def _groupby_partition_agg(
    df, gb_cols, agg_list, columns, split_out, dropna, out_to_host, sort, sep
):
    _agg_list = set()
    for agg in agg_list:
        if agg in ("mean", "std", "var"):
            _agg_list.add("count")
            _agg_list.add("sum")
        else:
            _agg_list.add(agg)
    _agg_dict = {col: list(_agg_list) for col in columns}

    if set(agg_list).intersection({"std", "var"}):
        for c in columns:
            pow2_name = _make_name(c, "pow2", sep=sep)
            df[pow2_name] = df[c].pow(2)
            _agg_dict[pow2_name] = ["sum"]

    gb = df.groupby(gb_cols, dropna=dropna, as_index=False, sort=sort).agg(
        _agg_dict
    )
    gb.columns = [_make_name(*name, sep=sep) for name in gb.columns]
    output = {}
    for j, split in enumerate(
        gb.partition_by_hash(gb_cols, split_out, keep_index=False)
    ):
        if out_to_host:
            output[j] = split.to_pandas()
        else:
            output[j] = split
    del gb
    return output


def _tree_node_agg(
    dfs, gb_cols, agg_list, split_out, dropna, out_to_host, sort, sep
):
    df = _concat(dfs, ignore_index=True)
    if out_to_host:
        df.reset_index(drop=True, inplace=True)
        df = cudf.from_pandas(df)
    agg_dict = {}
    for col in df.columns:
        if col in gb_cols:
            continue
        agg = col.split(sep)[-1]
        if agg in ("count", "sum"):
            agg_dict[col] = ["sum"]
        elif agg in ("min", "max"):
            agg_dict[col] = [agg]
        else:
            raise ValueError(f"Unexpected aggregation: {agg}")

    gb = df.groupby(gb_cols, dropna=dropna, as_index=False, sort=sort).agg(
        agg_dict
    )

    # Don't include the last aggregation in the column names
    gb.columns = [_make_name(*name[:-1], sep=sep) for name in gb.columns]
    return gb


def _finalize_gb_agg(gb, gb_cols, agg_list, columns, sep):

    # Deal with higher-order aggregations
    agg_set = set(agg_list)
    if agg_set.intersection({"mean", "std", "var"}):
        for col in columns:
            count_name = _make_name(col, "count", sep=sep)
            sum_name = _make_name(col, "sum", sep=sep)
            if agg_set.intersection({"std", "var"}):
                n = gb[count_name]
                x = gb[sum_name]
                pow2_sum_name = _make_name(col, "pow2", "sum", sep=sep)
                x2 = gb[pow2_sum_name]
                result = x2 - x ** 2 / n
                ddof = 1
                div = n - ddof
                div[div < 1] = 1
                result /= div
                result[(n - ddof) == 0] = np.nan
                if "var" in agg_list:
                    name_var = _make_name(col, "var", sep=sep)
                    gb[name_var] = result
                if "std" in agg_list:
                    name_std = _make_name(col, "std", sep=sep)
                    gb[name_std] = np.sqrt(result)
                gb.drop(columns=[pow2_sum_name], inplace=True)
            if "mean" in agg_list:
                mean_name = _make_name(col, "mean", sep=sep)
                gb[mean_name] = gb[sum_name] / gb[count_name]
            if "sum" not in agg_list:
                gb.drop(columns=[sum_name], inplace=True)
            if "count" not in agg_list:
                gb.drop(columns=[count_name], inplace=True)

    # Unflatten column names
    col_array = []
    agg_array = []
    for col in gb.columns:
        if col in gb_cols:
            col_array.append(col)
            agg_array.append("")
        else:
            name, agg = col.split(sep)
            col_array.append(name)
            agg_array.append(agg)
    gb.columns = pd.MultiIndex.from_arrays([col_array, agg_array])

    return gb
