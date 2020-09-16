# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd

from dask.dataframe.core import _concat

import cudf


def _make_name(*args, sep="_"):
    _args = (arg for arg in args if arg != "")
    return sep.join(_args)


def _groupby_partition_agg(
    df, gb_cols, agg_list, split_out, dropna, out_to_host, sep
):
    _agg_list = set()
    for agg in agg_list:
        if agg == "mean":
            _agg_list.add("count")
            _agg_list.add("sum")
        else:
            _agg_list.add(agg)
    _agg_list = list(_agg_list)

    gb = df.groupby(gb_cols, dropna=dropna, as_index=False).agg(_agg_list)
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
    dfs, gb_cols, agg_list, split_out, dropna, out_to_host, sep
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

    gb = df.groupby(gb_cols, dropna=dropna, as_index=False).agg(agg_dict)

    # Don't include the last aggregation in the column names
    gb.columns = [_make_name(*name[:-1], sep=sep) for name in gb.columns]
    return gb


def _finalize_gb_agg(gb, gb_cols, agg_list, sep):

    # Deal with "mean"
    if "mean" in agg_list:
        for col in gb.columns:
            if col in gb_cols:
                continue
            name = col.split(sep)
            # import pdb; pdb.set_trace()
            if name[-1] == "sum":
                mean_name = _make_name(*(name[:-1] + ["mean"]), sep=sep)
                count_name = _make_name(*(name[:-1] + ["count"]), sep=sep)
                sum_name = _make_name(*name, sep=sep)
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
