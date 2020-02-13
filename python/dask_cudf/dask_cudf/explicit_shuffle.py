import asyncio
import warnings

import pandas as pd

from dask.dataframe.utils import group_split_dispatch
from dask_cuda.explicit_comms import comms
from distributed.protocol import to_serialize

import cudf


def _cleanup_parts(df_parts):
    for part in df_parts:
        if part:
            del part


async def send_df(ep, df):
    if df is None:
        return await ep.write("empty")
    else:
        return await ep.write([to_serialize(df)])


async def recv_df(ep):
    ret = await ep.read()
    if ret == "empty":
        return None
    else:
        return ret[0]


async def send_parts(eps, parts):
    futures = []
    for rank, ep in eps.items():
        futures.append(send_df(ep, parts[rank]))
    await asyncio.gather(*futures)


async def recv_parts(eps, parts):
    futures = []
    for ep in eps.values():
        futures.append(recv_df(ep))
    parts.extend(await asyncio.gather(*futures))


async def exchange_and_concat_parts(rank, eps, parts, sort_by):
    ret = [parts[rank]]
    await asyncio.gather(recv_parts(eps, ret), send_parts(eps, parts))
    return concat(
        [df for df in ret if df is not None and len(df)], sort_by=sort_by
    )


def concat(df_list, sort_by=None):
    if len(df_list) == 0:
        return None
    if isinstance(df_list[0], cudf.DataFrame):
        if sort_by:
            new_df = cudf.merge_sorted(df_list, keys=sort_by)
        else:
            new_df = cudf.concat(df_list)
    else:
        new_df = pd.concat(df_list)
    _cleanup_parts(df_list)
    return new_df


def partition_by_column(df, column, n_chunks):
    if df is None:
        return [None] * n_chunks
    else:
        return group_split_dispatch(df, column, n_chunks)


async def distributed_shuffle(
    n_chunks, rank, eps, table, partitions, index, sort_by
):
    if sort_by:
        parts = [
            table.iloc[partitions[i] : partitions[i + 1]].copy(deep=False)
            for i in range(0, len(partitions) - 1)
        ]
    else:
        parts = partition_by_column(table, partitions, n_chunks)
    return await exchange_and_concat_parts(rank, eps, parts, sort_by)


async def _explicit_shuffle(
    s, df_nparts, df_parts, index, sort_by, divisions, to_cpu
):
    def df_concat(df_parts, sort_by=None):
        """Making sure df_parts is a single dataframe or None"""
        if len(df_parts) == 0:
            return None
        elif len(df_parts) == 1:
            return df_parts[0]
        else:
            return concat(df_parts, sort_by=sort_by)

    # Concatenate all parts owned by this worker into
    # a single cudf DataFrame
    if to_cpu:
        df = df_concat([dfp.to_pandas() for dfp in df_parts[0]])
    else:
        df = df_concat(df_parts[0], sort_by=sort_by)

    # Calculate new partition mapping
    if df is not None:
        divisions = df._constructor_sliced(divisions, dtype=df[index].dtype)
        if sort_by:
            splits = df[index].searchsorted(divisions, side="left")
            splits[-1] = len(df[index])
            partitions = splits.tolist()
        else:
            partitions = divisions.searchsorted(df[index], side="right") - 1
            partitions[(df[index] >= divisions.iloc[-1]).values] = (
                len(divisions) - 2
            )
        del divisions
    else:
        partitions = None

    # Run distributed shuffle and set_index algorithm
    new_df = await distributed_shuffle(
        s["nworkers"], s["rank"], s["eps"], df, partitions, index, sort_by
    )
    del df
    if to_cpu:
        return cudf.from_pandas(new_df)
    return new_df


def explicit_sorted_shuffle(df, index, divisions, sort_by, client, **kwargs):
    # Explict-comms shuffle
    # TODO: Fast repartition back to df.npartitions using views...
    # client.rebalance(futures=df.to_delayed())
    to_cpu = kwargs.get("to_cpu", False)
    if to_cpu:
        warnings.warn("Using CPU for shuffling. Performance will suffer!")
    return comms.default_comms().dataframe_operation(
        _explicit_shuffle,
        df_list=(df,),
        extra_args=(index, sort_by, divisions, to_cpu),
    )
