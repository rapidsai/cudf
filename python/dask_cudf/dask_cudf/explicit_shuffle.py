import asyncio
import warnings

import cupy
import pandas as pd

import distributed
from dask.distributed import wait
from dask_cuda.explicit_comms import comms
from distributed.protocol import to_serialize

import cudf
import rmm

cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)


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
    del parts[rank]
    parts[rank] = None
    await asyncio.gather(recv_parts(eps, ret), send_parts(eps, parts))
    for rank in list(parts):
        del parts[rank]
    new_df = concat(
        [df.copy(deep=False) for df in ret if df is not None and len(df)],
        sort_by=sort_by,
    )
    del ret
    return new_df


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


def partition_table(df, partitions, n_chunks, sort_by=None):
    if df is None:
        result = [None] * n_chunks
    elif sort_by:
        result = {
            i: df.iloc[partitions[i] : partitions[i + 1]].copy(deep=False)
            for i in range(0, len(partitions) - 1)
        }
    else:
        result = dict(
            zip(
                range(n_chunks),
                df.scatter_by_map(partitions, map_size=n_chunks),
            )
        )
    return result


async def distributed_shuffle(
    n_chunks, rank, eps, table, partitions, index, sort_by
):
    parts = partition_table(table, partitions, n_chunks, sort_by=sort_by)
    return await exchange_and_concat_parts(rank, eps, parts, sort_by)


async def _explicit_shuffle(
    s, df_nparts, df_parts, index, sort_by, divisions, to_cpu
):
    if len(df_parts[0]) == 0:
        df = None
    else:
        df = df_parts[0][0].copy(deep=False)

    # Calculate new partition mapping
    if df is not None:
        divisions = df._constructor_sliced(divisions, dtype=df[index].dtype)
        if sort_by:
            splits = df[index].searchsorted(divisions, side="left")
            splits[-1] = len(df[index])
            partitions = splits.tolist()
            del splits
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

    if to_cpu:
        return cudf.from_pandas(new_df)
    return new_df


async def _explicit_aggregate(s, df_nparts, df_parts, sort_by, to_cpu):
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
        return df_concat([dfp.to_pandas() for dfp in df_parts[0]])
    else:
        return df_concat(df_parts[0], sort_by=sort_by)


def explicit_sorted_shuffle(df, index, divisions, sort_by, client, **kwargs):
    client.rebalance(futures=distributed.futures_of(df))
    to_cpu = kwargs.get("to_cpu", False)
    if to_cpu:
        warnings.warn("Using CPU for shuffling. Performance will suffer!")

    # Explict-comms Partition Aggregation
    df2 = comms.default_comms().dataframe_operation(
        _explicit_aggregate, df_list=(df,), extra_args=(sort_by, to_cpu)
    )
    wait(df2.persist())
    wait(client.cancel(df))
    del df

    # Explict-comms shuffle
    df3 = comms.default_comms().dataframe_operation(
        _explicit_shuffle,
        df_list=(df2,),
        extra_args=(index, sort_by, divisions, to_cpu),
    )
    wait(df3.persist())
    wait(client.cancel(df2))
    del df2
    return df3
