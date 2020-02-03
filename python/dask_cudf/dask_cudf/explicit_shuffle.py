import asyncio

import cupy

from dask_cuda.explicit_comms import comms
from distributed.protocol import to_serialize

import cudf
import rmm

cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)


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


async def exchange_and_concat_parts(rank, eps, parts):
    ret = [parts[rank]]
    await asyncio.gather(recv_parts(eps, ret), send_parts(eps, parts))
    return concat(list(filter(None, ret)))


def concat(df_list):
    if len(df_list) == 0:
        return None
    return cudf.concat(df_list)


def partition_by_column(df, column, n_chunks):
    if df is None:
        return [None] * n_chunks
    else:
        return df.scatter_by_map(column, map_size=n_chunks)


async def distributed_shuffle(n_chunks, rank, eps, table, partitions):
    parts = partition_by_column(table, partitions, n_chunks)
    del table
    return await exchange_and_concat_parts(rank, eps, parts)


async def _explicit_shuffle(s, df_nparts, df_parts, index, divisions):
    def df_concat(df_parts):
        """Making sure df_parts is a single dataframe or None"""
        if len(df_parts) == 0:
            return None
        elif len(df_parts) == 1:
            return df_parts[0]
        else:
            return concat(df_parts)

    # Concatenate all parts owned by this worker into
    # a single cudf DataFrame
    df = df_concat(df_parts[0])
    del df_parts

    # Calculate new partition mapping
    if df:
        divisions = cudf.Series(divisions)
        partitions = divisions.searchsorted(df[index], side="right") - 1
        partitions[(df[index] >= divisions.iloc[-1]).values] = (
            len(divisions) - 2
        )
    else:
        partitions = None

    # Run distributed shuffle and set_index algorithm
    return await distributed_shuffle(
        s["nworkers"], s["rank"], s["eps"], df, partitions
    )


def explicit_sorted_shuffle(df, index, divisions, sort_by, client, **kwargs):
    # Explict-comms shuffle
    # TODO: Fast repartition back to df.npartitions using views...
    client.rebalance(futures=df.to_delayed())
    return comms.default_comms().dataframe_operation(
        _explicit_shuffle, df_list=(df,), extra_args=(index, divisions)
    )
