import asyncio

from distributed.protocol import to_serialize

import cudf

from dask_cudf.explicit_comms import comms


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


async def exchange_and_concat_parts(rank, eps, parts, sort=False):
    ret = [parts[rank]]
    await asyncio.gather(recv_parts(eps, ret), send_parts(eps, parts))
    df = concat(list(filter(None, ret)))
    if sort:
        return df.sort_values(sort)
    return df


def concat(df_list):
    if len(df_list) == 0:
        return None
    return cudf.concat(df_list)


def partition_by_column(df, column, n_chunks):
    if df is None:
        return [None] * n_chunks
    elif hasattr(df, "scatter_by_map"):
        return df.scatter_by_map(column, map_size=n_chunks)
    else:
        raise NotImplementedError(
            "partition_by_column not yet implemented for pandas backend.\n"
        )


async def distributed_shuffle(
    n_chunks, rank, eps, table, partitions, index, sort_by, sorted_split
):
    if sorted_split:
        parts = [
            table.iloc[partitions[i] : partitions[i + 1]]
            for i in range(0, len(partitions) - 1)
        ]
    else:
        parts = partition_by_column(table, partitions, n_chunks)
    return await exchange_and_concat_parts(rank, eps, parts, sort=sort_by)


async def _explicit_shuffle(
    s, df_nparts, df_parts, index, divisions, sort_by, sorted_split
):
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

    divisions = cudf.Series(divisions)
    if sorted_split:
        # Avoid `scatter_by_map` by sorting the dataframe here
        # (Can just use iloc to split into groups)
        if len(df_parts) > 1:
            # Need to sort again after concatenation
            df = df.sort_values(sort_by)
        splits = df[index].searchsorted(divisions, side="left")
        splits[-1] = len(df[index])
        partitions = splits.tolist()
    else:
        partitions = divisions.searchsorted(df[index], side="right") - 1
        partitions[(df[index] >= divisions.iloc[-1]).values] = (
            len(divisions) - 2
        )

    # Run distributed shuffle and set_index algorithm
    return await distributed_shuffle(
        s["nworkers"],
        s["rank"],
        s["eps"],
        df,
        partitions,
        index,
        sort_by,
        sorted_split,
    )


def explicit_sorted_shuffle(
    df, index, divisions, sort_by, sorted_split=False, **kwargs
):
    # Explict-comms shuffle
    # TODO: Fast repartition back to df.npartitions using views...
    df.persist()
    return comms.default_comms().dataframe_operation(
        _explicit_shuffle,
        df_list=(df,),
        extra_args=(index, divisions, sort_by, sorted_split),
    )
