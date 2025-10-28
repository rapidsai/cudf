# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""IO logic for the RapidsMPF streaming runtime."""

from __future__ import annotations

import asyncio
import dataclasses
import math
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from cudf_polars.dsl.ir import IR, DataFrameScan, Scan
from cudf_polars.experimental.base import (
    IOPartitionFlavor,
    IOPartitionPlan,
    PartitionInfo,
)
from cudf_polars.experimental.io import SplitScan, scan_partition_plan
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
    lower_ir_node,
)
from cudf_polars.experimental.rapidsmpf.nodes import define_py_node, shutdown_on_error
from cudf_polars.experimental.rapidsmpf.utils import ChannelManager

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.dispatch import LowerIRTransformer
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair
    from cudf_polars.utils.config import ParquetOptions


@lower_ir_node.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'lower_ir_node_rapidsmpf'"
    )

    # NOTE: We calculate the expected partition count
    # to help trigger fallback warnings in lower_ir_graph.
    # The generate_ir_sub_network logic is NOT required
    # to obey this partition count. However, the count
    # WILL match after an IO operation (for now).
    rows_per_partition = config_options.executor.max_rows_per_partition
    nrows = max(ir.df.shape()[0], 1)
    count = math.ceil(nrows / rows_per_partition)

    return ir, {ir: PartitionInfo(count=count)}


@define_py_node()
async def dataframescan_node(
    context: Context,
    ir: DataFrameScan,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    *,
    max_io_threads: int,
    rows_per_partition: int,
) -> None:
    """
    DataFrameScan node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The DataFrameScan node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    max_io_threads
        The maximum number of IO threads to use
        concurrently for a single DataFrameScan node.
    rows_per_partition
        The number of rows per partition.
    """
    # TODO: Use multiple streams
    nrows = max(ir.df.shape()[0], 1)
    global_count = math.ceil(nrows / rows_per_partition)

    # For single rank, simplify the logic
    if context.comm().nranks == 1:
        local_count = global_count
        local_offset = 0
    else:
        local_count = math.ceil(global_count / context.comm().nranks)
        local_offset = local_count * context.comm().rank

    async with shutdown_on_error(context, ch_out.data):
        io_throttle = asyncio.Semaphore(max_io_threads)
        for seq_num in range(local_count):
            offset = local_offset * rows_per_partition + seq_num * rows_per_partition
            if offset >= nrows:
                break

            ir_slice = DataFrameScan(
                ir.schema,
                ir.df.slice(offset, rows_per_partition),
                ir.projection,
            )
            await read_chunk(
                context, io_throttle, ir_slice, seq_num, ch_out.data, ir_context
            )

        # Drain data channel
        await ch_out.data.drain(context)


@generate_ir_sub_network.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: SubNetGenerator
) -> tuple[list[Any], dict[IR, ChannelManager]]:
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_sub_network'"
    )
    rows_per_partition = config_options.executor.max_rows_per_partition
    max_io_threads = rec.state["max_io_threads"]

    context = rec.state["context"]
    ir_context = rec.state["ir_context"]
    channels: dict[IR, ChannelManager] = {ir: ChannelManager()}
    nodes: list[Any] = [
        dataframescan_node(
            context,
            ir,
            ir_context,
            channels[ir].reserve_input_slot(),
            max_io_threads=max_io_threads,
            rows_per_partition=rows_per_partition,
        )
    ]

    return nodes, channels


@lower_ir_node.register(Scan)
def _(
    ir: Scan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    config_options = rec.state["config_options"]
    if (
        ir.typ in ("csv", "parquet", "ndjson")
        and ir.n_rows == -1
        and ir.skip_rows == 0
        and ir.row_index is None
    ):
        # NOTE: We calculate the expected partition count
        # to help trigger fallback warnings in lower_ir_graph.
        # The generate_ir_sub_network logic is NOT required
        # to obey this partition count. However, the count
        # WILL match after an IO operation (for now).
        plan = scan_partition_plan(ir, rec.state["stats"], config_options)
        paths = list(ir.paths)
        if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
            count = plan.factor * len(paths)
        else:
            count = math.ceil(len(paths) / plan.factor)

        return ir, {ir: PartitionInfo(count=count, io_plan=plan)}
    else:
        plan = IOPartitionPlan(
            flavor=IOPartitionFlavor.SINGLE_READ, factor=len(ir.paths)
        )
        return ir, {ir: PartitionInfo(count=1, io_plan=plan)}


async def read_chunk(
    context: Context,
    io_throttle: asyncio.Semaphore,
    scan: IR,
    seq_num: int,
    ch_out: Channel[TableChunk],
    ir_context: IRExecutionContext,
) -> None:
    """
    Read a chunk from disk and send it to the output channel.

    Parameters
    ----------
    context
        The rapidsmpf context.
    io_throttle
        The IO throttle.
    scan
        The Scan or DataFrameScan node.
    seq_num
        The sequence number.
    ch_out
        The output channel.
    ir_context
        The execution context for the IR node.
    """
    async with io_throttle:
        # Evaluate and send the Scan-node result
        df = await asyncio.to_thread(
            scan.do_evaluate,
            *scan._non_child_args,
            context=ir_context,
        )
        await ch_out.send(
            context,
            Message(
                TableChunk.from_pylibcudf_table(
                    seq_num,
                    df.table,
                    df.stream,
                    exclusive_view=True,
                )
            ),
        )


@define_py_node()
async def scan_node(
    context: Context,
    ir: Scan,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    *,
    max_io_threads: int,
    plan: IOPartitionPlan,
    parquet_options: ParquetOptions,
) -> None:
    """
    Scan node for rapidsmpf.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The Scan node.
    ir_context
        The execution context for the IR node.
    ch_out
        The output ChannelPair.
    max_io_threads
        The maximum number of IO threads to use
        concurrently for a single Scan node.
    plan
        The partitioning plan.
    parquet_options
        The Parquet options.
    """
    # TODO: Use multiple streams
    async with shutdown_on_error(context, ch_out.data):
        # Build a list of local Scan operations
        scans: list[Scan | SplitScan] = []
        if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
            count = plan.factor * len(ir.paths)
            local_count = math.ceil(count / context.comm().nranks)
            local_offset = local_count * context.comm().rank
            path_offset = local_offset // plan.factor
            path_count = math.ceil(local_count / plan.factor)
            local_paths = ir.paths[path_offset : path_offset + path_count]
            sindex = local_offset % plan.factor
            for path in local_paths:
                base_scan = Scan(
                    ir.schema,
                    ir.typ,
                    ir.reader_options,
                    ir.cloud_options,
                    [path],
                    ir.with_columns,
                    ir.skip_rows,
                    ir.n_rows,
                    ir.row_index,
                    ir.include_file_paths,
                    ir.predicate,
                    parquet_options,
                )
                while sindex < plan.factor:
                    scans.append(
                        SplitScan(
                            ir.schema,
                            base_scan,
                            sindex,
                            plan.factor,
                            parquet_options,
                        )
                    )
                    sindex += 1
                sindex = 0

        else:
            count = math.ceil(len(ir.paths) / plan.factor)
            local_count = math.ceil(count / context.comm().nranks)
            local_offset = local_count * context.comm().rank
            paths_offset_start = local_offset * plan.factor
            paths_offset_end = paths_offset_start + plan.factor * local_count
            for offset in range(paths_offset_start, paths_offset_end, plan.factor):
                local_paths = ir.paths[offset : offset + plan.factor]
                scans.append(
                    Scan(
                        ir.schema,
                        ir.typ,
                        ir.reader_options,
                        ir.cloud_options,
                        local_paths,
                        ir.with_columns,
                        ir.skip_rows,
                        ir.n_rows,
                        ir.row_index,
                        ir.include_file_paths,
                        ir.predicate,
                        parquet_options,
                    )
                )

        # Read data (using io_throttle).
        # In some cases, we may may be able to read data
        # concurrently. We are reading the chunks
        # sequentially for now, because some tests assume
        # chunks are read in order.
        io_throttle = asyncio.Semaphore(max_io_threads)
        tasks = []
        for seq_num, scan in enumerate(scans):
            tasks.append(
                read_chunk(context, io_throttle, scan, seq_num, ch_out.data, ir_context)
            )

        # Drain the output data channel
        await asyncio.gather(*tasks)
        await ch_out.data.drain(context)


@generate_ir_sub_network.register(Scan)
def _(ir: Scan, rec: SubNetGenerator) -> tuple[list[Any], dict[IR, ChannelManager]]:
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_sub_network'"
    )
    parquet_options = config_options.parquet_options
    partition_info = rec.state["partition_info"][ir]
    max_io_threads = rec.state["max_io_threads"]

    assert partition_info.io_plan is not None, "Scan node must have a partition plan"
    plan: IOPartitionPlan = partition_info.io_plan
    if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
        parquet_options = dataclasses.replace(parquet_options, chunked=False)

    channels: dict[IR, ChannelManager] = {ir: ChannelManager()}
    nodes: list[Any] = [
        scan_node(
            rec.state["context"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            max_io_threads=max_io_threads,
            plan=plan,
            parquet_options=parquet_options,
        )
    ]
    return nodes, channels
