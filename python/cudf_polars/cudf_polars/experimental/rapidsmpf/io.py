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

import pylibcudf as plc

from cudf_polars.dsl.ir import (
    IR,
    DataFrameScan,
    Scan,
    _cast_literals_to_physical_types,
    _parquet_physical_types,
)
from cudf_polars.dsl.to_ast import to_parquet_filter
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
from cudf_polars.experimental.rapidsmpf.nodes import (
    define_py_node,
    shutdown_on_error,
)
from cudf_polars.experimental.rapidsmpf.utils import ChannelManager

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.experimental.base import ColumnStat, StatsCollector
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.dispatch import LowerIRTransformer
    from cudf_polars.experimental.rapidsmpf.utils import ChannelPair
    from cudf_polars.utils.config import ParquetOptions


class Lineariser:
    """
    Linearizer that ensures ordered delivery from multiple concurrent producers.

    Creates one input channel per producer and streams messages to output
    in sequence-number order, buffering only out-of-order arrivals.
    """

    def __init__(
        self, context: Context, ch_out: Channel[TableChunk], num_producers: int
    ):
        self.context = context
        self.ch_out = ch_out
        self.num_producers = num_producers
        self.input_channels = [context.create_channel() for _ in range(num_producers)]

    async def drain(self) -> None:
        """
        Drain producer channels and forward messages in sequence-number order.

        Streams messages to output as soon as they arrive in order, buffering
        only out-of-order messages to minimize memory pressure.
        """
        next_seq = 0
        buffer = {}

        pending_tasks = {
            asyncio.create_task(ch.recv(self.context)): ch for ch in self.input_channels
        }

        while pending_tasks:
            done, _ = await asyncio.wait(
                pending_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                ch = pending_tasks.pop(task)
                msg = await task

                if msg is not None:
                    buffer[msg.sequence_number] = msg
                    new_task = asyncio.create_task(ch.recv(self.context))
                    pending_tasks[new_task] = ch

            # Forward consecutive messages
            while next_seq in buffer:
                await self.ch_out.send(self.context, buffer.pop(next_seq))
                next_seq += 1

        # Forward any remaining buffered messages
        for seq in sorted(buffer.keys()):
            await self.ch_out.send(self.context, buffer[seq])

        await self.ch_out.drain(self.context)


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
    num_producers: int,
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
    num_producers
        The number of producers to use for the DataFrameScan node.
    rows_per_partition
        The number of rows per partition.
    """
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
        # Build list of IR slices to read
        ir_slices = []
        for seq_num in range(local_count):
            offset = local_offset * rows_per_partition + seq_num * rows_per_partition
            if offset >= nrows:
                break
            ir_slices.append(
                DataFrameScan(
                    ir.schema,
                    ir.df.slice(offset, rows_per_partition),
                    ir.projection,
                )
            )

        # Use Lineariser to ensure ordered delivery
        num_producers = min(num_producers, len(ir_slices))
        lineariser = Lineariser(context, ch_out.data, num_producers)

        # Assign tasks to producers using round-robin
        producer_tasks: list[list[tuple[int, DataFrameScan]]] = [
            [] for _ in range(num_producers)
        ]
        for task_idx, ir_slice in enumerate(ir_slices):
            producer_id = task_idx % num_producers
            producer_tasks[producer_id].append((task_idx, ir_slice))

        async def _producer(producer_id: int, ch_out: Channel) -> None:
            for task_idx, ir_slice in producer_tasks[producer_id]:
                await read_chunk(
                    context,
                    ir_slice,
                    task_idx,
                    ch_out,
                    ir_context,
                )
            await ch_out.drain(context)

        tasks = [lineariser.drain()]
        tasks.extend(
            _producer(i, ch_in) for i, ch_in in enumerate(lineariser.input_channels)
        )
        await asyncio.gather(*tasks)


@generate_ir_sub_network.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: SubNetGenerator
) -> tuple[list[Any], dict[IR, ChannelManager]]:
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_sub_network'"
    )
    rows_per_partition = config_options.executor.max_rows_per_partition
    num_producers = rec.state["max_io_threads"]

    context = rec.state["context"]
    ir_context = rec.state["ir_context"]
    channels: dict[IR, ChannelManager] = {ir: ChannelManager(rec.state["context"])}
    nodes: list[Any] = [
        dataframescan_node(
            context,
            ir,
            ir_context,
            channels[ir].reserve_input_slot(),
            num_producers=num_producers,
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
    scan
        The Scan or DataFrameScan node.
    seq_num
        The sequence number.
    ch_out
        The output channel.
    ir_context
        The execution context for the IR node.
    """
    # Evaluate and send the Scan-node result
    df = await asyncio.to_thread(
        scan.do_evaluate,
        *scan._non_child_args,
        context=ir_context,
    )
    await ch_out.send(
        context,
        Message(
            seq_num,
            TableChunk.from_pylibcudf_table(
                df.table,
                df.stream,
                exclusive_view=True,
            ),
        ),
    )


@define_py_node()
async def scan_node(
    context: Context,
    ir: Scan,
    ir_context: IRExecutionContext,
    ch_out: ChannelPair,
    *,
    num_producers: int,
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
    num_producers
        The number of producers to use for the scan node.
    plan
        The partitioning plan.
    parquet_options
        The Parquet options.
    """
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

        # Use Lineariser to ensure ordered delivery
        num_producers = min(num_producers, len(scans))
        lineariser = Lineariser(context, ch_out.data, num_producers)

        # Assign tasks to producers using round-robin
        producer_tasks: list[list[tuple[int, Scan | SplitScan]]] = [
            [] for _ in range(num_producers)
        ]
        for task_idx, scan in enumerate(scans):
            producer_id = task_idx % num_producers
            producer_tasks[producer_id].append((task_idx, scan))

        async def _producer(producer_id: int, ch_out: Channel) -> None:
            for task_idx, scan in producer_tasks[producer_id]:
                await read_chunk(
                    context,
                    scan,
                    task_idx,
                    ch_out,
                    ir_context,
                )
            await ch_out.drain(context)

        tasks = [lineariser.drain()]
        tasks.extend(
            _producer(i, ch_in) for i, ch_in in enumerate(lineariser.input_channels)
        )
        await asyncio.gather(*tasks)


def make_rapidsmpf_read_parquet_node(
    context: Context,
    ir: Scan,
    num_producers: int,
    ch_out: ChannelPair,
    stats: StatsCollector,
    partition_info: PartitionInfo,
) -> Any | None:
    """
    Make a RapidsMPF read parquet node.

    Parameters
    ----------
    context
        The rapidsmpf context.
    ir
        The Scan node.
    num_producers
        The number of producers to use for the scan node.
    ch_out
        The output ChannelPair.
    stats
        The statistics collector.
    partition_info
        The partition information.

    Returns
    -------
    The RapidsMPF read parquet node, or None if the predicate cannot be
    converted to a parquet filter (caller should fall back to scan_node).
    """
    from rapidsmpf.streaming.cudf.parquet import Filter, read_parquet

    # Build ParquetReaderOptions
    try:
        stream = context.get_stream_from_pool()
        parquet_reader_options = plc.io.parquet.ParquetReaderOptions.builder(
            plc.io.SourceInfo(ir.paths)
        ).build()

        if ir.with_columns is not None:
            parquet_reader_options.set_columns(ir.with_columns)

        # Build predicate filter if present (passed separately to read_parquet)
        filter_obj = None
        if ir.predicate is not None:
            filter_expr = to_parquet_filter(
                _cast_literals_to_physical_types(
                    ir.predicate.value,
                    _parquet_physical_types(
                        ir.schema,
                        ir.paths,
                        ir.with_columns or list(ir.schema.keys()),
                        stream,
                    ),
                ),
                stream=stream,
            )
            if filter_expr is None:
                # Predicate cannot be converted to parquet filter
                # Return None to signal fallback to scan_node
                return None
            filter_obj = Filter(stream, filter_expr)
    except Exception as e:
        raise ValueError(f"Failed to build ParquetReaderOptions: {e}") from e

    # Calculate num_rows_per_chunk from statistics
    # Default to a reasonable chunk size if statistics are unavailable
    estimated_row_count: ColumnStat[int] | None = stats.row_count.get(ir)
    if estimated_row_count is None:
        for cs in stats.column_stats.get(ir, {}).values():
            if cs.source_info.row_count.value is not None:
                estimated_row_count = cs.source_info.row_count
                break
    if estimated_row_count is not None and estimated_row_count.value is not None:
        num_rows_per_chunk = int(
            max(1, estimated_row_count.value // partition_info.count)
        )
    else:
        # Fallback: use a default chunk size if statistics are not available
        num_rows_per_chunk = 1_000_000  # 1 million rows as default

    # Validate inputs
    if num_rows_per_chunk <= 0:
        raise ValueError(f"Invalid num_rows_per_chunk: {num_rows_per_chunk}")
    if num_producers <= 0:
        raise ValueError(f"Invalid num_producers: {num_producers}")

    try:
        return read_parquet(
            context,
            ch_out.data,
            num_producers,
            parquet_reader_options,
            num_rows_per_chunk,
            filter=filter_obj,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create read_parquet node: {e}\n"
            f"  paths: {ir.paths}\n"
            f"  num_producers: {num_producers}\n"
            f"  num_rows_per_chunk: {num_rows_per_chunk}\n"
            f"  partition_count: {partition_info.count}\n"
            f"  filter: {filter_obj}"
        ) from e


@generate_ir_sub_network.register(Scan)
def _(ir: Scan, rec: SubNetGenerator) -> tuple[list[Any], dict[IR, ChannelManager]]:
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_sub_network'"
    )
    parquet_options = config_options.parquet_options
    partition_info = rec.state["partition_info"][ir]
    num_producers = rec.state["max_io_threads"]
    channels: dict[IR, ChannelManager] = {ir: ChannelManager(rec.state["context"])}

    # Use rapidsmpf native read_parquet for multi-partition Parquet scans.
    ch_pair = channels[ir].reserve_input_slot()
    nodes: list[Any]
    native_node: Any = None
    if (
        partition_info.count > 1
        and ir.typ == "parquet"
        and ir.row_index is None
        and ir.include_file_paths is None
        and ir.n_rows == -1
        and ir.skip_rows == 0
    ):
        native_node = make_rapidsmpf_read_parquet_node(
            rec.state["context"],
            ir,
            num_producers,
            ch_pair,
            rec.state["stats"],
            partition_info,
        )

    if native_node is not None:
        nodes = [native_node]
    else:
        # Fall back to scan_node (predicate not convertible, or other constraint)
        assert partition_info.io_plan is not None, (
            "Scan node must have a partition plan"
        )
        plan: IOPartitionPlan = partition_info.io_plan
        if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
            parquet_options = dataclasses.replace(parquet_options, chunked=False)

        nodes = [
            scan_node(
                rec.state["context"],
                ir,
                rec.state["ir_context"],
                ch_pair,
                num_producers=num_producers,
                plan=plan,
                parquet_options=parquet_options,
            )
        ]
    return nodes, channels
