# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""IO logic for the RapidsMPF streaming engine."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.dsl.ir import DataFrameScan, Scan
from cudf_polars.experimental.base import (
    IOPartitionFlavor,
    PartitionInfo,
)
from cudf_polars.experimental.io import SplitScan, scan_partition_plan
from cudf_polars.experimental.rapidsmpf.dispatch import (
    generate_ir_sub_network,
    lower_ir_node,
)
from cudf_polars.experimental.rapidsmpf.nodes import define_py_node, shutdown_on_error

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import IOPartitionPlan
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.dispatch import LowerIRTransformer
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
async def dataframe_scan_node(
    ctx: Context,
    ch_out: Channel[TableChunk],
    ir: DataFrameScan,
    rows_per_partition: int,
) -> None:
    """
    DataFrameScan node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ch_out
        The output channel.
    ir
        The DataFrameScan node.
    rows_per_partition
        The number of rows per partition.
    """
    # TODO: Use (throttled) thread pool
    # TODO: Use multiple streams
    nrows = max(ir.df.shape()[0], 1)
    global_count = math.ceil(nrows / rows_per_partition)
    local_count = math.ceil(global_count / ctx.comm().nranks)
    local_offset = local_count * ctx.comm().rank

    async with shutdown_on_error(ctx, ch_out):
        for seq_num in range(local_count):
            offset = local_offset + seq_num * rows_per_partition
            if offset >= nrows:
                break  # pragma: no cover; Requires multiple ranks
            ir_slice = DataFrameScan(
                ir.schema,
                ir.df.slice(offset, rows_per_partition),
                ir.projection,
            )

            # Evaluate the IR node
            df: DataFrame = ir_slice.do_evaluate(*ir_slice._non_child_args)

            # Return the output chunk
            chunk = TableChunk.from_pylibcudf_table(seq_num, df.table, DEFAULT_STREAM)
            await ch_out.send(ctx, Message(chunk))
        await ch_out.drain(ctx)


@generate_ir_sub_network.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_sub_network'"
    )
    rows_per_partition = config_options.executor.max_rows_per_partition

    ctx = rec.state["ctx"]
    ch_out = Channel()
    nodes: dict[IR, list[Any]] = {
        ir: [dataframe_scan_node(ctx, ch_out, ir, rows_per_partition)]
    }
    channels: dict[IR, list[Any]] = {ir: [ch_out]}
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
        plan = scan_partition_plan(ir, rec.state["stats"], config_options)
        paths = list(ir.paths)

        # NOTE: We calculate the expected partition count
        # to help trigger fallback warnings in lower_ir_graph.
        # The generate_ir_sub_network logic is NOT required
        # to obey this partition count. However, the count
        # WILL match after an IO operation (for now).
        if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
            count = plan.factor * len(paths)
        else:
            count = math.ceil(len(paths) / plan.factor)

        return ir, {ir: PartitionInfo(count=count, io_plan=plan)}

    return ir, {ir: PartitionInfo(count=1)}  # pragma: no cover


@define_py_node()
async def scan_node(
    ctx: Context,
    ch_out: Channel[TableChunk],
    ir: Scan,
    plan: IOPartitionPlan,
    parquet_options: ParquetOptions,
) -> None:
    """
    Scan node for rapidsmpf.

    Parameters
    ----------
    ctx
        The context.
    ch_out
        The output channel.
    ir
        The Scan node.
    plan
        The partitioning plan.
    parquet_options
        The Parquet options.
    """
    # TODO: Use (throttled) thread pool
    # TODO: Use multiple streams
    async with shutdown_on_error(ctx, ch_out):
        # Build a list of local Scan operations
        scans: list[Scan | SplitScan] = []
        if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
            count = plan.factor * len(ir.paths)
            local_count = math.ceil(count / ctx.comm().nranks)
            local_offset = local_count * ctx.comm().rank
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
            local_count = math.ceil(count / ctx.comm().nranks)
            local_offset = local_count * ctx.comm().rank
            paths_offset_start = local_offset * plan.factor
            paths_offset_end = paths_offset_start + plan.factor
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

        # Read data and push to the output channel
        for seq_num, scan in enumerate(scans):
            # Evaluate the IR node
            df: DataFrame = scan.do_evaluate(*scan._non_child_args)

            # Return the output chunk
            chunk = TableChunk.from_pylibcudf_table(seq_num, df.table, DEFAULT_STREAM)
            await ch_out.send(ctx, Message(chunk))

        await ch_out.drain(ctx)


@generate_ir_sub_network.register(Scan)
def _(ir: Scan, rec: SubNetGenerator) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    config_options = rec.state["config_options"]
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_sub_network'"
    )
    parquet_options = config_options.parquet_options
    partition_info = rec.state["partition_info"][ir]

    assert partition_info.io_plan is not None, "Scan node must have a partition plan"
    plan: IOPartitionPlan = partition_info.io_plan
    if plan.flavor == IOPartitionFlavor.SPLIT_FILES:
        parquet_options = dataclasses.replace(parquet_options, chunked=False)

    ch_out = Channel()
    nodes: dict[IR, list[Any]] = {
        ir: [
            scan_node(
                rec.state["ctx"],
                ch_out,
                ir,
                plan,
                parquet_options,
            )
        ]
    }
    channels: dict[IR, list[Any]] = {ir: [ch_out]}
    return nodes, channels
