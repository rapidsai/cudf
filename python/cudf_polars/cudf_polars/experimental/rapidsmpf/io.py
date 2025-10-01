# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""IO logic for the RAPIDS-MPF streaming engine."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from rapidsmpf.streaming.core.channel import Channel, Message
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

from rmm.pylibrmm.stream import DEFAULT_STREAM

from cudf_polars.dsl.ir import DataFrameScan, Scan
from cudf_polars.experimental.base import PartitionInfo
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
    from cudf_polars.experimental.rapidsmpf.core import SubNetGenerator
    from cudf_polars.experimental.rapidsmpf.dispatch import LowerIRTransformer


@lower_ir_node.register(DataFrameScan)
def _(
    ir: DataFrameScan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    config_options = rec.state["config_options"]

    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'lower_ir_node_rapidsmpf'"
    )

    # TODO: Handle multiple workers.
    rows_per_partition = config_options.executor.max_rows_per_partition
    nrows = max(ir.df.shape()[0], 1)
    count = math.ceil(nrows / rows_per_partition)
    return ir, {ir: PartitionInfo(count=count)}


@lower_ir_node.register(Scan)
def _(
    ir: Scan, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:  # pragma: no cover
    raise NotImplementedError(
        "Scan is not yet supported in the RAPIDS-MPF streaming engine"
    )


@define_py_node()
async def dataframe_scan_node(
    ctx: Context,
    ch_out: Channel[TableChunk],
    ir: DataFrameScan,
    rows_per_partition: int,
) -> None:
    """
    DataFrame scan node for rapidsmpf.

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
    async with shutdown_on_error(ctx, ch_out):
        for seq_num, offset in enumerate(range(0, nrows, rows_per_partition)):
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
    channels: dict[IR, Any] = {ir: ch_out}
    return nodes, channels
