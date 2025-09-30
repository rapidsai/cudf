# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation with the RAPIDS-MPF streaming engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

from rapidsmpf.buffer.resource import BufferResource
from rapidsmpf.communicator.single import new_communicator
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.channel import Channel
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.leaf_node import pull_from_channel
from rapidsmpf.streaming.core.node import (
    run_streaming_pipeline,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import rmm

import cudf_polars.experimental.rapidsmpf.io  # noqa: F401
from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import (
    IR,
)
from cudf_polars.dsl.traversal import CachingVisitor
from cudf_polars.experimental.dispatch import (
    generate_ir_sub_network,
    lower_ir_node,
    lower_ir_node_rapidsmpf,
)
from cudf_polars.experimental.rapidsmpf.utils import (
    concatenate,
    pointwise_single_channel_node,
)
from cudf_polars.experimental.statistics import collect_statistics
from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.streaming.core.leaf_node import DeferredMessages

    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.experimental.dispatch import LowerIRTransformer, State
    from cudf_polars.experimental.parallel import ConfigOptions


class GenState(TypedDict):
    """
    State used for generating a streaming sub-network.

    Parameters
    ----------
    ctx
        The rapidsmpf context.
    config_options
        GPUEngine configuration options.
    partition_info
        Partition information.
    """

    ctx: Context
    config_options: ConfigOptions
    partition_info: MutableMapping[IR, PartitionInfo]


SubNetGenerator: TypeAlias = GenericTransformer[
    "IR", "tuple[dict[IR, list[Any]], dict[IR, Any]]", GenState
]
"""Protocol for Generating a streaming sub-network."""


def evaluate_logical_plan(ir: IR, config_options: ConfigOptions) -> DataFrame:
    """Evaluate a logical plan with RAPIDS-MPF."""
    # Configure the context
    # TODO: Mulit-GPU version may be very different.
    options = Options(get_environment_variables())
    comm = new_communicator(options)
    mr = RmmResourceAdaptor(rmm.mr.CudaAsyncMemoryResource())
    br = BufferResource(mr)
    rmm.mr.set_current_device_resource(mr)
    ctx = Context(comm, br, options)
    executor = ThreadPoolExecutor(max_workers=1)

    # Lower the IR graph
    ir, partition_info = lower_ir_graph_rapidsmpf(ir, config_options)

    # Generate network nodes
    nodes, output = generate_network(ctx, ir, partition_info, config_options)

    # Run the network
    run_streaming_pipeline(nodes=nodes, py_executor=executor)

    # Extract/return the result
    msgs = output.release()
    assert len(msgs) == 1, "Expected exactly one message"
    return DataFrame.from_table(
        TableChunk.from_message(msgs[0]).table_view(),
        list(ir.schema.keys()),
        list(ir.schema.values()),
    )


@lower_ir_node_rapidsmpf.register(IR)
def _(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:  # pragma: no cover
    # Default logic - Use ``lower_ir_node``, but
    # many IR types will need different logic.
    return lower_ir_node(ir, rec)


def lower_ir_graph_rapidsmpf(
    ir: IR, config_options: ConfigOptions
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    """
    Rewrite an IR graph and extract partitioning information.

    Parameters
    ----------
    ir
        Root of the graph to rewrite.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    new_ir, partition_info
        The rewritten graph, and a mapping from unique nodes
        in the new graph to associated partitioning information.

    Notes
    -----
    This function traverses the unique nodes of the graph with
    root `ir`, and applies :func:`lower_ir_node` to each node.

    See Also
    --------
    lower_ir_node
    """
    state: State = {
        "config_options": config_options,
        "stats": collect_statistics(ir, config_options),
    }
    mapper: LowerIRTransformer = CachingVisitor(lower_ir_node_rapidsmpf, state=state)
    return mapper(ir)


def generate_network(
    ctx: Context,
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
) -> tuple[list[Any], DeferredMessages]:
    """
    Generate network nodes for a logical plan.

    Parameters
    ----------
    ctx
        The context.
    ir
        The IR node.
    partition_info
        The partition information.
    config_options
        The configuration options.
    """
    # Generate the network
    state: GenState = {
        "ctx": ctx,
        "config_options": config_options,
        "partition_info": partition_info,
    }
    mapper: SubNetGenerator = CachingVisitor(generate_ir_sub_network, state=state)
    node_mapping, channels = mapper(ir)
    nodes = [node for sublist in node_mapping.values() for node in sublist]
    ch_out = channels[ir]

    # Add node to concatenate multiple chunks
    choncat_ch_out = Channel()
    nodes.append(concatenate(ctx, ch_in=ch_out, ch_out=choncat_ch_out))
    ch_out = choncat_ch_out

    # Add final node to pull from the output channel
    output_node, output = pull_from_channel(ctx, ch_in=ch_out)
    nodes.append(output_node)

    # Return network and output hook
    return nodes, output


@generate_ir_sub_network.register(IR)
def _(ir: IR, rec: SubNetGenerator) -> tuple[dict[IR, list[Any]], dict[IR, Any]]:
    # Process children
    if len(ir.children) != 1:
        raise NotImplementedError(f"Unsupported IR node for rapidsmpf: {type(ir)}.")
    nodes, channels = rec(ir.children[0])

    # Create output channel
    channels[ir] = Channel()

    # Add simple python node
    nodes[ir] = [
        pointwise_single_channel_node(
            rec.state["ctx"],
            ir,
            channels[ir.children[0]],
            channels[ir],
            rec.state["partition_info"],
        )
    ]

    return nodes, channels
