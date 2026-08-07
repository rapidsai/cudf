# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone execution of optional pushdown-filter hints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_streaming import CardinalityEstimator
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.dsl.ir import Join, Projection
from cudf_polars.dsl.utils.naming import names_to_indices
from cudf_polars.streaming.actor_graph.dispatch import generate_ir_sub_network
from cudf_polars.streaming.actor_graph.join import JoinStrategy, broadcast_join
from cudf_polars.streaming.actor_graph.prefilter import (
    PrefilterExecution,
    add_bloom_prefilter,
    choose_prefilter_method,
)
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    _sample_chunks,
    aggregate_table_size_stats,
    gather_in_task_group,
    process_children,
    recv_metadata,
    replay_buffered_channel,
    shutdown_on_error,
)
from cudf_polars.streaming.filter_hint import PushdownFilterHint

if TYPE_CHECKING:
    from cudf_streaming.channel_metadata import ChannelMetadata
    from cudf_streaming.table_chunk import TableChunk
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator
    from cudf_polars.streaming.actor_graph.prefilter import PrefilterDecision
    from cudf_polars.streaming.actor_graph.utils import TableSizeStats
    from cudf_polars.utils.config import StreamingExecutor


def make_broadcast_semi_join(ir: PushdownFilterHint) -> Join:
    """Build the synthetic semi-join used for exact filtering."""
    target, domain = ir.children
    domain_schema = {key.name: key.value.dtype for key in ir.domain_on}
    if len(domain_schema) != len(ir.domain_on):
        raise ValueError("Broadcast semi-join keys must have unique names")
    projected_domain = Projection(domain_schema, domain)
    return Join(
        target.schema,
        ir.target_on,
        ir.domain_on,
        ("Semi", ir.nulls_equal, None, "", False, "none"),
        target,
        projected_domain,
    )


async def sample_prefilter_inputs(
    context: Context,
    comm: Communicator,
    ir: PushdownFilterHint,
    ch_target: Channel[TableChunk],
    ch_domain: Channel[TableChunk],
    target_metadata: ChannelMetadata,
    domain_metadata: ChannelMetadata,
    executor: StreamingExecutor,
    collective_id: int,
) -> tuple[TableSizeStats, TableSizeStats]:
    """Sample and aggregate a standalone hint's target and domain inputs."""
    dynamic_planning = executor.dynamic_planning
    if dynamic_planning is None:
        raise ValueError("Standalone prefilters require dynamic planning")
    local_samples = await gather_in_task_group(
        _sample_chunks(
            context,
            ch_target,
            dynamic_planning.sample_chunk_count,
            executor.target_partition_size,
            target_metadata.local_count,
        ),
        _sample_chunks(
            context,
            ch_domain,
            dynamic_planning.sample_chunk_count,
            executor.target_partition_size,
            domain_metadata.local_count,
            cardinality_estimator=CardinalityEstimator(
                context,
                comm,
                tag=collective_id,
            ),
            cardinality_columns=names_to_indices(ir.domain_on, ir.children[1].schema),
        ),
    )
    target_sample, domain_sample = await aggregate_table_size_stats(
        context,
        comm,
        tuple(local_samples),
        collective_id,
    )
    return target_sample, domain_sample


async def replay_skipped_prefilter(
    context: Context,
    ir: PushdownFilterHint,
    ch_out: Channel[TableChunk],
    ch_target: Channel[TableChunk],
    ch_domain: Channel[TableChunk],
    target_metadata: ChannelMetadata,
    target_sample: TableSizeStats,
    domain_sample: TableSizeStats,
) -> None:
    """Replay an unfiltered target and stop its unused domain input."""
    domain_sample.chunks.clear()
    await gather_in_task_group(
        replay_buffered_channel(
            context,
            ch_out,
            ch_target,
            target_sample.chunks,
            target_metadata,
            trace_ir=ir,
        ),
        ch_domain.shutdown(context),
    )


async def apply_prefilter(
    context: Context,
    comm: Communicator,
    ir: PushdownFilterHint,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_target: Channel[TableChunk],
    ch_domain: Channel[TableChunk],
    target_metadata: ChannelMetadata,
    domain_metadata: ChannelMetadata,
    target_sample: TableSizeStats,
    domain_sample: TableSizeStats,
    decision: PrefilterDecision,
    collective_id: int,
    trace_stats: dict[str, Any] | None,
) -> None:
    """Apply the selected standalone prefilter implementation."""
    target, domain = ir.children
    domain_indices = names_to_indices(ir.domain_on, domain.schema)
    if domain_indices != tuple(range(len(domain.schema))):
        raise ValueError("Pushdown filter domains must contain only keys")

    execution = PrefilterExecution(context)
    ch_target_replay: Channel[TableChunk] = context.create_channel()
    ch_domain_replay: Channel[TableChunk] = context.create_channel()
    execution.add_channel(ch_target_replay)
    execution.add_channel(ch_domain_replay)
    execution.add_task(
        replay_buffered_channel(
            context,
            ch_target_replay,
            ch_target,
            target_sample.chunks,
            target_metadata,
            trace_ir=ir,
        )
    )
    execution.add_task(
        replay_buffered_channel(
            context,
            ch_domain_replay,
            ch_domain,
            domain_sample.chunks,
            domain_metadata,
            trace_ir=ir,
        )
    )

    if decision.method == "bloom":
        if decision.bloom_bytes is None:
            raise ValueError("Bloom prefilter decision has no filter size")
        add_bloom_prefilter(
            context,
            comm,
            decision.bloom_bytes,
            execution,
            names_to_indices(ir.target_on, target.schema),
            ch_domain_replay,
            ch_target_replay,
            ch_out,
            collective_id,
            trace_stats,
        )
    elif decision.method == "broadcast_semi_join":
        execution.add_task(
            broadcast_join(
                context,
                comm,
                make_broadcast_semi_join(ir),
                ir_context,
                ch_out,
                ch_target_replay,
                ch_domain_replay,
                JoinStrategy(broadcast_side="right"),
                collective_id,
                target_partition_size=None,
                tracer=None,
                trace_stats=trace_stats,
            )
        )
    else:
        raise ValueError(f"Cannot apply prefilter method {decision.method!r}")

    async with shutdown_on_error(
        context,
        *execution.channels,
        trace_ir=ir,
        ir_context=ir_context,
    ):
        await gather_in_task_group(*execution.tasks)


@define_actor()
async def pushdown_filter_actor(
    context: Context,
    comm: Communicator,
    ir: PushdownFilterHint,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_target: Channel[TableChunk],
    ch_domain: Channel[TableChunk],
    executor: StreamingExecutor,
    collective_id: int,
) -> None:
    """Choose and optionally execute one standalone pushdown-filter hint."""
    samples: tuple[TableSizeStats, TableSizeStats] | None = None
    async with shutdown_on_error(
        context,
        ch_out,
        ch_target,
        ch_domain,
        trace_ir=ir,
        ir_context=ir_context,
    ) as tracer:
        try:
            target_metadata, domain_metadata = await gather_in_task_group(
                recv_metadata(ch_target, context),
                recv_metadata(ch_domain, context),
            )
            samples = await sample_prefilter_inputs(
                context,
                comm,
                ir,
                ch_target,
                ch_domain,
                target_metadata,
                domain_metadata,
                executor,
                collective_id,
            )
            target_sample, domain_sample = samples
            config = executor.join_filter_pushdown
            if config is None:
                raise ValueError("Standalone prefilter has no runtime configuration")
            decision = choose_prefilter_method(
                ir.domain_on,
                target_sample,
                domain_sample,
                broadcast_limit=executor.broadcast_limit,
                bloom_filter_max_size=config.bloom_filter_max_size,
            )
            trace = decision.trace_details()
            trace["placement"] = "standalone"
            trace_stats = trace if tracer is not None else None
            if tracer is not None:
                tracer.decision = decision.method
                tracer.set_extra("prefilter", trace)

            if decision.method == "skip":
                await replay_skipped_prefilter(
                    context,
                    ir,
                    ch_out,
                    ch_target,
                    ch_domain,
                    target_metadata,
                    target_sample,
                    domain_sample,
                )
            else:
                await apply_prefilter(
                    context,
                    comm,
                    ir,
                    ir_context,
                    ch_out,
                    ch_target,
                    ch_domain,
                    target_metadata,
                    domain_metadata,
                    target_sample,
                    domain_sample,
                    decision,
                    collective_id,
                    trace_stats,
                )
        finally:
            if samples is not None:
                for sample in samples:
                    sample.chunks.clear()


@generate_ir_sub_network.register(PushdownFilterHint)
def generate_pushdown_filter_subnetwork(
    ir: PushdownFilterHint, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    """Generate the actor subnetwork for a standalone filter hint."""
    target, domain = ir.children
    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    (collective_id,) = rec.state["collective_id_map"][ir]
    actors[ir] = [
        pushdown_filter_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            rec.state["ir_context"],
            channels[ir].reserve_input_slot(),
            channels[target].reserve_output_slot(),
            channels[domain].reserve_output_slot(),
            rec.state["config_options"].executor,
            collective_id,
        )
    ]
    return actors, channels
