# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-partition evaluation."""

from __future__ import annotations

import itertools
import operator
from functools import partial, reduce
from typing import TYPE_CHECKING, Any

import cudf_polars.experimental.distinct
import cudf_polars.experimental.groupby
import cudf_polars.experimental.io
import cudf_polars.experimental.join
import cudf_polars.experimental.select
import cudf_polars.experimental.shuffle
import cudf_polars.experimental.sort  # noqa: F401
from cudf_polars.dsl.ir import (
    IR,
    Cache,
    Filter,
    HConcat,
    HStack,
    IRExecutionContext,
    MapFunction,
    Projection,
    Slice,
    Union,
)
from cudf_polars.dsl.traversal import CachingVisitor, traversal
from cudf_polars.experimental.base import PartitionInfo, get_key_name
from cudf_polars.experimental.dispatch import (
    generate_ir_tasks,
    lower_ir_node,
)
from cudf_polars.experimental.io import _clear_source_info_cache
from cudf_polars.experimental.repartition import Repartition
from cudf_polars.experimental.statistics import collect_statistics
from cudf_polars.experimental.utils import (
    _concat,
    _contains_over,
    _dynamic_planning_on,
    _lower_ir_fallback,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from typing import Any

    import polars as pl

    from cudf_polars.experimental.base import StatsCollector
    from cudf_polars.experimental.dispatch import LowerIRTransformer, State
    from cudf_polars.utils.config import ConfigOptions


@lower_ir_node.register(IR)
def _(
    ir: IR, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:  # pragma: no cover
    # Default logic - Requires single partition
    return _lower_ir_fallback(
        ir, rec, msg=f"Class {type(ir)} does not support multiple partitions."
    )


def lower_ir_graph(
    ir: IR, config_options: ConfigOptions
) -> tuple[IR, MutableMapping[IR, PartitionInfo], StatsCollector]:
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
    new_ir, partition_info, stats
        The rewritten graph, a mapping from unique nodes
        in the new graph to associated partitioning information,
        and the statistics collector.

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
    mapper: LowerIRTransformer = CachingVisitor(lower_ir_node, state=state)
    return *mapper(ir), state["stats"]


def task_graph(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
) -> tuple[MutableMapping[Any, Any], str | tuple[str, int]]:
    """
    Construct a task graph for evaluation of an IR graph.

    Parameters
    ----------
    ir
        Root of the graph to rewrite.
    partition_info
        A mapping from all unique IR nodes to the
        associated partitioning information.
    config_options
        GPUEngine configuration options.
    context
        Runtime context for IR node execution.

    Returns
    -------
    graph
        A Dask-compatible task graph for the entire
        IR graph with root `ir`.

    Notes
    -----
    This function traverses the unique nodes of the
    graph with root `ir`, and extracts the tasks for
    each node with :func:`generate_ir_tasks`.

    The output is passed into :func:`post_process_task_graph` to
    add any additional processing that is specific to the executor.

    See Also
    --------
    generate_ir_tasks
    """
    context = IRExecutionContext.from_config_options(config_options)
    graph = reduce(
        operator.or_,
        (
            generate_ir_tasks(node, partition_info, context=context)
            for node in traversal([ir])
        ),
    )

    key_name = get_key_name(ir)
    partition_count = partition_info[ir].count

    key: str | tuple[str, int]
    if partition_count > 1:
        graph[key_name] = (
            partial(_concat, context=context),
            *partition_info[ir].keys(ir),
        )
        key = key_name
    else:
        key = (key_name, 0)

    graph = post_process_task_graph(graph, key, config_options)
    return graph, key


# The true type signature for get_scheduler() needs an overload. Not worth it.


def get_scheduler(config_options: ConfigOptions) -> Any:
    """Get appropriate task scheduler."""
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'generate_ir_tasks'"
    )

    cluster = config_options.executor.cluster

    if (
        cluster == "distributed"
    ):  # pragma: no cover; block depends on executor type and Distributed cluster
        from distributed import get_client

        from cudf_polars.experimental.dask_registers import DaskRegisterManager

        client = get_client()
        DaskRegisterManager.register_once()
        DaskRegisterManager.run_on_cluster(client)
        return client.get
    elif cluster == "single":
        from cudf_polars.experimental.scheduler import synchronous_scheduler

        return synchronous_scheduler
    else:  # pragma: no cover
        raise ValueError(f"{cluster} not a supported cluster option.")


def post_process_task_graph(
    graph: MutableMapping[Any, Any],
    key: str | tuple[str, int],
    config_options: ConfigOptions,
) -> MutableMapping[Any, Any]:
    """
    Post-process the task graph.

    Parameters
    ----------
    graph
        Task graph to post-process.
    key
        Output key for the graph.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    graph
        A Dask-compatible task graph.
    """
    assert config_options.executor.name == "streaming", (
        "'in-memory' executor not supported in 'post_process_task_graph'"
    )

    if config_options.executor.rapidsmpf_spill:  # pragma: no cover
        from cudf_polars.experimental.spilling import wrap_dataframe_in_spillable

        return wrap_dataframe_in_spillable(
            graph, ignore_key=key, config_options=config_options
        )
    return graph


def evaluate_rapidsmpf(
    ir: IR,
    config_options: ConfigOptions,
) -> pl.DataFrame:  # pragma: no cover; rapidsmpf runtime not tested in CI yet
    """
    Evaluate with the RapidsMPF streaming runtime.

    Parameters
    ----------
    ir
        Logical plan to evaluate.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    A cudf-polars DataFrame object.
    """
    from cudf_polars.experimental.rapidsmpf.core import evaluate_logical_plan

    result, _ = evaluate_logical_plan(ir, config_options, collect_metadata=False)
    return result


def evaluate_streaming(
    ir: IR,
    config_options: ConfigOptions,
) -> pl.DataFrame:
    """
    Evaluate an IR graph with partitioning.

    Parameters
    ----------
    ir
        Logical plan to evaluate.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    A cudf-polars DataFrame object.
    """
    # Clear source info cache in case data was overwritten
    _clear_source_info_cache()

    assert config_options.executor.name == "streaming", "Executor must be streaming"
    if (
        config_options.executor.runtime == "rapidsmpf"
    ):  # pragma: no cover; rapidsmpf runtime not tested in CI yet
        # Using the RapidsMPF streaming runtime.
        return evaluate_rapidsmpf(ir, config_options)
    else:
        # Using the default task engine.
        ir, partition_info, _ = lower_ir_graph(ir, config_options)

        graph, key = task_graph(ir, partition_info, config_options)

        return get_scheduler(config_options)(graph, key).to_polars()


@generate_ir_tasks.register(IR)
def _(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    context: IRExecutionContext,
) -> MutableMapping[Any, Any]:
    # Generate pointwise (embarrassingly-parallel) tasks by default
    child_names = [get_key_name(c) for c in ir.children]
    bcast_child = [partition_info[c].count == 1 for c in ir.children]

    return {
        key: (
            partial(ir.do_evaluate, context=context),
            *ir._non_child_args,
            *[
                (child_name, 0 if bcast_child[j] else i)
                for j, child_name in enumerate(child_names)
            ],
        )
        for i, key in enumerate(partition_info[ir].keys(ir))
    }


@lower_ir_node.register(Union)
def _(
    ir: Union, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Check zlice
    if ir.zlice is not None:
        return rec(
            Slice(
                ir.schema,
                *ir.zlice,
                Union(ir.schema, None, *ir.children),
            )
        )

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)

    # Partition count is the sum of all child partitions
    count = sum(partition_info[c].count for c in children)

    # Return reconstructed node and partition-info dict
    new_node = ir.reconstruct(children)
    partition_info[new_node] = PartitionInfo(count=count)
    return new_node, partition_info


@generate_ir_tasks.register(Union)
def _(
    ir: Union,
    partition_info: MutableMapping[IR, PartitionInfo],
    context: IRExecutionContext,
) -> MutableMapping[Any, Any]:
    key_name = get_key_name(ir)
    partition = itertools.count()
    return {
        (key_name, next(partition)): child_key
        for child in ir.children
        for child_key in partition_info[child].keys(child)
    }


@lower_ir_node.register(MapFunction)
def _(
    ir: MapFunction, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Allow pointwise operations
    if ir.name in ("rename", "explode"):
        return _lower_ir_pwise(ir, rec)

    # Fallback for everything else
    return _lower_ir_fallback(
        ir, rec, msg=f"{ir.name} is not supported for multiple partitions."
    )


def _lower_ir_pwise(
    ir: IR, rec: LowerIRTransformer, *, preserve_partitioning: bool = False
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Lower a partition-wise (i.e. embarrassingly-parallel) IR node

    # Lower children
    children, _partition_info = zip(*(rec(c) for c in ir.children), strict=True)
    partition_info = reduce(operator.or_, _partition_info)
    counts = {partition_info[c].count for c in children}

    # Check that child partitioning is supported
    if len(counts) > 1:  # pragma: no cover
        return _lower_ir_fallback(
            ir,
            rec,
            msg=f"Class {type(ir)} does not support children with mismatched partition counts.",
        )

    # Preserve child partition_info if possible
    if preserve_partitioning and len(children) == 1:
        partition = partition_info[children[0]]
    else:
        partition = PartitionInfo(count=max(counts))

    # Return reconstructed node and partition-info dict
    new_node = ir.reconstruct(children)
    partition_info[new_node] = partition
    return new_node, partition_info


_lower_ir_pwise_preserve = partial(_lower_ir_pwise, preserve_partitioning=True)
lower_ir_node.register(Projection, _lower_ir_pwise_preserve)
lower_ir_node.register(Cache, _lower_ir_pwise_preserve)
lower_ir_node.register(HConcat, _lower_ir_pwise)


@lower_ir_node.register(Filter)
def _(
    ir: Filter, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])

    if partition_info[child].count > 1 and _contains_over([ir.mask.value]):
        # mask contains .over(...), collapse to single partition
        return _lower_ir_fallback(
            ir.reconstruct([child]),
            rec,
            msg=(
                "over(...) inside filter is not supported for multiple partitions; "
                "falling back to in-memory evaluation."
            ),
        )

    if partition_info[child].count > 1 and not all(
        expr.is_pointwise for expr in traversal([ir.mask.value])
    ):
        # TODO: Use expression decomposition to lower Filter
        # See: https://github.com/rapidsai/cudf/issues/20076
        return _lower_ir_fallback(
            ir, rec, msg="This filter is not supported for multiple partitions."
        )

    new_node = ir.reconstruct([child])
    partition_info[new_node] = partition_info[child]
    return new_node, partition_info


@lower_ir_node.register(Slice)
def _(
    ir: Slice, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    # Check for dynamic planning - may have more partitions at runtime
    config_options = rec.state["config_options"]
    dynamic_planning = _dynamic_planning_on(config_options)

    if ir.offset == 0:
        # Taking the first N rows.
        # We don't know how large each partition is, so we reduce.
        new_node, partition_info = _lower_ir_pwise(ir, rec)
        if partition_info[new_node].count > 1 or dynamic_planning:
            # Collapse down to single partition
            inter = Repartition(new_node.schema, new_node)
            partition_info[inter] = PartitionInfo(count=1)
            # Slice reduced partition
            new_node = ir.reconstruct([inter])
            partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info

    # Fallback
    return _lower_ir_fallback(
        ir, rec, msg="This slice not supported for multiple partitions."
    )


@lower_ir_node.register(HStack)
def _(
    ir: HStack, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    if not all(expr.is_pointwise for expr in traversal([e.value for e in ir.columns])):
        # TODO: Avoid fallback if/when possible
        return _lower_ir_fallback(
            ir, rec, msg="This HStack not supported for multiple partitions."
        )

    child, partition_info = rec(ir.children[0])
    new_node = ir.reconstruct([child])
    partition_info[new_node] = partition_info[child]
    return new_node, partition_info
