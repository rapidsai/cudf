# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Quent telemetry tracing."""

from __future__ import annotations

import concurrent.futures
import uuid
from typing import TYPE_CHECKING

import pytest

import polars as pl

import cudf_polars.quent
import cudf_polars.quent._logging
from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import DataFrameScan, Filter
from cudf_polars.dsl.translate import Translator
from cudf_polars.quent._context import (
    LocalQuentContext,
    ProcessorRegistry,
    QuentContext,
    QuentIRExecutionContext,
    declare_network_channels,
    finalize_network_channels,
)
from cudf_polars.quent._plan import build_plan, port_names_for_node
from cudf_polars.quent._types import (
    Attribute,
    Channel,
    Engine,
    Implementation,
    Memory,
    Network,
    Operator,
    Plan,
    Port,
    Query,
    Statistics,
    Task,
    Worker,
    _deserialize_value,
)
from cudf_polars.utils.config import ConfigOptions
from cudf_polars.utils.cuda_stream import get_cuda_stream

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR
    from cudf_polars.quent._types import Processor
    from cudf_polars.utils.config import StreamingExecutor


def _make_worker() -> Worker:
    return Worker(
        id=uuid.uuid4(),
        engine=Engine(id=uuid.uuid4()),
        instance_name="test-worker",
    )


def _make_dataframe(pl_df: pl.DataFrame) -> DataFrame:
    return DataFrame.from_polars(pl_df, get_cuda_stream())


def _make_quent_ir_execution_context(
    *,
    operator_id: uuid.UUID | None = None,
    disk_to_device_channel: Channel | None = None,
) -> tuple[cudf_polars.quent._logging.QuentLogger, QuentIRExecutionContext]:
    pytest.importorskip("structlog")
    logger = cudf_polars.quent._logging.QuentLogger()
    context = QuentContext()
    engine_id = context.engine.id
    worker_id = uuid.uuid4()
    pool_id = uuid.uuid4()
    if disk_to_device_channel is not None:
        device_memory = disk_to_device_channel.target
    else:
        device_memory = Memory(
            instance_name="device",
            resource_type_name="memory",
            parent_group_id=engine_id,
        )
    operator_id = operator_id or uuid.uuid4()
    query = context.query_for(uuid.uuid4())
    plan = Plan(
        id=uuid.uuid4(),
        query=query,
        parent_plan=None,
        instance_name="logical",
        edges=[],
        worker=None,
    )
    operator = Operator(
        id=operator_id,
        plan=plan,
        parent_operators=[],
        type_name="Filter",
    )
    local_context = LocalQuentContext(
        context=context,
        query=query,
        worker=Worker(id=worker_id, engine=context.engine, instance_name="rank-0"),
        logger=logger,
        thread_pool_id=pool_id,
        processor_registry=ProcessorRegistry(),
        device_memory=device_memory,
        disk_to_device_channel=disk_to_device_channel,
    )
    quent_ir_execution_context = QuentIRExecutionContext.from_execution_context(
        local_context, operator
    )
    return logger, quent_ir_execution_context


@pytest.mark.parametrize(
    "value,expected_variant",
    [
        (0, "U8"),
        (2**8 - 1, "U8"),
        (2**8, "U16"),
        (2**16 - 1, "U16"),
        (2**16, "U32"),
        (2**32 - 1, "U32"),
        (2**32, "U64"),
        (2**64 - 1, "U64"),
        (-(2**7), "I8"),
        (2**7 - 1, "U8"),
        (-(2**15), "I16"),
        (-(2**15) - 1, "I32"),
        (-(2**31), "I32"),
        (-(2**31) - 1, "I64"),
        (-(2**63), "I64"),
    ],
)
def test_attribute_integer_serialization_variants(
    value: int, expected_variant: str
) -> None:
    serialized = Attribute("x", value).serialize()
    assert serialized["key"] == "x"
    assert serialized["value"] == {expected_variant: value}


@pytest.mark.parametrize("value", [2**64, -(2**63) - 1])
def test_attribute_integer_serialization_overflow(value: int) -> None:
    with pytest.raises(
        ValueError,
        match="does not fit any Quent integer type",
    ):
        Attribute("x", value).serialize()


def test_attribute_serialization_uses_quent_value_envelope() -> None:
    assert Attribute("ratio", 1.5).serialize() == {
        "key": "ratio",
        "value": {"F64": 1.5},
    }
    assert Attribute("name", "scan").serialize() == {
        "key": "name",
        "value": {"String": "scan"},
    }
    assert Attribute("enabled", value=True).serialize() == {
        "key": "enabled",
        "value": {"U8": 1},
    }


def test_deserialize_value_requires_single_variant() -> None:
    with pytest.raises(
        ValueError,
        match=r"Expected Quent attribute value envelope with exactly one variant, got '2' instead.",
    ):
        _deserialize_value({"U8": 1, "I8": -1})


def test_deserialize_value_raises_on_unknown_variant() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported Quent custom attribute variant: 'UnsupportedVariant'",
    ):
        _deserialize_value({"UnsupportedVariant": "x"})


@pytest.fixture
def ir_and_config() -> tuple[IR, ConfigOptions[StreamingExecutor]]:
    q = pl.LazyFrame({"x": [1, 2]}).filter(pl.col("x") > 1)
    engine = pl.GPUEngine(executor="streaming")
    config_options = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    return ir, config_options


def test_build_plan_returns_correct_types(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    query_id = uuid.uuid4()
    plan_id = uuid.uuid4()
    worker = _make_worker()

    plan, operators, ports, _ = build_plan(
        ir, config_options, Query(id=query_id), plan_id, worker
    )

    assert isinstance(plan, Plan)
    assert plan.id == plan_id
    assert plan.query_id == query_id
    assert plan.instance_name == "logical"
    assert plan.parent_plan_id is None
    assert plan.worker_id == worker.id
    assert len(operators) > 0
    assert len(ports) > 0
    assert all(isinstance(op, Operator) for op in operators)
    assert all(isinstance(p, Port) for p in ports)


def test_build_plan_operator_plan_ids(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    plan_id = uuid.uuid4()
    _plan, operators, _, _ = build_plan(
        ir, config_options, Query(), plan_id, _make_worker()
    )

    for op in operators:
        assert op.plan_id == plan_id


def test_build_plan_edges_reference_ports(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    plan, _operators, ports, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )
    port_ids = {p.id for p in ports}
    for edge in plan.edges:
        assert edge.source.id in port_ids
        assert edge.target.id in port_ids


def test_build_plan_edge_direction(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    """Edges go from child 'out' port to parent input port."""
    ir, config_options = ir_and_config
    plan, _operators, _ports, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )
    for edge in plan.edges:
        assert edge.source.instance_name == "out"
        assert edge.target.instance_name != "out"


def test_build_plan_filter_topology(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    """Filter(DataFrameScan) should produce 2 operators and 1 edge."""
    ir, config_options = ir_and_config
    plan, operators, _, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )
    type_names = {op.type_name for op in operators}
    assert "Filter" in type_names
    assert "DataFrameScan" in type_names
    assert len(plan.edges) == 1


def test_plan_declare_serialization(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    query_id = uuid.uuid4()
    plan_id = uuid.uuid4()
    plan, _, _, _ = build_plan(
        ir, config_options, Query(id=query_id), plan_id, _make_worker()
    )

    event = plan.declare(timestamp=12345)
    d = event.to_dict()
    assert d["id"] == str(plan_id)
    assert d["timestamp"] == 12345

    decl = d["data"]["Plan"]["Declaration"]
    assert decl["parent"]["query_id"] == str(query_id)
    assert decl["parent"]["plan_id"] is None
    assert decl["instance_name"] == "logical"
    assert len(decl["edges"]) == 1


def test_operator_declare_serialization(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    _, operators, _, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )

    for op in operators:
        event = op.declare(timestamp=99)
        d = event.to_dict()
        assert d["id"] == str(op.id)
        decl = d["data"]["Operator"]["Declaration"]
        assert decl["plan_id"] == str(op.plan_id)
        assert decl["type_name"] == op.type_name


def test_operator_statistics_serialization(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    _, operators, _, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )
    op = operators[0]
    stats = Statistics(input_bytes=123, output_bytes=456, output_rows=7)

    event = op.statistics(stats, timestamp=101)
    d = event.to_dict()

    assert d["id"] == str(op.id)
    payload = d["data"]["Operator"]["Statistics"]["custom_attributes"]
    assert payload == [
        {"key": "input_bytes", "value": {"U64": 123}},
        {"key": "output_bytes", "value": {"U64": 456}},
        {"key": "output_rows", "value": {"U64": 7}},
    ]


def test_memory_lifecycle_events() -> None:
    memory = Memory(
        instance_name="device",
        resource_type_name="memory",
        parent_group_id=uuid.uuid4(),
    )
    assert memory.initializing().to_dict()["data"]["Memory"]["seq"] == 0
    assert memory.operating(1024).to_dict()["data"]["Memory"]["seq"] == 1
    assert memory.finalizing().to_dict()["data"]["Memory"]["seq"] == 2
    assert memory.exit().to_dict()["data"]["Memory"]["seq"] == 3


def test_task_lifecycle_events() -> None:
    operator_id = uuid.uuid4()
    task = Task(operator_id=operator_id, instance_name="task-0")
    queue = task.queueing().to_dict()
    assert queue["data"]["Task"]["state"]["Queueing"]["operator_id"] == str(operator_id)
    assert queue["data"]["Task"]["seq"] == 0
    # ``seq`` is a per-instance counter that increments by one on each
    # transition, in emission order (queueing == 0, allocating == 1, exit == 2).
    assert task.allocating(uuid.uuid4()).to_dict()["data"]["Task"]["seq"] == 1
    assert task.exit().to_dict()["data"]["Task"]["seq"] == 2


def test_port_declare_serialization(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    _, _, ports, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )

    for port in ports:
        event = port.declare(timestamp=42)
        d = event.to_dict()
        assert d["id"] == str(port.id)
        decl = d["data"]["Port"]["Declaration"]
        assert decl["operator_id"] == str(port.operator.id)
        assert decl["instance_name"] == port.instance_name


def test_build_physical_plan(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    query_id = uuid.uuid4()
    logical_plan_id = uuid.uuid4()
    physical_plan_id = uuid.uuid4()

    logical_plan, _, _, _ = build_plan(
        ir, config_options, Query(id=query_id), logical_plan_id, _make_worker()
    )
    assert logical_plan.instance_name == "logical"
    assert logical_plan.parent_plan_id is None

    physical_plan, _, _, _ = build_plan(
        ir,
        config_options,
        Query(id=query_id),
        physical_plan_id,
        _make_worker(),
        instance_name="physical",
        parent_plan=logical_plan,
    )
    assert physical_plan.instance_name == "physical"
    assert physical_plan.parent_plan_id == logical_plan_id
    assert physical_plan.query_id == query_id


def test_physical_plan_declare_serialization(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    query_id = uuid.uuid4()
    logical_plan_id = uuid.uuid4()
    physical_plan_id = uuid.uuid4()

    physical_plan, _, _, _ = build_plan(
        ir,
        config_options,
        Query(id=query_id),
        physical_plan_id,
        _make_worker(),
        instance_name="physical",
        parent_plan=Plan(
            id=logical_plan_id,
            query=Query(id=query_id),
            parent_plan=None,
            instance_name="logical",
            edges=[],
            worker=None,
        ),
    )

    event = physical_plan.declare(timestamp=99999)
    d = event.to_dict()
    decl = d["data"]["Plan"]["Declaration"]
    assert decl["instance_name"] == "physical"
    assert decl["parent"]["query_id"] == str(query_id)
    assert decl["parent"]["plan_id"] == str(logical_plan_id)


def test_build_plan_with_parent_operators(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    """Physical operators reference their logical parent operators."""
    ir, config_options = ir_and_config
    _, logical_ops, _, logical_op_by_id = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )

    parent_operators_by_node_id = {sid: [op] for sid, op in logical_op_by_id.items()}

    _, physical_ops, _, _ = build_plan(
        ir,
        config_options,
        Query(),
        uuid.uuid4(),
        _make_worker(),
        instance_name="physical",
        parent_plan=Plan(
            id=uuid.uuid4(),
            query=Query(),
            parent_plan=None,
            instance_name="logical",
            edges=[],
            worker=None,
        ),
        parent_operators_by_node_id=parent_operators_by_node_id,
    )

    for logical_op, physical_op in zip(logical_ops, physical_ops, strict=True):
        assert len(physical_op.parent_operators) == 1
        assert physical_op.parent_operators[0] is logical_op


def test_build_plan_parent_operators_serialization(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    """parent_operator_ids appear in the serialized Operator declaration."""
    ir, config_options = ir_and_config
    _, logical_ops, _, logical_op_by_id = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )

    parent_operators_by_node_id = {sid: [op] for sid, op in logical_op_by_id.items()}

    _, physical_ops, _, _ = build_plan(
        ir,
        config_options,
        Query(),
        uuid.uuid4(),
        _make_worker(),
        instance_name="physical",
        parent_operators_by_node_id=parent_operators_by_node_id,
    )

    for logical_op, physical_op in zip(logical_ops, physical_ops, strict=True):
        d = physical_op.declare(timestamp=1).to_dict()
        parent_ids = d["data"]["Operator"]["Declaration"]["parent_operator_ids"]
        assert parent_ids == [str(logical_op.id)]


def test_build_plan_without_parent_operators_has_empty_list(
    ir_and_config: tuple[IR, ConfigOptions[StreamingExecutor]],
) -> None:
    ir, config_options = ir_and_config
    _, operators, _, _ = build_plan(
        ir, config_options, Query(), uuid.uuid4(), _make_worker()
    )
    for op in operators:
        assert op.parent_operators == []


def test_port_names_for_node_leaf() -> None:
    from cudf_polars.streaming.explain import SerializableIRNode

    node = SerializableIRNode(
        id="1", children=[], schema={}, properties={}, type="Scan"
    )
    assert port_names_for_node(len(node.children), node.type) == ("out",)


def test_port_names_for_node_single_child() -> None:
    from cudf_polars.streaming.explain import SerializableIRNode

    node = SerializableIRNode(
        id="1", children=["2"], schema={}, properties={}, type="Filter"
    )
    assert port_names_for_node(len(node.children), node.type) == ("out", "in")


def test_port_names_for_node_join() -> None:
    from cudf_polars.streaming.explain import SerializableIRNode

    node = SerializableIRNode(
        id="1", children=["2", "3"], schema={}, properties={}, type="Join"
    )
    assert port_names_for_node(len(node.children), node.type) == (
        "out",
        "left",
        "right",
    )


def test_port_names_for_node_multi_child() -> None:
    from cudf_polars.streaming.explain import SerializableIRNode

    node = SerializableIRNode(
        id="1",
        children=["2", "3", "4"],
        schema={},
        properties={},
        type="Union",
    )

    assert port_names_for_node(len(node.children), node.type) == (
        "out",
        "in_0",
        "in_1",
        "in_2",
    )


def test_lower_ir_graph_with_node_map() -> None:
    from cudf_polars.streaming.parallel import lower_ir_graph_with_node_map
    from cudf_polars.streaming.statistics import collect_statistics

    q = pl.LazyFrame({"x": [1, 2]}).filter(pl.col("x") > 1)
    engine = pl.GPUEngine(executor="streaming")
    config_options = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    stats = collect_statistics(
        ir, config_options, concurrent.futures.ThreadPoolExecutor()
    )

    lowering, node_map = lower_ir_graph_with_node_map(ir, config_options, stats)

    assert lowering.optimized is ir
    assert len(node_map) > 0
    for physical_sid, logical_sids in node_map.items():
        assert isinstance(physical_sid, str)
        assert isinstance(logical_sids, list)
        assert all(isinstance(s, str) for s in logical_sids)


def test_engine_lifecycle() -> None:
    engine_id = uuid.uuid4()
    impl = Implementation()
    engine = Engine(id=engine_id, implementation=impl)

    init_event = engine._init()
    d = init_event.to_dict()
    init = d["data"]["Engine"]["Init"]
    assert init["implementation"]["name"] == "cudf-polars"
    assert init["instance_name"].startswith("cudf-polars-")

    exit_event = engine._exit()
    d = exit_event.to_dict()
    assert d["data"]["Engine"]["Exit"] is None


def test_worker_lifecycle() -> None:
    engine_id = uuid.uuid4()
    worker_id = uuid.uuid4()
    worker = Worker(id=worker_id, engine=Engine(id=engine_id), instance_name="rank-0")

    init_event = worker._init()
    d = init_event.to_dict()
    assert d["id"] == str(worker_id)
    init_data = d["data"]["Worker"]["Init"]
    assert init_data["parent_engine_id"] == str(engine_id)
    assert init_data["instance_name"] == "rank-0"

    exit_event = worker._exit()
    d = exit_event.to_dict()
    assert d["id"] == str(worker_id)
    assert d["data"]["Worker"]["Exit"] is None


def test_query_lifecycle() -> None:
    query_id = uuid.uuid4()
    group_id = uuid.uuid4()
    query = Query(id=query_id)

    init_event = query._init(query_group=cudf_polars.quent.QueryGroup(id=group_id))
    assert init_event.to_dict()["data"]["Query"]["seq"] == 0
    assert query._planning().to_dict()["data"]["Query"]["seq"] == 1
    assert query._executing().to_dict()["data"]["Query"]["seq"] == 2
    assert query._exit().to_dict()["data"]["Query"]["seq"] == 3


@pytest.fixture
def quent_context() -> QuentContext:
    return QuentContext(
        query_group=cudf_polars.quent.QueryGroup(instance_name="test_query_group"),
        query=cudf_polars.quent.Query(instance_name="test_query"),
    )


def test_quent_context_serialization() -> None:
    quent_context = QuentContext(
        query_group=cudf_polars.quent.QueryGroup(instance_name="test_query_group"),
        query=cudf_polars.quent.Query(instance_name="test_query"),
    )
    data = quent_context.serialize()

    new = QuentContext.deserialize(data)
    assert new == quent_context


def test_quent_context_serialization_with_custom_attributes() -> None:
    engine = Engine(
        implementation=Implementation(
            name="test-impl",
            version="1.2.3",
            custom_attributes=[
                Attribute("count", 3),
                Attribute("ratio", 1.5),
                Attribute("name", "demo"),
                Attribute("optional", None),
            ],
        )
    )
    quent_context = QuentContext(
        engine=engine,
        query_group=cudf_polars.quent.QueryGroup(instance_name="test_query_group"),
        query=cudf_polars.quent.Query(instance_name="test_query"),
    )

    data = quent_context.serialize()
    new = QuentContext.deserialize(data)

    assert new == quent_context


def test_emit_query_group_events_idempotent(quent_context: QuentContext):
    pytest.importorskip("structlog")
    logger = cudf_polars.quent._logging.QuentLogger()
    quent_context._emit_query_group_events(logger)
    quent_context._emit_query_group_events(logger)
    assert len(logger._buffer) == 1


def test_processor_registry_declares_once_per_thread() -> None:
    pytest.importorskip("structlog")

    logger = cudf_polars.quent._logging.QuentLogger()
    registry = ProcessorRegistry()
    pool_id = uuid.uuid4()
    thread_ident = 42

    processor_a = registry.get_or_declare_processor(
        logger, thread_ident=thread_ident, pool_id=pool_id
    )
    processor_b = registry.get_or_declare_processor(
        logger, thread_ident=thread_ident, pool_id=pool_id
    )

    assert processor_a is processor_b
    processor_events = [x for x in _drained_events(logger) if "Processor" in x["data"]]
    assert len(processor_events) == 2
    assert processor_events[0]["data"]["Processor"]["state"] == {
        "ProcessorInitializing": {
            "instance_name": f"Thread {processor_a.id.hex[:8]}",
            "parent_group_id": str(pool_id),
            "resource_type_name": "processor",
        }
    }
    assert processor_events[1]["data"]["Processor"]["state"] == {
        "ProcessorOperating": None
    }


def test_processor_registry_concurrent_first_use_declares_once() -> None:
    pytest.importorskip("structlog")

    logger = cudf_polars.quent._logging.QuentLogger()
    registry = ProcessorRegistry()
    pool_id = uuid.uuid4()
    thread_ident = 123

    def get_processor(_: int) -> Processor:
        return registry.get_or_declare_processor(
            logger, thread_ident=thread_ident, pool_id=pool_id
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        processors = list(executor.map(get_processor, range(32)))

    assert len({processor.id for processor in processors}) == 1
    processor_events = [x for x in _drained_events(logger) if "Processor" in x["data"]]
    assert len(processor_events) == 2


def test_processor_registry_reused_across_quent_contexts() -> None:
    pytest.importorskip("structlog")
    logger = cudf_polars.quent._logging.QuentLogger()
    registry = ProcessorRegistry()
    pool_id = uuid.uuid4()
    thread_ident = 99

    context_a = QuentContext()
    context_b = QuentContext()
    local_a = LocalQuentContext(
        context=context_a,
        query=context_a.query_for(uuid.uuid4()),
        worker=Worker(id=uuid.uuid4(), engine=context_a.engine, instance_name="rank-0"),
        logger=logger,
        thread_pool_id=pool_id,
        processor_registry=registry,
        device_memory=Memory(
            instance_name="device",
            resource_type_name="memory",
            parent_group_id=context_a.engine.id,
        ),
    )
    local_b = LocalQuentContext(
        context=context_b,
        query=context_b.query_for(uuid.uuid4()),
        worker=Worker(id=uuid.uuid4(), engine=context_b.engine, instance_name="rank-0"),
        logger=logger,
        thread_pool_id=pool_id,
        processor_registry=registry,
        device_memory=Memory(
            instance_name="device",
            resource_type_name="memory",
            parent_group_id=context_b.engine.id,
        ),
    )

    processor_a = local_a.get_or_declare_processor(thread_ident=thread_ident)
    processor_b = local_b.get_or_declare_processor(thread_ident=thread_ident)

    assert processor_a is processor_b
    processor_events = [x for x in _drained_events(logger) if "Processor" in x["data"]]
    assert len(processor_events) == 2


def test_processor_registry_exit_events() -> None:
    pytest.importorskip("structlog")
    from cudf_polars.quent._context import ProcessorRegistry

    logger = cudf_polars.quent._logging.QuentLogger()
    registry = ProcessorRegistry()
    pool_id = uuid.uuid4()

    registry.get_or_declare_processor(logger, thread_ident=1, pool_id=pool_id)
    registry.get_or_declare_processor(logger, thread_ident=2, pool_id=pool_id)

    registry._emit_processor_exit_events(logger)

    events = _drained_events(logger)
    finalizing_events = [
        x
        for x in events
        if "Processor" in x["data"]
        and x["data"]["Processor"]["state"] == {"ProcessorFinalizing": None}
    ]
    exit_events = [
        x
        for x in events
        if "Processor" in x["data"] and x["data"]["Processor"]["state"] == "Exit"
    ]
    assert len(finalizing_events) == 2
    assert len(exit_events) == 2


def _drained_events(
    logger: cudf_polars.quent._logging.QuentLogger,
) -> list[dict]:
    """Drain Quent logger events into the same shape as engine._quent_events."""
    return [x["event"] for x in logger.drain()]


def test_serialize_list_raises():
    with pytest.raises(NotImplementedError, match="not supported yet"):
        Attribute("list", [1, 2]).serialize()


def test_serialize_dict_raises():
    with pytest.raises(NotImplementedError, match="not supported yet"):
        Attribute("dict", {"a": 1, "b": 2}).serialize()


def test_quent_serialize_none():
    assert Attribute("none", None).serialize() == {
        "key": "none",
        "value": None,
    }


def test_task_from_ir() -> None:
    operator_id = uuid.uuid4()
    _logger, quent_ir_execution_context = _make_quent_ir_execution_context(
        operator_id=operator_id
    )

    task = Task.from_ir(Filter, quent_ir_execution_context)

    assert task is not None
    assert task.operator_id == operator_id
    assert task.instance_name is not None
    assert task.instance_name.startswith("Filter-")
    assert operator_id.hex[:8] in task.instance_name


def test_task_loading_serialization(
    processor: Processor,
    device_memory: Memory,
    disk_to_device_channel: Channel,
) -> None:
    operator_id = uuid.uuid4()
    task = Task(operator_id=operator_id, instance_name="scan-task")

    event = task.loading(
        use_thread=processor,
        use_channel=disk_to_device_channel,
        channel_capacity_bytes=4096,
        use_memory=device_memory,
        memory_capacity_bytes=8192,
        timestamp=100,
    )
    d = event.to_dict()

    assert d["id"] == str(task.id)
    loading = d["data"]["Task"]["state"]["Loading"]
    assert loading["use_thread"] == {
        "resource_id": str(processor.id),
        "capacity": None,
    }
    assert loading["use_fs_to_mem"] == {
        "resource_id": str(disk_to_device_channel.id),
        "capacity": {"capacity_bytes": 4096},
    }
    assert loading["use_memory"] == {
        "resource_id": str(device_memory.id),
        "capacity": {"capacity_bytes": 8192},
    }


def test_task_computing_serialization(
    processor: Processor,
    device_memory: Memory,
) -> None:
    operator_id = uuid.uuid4()
    task = Task(operator_id=operator_id, instance_name="filter-task")

    event = task.computing(
        use_thread=processor,
        use_memory=device_memory,
        memory_capacity_bytes=16384,
        timestamp=101,
    )
    d = event.to_dict()

    computing = d["data"]["Task"]["state"]["Computing"]
    assert computing["use_thread"] == {
        "resource_id": str(processor.id),
        "capacity": None,
    }
    assert computing["use_memory"] == {
        "resource_id": str(device_memory.id),
        "capacity": {"capacity_bytes": 16384},
    }


def test_task_sending_serialization(
    processor: Processor,
    device_memory: Memory,
) -> None:
    operator_id = uuid.uuid4()
    task = Task(operator_id=operator_id, instance_name="shuffle-task")
    link = Channel(
        instance_name="rank-0 -> rank-1",
        resource_type_name="Link",
        parent_group_id=uuid.uuid4(),
        source=device_memory,
        target=device_memory,
    )

    event = task.sending(
        use_thread=processor,
        use_link=link,
        link_capacity_bytes=2048,
        timestamp=102,
    )
    d = event.to_dict()

    sending = d["data"]["Task"]["state"]["Sending"]
    assert sending["use_thread"] == {
        "resource_id": str(processor.id),
        "capacity": None,
    }
    assert sending["use_link"] == {
        "resource_id": str(link.id),
        "capacity": {"capacity_bytes": 2048},
    }


def test_network_declare_serialization() -> None:
    engine_id = uuid.uuid4()
    network = Network(engine_id=engine_id)

    event = network.declare(timestamp=555)
    d = event.to_dict()

    assert d["id"] == str(network.id)
    assert d["timestamp"] == 555
    assert d["data"]["Network"]["Declaration"] == {
        "instance_name": "Network",
        "parent_group_id": str(engine_id),
    }


def test_declare_network_channels_single_rank(device_memory: Memory) -> None:
    pytest.importorskip("structlog")
    logger = cudf_polars.quent._logging.QuentLogger()

    network, link_channels = declare_network_channels(
        logger,
        rank=0,
        nranks=1,
        engine_id=uuid.uuid4(),
        device_memory=device_memory,
    )

    assert network is None
    assert link_channels == {}
    assert _drained_events(logger) == []


@pytest.mark.parametrize(
    "rank,nranks,expected_targets", [(0, 3, [1, 2]), (1, 3, [0, 2])]
)
def test_declare_network_channels_multi_rank(
    device_memory: Memory,
    rank: int,
    nranks: int,
    expected_targets: list[int],
) -> None:
    pytest.importorskip("structlog")
    logger = cudf_polars.quent._logging.QuentLogger()
    engine_id = uuid.uuid4()

    network, link_channels = declare_network_channels(
        logger,
        rank=rank,
        nranks=nranks,
        engine_id=engine_id,
        device_memory=device_memory,
    )

    assert network is not None
    assert set(link_channels) == set(expected_targets)
    for target_rank, link in link_channels.items():
        assert link.instance_name == f"rank-{rank} -> rank-{target_rank}"
        assert link.resource_type_name == "Link"
        assert link.parent_group_id == network.id
        assert link.source is device_memory
        assert link.target is device_memory

    events = _drained_events(logger)
    network_events = [event for event in events if "Network" in event["data"]]
    channel_events = [event for event in events if "Channel" in event["data"]]
    assert len(network_events) == 1
    assert network_events[0]["data"]["Network"]["Declaration"][
        "parent_group_id"
    ] == str(engine_id)
    assert len(channel_events) == len(expected_targets) * 2


def test_finalize_network_channels(device_memory: Memory) -> None:
    pytest.importorskip("structlog")
    logger = cudf_polars.quent._logging.QuentLogger()
    link_channels = {
        target_rank: Channel(
            instance_name=f"rank-0 -> rank-{target_rank}",
            resource_type_name="Link",
            parent_group_id=uuid.uuid4(),
            source=device_memory,
            target=device_memory,
        )
        for target_rank in (1, 2)
    }

    finalize_network_channels(logger, link_channels=link_channels)

    events = _drained_events(logger)
    finalizing_events = [
        event
        for event in events
        if event["data"]["Channel"]["state"] == {"ChannelFinalizing": None}
    ]
    exit_events = [
        event for event in events if event["data"]["Channel"]["state"] == "Exit"
    ]
    assert len(finalizing_events) == 2
    assert len(exit_events) == 2


def test_emit_task_events_computing_node() -> None:
    logger, quent_ir_execution_context = _make_quent_ir_execution_context()
    task = Task.from_ir(Filter, quent_ir_execution_context)
    assert task is not None

    quent_ir_execution_context.context._emit_task_begin_events(
        Filter,
        task,
        quent_ir_execution_context,
        input_frames_bytes=0,
    )

    # Simulate the result
    result = _make_dataframe(pl.DataFrame({"y": list(range(7))}))

    quent_ir_execution_context.context._emit_task_end_events(
        Filter,
        task,
        quent_ir_execution_context,
        [],
        result,
    )

    events = _drained_events(logger)
    task_events = [event for event in events if "Task" in event["data"]]
    # queueing -> allocating -> computing -> exit
    assert [event["data"]["Task"]["seq"] for event in task_events] == [0, 1, 2, 3, 4]
    assert "Queueing" in task_events[0]["data"]["Task"]["state"]
    assert "Allocating" in task_events[1]["data"]["Task"]["state"]
    assert "Computing" in task_events[2]["data"]["Task"]["state"]
    assert "Exit" in task_events[3]["data"]["Task"]["state"]
    assert "Statistics" in task_events[4]["data"]["Task"]["state"]
    processor_events = [event for event in events if "Processor" in event["data"]]
    assert len(processor_events) == 2


def test_emit_task_events_io_node(disk_to_device_channel: Channel) -> None:
    logger, quent_ir_execution_context = _make_quent_ir_execution_context(
        disk_to_device_channel=disk_to_device_channel
    )
    task = Task.from_ir(DataFrameScan, quent_ir_execution_context)
    assert task is not None

    quent_ir_execution_context.context._emit_task_begin_events(
        DataFrameScan,
        task,
        quent_ir_execution_context,
        input_frames_bytes=0,
    )

    # Simulate the result
    result = _make_dataframe(pl.DataFrame({"y": list(range(7))}))
    quent_ir_execution_context.context._emit_task_end_events(
        DataFrameScan,
        task,
        quent_ir_execution_context,
        [],
        result,
    )

    events = _drained_events(logger)
    # queueing -> allocating -> loading -> computing -> exit
    task_events = [event for event in events if "Task" in event["data"]]
    assert [event["data"]["Task"]["seq"] for event in task_events] == [0, 1, 2, 3, 4, 5]
    assert "Queueing" in task_events[0]["data"]["Task"]["state"]
    assert "Allocating" in task_events[1]["data"]["Task"]["state"]
    assert "Loading" in task_events[2]["data"]["Task"]["state"]
    assert "Computing" in task_events[3]["data"]["Task"]["state"]
    assert "Exit" in task_events[4]["data"]["Task"]["state"]
    assert "Statistics" in task_events[5]["data"]["Task"]["state"]
