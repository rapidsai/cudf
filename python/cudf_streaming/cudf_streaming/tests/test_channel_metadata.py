# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming metadata types (Partitioning and ChannelMetadata)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc
import pytest

from cudf_streaming import (
    ChannelMetadata,
    HashScheme,
    OrderKey,
    Partitioning,
    PartitionOrdering,
    RangeScheme,
    RowOrdering,
    RowOrderingScope,
    TableChunk,
)
from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.streaming.core.message import Message

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.context import Context


def _make_boundaries(context: Context, table: plc.Table) -> TableChunk:
    stream = context.br().stream_pool.get_stream()
    return TableChunk.from_pylibcudf_table(
        table,
        stream,
        exclusive_view=False,
        br=context.br(),
    )


def _column_from_values(
    values: list[int | str | None],
    dtype: plc.DataType,
    non_null_value: int | str,
) -> plc.Column:
    if values and all(value is None for value in values):
        return plc.Column.all_null_like(
            plc.Column.from_iterable_of_py([non_null_value], dtype),
            len(values),
        )
    return plc.Column.from_iterable_of_py(values, dtype)


def _two_key_partition_ordering(
    context: Context,
    int_values: list[int | None],
    string_values: list[str | None],
    *,
    strict_boundaries: bool = False,
) -> PartitionOrdering:
    """Return a two-key partition ordering with mixed boundary columns."""
    return PartitionOrdering(
        [
            OrderKey(
                0,
                plc.types.Order.ASCENDING,
                plc.types.NullOrder.BEFORE,
            ),
            OrderKey(
                1,
                plc.types.Order.DESCENDING,
                plc.types.NullOrder.AFTER,
            ),
        ],
        _make_boundaries(
            context,
            plc.Table(
                [
                    _column_from_values(
                        int_values, plc.DataType(plc.TypeId.INT64), 0
                    ),
                    _column_from_values(
                        string_values, plc.DataType(plc.TypeId.STRING), "x"
                    ),
                ]
            ),
        ),
        strict_boundaries=strict_boundaries,
    )


def _two_key_range_scheme(
    context: Context,
    *,
    strict_boundaries: bool = False,
) -> RangeScheme:
    """Return a two-key range scheme with one boundary row."""
    return RangeScheme(
        [
            _two_key_partition_ordering(
                context,
                [100],
                ["abc"],
                strict_boundaries=strict_boundaries,
            )
        ]
    )


def test_hash_scheme() -> None:
    """Test HashScheme construction, properties, equality, and repr."""
    h1 = HashScheme((0, 1), 16)
    assert h1.column_indices == (0, 1)
    assert h1.modulus == 16
    assert repr(h1) == "HashScheme((0, 1), 16)"

    # Equality
    assert h1 == HashScheme((0, 1), 16)
    assert h1 != HashScheme((0, 1), 32)
    assert h1 != HashScheme((2,), 16)

    with pytest.raises(ValueError):
        HashScheme((), 16)
    with pytest.raises(ValueError):
        HashScheme((0, 0), 16)
    with pytest.raises(ValueError):
        HashScheme((-1,), 16)
    with pytest.raises(ValueError):
        HashScheme((0,), 0)


def test_order_key() -> None:
    k = OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)
    assert k.column_index == 0
    assert k.order == plc.types.Order.ASCENDING
    assert k.null_order == plc.types.NullOrder.BEFORE
    assert k == OrderKey(
        0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE
    )
    assert k != OrderKey(
        1, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE
    )
    assert "OrderKey" in repr(k)


def test_row_ordering() -> None:
    """Test row-ordering properties, equality, and repr."""
    time = OrderKey(2, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)
    global_ordering = RowOrdering([time], RowOrderingScope.GLOBAL)
    grouped = RowOrdering(
        [time], RowOrderingScope.RANK_LOCAL, group_keys=(0, 1)
    )

    assert global_ordering.order_keys == (time,)
    assert global_ordering.scope == RowOrderingScope.GLOBAL
    assert global_ordering.group_keys == ()
    assert grouped.scope == RowOrderingScope.RANK_LOCAL
    assert grouped.group_keys == (0, 1)
    assert grouped == RowOrdering(
        [time], RowOrderingScope.RANK_LOCAL, group_keys=(0, 1)
    )
    assert grouped == RowOrdering(
        [time], RowOrderingScope.RANK_LOCAL, group_keys=(1, 0)
    )
    assert grouped != global_ordering
    assert grouped != object()
    assert "RowOrdering" in repr(grouped)


@pytest.mark.parametrize(
    ("order_indices", "group_indices", "match"),
    [
        ((), (), "must not be empty"),
        ((-1,), (), "non-negative"),
        ((0, 0), (), "unique"),
        ((1,), (-1,), "non-negative"),
        ((1,), (0, 0), "unique"),
        ((0,), (0,), "unique"),
    ],
)
def test_row_ordering_validation(
    order_indices: tuple[int, ...],
    group_indices: tuple[int, ...],
    match: str,
) -> None:
    """Reject invalid row-ordering keys."""
    order_keys = tuple(
        OrderKey(
            index,
            plc.types.Order.ASCENDING,
            plc.types.NullOrder.BEFORE,
        )
        for index in order_indices
    )
    with pytest.raises(ValueError, match=match):
        RowOrdering(
            order_keys,
            RowOrderingScope.GLOBAL,
            group_keys=group_indices,
        )


def test_range_scheme(context: Context) -> None:
    """Test RangeScheme construction, properties, and repr."""
    o1 = _two_key_range_scheme(context)
    partition_ordering = o1.partition_orderings[0]
    assert partition_ordering.keys == (
        OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(1, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    )
    assert partition_ordering.column_indices == (0, 1)
    assert not partition_ordering.strict_boundaries
    assert partition_ordering.num_boundaries == 1
    assert "PartitionOrdering" in repr(partition_ordering)
    assert "RangeScheme" in repr(o1)

    assert o1.partition_orderings[0].boundaries_aligned_with(
        _two_key_range_scheme(context).partition_orderings[0], context.br()
    )

    o_strict = _two_key_range_scheme(context, strict_boundaries=True)
    assert o_strict.partition_orderings[0].strict_boundaries
    assert not o1.partition_orderings[0].boundaries_aligned_with(
        o_strict.partition_orderings[0], context.br()
    )
    assert o_strict.partition_orderings[0].boundaries_aligned_with(
        _two_key_range_scheme(
            context, strict_boundaries=True
        ).partition_orderings[0],
        context.br(),
    )

    with pytest.raises(TypeError, match="OrderKey"):
        RangeScheme(
            [
                PartitionOrdering(
                    [
                        (
                            0,
                            plc.types.Order.ASCENDING,
                            plc.types.NullOrder.BEFORE,
                        )
                    ],  # ty: ignore[invalid-argument-type]
                    _make_boundaries(
                        context,
                        plc.Table(
                            [
                                plc.Column.from_iterable_of_py(
                                    [0], plc.DataType(plc.TypeId.INT64)
                                )
                            ]
                        ),
                    ),
                )
            ],  # ty: ignore[invalid-argument-type]
        )

    with pytest.raises(ValueError, match="empty"):
        RangeScheme([])


def test_range_scheme_multiple_orderings(context: Context) -> None:
    """RangeScheme stores partition orderings valid for the same stream."""
    first = PartitionOrdering(
        [OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)],
        _make_boundaries(
            context,
            plc.Table(
                [
                    plc.Column.from_iterable_of_py(
                        [100], plc.DataType(plc.TypeId.INT64)
                    )
                ]
            ),
        ),
        strict_boundaries=True,
    )
    second = PartitionOrdering(
        [OrderKey(2, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER)],
        _make_boundaries(
            context,
            plc.Table(
                [
                    plc.Column.from_iterable_of_py(
                        [200], plc.DataType(plc.TypeId.INT64)
                    )
                ]
            ),
        ),
    )
    scheme = RangeScheme([first, second])

    assert len(scheme.partition_orderings) == 2
    assert scheme.partition_orderings[0].keys == first.keys
    assert (
        scheme.partition_orderings[0].strict_boundaries
        == first.strict_boundaries
    )
    assert scheme.partition_orderings[0].num_boundaries == first.num_boundaries
    assert scheme.partition_orderings[1].keys == second.keys


def test_range_scheme_get_boundaries(context: Context) -> None:
    scheme = _two_key_range_scheme(context)
    partition_ordering = scheme.partition_orderings[0]
    chunk = partition_ordering.get_boundaries(context.br())
    assert chunk.table_view().num_columns() == 2
    assert chunk.table_view().num_rows() == 1
    scheme2 = RangeScheme(
        [
            PartitionOrdering(
                partition_ordering.keys,
                chunk,
                strict_boundaries=partition_ordering.strict_boundaries,
            )
        ]
    )
    assert scheme2.partition_orderings[0].boundaries_aligned_with(
        scheme.partition_orderings[0], context.br()
    )


def test_partition_ordering_with_keys(context: Context) -> None:
    """with_keys shares boundaries and updates column indices."""
    o1 = _two_key_range_scheme(context)
    partition_ordering = o1.partition_orderings[0]
    new_keys = [
        OrderKey(5, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(3, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    ]
    ordering2 = partition_ordering.with_keys(new_keys)
    assert ordering2.keys[0].column_index == 5
    assert ordering2.keys[1].column_index == 3
    assert ordering2.num_boundaries == partition_ordering.num_boundaries
    assert ordering2.strict_boundaries == partition_ordering.strict_boundaries
    # Schemes with different key indices but shared boundaries are boundary-aligned
    assert partition_ordering.boundaries_aligned_with(ordering2, context.br())

    with pytest.raises(ValueError, match="preserve ordering semantics"):
        partition_ordering.with_keys(
            [
                OrderKey(
                    5, plc.types.Order.DESCENDING, plc.types.NullOrder.BEFORE
                ),
                new_keys[1],
            ]
        )


@pytest.mark.parametrize(
    "int_values,string_values",
    [
        ([100], ["abc"]),
        ([], []),
        ([None], [None]),
    ],
)
def test_partition_ordering_with_strict_boundaries(
    context: Context,
    int_values: list[int | None],
    string_values: list[str | None],
) -> None:
    """with_strict_boundaries shares boundaries and marks them strict."""
    partition_ordering = _two_key_partition_ordering(
        context, int_values, string_values
    )
    strict_ordering = partition_ordering.with_strict_boundaries()
    strict_ordering2 = _two_key_partition_ordering(
        context,
        int_values,
        string_values,
        strict_boundaries=True,
    )
    assert strict_ordering.keys == partition_ordering.keys
    assert strict_ordering.num_boundaries == partition_ordering.num_boundaries
    assert strict_ordering.strict_boundaries
    assert not partition_ordering.boundaries_aligned_with(
        strict_ordering, context.br()
    )
    assert strict_ordering.boundaries_aligned_with(
        strict_ordering2, context.br()
    )


def test_partition_ordering_boundaries_aligned_with(context: Context) -> None:
    """Boundary comparison ignores key indices but checks values and ordering."""
    df = plc.Table(
        [
            plc.Column.from_iterable_of_py(
                [100, 200], plc.DataType(plc.TypeId.INT64)
            ),
            plc.Column.from_iterable_of_py(
                ["abc", "xyz"], plc.DataType(plc.TypeId.STRING)
            ),
        ]
    )
    keys = [
        OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(1, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    ]
    o1 = PartitionOrdering(keys, _make_boundaries(context, df))
    o2 = PartitionOrdering(keys, _make_boundaries(context, df))
    assert o1.boundaries_aligned_with(o2, context.br())

    # Different key column indices but same boundary values → still aligned
    shifted_keys = [
        OrderKey(2, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(3, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    ]
    o_shifted = PartitionOrdering(shifted_keys, _make_boundaries(context, df))
    assert o1.boundaries_aligned_with(o_shifted, context.br())

    # Different boundary values → not aligned (shape matches, values differ)
    df_diff = plc.Table(
        [
            plc.Column.from_iterable_of_py(
                [100, 300], plc.DataType(plc.TypeId.INT64)
            ),
            plc.Column.from_iterable_of_py(
                ["abc", "xyz"], plc.DataType(plc.TypeId.STRING)
            ),
        ]
    )
    o3 = PartitionOrdering(keys, _make_boundaries(context, df_diff))
    assert not o1.boundaries_aligned_with(o3, context.br())

    # Different strict_boundaries → not aligned
    o_strict = PartitionOrdering(
        keys,
        _make_boundaries(context, df),
        strict_boundaries=True,
    )
    assert not o1.boundaries_aligned_with(o_strict, context.br())


def test_range_scheme_key_column_mismatch(context: Context) -> None:
    """RangeScheme rejects key/column count mismatch."""
    boundaries = _make_boundaries(
        context,
        plc.Table(
            [
                plc.Column.from_iterable_of_py(
                    [0], plc.DataType(plc.TypeId.INT64)
                )
            ]
        ),
    )
    with pytest.raises(ValueError, match="keys must match"):
        RangeScheme(
            [
                PartitionOrdering(
                    [
                        OrderKey(
                            0,
                            plc.types.Order.ASCENDING,
                            plc.types.NullOrder.BEFORE,
                        ),
                        OrderKey(
                            1,
                            plc.types.Order.DESCENDING,
                            plc.types.NullOrder.AFTER,
                        ),
                    ],
                    boundaries,  # 1 column, but 2 keys
                )
            ]
        )


def test_partitioning_scenarios(context: Context) -> None:
    """Test various partitioning configurations."""
    # Default / None
    p_default = Partitioning()
    assert p_default.inter_rank is None
    assert p_default.local is None
    assert Partitioning(None, None).inter_rank is None
    assert Partitioning(None, None).local is None

    # Direct global shuffle: inter_rank=Hash, local=Aligned
    p_global = Partitioning(HashScheme((0,), 16), "inherit")
    assert p_global.inter_rank == HashScheme((0,), 16)
    assert p_global.local == "inherit"

    # Two-stage shuffle: inter_rank=Hash(nranks), local=Hash(N_l)
    p_twostage = Partitioning(HashScheme((0,), 4), HashScheme((0,), 8))
    assert p_twostage.inter_rank == HashScheme((0,), 4)
    assert p_twostage.local == HashScheme((0,), 8)

    # Order-based partitioning (range partitioned / sorted)
    range_scheme = _two_key_range_scheme(context)
    p_ordered = Partitioning(range_scheme, "inherit")
    assert isinstance(p_ordered.inter_rank, RangeScheme)
    assert p_ordered.inter_rank.partition_orderings[0].boundaries_aligned_with(
        range_scheme.partition_orderings[0], context.br()
    )
    assert p_ordered.local == "inherit"

    # Mixed: inter_rank=Order, local=Hash
    p_mixed = Partitioning(
        _two_key_range_scheme(context),
        HashScheme((1,), 8),
    )
    assert isinstance(p_mixed.inter_rank, RangeScheme)
    assert isinstance(p_mixed.local, HashScheme)

    # Repr
    assert "Partitioning" in repr(p_global)
    assert "inter_rank" in repr(p_global)

    # Invalid type
    with pytest.raises(TypeError):
        Partitioning("invalid", None)  # ty: ignore[invalid-argument-type]


def test_channel_metadata() -> None:
    """Test ChannelMetadata construction and properties."""
    # Basic construction
    m = ChannelMetadata(local_count=4)
    assert m.local_count == 4
    assert m.row_orderings == ()
    assert not m.duplicated

    # With partitioning and duplicated
    p = Partitioning(HashScheme((0,), 16), "inherit")
    m_full = ChannelMetadata(local_count=4, partitioning=p, duplicated=True)
    assert m_full.partitioning.inter_rank == HashScheme((0,), 16)
    assert m_full.partitioning.local == "inherit"
    assert m_full.duplicated

    key = OrderKey(1, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)
    row_ordering = RowOrdering(
        [key], RowOrderingScope.RANK_LOCAL, group_keys=(0,)
    )
    m_ordered = ChannelMetadata(local_count=4, row_orderings=(row_ordering,))
    assert m_ordered.row_orderings == (row_ordering,)
    assert "row_orderings" in repr(m_ordered)

    # Field comparisons (ChannelMetadata.__eq__ removed)
    m2 = ChannelMetadata(local_count=4)
    assert m.local_count == m2.local_count
    assert m.duplicated == m2.duplicated
    assert ChannelMetadata(local_count=8).local_count != m.local_count
    assert "local_count=4" in repr(m)

    # Validation
    with pytest.raises(ValueError, match="local_count must be non-negative"):
        ChannelMetadata(local_count=-1)


def test_message_roundtrip(context: Context) -> None:
    """Test ChannelMetadata can be copied and moved through Message."""
    key = OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE)
    row_ordering = RowOrdering([key], RowOrderingScope.GLOBAL)
    m = ChannelMetadata(
        local_count=4,
        partitioning=Partitioning(HashScheme((0,), 16), "inherit"),
        row_orderings=(row_ordering,),
        duplicated=True,
    )
    msg_m = Message(99, m)
    assert msg_m.sequence_number == 99

    assert msg_m.copy_cost() == 0
    reservation, _ = context.br().reserve(
        MemoryType.DEVICE, 0, allow_overbooking=True
    )
    msg_copy = msg_m.copy(reservation)
    copied_m = ChannelMetadata.from_message(msg_copy)
    assert copied_m.local_count == 4
    assert copied_m.duplicated
    assert copied_m.partitioning.inter_rank == HashScheme((0,), 16)
    assert copied_m.row_orderings == (row_ordering,)
    assert not msg_m.empty()

    got_m = ChannelMetadata.from_message(msg_m)
    assert got_m.local_count == 4
    assert got_m.duplicated
    assert got_m.partitioning.inter_rank == HashScheme((0,), 16)
    assert got_m.row_orderings == (row_ordering,)
    assert msg_m.empty()


def test_message_roundtrip_with_range_scheme(context: Context) -> None:
    """Test ChannelMetadata with RangeScheme can round-trip through Message."""
    table = plc.Table(
        [
            plc.Column.from_iterable_of_py(
                [100, 200], plc.DataType(plc.TypeId.INT64)
            ),
            plc.Column.from_iterable_of_py(
                ["abc", "xyz"], plc.DataType(plc.TypeId.STRING)
            ),
        ]
    )
    boundaries = _make_boundaries(context, table)
    keys = (
        OrderKey(0, plc.types.Order.ASCENDING, plc.types.NullOrder.BEFORE),
        OrderKey(1, plc.types.Order.DESCENDING, plc.types.NullOrder.AFTER),
    )
    range_scheme = RangeScheme(
        [
            PartitionOrdering(
                keys,
                boundaries,
                strict_boundaries=True,
            )
        ]
    )
    m = ChannelMetadata(
        local_count=8,
        partitioning=Partitioning(range_scheme, "inherit"),
        row_orderings=(RowOrdering(keys, RowOrderingScope.GLOBAL),),
        duplicated=True,
    )
    msg_m = Message(42, m)
    assert msg_m.sequence_number == 42
    got_m = ChannelMetadata.from_message(msg_m)
    assert got_m.local_count == 8
    assert got_m.duplicated
    assert isinstance(got_m.partitioning.inter_rank, RangeScheme)
    partition_ordering = got_m.partitioning.inter_rank.partition_orderings[0]
    assert partition_ordering.keys == keys
    assert got_m.partitioning.local == "inherit"
    assert partition_ordering.strict_boundaries
    assert partition_ordering.num_boundaries == 2
    assert got_m.row_orderings == (RowOrdering(keys, RowOrderingScope.GLOBAL),)
    assert got_m.partitioning.inter_rank.partition_orderings[
        0
    ].boundaries_aligned_with(
        range_scheme.partition_orderings[0], context.br()
    )
    assert msg_m.empty()


def test_range_scheme_roundtrip_from_metadata(context: Context) -> None:
    """A range scheme read from metadata can be used in new partitioning."""
    src = ChannelMetadata(
        local_count=1,
        partitioning=Partitioning(_two_key_range_scheme(context), "inherit"),
    )
    scheme = src.partitioning.inter_rank
    assert isinstance(scheme, RangeScheme)

    p2 = Partitioning(scheme, None)
    assert isinstance(p2.inter_rank, RangeScheme)
    assert p2.inter_rank.partition_orderings[0].boundaries_aligned_with(
        _two_key_range_scheme(context).partition_orderings[0], context.br()
    )


def test_access_after_move_raises() -> None:
    """Test that accessing a released ChannelMetadata raises ValueError."""
    m = ChannelMetadata(
        local_count=4,
        partitioning=Partitioning(HashScheme((0,), 16), "inherit"),
    )
    # Move into a message (releases the handle)
    _ = Message(0, m)

    # Accessing any property should raise ValueError
    with pytest.raises(ValueError, match="uninitialized"):
        _ = m.local_count

    with pytest.raises(ValueError, match="uninitialized"):
        _ = m.partitioning

    with pytest.raises(ValueError, match="uninitialized"):
        _ = m.row_orderings

    with pytest.raises(ValueError, match="uninitialized"):
        _ = m.duplicated

    with pytest.raises(ValueError, match="uninitialized"):
        repr(m)
