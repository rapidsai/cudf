# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Datatype utilities."""

from __future__ import annotations

import pyarrow as pa

import pylibcudf as plc
from pylibcudf.traits import (
    is_floating_point,
    is_integral_not_bool,
    is_numeric_not_bool,
)

__all__ = [
    "can_cast",
    "downcast_arrow_lists",
    "is_order_preserving_cast",
]


def downcast_arrow_lists(typ: pa.DataType) -> pa.DataType:
    """
    Sanitize an arrow datatype from polars.

    Parameters
    ----------
    typ
        Arrow type to sanitize

    Returns
    -------
    Sanitized arrow type

    Notes
    -----
    As well as arrow ``ListType``s, polars can produce
    ``LargeListType``s and ``FixedSizeListType``s, these are not
    currently handled by libcudf, so we attempt to cast them all into
    normal ``ListType``s on the arrow side before consuming the arrow
    data.
    """
    if isinstance(typ, pa.LargeListType):
        return pa.list_(downcast_arrow_lists(typ.value_type))
    # We don't have to worry about diving into struct types for now
    # since those are always NotImplemented before we get here.
    assert not isinstance(typ, pa.StructType)
    return typ


def can_cast(from_: plc.DataType, to: plc.DataType) -> bool:
    """
    Can we cast (via :func:`~.pylibcudf.unary.cast`) between two datatypes.

    Parameters
    ----------
    from_
        Source datatype
    to
        Target datatype

    Returns
    -------
    True if casting is supported, False otherwise
    """
    to_is_empty = to.id() == plc.TypeId.EMPTY
    from_is_empty = from_.id() == plc.TypeId.EMPTY
    has_empty = to_is_empty or from_is_empty
    return (
        (
            from_ == to
            or (
                not has_empty
                and (
                    plc.traits.is_fixed_width(to)
                    and plc.traits.is_fixed_width(from_)
                    and plc.unary.is_supported_cast(from_, to)
                )
            )
        )
        or (
            from_.id() == plc.TypeId.STRING
            and not to_is_empty
            and is_numeric_not_bool(to)
        )
        or (
            to.id() == plc.TypeId.STRING
            and not from_is_empty
            and is_numeric_not_bool(from_)
        )
    )


def is_order_preserving_cast(from_: plc.DataType, to: plc.DataType) -> bool:
    """
    Determine if a cast would preserve the order of the source data.

    Parameters
    ----------
    from_
        Source datatype
    to
        Target datatype

    Returns
    -------
    True if the cast is order-preserving, False otherwise
    """
    if from_.id() == to.id():
        return True

    if is_integral_not_bool(from_) and is_integral_not_bool(to):
        # True if signedness is the same and the target is larger
        if plc.traits.is_unsigned(from_) == plc.traits.is_unsigned(to):
            if plc.types.size_of(to) >= plc.types.size_of(from_):
                return True
        elif (plc.traits.is_unsigned(from_) and not plc.traits.is_unsigned(to)) and (
            plc.types.size_of(to) > plc.types.size_of(from_)
        ):
            # Unsigned to signed is order preserving if target is large enough
            # But signed to unsigned is never order preserving due to negative values
            return True
    elif (
        is_floating_point(from_)
        and is_floating_point(to)
        and (plc.types.size_of(to) >= plc.types.size_of(from_))
    ):
        # True if the target is larger
        return True
    return (is_integral_not_bool(from_) and is_floating_point(to)) or (
        is_floating_point(from_) and is_integral_not_bool(to)
    )
