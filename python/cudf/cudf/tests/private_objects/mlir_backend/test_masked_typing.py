# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

from cudf.core.udf.mlir_backend.masked_typing import (
    _SUPPORTED_MASKED_VALUE_TYPE_CLASSES,
    MaskedType,
    _supported_value_type_instances,
)


def test_masked_type_repr():
    assert repr(MaskedType(types.int64)) == "Masked(int64)"
    assert repr(MaskedType(types.float64)) == "Masked(float64)"
    assert repr(MaskedType(types.boolean)) == "Masked(bool)"


def test_masked_type_equality_by_value_type():
    """Two MaskedType instances are equal iff their value_type is equal."""
    a = MaskedType(types.int64)
    b = MaskedType(types.int64)
    c = MaskedType(types.float64)
    # Numba ``Type`` equality is structural: same name -> equal.
    assert a == b
    assert a != c


def test_masked_type_hash_keys_by_value_type():
    """Numba caches typed signatures by hash; matching params must hash equal."""
    assert hash(MaskedType(types.int64)) == hash(MaskedType(types.int64))
    assert hash(MaskedType(types.int64)) != hash(MaskedType(types.float64))


def test_masked_type_unliterals_value():
    """A literal value type is unliteralled before being stored."""
    lit = types.IntegerLiteral(7)
    masked = MaskedType(lit)
    # The stored value_type is the underlying int64 (or whatever the
    # literal unwraps to), not the literal itself.
    assert not isinstance(masked.value_type, types.Literal)


def test_masked_type_unsupported_value_becomes_poison():
    """Unsupported value types are wrapped in Poison (not raised here).

    The typing pass may try to construct ``MaskedType`` for unsupported
    column dtypes; we want that to succeed at construction so a
    descriptive error can surface later when the user actually performs
    an op on the value. The poison sentinel is the carrier.
    """
    masked = MaskedType(types.unicode_type)
    assert isinstance(masked.value_type, types.Poison)


def test_supported_value_type_classes_are_numeric_and_boolean_only():
    """At this PR layer, value types are restricted to numeric + boolean."""
    assert _SUPPORTED_MASKED_VALUE_TYPE_CLASSES == (
        types.Number,
        types.Boolean,
    )


def test_supported_value_type_instances_includes_typical_numerics():
    """Spot-check that the instance set covers common scalar types."""
    expected = {
        types.int8,
        types.int16,
        types.int32,
        types.int64,
        types.uint8,
        types.uint16,
        types.uint32,
        types.uint64,
        types.float32,
        types.float64,
        types.boolean,
    }
    missing = expected - _supported_value_type_instances
    assert not missing, f"missing instance entries: {missing}"


def test_supported_value_type_instances_excludes_strings():
    """String-typed values aren't supported at this layer."""
    assert types.unicode_type not in _supported_value_type_instances
