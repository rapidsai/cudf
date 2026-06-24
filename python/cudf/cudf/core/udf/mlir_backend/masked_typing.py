# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from numba_cuda_mlir import types
from numba_cuda_mlir.extending import typing_registry
from numba_cuda_mlir.numba_cuda import types as nb_types
from numba_cuda_mlir.numba_cuda.types.misc import unliteral
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
)
from numba_cuda_mlir.typing import signature as nb_signature

from cudf.core.udf.api import Masked

_SUPPORTED_MASKED_VALUE_TYPE_CLASSES = (
    types.Number,
    types.Boolean,
)


_supported_value_type_instances = (
    nb_types.integer_domain | nb_types.real_domain | {nb_types.boolean}
)


class MaskedType(types.Type):
    """Logical struct type used for propagation of nulls. Semantically carries
    the column value and corresponding validity bit from the columns bitmask.
    Operations over this type such as arithmetic are implemented to be sensitive
    to the nullity of the output value.

    Instances are parameterized by value type mapping to the type of the source
    column.
    """

    def __init__(self, value):
        if isinstance(value, types.Literal):
            value = unliteral(value)
        if isinstance(value, _SUPPORTED_MASKED_VALUE_TYPE_CLASSES):
            self.value_type = value
        else:
            self.value_type = types.Poison(value)
        super().__init__(name=f"Masked({self.value_type})")

    def __hash__(self):
        # Two MaskedType instances compare equal (and hash equal) iff their
        # parameter ``value_type`` matches, so numba can cache them by repr.
        return hash(repr(self))


# ``Masked(value, valid)`` constructor: produces a ``Masked(value_ty)``.
class MaskedConstructor(ConcreteTemplate):
    key = Masked
    cases = [
        nb_signature(MaskedType(t), t, types.boolean)
        for t in _supported_value_type_instances
    ]


# ``m.value`` -> inner value type; ``m.valid`` -> boolean. ``AttributeTemplate``
# dispatches ``resolve_<attr>`` methods automatically.
class MaskedTypeAttrs(AttributeTemplate):
    key = MaskedType

    def resolve_value(self, typ):
        return typ.value_type

    def resolve_valid(self, typ):
        return types.boolean


def _register():
    """Register typing for ``Masked`` and ``MaskedType`` attributes with
    ``numba_cuda_mlir``. Called once at module import.
    """
    typing_registry.register_global(Masked, types.Function(MaskedConstructor))
    typing_registry.register_attr(MaskedTypeAttrs)


_register()
