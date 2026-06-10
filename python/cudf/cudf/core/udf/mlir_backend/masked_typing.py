# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
MLIR (numba_cuda_mlir) typing for ``MaskedType`` -- the foundational
"value + validity bit" wrapper used by every UDF input/return value.

Scope of this PR (intentionally minimal):

* the ``MaskedType`` class itself, parameterized by a value type;
* numeric value types only (``int8..int64``, ``uint8..uint64``,
  ``float32``, ``float64``, ``boolean``) -- string / datetime / timedelta
  cases are layered in later PRs;
* the ``Masked(value, valid)`` constructor;
* the ``.value`` and ``.valid`` attributes.

Out of scope (deferred to subsequent PRs in the stack):

* ``pack_return`` (the scalar-vs-Masked return bridge for UDFs);
* ``NAType`` and ``Masked + NA`` semantics;
* binary / unary / comparison / bitwise op typing;
* string- and datetime-flavored value types;
* casts between Masked values of different inner types.

Importing this module registers typing with ``numba_cuda_mlir``.
"""

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

# Numba type classes that may serve as the inner value of a ``MaskedType``
# at this layer. Subsequent PRs extend this tuple as new value types
# (string, datetime, timedelta) come online.
_SUPPORTED_MASKED_VALUE_TYPE_CLASSES = (
    types.Number,
    types.Boolean,
)

# Concrete instances used for the ``Masked`` constructor's ConcreteTemplate
# cases. ``ConcreteTemplate.cases`` requires fully-qualified instances rather
# than type classes. ``integer_domain`` / ``real_domain`` are exposed on the
# vendored ``numba_cuda`` types module (not on ``numba_cuda_mlir.types``).
_supported_value_type_instances = (
    nb_types.integer_domain | nb_types.real_domain | {nb_types.boolean}
)


class MaskedType(types.Type):
    """A Numba type for ``Masked(value_type, valid: bool)`` values.

    Parameterized by ``value_type`` so that ``MaskedType(int64)`` and
    ``MaskedType(float64)`` are distinct types that cache and dispatch
    independently.

    Unsupported inner types are wrapped in ``types.Poison`` so the
    typing pass can still construct a ``MaskedType`` (e.g. while
    inferring the type of an unsupported column) and surface a
    descriptive error later when the user actually performs an
    operation.
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


def _register():
    """Register typing for ``Masked`` and ``MaskedType`` attributes with
    ``numba_cuda_mlir``. Called once at module import.
    """

    # ``Masked(value, valid)`` constructor: produces a ``Masked(value_ty)``.
    class MaskedConstructor(ConcreteTemplate):
        key = Masked
        cases = [
            nb_signature(MaskedType(t), t, types.boolean)
            for t in _supported_value_type_instances
        ]

    typing_registry.register_global(Masked, types.Function(MaskedConstructor))

    # ``m.value`` -> inner value type; ``m.valid`` -> boolean.
    @typing_registry.register_attr
    class MaskedTypeAttrs(AttributeTemplate):
        key = MaskedType

        def generic_resolve(self, typ, attr):
            if attr == "value":
                return typ.value_type
            if attr == "valid":
                return types.boolean
            return None


_register()
