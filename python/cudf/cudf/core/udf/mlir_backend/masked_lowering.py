# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
MLIR (numba_cuda_mlir) lowering for ``MaskedType`` -- mirrors the typing
surface introduced in :mod:`cudf.core.udf.mlir_backend.masked_typing`.

Scope of this PR (intentionally minimal):

* a ``PrimitiveModel`` data model registering ``MaskedType`` as a
  two-field LLVM struct ``{value_ty, i1}``;
* lowering of the ``Masked(value, valid)`` constructor;
* lowering of ``.value`` and ``.valid`` accessors via a generic
  getattr;
* lowering of ``pack_return`` for ``MaskedType`` inputs (identity)
  and for plain numeric/boolean scalar inputs (wrap with valid=True).

Out of scope (deferred to subsequent PRs in the stack):

* binary / unary / comparison op lowerings;
* casts between ``MaskedType`` value types;
* scalar -> ``Masked`` implicit casts (used by ``return literal``
  patterns); ``pack_return`` covers the explicit case at this layer.

Importing this module registers lowerings with ``numba_cuda_mlir``.
"""

from __future__ import annotations

from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import arith, llvm
from numba_cuda_mlir.extending import lowering_registry
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.models import PrimitiveModel, register_model

from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.mlir_backend.masked_typing import MaskedType


def _pack_masked_result(builder, target_type, result_value, result_valid):
    """Build a ``MaskedType`` struct value from raw ``value`` and ``valid``.

    Used by the constructor lowering and by ``pack_return`` for plain
    scalars.
    """
    struct_ty = builder.get_mlir_type(target_type)
    undef = llvm.UndefOp(struct_ty)
    with_value = llvm.insertvalue(
        container=undef,
        value=result_value,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    with_valid = llvm.insertvalue(
        container=with_value,
        value=result_valid,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    return with_valid


def _register():
    """Register the data model and lowerings with ``numba_cuda_mlir``.

    Called once at module import; deferred from module top-level so
    helper definitions above remain available for tests/imports without
    needing the registry to be ready.
    """
    lower = lowering_registry.lower
    lower_getattr_generic = lowering_registry.lower_getattr_generic

    # Data model: a MaskedType is an LLVM struct {value_ty, i1}. Each
    # parameterization (Masked(int64), Masked(float64), ...) gets its
    # own identified struct so MLIR can keep them distinct.
    @register_model(MaskedType)
    class MaskedTypeModel(PrimitiveModel):
        def __init__(self, dmm, fe_type):
            value_mlir = dmm.lookup(fe_type.value_type).get_value_type()
            valid_mlir = dmm.lookup(types.boolean).get_value_type()
            be_type = llvm.StructType.new_identified(
                fe_type.name, [value_mlir, valid_mlir]
            )
            super().__init__(dmm, fe_type, be_type)

    # ``Masked(value, valid)`` -> packed struct.
    def _lower_masked_constructor(builder, target, args, kwargs):
        target_type = builder.get_numba_type(target.name)
        val_var, valid_var = args

        # Coerce the raw value to the target's declared MLIR value type
        # (handles e.g. Literal[bool](True) flowing in for ``valid``).
        value_mlir_ty = builder.get_mlir_type(target_type.value_type)
        valid_mlir_ty = builder.get_mlir_type(types.boolean)
        value_mlir = convert(builder.load_var(val_var), value_mlir_ty)
        valid_mlir = convert(builder.load_var(valid_var), valid_mlir_ty)

        struct_val = _pack_masked_result(
            builder, target_type, value_mlir, valid_mlir
        )
        builder.store_var(target, struct_val)

    lower(Masked, types.Any, types.boolean)(_lower_masked_constructor)
    # Row-apply may pass a literal True/False for ``valid``, typed as
    # ``Literal[bool]``; accept that shape too.
    lower(Masked, types.Any, types.Literal)(_lower_masked_constructor)

    # Generic getattr for ``.value`` and ``.valid``.
    @lower_getattr_generic(MaskedType)
    def _lower_masked_getattr(context, builder, target, value, attr):
        struct_value = builder.load_var(value)
        if attr == "value":
            field_index = 0
        elif attr == "valid":
            field_index = 1
        else:
            raise AttributeError(f"MaskedType has no attribute {attr!r}")
        struct_ty = llvm.StructType(struct_value.type)
        field_mlir_ty = struct_ty.body[field_index]
        field_value = llvm.extractvalue(
            res=field_mlir_ty,
            container=struct_value,
            position=mlir_ir.DenseI64ArrayAttr.get([field_index]),
        )
        target_mlir_ty = builder.get_mlir_type(
            builder.get_numba_type(target.name)
        )
        builder.store_var(target, convert(field_value, target_mlir_ty))

    # ``pack_return(masked)`` -> identity. This gets called by user UDFs
    # whose return value is already a ``Masked``.
    def _lower_pack_return_masked(builder, target, args, kwargs):
        builder.store_var(target, builder.load_var(args[0]))

    lower(pack_return, MaskedType)(_lower_pack_return_masked)

    # ``pack_return(scalar)`` -> ``Masked(scalar, True)``. Allows UDFs
    # that ``return some_int`` to type-unify with a Masked return shape.
    def _lower_pack_return_scalar(builder, target, args, kwargs):
        target_type = builder.get_numba_type(target.name)
        value_mlir_ty = builder.get_mlir_type(target_type.value_type)
        scalar_val = convert(builder.load_var(args[0]), value_mlir_ty)
        valid_one = arith.constant(
            result=builder.get_mlir_type(types.boolean), value=1
        )
        struct_val = _pack_masked_result(
            builder, target_type, scalar_val, valid_one
        )
        builder.store_var(target, struct_val)

    # Register one entry per concrete numeric scalar shape so the
    # dispatcher finds an exact match rather than falling through to
    # ``types.Number`` (which would shadow Boolean).
    for scalar_ty in (
        types.Integer,
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
    ):
        lower(pack_return, scalar_ty)(_lower_pack_return_scalar)


_register()
