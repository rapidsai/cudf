# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.extending import lowering_registry
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.models import PrimitiveModel, register_model

from cudf.core.udf.api import Masked
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


class MaskedTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        value_mlir = dmm.lookup(fe_type.value_type).get_value_type()
        valid_mlir = dmm.lookup(types.boolean).get_value_type()
        be_type = llvm.StructType.new_identified(
            fe_type.name, [value_mlir, valid_mlir]
        )
        super().__init__(dmm, fe_type, be_type)


def _lower_masked_constructor(builder, target, args, kwargs):
    if kwargs:
        raise TypeError(
            "Masked(value, valid) does not accept keyword arguments; "
            f"got {kwargs!r}"
        )
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


def _register():
    """Register the data model and lowerings with ``numba_cuda_mlir``.

    Called once at module import; deferred from module top-level so the
    definitions above remain available for tests/imports without needing
    the registry to be ready.
    """
    lower = lowering_registry.lower
    lower_getattr_generic = lowering_registry.lower_getattr_generic

    register_model(MaskedType)(MaskedTypeModel)

    lower(Masked, types.Any, types.boolean)(_lower_masked_constructor)
    # Row apply may pass a literal True/False for ``valid``, typed as
    # ``Literal[bool]``; accept that shape too.
    lower(Masked, types.Any, types.Literal)(_lower_masked_constructor)

    lower_getattr_generic(MaskedType)(_lower_masked_getattr)


_register()
