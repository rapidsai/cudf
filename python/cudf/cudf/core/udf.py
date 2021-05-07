import operator
import numba
import numpy as np
from numba import cuda, njit
from numba.core import cgutils
from numba.core.extending import (
    lower_builtin,
    models,
    register_model,
    type_callable,
    typeof_impl,
)
from numba.core.typing import signature as nb_signature
from inspect import signature as py_signature
from numba.core.typing.templates import AbstractTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudaimpl import lower as cuda_lower, registry as cuda_lowering_registry
from numba.extending import types
import inspect

from llvmlite import ir
from cudf.core.scalar import _NAType

from numba.core.extending import make_attribute_wrapper

class Masked(object):
    def __init__(self, value, valid):
        self.value = value
        self.valid = valid


class MaskedType(types.Type):
    def __init__(self, value):
        # MaskedType in numba shall be parameterized
        # with a value type
        self.value_type = value
        super().__init__(name="Masked")
    
    def __repr__(self):
        return f"MaskedType({self.value_type})"

class NAType(types.Type):
    def __init__(self):
        super().__init__(name="NA")

numba_masked = MaskedType(types.int64)  # name this something more natural - GM
numba_na = NAType()

@typeof_impl.register(Masked)
def typeof_masked(val, c):
    return Masked(val.value)

@typeof_impl.register(_NAType)
def typeof_na(val, c):
    return numba_na

@type_callable(Masked)
def type_masked(context):
    def typer(value, valid):
        if isinstance(value, types.Integer) and isinstance(
            valid, types.Boolean
        ):
            return  Masked(value)

    return typer

make_attribute_wrapper(MaskedType, "value", "value")
make_attribute_wrapper(MaskedType, "valid", "valid")

@register_model(MaskedType)
class MaskedModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("value", fe_type.value_type), ("valid", types.bool_)]
        models.StructModel.__init__(self, dmm, fe_type, members)

register_model(NAType)(models.OpaqueModel)

# types.int64 instance, change to typeclass (types.Integer, types.Number, etc)
@lower_builtin(Masked, types.int64, types.bool_)
def impl_masked_constructor(context, builder, sig, args):
    typ = sig.return_type
    value, valid = args

    masked = cgutils.create_struct_proxy(typ)(context, builder)
    masked.value = value
    masked.valid = valid
    return masked._getvalue()  # return a pointer to the struct I created


@cuda_registry.register_global(operator.add)
class MaskedScalarAdd(AbstractTemplate):
    # abstracttemplate vs concretetemplate
    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(args[1], MaskedType):
            # In the case of op(Masked, Masked), the return type is a Masked
            # such that Masked.value 
            return_type = self.context.resolve_function_type(
                self.key,
                (
                    args[0].value_type,
                    args[1].value_type
                ),
                kws
            ).return_type
            return nb_signature(MaskedType(return_type), MaskedType(args[0].value_type), MaskedType(args[1].value_type))


@cuda_registry.register_global(operator.add)
class MaskedScalarAddNull(AbstractTemplate):
    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
            # In the case of op(Masked, NA), the result has the same 
            # dtype as the original regardless of what it is
            return_type = args[0].value_type
            return nb_signature(MaskedType(return_type), MaskedType(args[0].value_type), numba_na)

@cuda_lower(operator.add, MaskedType, MaskedType)
def masked_scalar_add_impl(context, builder, sig, args):
    # get the types from the signature

    masked_type_1, masked_type_2 = sig.args
    masked_return_type = sig.return_type

    # create LLVM IR structs
    m1 = cgutils.create_struct_proxy(masked_type_1)(
        context, builder, value=args[0]
    )
    m2 = cgutils.create_struct_proxy(masked_type_2)(
        context, builder, value=args[1]
    )
    result = cgutils.create_struct_proxy(masked_return_type)(context, builder)

    valid = builder.and_(m1.valid, m2.valid)
    result.valid = valid
    with builder.if_then(valid):
        # result.value = numba_op(m1.value, m2.value)
        result.value = builder.add(m1.value, m2.value)

    return result._getvalue()


@cuda_lower(operator.add, MaskedType, NAType)
def masked_scalar_add_na_impl(context, builder, sig, args):
    return_type = sig.return_type
    # use context to get llvm type for a bool
    result = cgutils.create_struct_proxy(MaskedType(return_type.value_type))(context, builder)
    result.valid = context.get_constant(types.boolean, 0)
    return result._getvalue()

@cuda_lowering_registry.lower_constant(NAType)
def constant_dummy(context, builder, ty, pyval):
    # This handles None, etc.
    return context.get_dummy_value()

@cuda_registry.register_global(operator.add)
class MaskedScalarAddConstant(AbstractTemplate):
    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(args[1], types.Integer):
            # In the case of op(Masked, constant), we resolve the type between
            # the Masked value_type and the constant's type directly
            return_type = self.context.resolve_function_type(
                self.key,
                (
                    args[0].value_type,
                    args[1]
                ),
                kws
            ).return_type
            return nb_signature(MaskedType(return_type), MaskedType(args[0].value_type), args[1])

@cuda_lower(operator.add, MaskedType, types.Integer)
def masked_scalar_add_constant_impl(context, builder, sig, input_values):
    masked_type, const_type = sig.args

    indata = cgutils.create_struct_proxy(MaskedType(masked_type.value_type))(context, builder, value=input_values[0])
    result = cgutils.create_struct_proxy(MaskedType(masked_type.value_type))(context, builder)
    #to_add_const = context.get_constant(const_type, input_values[1])

    result.valid = context.get_constant(types.boolean, 0)
    with builder.if_then(indata.valid):
        result.value = builder.add(indata.value, input_values[1])
        result.valid = context.get_constant(types.boolean, 1)

    return result._getvalue()


def compile_udf(func):
    n_params = len(py_signature(func).parameters)
    to_compiler_sig = tuple(numba_masked for arg in range(n_params))

    # Get the inlineable PTX function
    ptx, _ = cuda.compile_ptx_for_current_device(func, to_compiler_sig, device=True)

    # get the kernel that calls the inlineable function
    kernel = None
    return kernel, ptx

NA = _NAType()

def nulludf(func):
    def wrapper(*args):
        from cudf import DataFrame
        to_udf_table = DataFrame({idx: arg for idx, arg in zip(range(len(args)), args)})
        return to_udf_table._apply(func)
    return wrapper
