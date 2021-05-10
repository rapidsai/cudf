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
from numba.cuda.cudadecl import registry as cuda_decl_registry
from numba.cuda.cudaimpl import registry as cuda_impl_registry

from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry,
)
from numba.extending import types
import inspect

from numba.np import numpy_support

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
        super().__init__(name="Masked")
        self.value_type = value

    def __repr__(self):
        return f"MaskedType({self.value_type})"

    def __hash__(self):
        return self.__repr__().__hash__()

    def unify(self, context, other):
        '''
        Logic for sorting out what to do when the UDF conditionally
        returns a `MaskedType`, an `NAType`, or a literal based off 
        the data at runtime.

        In this framework, every input column is treated as having
        type `MaskedType`. Operations like `x + y` are understood 
        as translating to:

        `Masked(value=x, valid=True) + Masked(value=y, valid=True)`

        This means if the user writes a function such as 
        def f(x, y):
            return x + y
            
        numba sees this function as:
        f(x: MaskedType, y: MaskedType) -> MaskedType
        
        However if the user writes something like:
        def f(x, y):
            if x > 5:
                return 42
            else:
                return x + y
        
        numba now sees this as
        f(x: MaskedType, y: MaskedType) -> MaskedType OR literal 

        In general, when numba sees a function that could return
        more than a single type, it invokes unification to find a
        common type that can hold all possible results, eg given 
        a function that could return an int8 or an int16, numba 
        will resolve the type to int16. However it does not know
        how to unify MaskedType with primitive types. We need to 
        actually define the common type between MaskedType and
        literals, as well as between a Masked and an NA. For full
        generality, we'd need to unify NA and literal as well. 
        '''
        
        # If we have Masked and NA, the output should be a 
        # MaskedType with the original type as its value_type
        if isinstance(other, NAType):
            return MaskedType(self.value_type)

        # if we have MaskedType and Literal, the output should be
        # determined from the MaskedType.value_type (which is a 
        # primitive type) and other
        unified = context.unify_pairs(self.value_type, other)
        if unified is None:
            return None

        return MaskedType(unified)

class NAType(types.Type):
    def __init__(self):
        super().__init__(name="NA")

    def unify(self, context, other):
        '''
        Masked  <-> NA works from above
        Literal <-> NA -> Masked
        '''
        if isinstance(other, types.abstract.Literal):
            return MaskedType(other.literal_type)

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
            return Masked(value)

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
"""
@lower_builtin(Masked, types.int64, types.bool_)
def impl_masked_constructor(context, builder, sig, args):
    typ = sig.return_type
    value, valid = args

    masked = cgutils.create_struct_proxy(typ)(context, builder)
    masked.value = value
    masked.valid = valid
    return masked._getvalue()  # return a pointer to the struct I created
"""


@cuda_decl_registry.register_global(operator.add)
class MaskedScalarAdd(AbstractTemplate):
    # abstracttemplate vs concretetemplate
    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(args[1], MaskedType):
            # In the case of op(Masked, Masked), the return type is a Masked
            # such that Masked.value
            return_type = self.context.resolve_function_type(
                self.key, (args[0].value_type, args[1].value_type), kws
            ).return_type
            return nb_signature(
                MaskedType(return_type),
                MaskedType(args[0].value_type),
                MaskedType(args[1].value_type),
            )


@cuda_decl_registry.register_global(operator.add)
class MaskedScalarAddNull(AbstractTemplate):
    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
            # In the case of op(Masked, NA), the result has the same
            # dtype as the original regardless of what it is
            return_type = args[0].value_type
            return nb_signature(
                MaskedType(return_type),
                MaskedType(args[0].value_type),
                numba_na,
            )


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
        result.value = context.compile_internal(
            builder, 
            lambda x, y: x + y, 
            nb_signature(masked_return_type.value_type, masked_type_1.value_type, masked_type_2.value_type), (m1.value, m2.value)
        )

    return result._getvalue()


@cuda_lower(operator.add, MaskedType, NAType)
def masked_scalar_add_na_impl(context, builder, sig, args):
    return_type = sig.return_type
    # use context to get llvm type for a bool
    result = cgutils.create_struct_proxy(MaskedType(return_type.value_type))(
        context, builder
    )
    result.valid = context.get_constant(types.boolean, 0)
    return result._getvalue()


@cuda_lowering_registry.lower_constant(NAType)
def constant_dummy(context, builder, ty, pyval):
    # This handles None, etc.
    return context.get_dummy_value()


@cuda_decl_registry.register_global(operator.add)
class MaskedScalarAddConstant(AbstractTemplate):
    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(
            args[1], types.Integer
        ):
            # In the case of op(Masked, constant), we resolve the type between
            # the Masked value_type and the constant's type directly
            return_type = self.context.resolve_function_type(
                self.key, (args[0].value_type, args[1]), kws
            ).return_type
            return nb_signature(
                MaskedType(return_type),
                MaskedType(args[0].value_type),
                args[1],
            )


@cuda_lower(operator.add, MaskedType, types.Integer)
def masked_scalar_add_constant_impl(context, builder, sig, input_values):
    masked_type, const_type = sig.args

    indata = cgutils.create_struct_proxy(MaskedType(masked_type.value_type))(
        context, builder, value=input_values[0]
    )
    result = cgutils.create_struct_proxy(MaskedType(masked_type.value_type))(
        context, builder
    )
    # to_add_const = context.get_constant(const_type, input_values[1])

    result.valid = context.get_constant(types.boolean, 0)
    with builder.if_then(indata.valid):
        result.value = builder.add(indata.value, input_values[1])
        result.valid = context.get_constant(types.boolean, 1)

    return result._getvalue()

# To handle the unification, we need to support casting from any type to an
# extension type. The cast implementation takes the value passed in and returns
# an extension struct wrapping that value.
@cuda_impl_registry.lower_cast(types.Any, MaskedType)
def cast_primitive_to_masked(context, builder, fromty, toty, val):
    casted = context.cast(builder, val, fromty, toty.value_type)
    ext = cgutils.create_struct_proxy(toty)(context, builder)
    ext.value = casted
    return ext._getvalue()

@cuda_impl_registry.lower_cast(NAType, MaskedType)
def cast_na_to_masked(context, builder, fromty, toty, val):
    result = cgutils.create_struct_proxy(toty)(context, builder)
    result.valid = context.get_constant(types.boolean, 0)

    return result._getvalue()

def compile_udf(func, dtypes):
    n_params = len(py_signature(func).parameters)
    to_compiler_sig = tuple(
        MaskedType(arg)
        for arg in (numpy_support.from_dtype(np_type) for np_type in dtypes)
    )
    # Get the inlineable PTX function
    ptx, numba_output_type = cuda.compile_ptx_for_current_device(
        func, to_compiler_sig, device=True
    )
    numpy_output_type = numpy_support.as_dtype(
        numba_output_type.value_type
    )

    return numpy_output_type, ptx


NA = _NAType()


def nulludf(func):
    def wrapper(*args):
        from cudf import DataFrame

        to_udf_table = DataFrame(
            {idx: arg for idx, arg in zip(range(len(args)), args)}
        )
        return to_udf_table._apply(func)

    return wrapper
