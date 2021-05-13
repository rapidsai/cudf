from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry,
)
from numba.core.typing import signature as nb_signature
from cudf.core.udf.typing import MaskedType, NAType
from numba.core import cgutils
from numba.cuda.cudaimpl import registry as cuda_impl_registry
import operator
from numba.extending import types

@cuda_lowering_registry.lower_constant(NAType)
def constant_dummy(context, builder, ty, pyval):
    # This handles None, etc.
    return context.get_dummy_value()

# In the typing phase, we declared that a `MaskedType` can be 
# added to another `MaskedType` and specified what kind of 
# `MaskedType` would result. Now we have to actually fill in 
# the implementation details of how to do that. This is where
# we can involve both validities in constructing the answer
@cuda_lower(operator.add, MaskedType, MaskedType)
def masked_scalar_add_impl(context, builder, sig, args):
    '''
    Implement `MaskedType` + `MaskedType`
    '''

    masked_type_1, masked_type_2 = sig.args # MaskedType(...), MaskedType(...)
    masked_return_type = sig.return_type # MaskedType(...)

    # Let there be two actual LLVM structs backing the two inputs
    # https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/en/latest/basic-constructs/structures.html
    m1 = cgutils.create_struct_proxy(masked_type_1)(
        context, builder, value=args[0]
    )
    m2 = cgutils.create_struct_proxy(masked_type_2)(
        context, builder, value=args[1]
    )

    # we will return an output struct
    result = cgutils.create_struct_proxy(masked_return_type)(context, builder)

    # compute output validity
    valid = builder.and_(m1.valid, m2.valid)
    result.valid = valid
    with builder.if_then(valid):
        # Let numba handle generating the extra LLVM needed to perform
        # operations on mixed types, by compiling the final core op between
        # the two primitive values as a separate function and calling it
        result.value = context.compile_internal(
            builder, 
            lambda x, y: x + y, 
            nb_signature(
                masked_return_type.value_type, 
                masked_type_1.value_type, 
                masked_type_2.value_type
            ),
            (m1.value, m2.value)
        )
    return result._getvalue()

@cuda_lower(operator.add, MaskedType, NAType)
def masked_scalar_add_na_impl(context, builder, sig, args):
    '''
    Implement `MaskedType` + `NAType`
    The answer to this is known up front so no actual addition
    needs to take place
    '''

    return_type = sig.return_type # MaskedType(...)
    result = cgutils.create_struct_proxy(MaskedType(return_type.value_type))(
        context, builder
    )

    # Invalidate the struct and leave `value` uninitialized
    result.valid = context.get_constant(types.boolean, 0)
    return result._getvalue()

@cuda_lower(operator.add, MaskedType, types.Integer)
def masked_scalar_add_constant_impl(context, builder, sig, input_values):
    '''
    Implement `MaskedType` + constant
    '''
    masked_type, const_type = sig.args

    indata = cgutils.create_struct_proxy(MaskedType(masked_type.value_type))(
        context, builder, value=input_values[0]
    )
    result = cgutils.create_struct_proxy(MaskedType(masked_type.value_type))(
        context, builder
    )
    result.valid = context.get_constant(types.boolean, 0)
    with builder.if_then(indata.valid):
        result.value = builder.add(indata.value, input_values[1])
        result.valid = context.get_constant(types.boolean, 1)

    return result._getvalue()

@cuda_lower(operator.is_, MaskedType, NAType)
def masked_scalar_is_null_impl(context, builder, sig, args):
    '''
    Implement `MaskedType` + constant
    '''
    masked_type, na = sig.args
    indata = cgutils.create_struct_proxy(masked_type)(
        context, builder, value=args[0]
    )

    with builder.if_else(indata.valid) as (then, otherwise):
        with then:
            result = context.get_constant(types.boolean, 0)
        with otherwise:
            result = context.get_constant(types.boolean, 1)
    return result

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
