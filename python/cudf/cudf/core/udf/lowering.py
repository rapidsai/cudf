import operator

from llvmlite import ir
from numba.core import cgutils
from numba.core.typing import signature as nb_signature
from numba.cuda.cudaimpl import (
    lower as cuda_lower,
    registry as cuda_lowering_registry,
)
from numba.extending import lower_builtin, types

from cudf.core.udf.typing import MaskedType, NAType

from . import classes
from ._ops import arith_ops, comparison_ops


@cuda_lowering_registry.lower_constant(NAType)
def constant_na(context, builder, ty, pyval):
    # This handles None, etc.
    return context.get_dummy_value()


# In the typing phase, we declared that a `MaskedType` can be
# added to another `MaskedType` and specified what kind of a
# `MaskedType` would result. Now we have to actually fill in
# the implementation details of how to do that. This is where
# we can involve both validities in constructing the answer


def make_arithmetic_op(op):
    """
    Make closures that implement arithmetic operations. See
    register_arithmetic_op for details.
    """

    def masked_scalar_op_impl(context, builder, sig, args):
        """
        Implement `MaskedType` <op> `MaskedType`
        """
        # MaskedType(...), MaskedType(...)
        masked_type_1, masked_type_2 = sig.args
        # MaskedType(...)
        masked_return_type = sig.return_type

        # Let there be two actual LLVM structs backing the two inputs
        # https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/en/latest/basic-constructs/structures.html
        m1 = cgutils.create_struct_proxy(masked_type_1)(
            context, builder, value=args[0]
        )
        m2 = cgutils.create_struct_proxy(masked_type_2)(
            context, builder, value=args[1]
        )

        # we will return an output struct
        result = cgutils.create_struct_proxy(masked_return_type)(
            context, builder
        )

        # compute output validity
        valid = builder.and_(m1.valid, m2.valid)
        result.valid = valid
        with builder.if_then(valid):
            # Let numba handle generating the extra IR needed to perform
            # operations on mixed types, by compiling the final core op between
            # the two primitive values as a separate function and calling it
            result.value = context.compile_internal(
                builder,
                lambda x, y: op(x, y),
                nb_signature(
                    masked_return_type.value_type,
                    masked_type_1.value_type,
                    masked_type_2.value_type,
                ),
                (m1.value, m2.value),
            )
        return result._getvalue()

    return masked_scalar_op_impl


def register_arithmetic_op(op):
    """
    Register a lowering implementation for the
    arithmetic op `op`.

    Because the lowering implementations compile the final
    op separately using a lambda and compile_internal, `op`
    needs to be tied to each lowering implementation using
    a closure.

    This function makes and lowers a closure for one op.

    """
    to_lower_op = make_arithmetic_op(op)
    cuda_lower(op, MaskedType, MaskedType)(to_lower_op)


def masked_scalar_null_op_impl(context, builder, sig, args):
    """
    Implement `MaskedType` <op> `NAType`
    or `NAType` <op> `MaskedType`
    The answer to this is known up front so no actual operation
    needs to take place
    """

    return_type = sig.return_type  # MaskedType(...)
    result = cgutils.create_struct_proxy(MaskedType(return_type.value_type))(
        context, builder
    )

    # Invalidate the struct and leave `value` uninitialized
    result.valid = context.get_constant(types.boolean, 0)
    return result._getvalue()


def make_const_op(op):
    def masked_scalar_const_op_impl(context, builder, sig, args):
        return_type = sig.return_type
        result = cgutils.create_struct_proxy(return_type)(context, builder)
        result.valid = context.get_constant(types.boolean, 0)
        if isinstance(sig.args[0], MaskedType):
            masked_type, const_type = sig.args
            masked_value, const_value = args

            indata = cgutils.create_struct_proxy(masked_type)(
                context, builder, value=masked_value
            )
            nb_sig = nb_signature(
                return_type.value_type, masked_type.value_type, const_type
            )
            compile_args = (indata.value, const_value)
        else:
            const_type, masked_type = sig.args
            const_value, masked_value = args
            indata = cgutils.create_struct_proxy(masked_type)(
                context, builder, value=masked_value
            )
            nb_sig = nb_signature(
                return_type.value_type, const_type, masked_type.value_type
            )
            compile_args = (const_value, indata.value)
        with builder.if_then(indata.valid):
            result.value = context.compile_internal(
                builder, lambda x, y: op(x, y), nb_sig, compile_args
            )
            result.valid = context.get_constant(types.boolean, 1)
        return result._getvalue()

    return masked_scalar_const_op_impl


def register_const_op(op):
    to_lower_op = make_const_op(op)
    cuda_lower(op, MaskedType, types.Number)(to_lower_op)
    cuda_lower(op, types.Number, MaskedType)(to_lower_op)

    # to_lower_op_reflected = make_reflected_const_op(op)
    # cuda_lower(op, types.Number, MaskedType)(to_lower_op_reflected)


# register all lowering at init
for op in arith_ops + comparison_ops:
    register_arithmetic_op(op)
    register_const_op(op)
    # null op impl can be shared between all ops
    cuda_lower(op, MaskedType, NAType)(masked_scalar_null_op_impl)
    cuda_lower(op, NAType, MaskedType)(masked_scalar_null_op_impl)


@cuda_lower(operator.is_, MaskedType, NAType)
@cuda_lower(operator.is_, NAType, MaskedType)
def masked_scalar_is_null_impl(context, builder, sig, args):
    """
    Implement `MaskedType` is `NA`
    """
    if isinstance(sig.args[1], NAType):
        masked_type, na = sig.args
        value = args[0]
    else:
        na, masked_type = sig.args
        value = args[1]

    indata = cgutils.create_struct_proxy(masked_type)(
        context, builder, value=value
    )
    result = cgutils.alloca_once(builder, ir.IntType(1))
    with builder.if_else(indata.valid) as (then, otherwise):
        with then:
            builder.store(context.get_constant(types.boolean, 0), result)
        with otherwise:
            builder.store(context.get_constant(types.boolean, 1), result)

    return builder.load(result)


@cuda_lower(operator.truth, MaskedType)
def masked_scalar_truth_impl(context, builder, sig, args):
    indata = cgutils.create_struct_proxy(MaskedType(types.boolean))(
        context, builder, value=args[0]
    )
    return indata.value


@cuda_lower(bool, MaskedType)
def masked_scalar_bool_impl(context, builder, sig, args):
    indata = cgutils.create_struct_proxy(MaskedType(types.boolean))(
        context, builder, value=args[0]
    )
    return indata.value


# To handle the unification, we need to support casting from any type to a
# masked type. The cast implementation takes the value passed in and returns
# a masked type struct wrapping that value.
@cuda_lowering_registry.lower_cast(types.Any, MaskedType)
def cast_primitive_to_masked(context, builder, fromty, toty, val):
    casted = context.cast(builder, val, fromty, toty.value_type)
    ext = cgutils.create_struct_proxy(toty)(context, builder)
    ext.value = casted
    ext.valid = context.get_constant(types.boolean, 1)
    return ext._getvalue()


@cuda_lowering_registry.lower_cast(NAType, MaskedType)
def cast_na_to_masked(context, builder, fromty, toty, val):
    result = cgutils.create_struct_proxy(toty)(context, builder)
    result.valid = context.get_constant(types.boolean, 0)

    return result._getvalue()


@cuda_lowering_registry.lower_cast(MaskedType, MaskedType)
def cast_masked_to_masked(context, builder, fromty, toty, val):
    """
    When numba encounters an op that expects a certain type and
    the input to the op is not of the expected type it will try
    to cast the input to the appropriate type. But, in our case
    the input may be a MaskedType, which numba doesn't natively
    know how to cast to a different MaskedType with a different
    `value_type`. This implements and registers that cast.
    """

    # We will
    operand = cgutils.create_struct_proxy(fromty)(context, builder, value=val)
    casted = context.cast(
        builder, operand.value, fromty.value_type, toty.value_type
    )
    ext = cgutils.create_struct_proxy(toty)(context, builder)
    ext.value = casted
    ext.valid = operand.valid
    return ext._getvalue()


# Masked constructor for use in a kernel for testing
@lower_builtin(classes.Masked, types.Number, types.boolean)
def masked_constructor(context, builder, sig, args):
    ty = sig.return_type
    value, valid = args
    masked = cgutils.create_struct_proxy(ty)(context, builder)
    masked.value = value
    masked.valid = valid
    return masked._getvalue()


# Allows us to make an instance of MaskedType a global variable
# and properly use it inside functions we will later compile
@cuda_lowering_registry.lower_constant(MaskedType)
def lower_constant_masked(context, builder, ty, val):
    masked = cgutils.create_struct_proxy(ty)(context, builder)
    masked.value = context.get_constant(ty.value_type, val.value)
    masked.valid = context.get_constant(types.boolean, val.valid)
    return masked._getvalue()
