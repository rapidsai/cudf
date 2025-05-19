# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import operator

import numpy as np
from numba import types
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    register_model,
    typeof_impl,
)
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    ConcreteTemplate,
)
from numba.core.typing.typeof import typeof
from numba.cuda.cudadecl import registry as cuda_decl_registry
from numba.np.numpy_support import from_dtype

from cudf.core.missing import NA
from cudf.core.udf import api
from cudf.core.udf._ops import (
    arith_ops,
    bitwise_ops,
    comparison_ops,
    unary_ops,
)
from cudf.core.udf.strings_typing import (
    StringView,
    UDFString,
    bool_binary_funcs,
    id_unary_funcs,
    int_binary_funcs,
    size_type,
    string_return_attrs,
    string_unary_funcs,
    string_view,
    udf_string,
)
from cudf.utils.dtypes import (
    DATETIME_TYPES,
    NUMERIC_TYPES,
    STRING_TYPES,
    TIMEDELTA_TYPES,
)

SUPPORTED_NUMPY_TYPES = (
    NUMERIC_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES | STRING_TYPES
)
supported_type_str = "\n".join(sorted([*list(SUPPORTED_NUMPY_TYPES), "bool"]))

_units = ["ns", "ms", "us", "s"]
_datetime_cases = {types.NPDatetime(u) for u in _units}
_timedelta_cases = {types.NPTimedelta(u) for u in _units}
_supported_masked_types = (
    types.integer_domain
    | types.real_domain
    | _datetime_cases
    | _timedelta_cases
    | {types.boolean}
    | {string_view, udf_string}
)


SUPPORTED_NUMBA_TYPES = (
    types.Number,
    types.Boolean,
    types.NPDatetime,
    types.NPTimedelta,
    StringView,
    UDFString,
)


def _format_error_string(err):
    """
    Wrap an error message in newlines and color it red.
    """
    return "\033[91m" + "\n" + err + "\n" + "\033[0m"


def _type_to_masked_type(t):
    if isinstance(t, SUPPORTED_NUMBA_TYPES):
        return t
    else:
        # Unsupported Dtype. Numba tends to print out the type info
        # for whatever operands and operation failed to type and then
        # output its own error message. Putting the message in the repr
        # then is one way of getting the true cause to the user
        err = _format_error_string(
            "Unsupported MaskedType. This is usually caused by "
            "attempting to use a column of unsupported dtype in a UDF. "
            f"Supported dtypes are:\n{supported_type_str}"
        )
        return types.Poison(err)


# Masked scalars of all types
class MaskedType(types.Type):
    """
    A Numba type consisting of a value of some primitive type
    and a validity boolean, over which we can define math ops
    """

    def __init__(self, value):
        # MaskedType in Numba shall be parameterized
        # with a value type
        self.value_type = _type_to_masked_type(value)
        super().__init__(name=f"Masked({self.value_type})")

    def __hash__(self):
        """
        Needed so that numba caches type instances with different
        `value_type` separately.
        """
        return hash(repr(self))

    def unify(self, context, other):
        """
        Often within a UDF an instance arises where a variable could
        be a `MaskedType`, an `NAType`, or a literal based off
        the data at runtime, for example the variable `ret` here:

        def f(x):
            if x == 1:
                ret = x
            elif x > 2:
                ret = 1
            else:
                ret = cudf.NA
            return ret

        When numba analyzes this function it will eventually figure
        out that the variable `ret` could be any of the three types
        from above. This scenario will only work if numba knows how
        to find some kind of common type between the possibilities,
        and this function implements that - the goal is to return a
        common type when comparing `self` to other.

        """

        # If we have Masked and NA, the output should be a
        # MaskedType with the original type as its value_type
        if isinstance(other, NAType):
            return self

        # two MaskedType unify to a new MaskedType whose value_type
        # is the result of unifying `self` and `other` `value_type`
        elif isinstance(other, MaskedType):
            return MaskedType(
                context.unify_pairs(self.value_type, other.value_type)
            )

        # if we have MaskedType and something that results in a
        # scalar, unify between the MaskedType's value_type
        # and that other thing
        unified = context.unify_pairs(self.value_type, other)
        if unified is None:
            # The value types don't unify, so there is no unified masked type
            return None

        return MaskedType(unified)

    def __eq__(self, other):
        # Equality is required for determining whether a cast is required
        # between two different types.
        if not isinstance(other, MaskedType):
            # Require a cast when the other type is not masked
            return False

        # Require a cast for another masked with a different value type
        return self.value_type == other.value_type


# For typing a Masked constant value defined outside a kernel (e.g. captured in
# a closure).
@typeof_impl.register(api.Masked)
def typeof_masked(val, c):
    return MaskedType(typeof(val.value))


# Implemented typing for Masked(value, valid) - the construction of a Masked
# type in a kernel.
@cuda_decl_registry.register
class MaskedConstructor(ConcreteTemplate):
    key = api.Masked
    cases = [
        nb_signature(MaskedType(t), t, types.boolean)
        for t in _supported_masked_types
    ]


# Typing for `api.Masked`
@cuda_decl_registry.register_attr
class ClassesTemplate(AttributeTemplate):
    key = types.Module(api)

    def resolve_Masked(self, mod):
        return types.Function(MaskedConstructor)


# Registration of the global is also needed for Numba to type api.Masked
cuda_decl_registry.register_global(api, types.Module(api))
# For typing bare Masked (as in `from .api import Masked`
cuda_decl_registry.register_global(
    api.Masked, types.Function(MaskedConstructor)
)


# Provide access to `m.value` and `m.valid` in a kernel for a Masked `m`.
make_attribute_wrapper(MaskedType, "value", "value")
make_attribute_wrapper(MaskedType, "valid", "valid")


# Tell numba how `MaskedType` is constructed on the backend in terms
# of primitive things that exist at the LLVM level
@register_model(MaskedType)
class MaskedModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # This struct has two members, a value and a validity
        # let the type of the `value` field be the same as the
        # `value_type` and let `valid` be a boolean
        members = [("value", fe_type.value_type), ("valid", types.bool_)]
        models.StructModel.__init__(self, dmm, fe_type, members)


class NAType(types.Type):
    """
    A type for handling ops against nulls
    Exists so we can:
    1. Teach numba that all occurrences of `cudf.NA` are
       to be read as instances of this type instead
    2. Define ops like `if x is cudf.NA` where `x` is of
       type `Masked` to mean `if x.valid is False`
    """

    def __init__(self):
        super().__init__(name="NA")

    def unify(self, context, other):
        """
        Masked  <-> NA is deferred to MaskedType.unify()
        Literal <-> NA -> Masked
        """
        if isinstance(other, MaskedType):
            # bounce to MaskedType.unify
            return None
        elif isinstance(other, NAType):
            # unify {NA, NA} -> NA
            return self
        else:
            return MaskedType(other)


na_type = NAType()


@typeof_impl.register(type(NA))
def typeof_na(val, c):
    """
    Tie instances of _NAType (cudf.NA) to our NAType.
    Effectively make it so numba sees `cudf.NA` as an
    instance of this NAType -> handle it accordingly.
    """
    return na_type


register_model(NAType)(models.OpaqueModel)


# Ultimately, we want numba to produce PTX code that specifies how to implement
# an operation on two singular `Masked` structs together, which is defined
# as producing a new `Masked` with the right validity and if valid,
# the correct value. This happens in two phases:
#   1. Specify that `Masked` <op> `Masked` exists and what it should return
#   2. Implement how to actually do (1) at the LLVM level
# The following code accomplishes (1) - it is really just a way of specifying
# that the <op> has a CUDA overload that accepts two `Masked` that
# are parameterized with `value_type` and what flavor of `Masked` to return.
class MaskedScalarArithOp(AbstractTemplate):
    def generic(self, args, kws):
        """
        Typing for `Masked` <op> `Masked`
        Numba expects a valid numba type to be returned if typing is successful
        else `None` signifies the error state (this pattern is commonly used
        in Numba)
        """
        if isinstance(args[0], MaskedType) and isinstance(args[1], MaskedType):
            # In the case of op(Masked, Masked), the return type is a Masked
            # such that Masked.value is the primitive type that would have
            # been resolved if we were just operating on the
            # `value_type`s.
            return_type = self.context.resolve_function_type(
                self.key, (args[0].value_type, args[1].value_type), kws
            ).return_type
            return nb_signature(MaskedType(return_type), args[0], args[1])


class MaskedScalarUnaryOp(AbstractTemplate):
    def generic(self, args, kws):
        if len(args) == 1 and isinstance(args[0], MaskedType):
            return_type = self.context.resolve_function_type(
                self.key, (args[0].value_type,), kws
            ).return_type
            return nb_signature(MaskedType(return_type), args[0])


class MaskedScalarNullOp(AbstractTemplate):
    def generic(self, args, kws):
        """
        Typing for `Masked` + `NA`
        Handles situations like `x + cudf.NA`
        """
        if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
            # In the case of op(Masked, NA), the result has the same
            # dtype as the original regardless of what it is
            return nb_signature(
                args[0],
                args[0],
                na_type,
            )
        elif isinstance(args[0], NAType) and isinstance(args[1], MaskedType):
            return nb_signature(args[1], na_type, args[1])


class MaskedScalarScalarOp(AbstractTemplate):
    def generic(self, args, kws):
        """
        Typing for `Masked` <op> a scalar (and vice-versa).
        handles situations like `x + 1`
        """
        # In the case of op(Masked, scalar), we resolve the type between
        # the Masked value_type and the scalar's type directly
        to_resolve_types = None
        if isinstance(args[0], MaskedType) and isinstance(
            args[1], SUPPORTED_NUMBA_TYPES
        ):
            to_resolve_types = (args[0].value_type, args[1])
        elif isinstance(args[0], SUPPORTED_NUMBA_TYPES) and isinstance(
            args[1], MaskedType
        ):
            to_resolve_types = (args[1].value_type, args[0])
        else:
            # fail typing
            return None
        return_type = self.context.resolve_function_type(
            self.key, to_resolve_types, kws
        ).return_type
        return nb_signature(
            MaskedType(return_type),
            args[0],
            args[1],
        )


@cuda_decl_registry.register_global(operator.is_)
class MaskedScalarIsNull(AbstractTemplate):
    """
    Typing for `Masked is cudf.NA`
    """

    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
            return nb_signature(types.boolean, args[0], na_type)
        elif isinstance(args[1], MaskedType) and isinstance(args[0], NAType):
            return nb_signature(types.boolean, na_type, args[1])


@cuda_decl_registry.register_global(operator.truth)
class MaskedScalarTruth(AbstractTemplate):
    """
    Typing for `if Masked`
    Used for `if x > y`
    The truthiness of a MaskedType shall be the truthiness
    of the `value` stored therein
    """

    def generic(self, args, kws):
        if isinstance(args[0], MaskedType):
            return nb_signature(types.boolean, MaskedType(types.boolean))


@cuda_decl_registry.register_global(float)
class MaskedScalarFloatCast(AbstractTemplate):
    """
    Typing for float(Masked)
    returns the result of calling "float" on the input
    TODO: retains the validity of the input rather than
    raising as in float(pd.NA)
    """

    def generic(self, args, kws):
        if isinstance(args[0], MaskedType):
            # following numpy convention np.dtype(float) -> dtype('float64')
            return nb_signature(MaskedType(types.float64), args[0])


@cuda_decl_registry.register_global(int)
class MaskedScalarIntCast(AbstractTemplate):
    """
    Typing for int(Masked)
    returns the result of calling "int" on the input
    TODO: retains the validity of the input rather than
    raising as in int(pd.NA)
    """

    def generic(self, args, kws):
        if isinstance(args[0], MaskedType):
            # following numpy convention np.dtype(int) -> dtype('int64')
            return nb_signature(MaskedType(types.int64), args[0])


@cuda_decl_registry.register_global(abs)
class MaskedScalarAbsoluteValue(AbstractTemplate):
    """
    Typing for the builtin function abs. Returns the same
    type as input except for boolean values which are converted
    to integer.

    This follows the expected result from the builtin abs function
    which differs from numpy - np.abs returns a bool whereas abs
    itself performs the cast.
    """

    def generic(self, args, kws):
        if isinstance(args[0], MaskedType):
            if isinstance(args[0].value_type, (StringView, UDFString)):
                # reject string types
                return
            else:
                return_type = self.context.resolve_function_type(
                    self.key, (args[0].value_type,), kws
                ).return_type
                if return_type in types.signed_domain:
                    # promote to unsigned to avoid overflow
                    return_type = from_dtype(np.dtype("u" + return_type.name))
                return nb_signature(MaskedType(return_type), args[0])


@cuda_decl_registry.register_global(api.pack_return)
class UnpackReturnToMasked(AbstractTemplate):
    """
    Turn a returned MaskedType into its value and validity
    or turn a scalar into the tuple (scalar, True).
    """

    def generic(self, args, kws):
        if isinstance(args[0], MaskedType):
            # MaskedType(dtype, valid) -> MaskedType(dtype, valid)
            return nb_signature(args[0], args[0])
        elif isinstance(args[0], SUPPORTED_NUMBA_TYPES):
            # scalar_type -> MaskedType(scalar_type, True)
            return_type = MaskedType(args[0])
            return nb_signature(return_type, args[0])


for binary_op in arith_ops + bitwise_ops + comparison_ops:
    # Every op shares the same typing class
    cuda_decl_registry.register_global(binary_op)(MaskedScalarArithOp)
    cuda_decl_registry.register_global(binary_op)(MaskedScalarNullOp)
    cuda_decl_registry.register_global(binary_op)(MaskedScalarScalarOp)

for unary_op in unary_ops:
    cuda_decl_registry.register_global(unary_op)(MaskedScalarUnaryOp)


# Strings functions and utilities
def _is_valid_string_arg(ty):
    return (
        isinstance(ty, MaskedType)
        and isinstance(ty.value_type, (StringView, UDFString))
    ) or isinstance(ty, types.StringLiteral)


def register_masked_string_function(func):
    """
    Helper function wrapping numba's low level extension API. Provides
    the boilerplate needed to associate a signature with a function or
    operator to be overloaded.
    """

    def deco(generic):
        class MaskedStringFunction(AbstractTemplate):
            pass

        MaskedStringFunction.generic = generic
        cuda_decl_registry.register_global(func)(MaskedStringFunction)

    return deco


@register_masked_string_function(len)
def len_typing(self, args, kws):
    if isinstance(args[0], MaskedType) and isinstance(
        args[0].value_type, (StringView, UDFString)
    ):
        return nb_signature(MaskedType(size_type), MaskedType(string_view))
    elif isinstance(args[0], types.StringLiteral) and len(args) == 1:
        return nb_signature(size_type, args[0])


@register_masked_string_function(operator.add)
def concat_typing(self, args, kws):
    if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
        return nb_signature(
            MaskedType(udf_string),
            MaskedType(string_view),
            MaskedType(string_view),
        )


@register_masked_string_function(operator.contains)
def contains_typing(self, args, kws):
    if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
        return nb_signature(
            MaskedType(types.boolean),
            MaskedType(string_view),
            MaskedType(string_view),
        )


class MaskedStringViewCmpOp(AbstractTemplate):
    """
    return the boolean result of `cmpop` between to strings
    since the typing is the same for every comparison operator,
    we can reuse this class for all of them.
    """

    def generic(self, args, kws):
        if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
            return nb_signature(
                MaskedType(types.boolean),
                MaskedType(string_view),
                MaskedType(string_view),
            )


for op in comparison_ops:
    cuda_decl_registry.register_global(op)(MaskedStringViewCmpOp)


def create_masked_binary_attr(attrname, retty):
    """
    Helper function wrapping numba's low level extension API. Provides
    the boilerplate needed to register a binary function of two masked
    string objects as an attribute of one, e.g. `string.func(other)`.
    """

    class MaskedStringViewBinaryAttr(AbstractTemplate):
        key = attrname

        def generic(self, args, kws):
            return nb_signature(
                MaskedType(retty), MaskedType(string_view), recvr=self.this
            )

    def attr(self, mod):
        return types.BoundFunction(
            MaskedStringViewBinaryAttr,
            MaskedType(string_view),
        )

    return attr


def create_masked_unary_attr(attrname, retty):
    """
    Helper function wrapping numba's low level extension API. Provides
    the boilerplate needed to register a unary function of a masked
    string object as an attribute, e.g. `string.func()`.
    """

    class MaskedStringViewIdentifierAttr(AbstractTemplate):
        key = attrname

        def generic(self, args, kws):
            return nb_signature(MaskedType(retty), recvr=self.this)

    def attr(self, mod):
        return types.BoundFunction(
            MaskedStringViewIdentifierAttr,
            MaskedType(string_view),
        )

    return attr


class MaskedStringViewCount(AbstractTemplate):
    key = "MaskedType.count"

    def generic(self, args, kws):
        return nb_signature(
            MaskedType(size_type), MaskedType(string_view), recvr=self.this
        )


class MaskedStringViewReplace(AbstractTemplate):
    key = "MaskedType.replace"

    def generic(self, args, kws):
        return nb_signature(
            MaskedType(udf_string),
            MaskedType(string_view),
            MaskedType(string_view),
            recvr=self.this,
        )


class MaskedStringViewAttrs(AttributeTemplate):
    key = MaskedType(string_view)

    def resolve_replace(self, mod):
        return types.BoundFunction(
            MaskedStringViewReplace, MaskedType(string_view)
        )

    def resolve_count(self, mod):
        return types.BoundFunction(
            MaskedStringViewCount, MaskedType(string_view)
        )

    def resolve_value(self, mod):
        return string_view

    def resolve_valid(self, mod):
        return types.boolean


# Build attributes for `MaskedType(string_view)`
for func in bool_binary_funcs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_binary_attr(f"MaskedType.{func}", types.boolean),
    )

for func in int_binary_funcs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_binary_attr(f"MaskedType.{func}", size_type),
    )

for func in string_return_attrs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_binary_attr(f"MaskedType.{func}", udf_string),
    )

for func in id_unary_funcs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_unary_attr(f"MaskedType.{func}", types.boolean),
    )

for func in string_unary_funcs:
    setattr(
        MaskedStringViewAttrs,
        f"resolve_{func}",
        create_masked_unary_attr(f"MaskedType.{func}", udf_string),
    )


class MaskedUDFStringAttrs(MaskedStringViewAttrs):
    key = MaskedType(udf_string)

    def resolve_value(self, mod):
        return udf_string


cuda_decl_registry.register_attr(MaskedStringViewAttrs)
cuda_decl_registry.register_attr(MaskedUDFStringAttrs)
