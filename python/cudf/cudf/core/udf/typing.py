import operator

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
from pandas._libs.missing import NAType as _NAType

from . import classes
from ._ops import arith_ops, comparison_ops


class MaskedType(types.Type):
    """
    A Numba type consisting of a value of some primitive type
    and a validity boolean, over which we can define math ops
    """

    def __init__(self, value):
        # MaskedType in Numba shall be parameterized
        # with a value type
        if not isinstance(value, (types.Number, types.Boolean)):
            raise TypeError("value_type must be a numeric scalar type")
        self.value_type = value
        super().__init__(name=f"Masked{self.value_type}")

    def __hash__(self):
        """
        Needed so that numba caches type instances with different
        `value_type` separately.
        """
        return self.__repr__().__hash__()

    def unify(self, context, other):
        """
        Often within a UDF an instance arises where a variable could
        be a `MaskedType`, an `NAType`, or a literal based off
        the data at runtime, for examplem the variable `ret` here:

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
@typeof_impl.register(classes.Masked)
def typeof_masked(val, c):
    return MaskedType(typeof(val.value))


# Implemented typing for Masked(value, valid) - the construction of a Masked
# type in a kernel.
@cuda_decl_registry.register
class MaskedConstructor(ConcreteTemplate):
    key = classes.Masked

    cases = [
        nb_signature(MaskedType(t), t, types.boolean)
        for t in (types.integer_domain | types.real_domain)
    ]


# Provide access to `m.value` and `m.valid` in a kernel for a Masked `m`.
make_attribute_wrapper(MaskedType, "value", "value")
make_attribute_wrapper(MaskedType, "valid", "valid")


# Typing for `classes.Masked`
@cuda_decl_registry.register_attr
class ClassesTemplate(AttributeTemplate):
    key = types.Module(classes)

    def resolve_Masked(self, mod):
        return types.Function(MaskedConstructor)


# Registration of the global is also needed for Numba to type classes.Masked
cuda_decl_registry.register_global(classes, types.Module(classes))
# For typing bare Masked (as in `from .classes import Masked`
cuda_decl_registry.register_global(
    classes.Masked, types.Function(MaskedConstructor)
)


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
    1. Teach numba that all occurances of `cudf.NA` are
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


@typeof_impl.register(_NAType)
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


class MaskedScalarNullOp(AbstractTemplate):
    def generic(self, args, kws):
        """
        Typing for `Masked` + `NA`
        Handles situations like `x + cudf.NA`
        """
        if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
            # In the case of op(Masked, NA), the result has the same
            # dtype as the original regardless of what it is
            return nb_signature(args[0], args[0], na_type,)
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
        if isinstance(args[0], MaskedType) and isinstance(
            args[1], types.Number
        ):
            to_resolve_types = (args[0].value_type, args[1])
        elif isinstance(args[0], types.Number) and isinstance(
            args[1], MaskedType
        ):
            to_resolve_types = (args[1].value_type, args[0])
        return_type = self.context.resolve_function_type(
            self.key, to_resolve_types, kws
        ).return_type
        return nb_signature(MaskedType(return_type), args[0], args[1],)


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


for op in arith_ops + comparison_ops:
    # Every op shares the same typing class
    cuda_decl_registry.register_global(op)(MaskedScalarArithOp)
    cuda_decl_registry.register_global(op)(MaskedScalarNullOp)
    cuda_decl_registry.register_global(op)(MaskedScalarScalarOp)
