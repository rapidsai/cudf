from numba import types
from cudf.core.scalar import _NAType
from numba.core.extending import typeof_impl, register_model, models
from numba.cuda.cudadecl import registry as cuda_decl_registry
from numba.core.typing.templates import AbstractTemplate
from numba.core.typing import signature as nb_signature

import operator

arith_ops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow

]


comparison_ops = [
    operator.eq, 
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge
]

class MaskedType(types.Type):
    '''
    A numba type consiting of a value of some primitive type 
    and a validity boolean, over which we can define math ops
    '''
    def __init__(self, value):
        # MaskedType in numba shall be parameterized
        # with a value type
        super().__init__(name="Masked")
        self.value_type = value

    def __repr__(self):
        return f"MaskedType({self.value_type})"

    def __hash__(self):
        '''
        Needed so that numba caches type instances with different
        `value_type` separately.  
        '''
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
        f(x: MaskedType(dtype_1), y: MaskedType(dtype_2))
          -> MaskedType(dtype_unified) 
        '''
        
        # If we have Masked and NA, the output should be a 
        # MaskedType with the original type as its value_type
        if isinstance(other, NAType):
            return self

        # if we have MaskedType and something that results in a
        # scalar, unify between the MaskedType's value_type
        # and that other thing
        unified = context.unify_pairs(self.value_type, other)
        if unified is None:
            return None

        return MaskedType(unified)

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
    '''
    A type for handling ops against nulls
    Exists so we can:
    1. Teach numba that all occurances of `cudf.NA` are 
       to be read as instances of this type instead
    2. Define ops like `if x is cudf.NA` where `x` is of 
       type `Masked` to mean `if x.valid is False`
    '''
    def __init__(self):
        super().__init__(name="NA")

    def unify(self, context, other):
        '''
        Masked  <-> NA works from above
        Literal <-> NA -> Masked
        '''
        breakpoint()
        if isinstance(other, MaskedType):
            # bounce to MaskedType.unify
            return None
        elif isinstance(other, NAType):
            # unify {NA, NA} -> NA
            return self
        else:
            return MaskedType(other)

@typeof_impl.register(_NAType)
def typeof_na(val, c):
    '''
    Tie instances of _NAType (cudf.NA) to our NAType.
    Effectively make it so numba sees `cudf.NA` as an
    instance of this NAType -> handle it accordingly.
    '''
    return NAType()

register_model(NAType)(models.OpaqueModel)


# Ultimately, we want numba to produce PTX code that specifies how to add
# two singular `Masked` structs together, which is defined as producing a
# new `Masked` with the right validity and if valid, the correct value. 
# This happens in two phases:
#   1. Specify that `Masked` + `Masked` exists and what it should return
#   2. Implement how to actually do (1) at the LLVM level
# The following code accomplishes (1) - it is really just a way of specifying
# that the `+` operation has a CUDA overload that accepts two `Masked` that
# are parameterized with `value_type` and what flavor of `Masked` to return.
class MaskedScalarArithOp(AbstractTemplate):
    def generic(self, args, kws):
        '''
        Typing for `Masked` + `Masked`
        Numba expects a valid numba type to be returned if typing is successful
        else `None` signifies the error state (this is common across numba)
        '''
        if isinstance(args[0], MaskedType) and isinstance(args[1], MaskedType):
            # In the case of op(Masked, Masked), the return type is a Masked
            # such that Masked.value is the primitive type that would have 
            # been resolved if we were just adding the `value_type`s. 
            return_type = self.context.resolve_function_type(
                self.key, (args[0].value_type, args[1].value_type), kws
            ).return_type
            return nb_signature(
                MaskedType(return_type),
                MaskedType(args[0].value_type),
                MaskedType(args[1].value_type),
            )

class MaskedScalarNullOp(AbstractTemplate):
    def generic(self, args, kws):
        '''
        Typing for `Masked` + `NA`
        Handles situations like `x + cudf.NA`
        '''
        if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
            # In the case of op(Masked, NA), the result has the same
            # dtype as the original regardless of what it is
            return_type = args[0].value_type
            return nb_signature(
                MaskedType(return_type),
                MaskedType(args[0].value_type),
                NAType(),
            )

class MaskedScalarConstOp(AbstractTemplate):
    def generic(self, args, kws):
        '''
        Typing for `Masked` + a constant literal
        handles situations like `x + 1`
        '''
        if isinstance(args[0], MaskedType) and isinstance(
            args[1], types.Number
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

@cuda_decl_registry.register_global(operator.is_)
class MaskedScalarIsNull(AbstractTemplate):
    '''
    Typing for `Masked is cudf.NA`
    '''
    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
            return nb_signature(
                types.boolean, 
                MaskedType(args[0].value_type), 
                NAType())

@cuda_decl_registry.register_global(operator.truth)
class MaskedScalarTruth(AbstractTemplate):
    '''
    Typing for `if Masked`
    Used for `if x > y`
    The truthiness of a MaskedType shall be the truthiness
    of the `value` stored therein
    '''
    def generic(self, args, kws):
        if isinstance(args[0], MaskedType):
            return nb_signature(
                types.boolean,
                MaskedType(types.boolean)
            )

for op in arith_ops + comparison_ops:
    # Every op shares the same typing class
    cuda_decl_registry.register_global(op)(MaskedScalarArithOp)
    cuda_decl_registry.register_global(op)(MaskedScalarNullOp)
    cuda_decl_registry.register_global(op)(MaskedScalarConstOp)
