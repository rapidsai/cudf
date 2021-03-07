import numba
from numba import cuda
from numba import njit

class Masked(object):
    def __init__(self, value, valid):
        self.value = value
        self.valid = valid

from numba.extending import types

class MaskedType(types.Type):
    def __init__(self):
        super().__init__(name="Masked")

numba_masked = MaskedType() # name this something more natural - GM

from numba.core.extending import typeof_impl
@typeof_impl.register(Masked)
def typeof_masked(val, c):
    # This can be dynamic and be based off the specifics of the instance
    # which is why val is a parameter
    # c is a typeof context and doesn't matter for this
    return numba_masked

# types.number -  more general numeric values
from numba.core.extending import type_callable
@type_callable(Masked)
def type_masked(context):
    def typer(value, valid):
        if isinstance(value, types.Integer) and isinstance(valid, types.Boolean):
            return numba_masked
    return typer

# {i8, ... }
from numba.core.extending import models, register_model
@register_model(MaskedType)
class MaskedModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('value', types.int64),
            ('valid', types.bool_)
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)

from numba.core.extending import lower_builtin
from numba.core import cgutils

@lower_builtin(Masked, types.int64, types.bool_)
def impl_masked_constructor(context, builder, sig, args):
    
    print("beginning:")
    print(builder.module)
    
    typ = sig.return_type
    value, valid = args    
    
    masked = cgutils.create_struct_proxy(typ)(context, builder)
    masked.value = value
    masked.valid = valid
    print(masked._getvalue())
    
    print("end:")
    print(builder.module)
    return masked._getvalue() # return a pointer to the struct I created

@njit
def create_masked(value, valid):
    Masked(value, valid)

import numpy as np
create_masked(1, True)

from numba.core.typing import signature

from numba.core.typing.templates import AbstractTemplate
from numba.cuda.cudadecl import registry as cuda_registry

import operator

@cuda_registry.register_global(operator.add)
class MaskedScalarAdd(AbstractTemplate):
    # abstracttemplate vs concretetemplate
    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(args[1], MaskedType):
            return signature(numba_masked, numba_masked, numba_masked)

from numba.cuda.cudaimpl import lower as cuda_lower

@cuda_lower(operator.add, MaskedType, MaskedType)
def masked_scalar_add_impl(context, builder, sig, args):
    # get the types from the signature

    masked_type_1, masked_type_2 = sig.args
    masked_return_type = sig.return_type
    
    
    # create LLVM IR structs
    m1 = cgutils.create_struct_proxy(masked_type_1)(context, builder, value=args[0])
    m2 = cgutils.create_struct_proxy(masked_type_2)(context, builder, value=args[1])
    result = cgutils.create_struct_proxy(masked_return_type)(context, builder)

    valid = builder.or_(m1.valid, m2.valid)
    result.valid = valid
    with builder.if_then(valid):
        result.value = builder.add(m1.value, m2.value)
        
    return result._getvalue()

@cuda.jit(numba_masked(numba_masked, numba_masked), device=True)
def masked_add_py(m1, m2):
    return m1 + m2

def masked_add_py_2(m1, m2):
    return m1 + m2
