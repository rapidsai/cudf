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
    def __init__(self):
        super().__init__(name="Masked")

class NAType(types.Type):
    def __init__(self):
        super().__init__(name="NA")

numba_masked = MaskedType()  # name this something more natural - GM
numba_na = NAType()

@typeof_impl.register(Masked)
def typeof_masked(val, c):
    return numba_masked

@typeof_impl.register(_NAType)
def typeof_na(val, c):
    return numba_na

@type_callable(Masked)
def type_masked(context):
    def typer(value, valid):
        if isinstance(value, types.Integer) and isinstance(
            valid, types.Boolean
        ):
            return numba_masked

    return typer

make_attribute_wrapper(MaskedType, "value", "value")
make_attribute_wrapper(MaskedType, "valid", "valid")

@register_model(MaskedType)
class MaskedModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("value", types.int64), ("valid", types.bool_)]
        models.StructModel.__init__(self, dmm, fe_type, members)

register_model(NAType)(models.OpaqueModel)

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
            return nb_signature(numba_masked, numba_masked, numba_masked)


@cuda_registry.register_global(operator.add)
class MaskedScalarAddNull(AbstractTemplate):
    def generic(self, args, kws):
        if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
            return nb_signature(numba_masked, numba_masked, numba_na)

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
        result.value = builder.add(m1.value, m2.value)

    return result._getvalue()


@cuda_lower(operator.add, MaskedType, NAType)
def masked_scalar_add_na_impl(context, builder, sig, args):
#    return_type = sig.return_type
    # use context to get llvm type for a bool
    result = cgutils.create_struct_proxy(numba_masked)(context, builder)
    result.valid = context.get_constant(types.boolean, 0)
    return result._getvalue()

@cuda_lowering_registry.lower_constant(NAType)
def constant_dummy(context, builder, ty, pyval):
    # This handles None, etc.
    return context.get_dummy_value()


@cuda.jit(numba_masked(numba_masked, numba_masked), device=True)
def masked_add_py(m1, m2):
    return m1 + m2


def masked_add_py_2(m1, m2):
    return m1 + m2

def compile_udf(func):
    n_params = len(py_signature(func).parameters)
    to_compiler_sig = tuple(numba_masked for arg in range(n_params))

    # Get the inlineable PTX function
    ptx, _ = cuda.compile_ptx_for_current_device(func, to_compiler_sig, device=True)

    # get the kernel that calls the inlineable function
    kernel = make_kernel(n_params)
    return kernel, ptx

NA = _NAType()

def make_kernel(n_params):
    '''
    create a string containing the right templated kernel
    for `func`
    '''
    
    indent = ' '*18
    
    # Hack together the template string
    result = ''
        
    templates = 'template <typename TypeOut, '
    for i in range(n_params):
        templates += f"typename Type{i}, "
    
    templates = templates[:-3] + f"{i}>"
    result += templates
    
    # Hack together the function signature
    sig = '\n__global__\nvoid genop_kernel(cudf::size_type size,\n'
    sig += indent + "TypeOut* out_data,\n"
    sig += indent + 'bool* out_mask,\n'
    for i in range(n_params):
        sig += indent + f"Type{i}* data_{i},\n"
        sig += indent + f"cudf::bitmask_type const* mask_{i},\n"
        sig += indent + f"cudf::size_type offset_{i},\n"
    sig = sig[:-2] + ') {'
    
    result += sig
    result += '\n'
    
    # standard thread block
    result += '\n'
    result += '\tint tid = threadIdx.x;\n'
    result += '\tint blkid = blockIdx.x;\n'
    result += '\tint blksz = blockDim.x;\n'
    result += '\tint gridsz = gridDim.x;\n'
    result += '\tint start = tid + blkid * blksz;\n'
    result += '\tint step = blksz * gridsz;\n'
    result += '\n'
    
    result += '\tMasked output;\n'
    
    for i in range(n_params):
        result += f"\tchar valid_{i};\n"

    # main loop
    result += "\tfor (cudf::size_type i=start; i<size; i+=step) {\n"
    
    for i in range(n_params):
        result += f"\t\tvalid_{i} = cudf::bit_is_set(mask_{i}, offset_{i} + i) : true;\n"
        
    # genop signature
    genop_sig = "\t\tGENERIC_OP(&output.value, "
    for i in range(n_params):
        genop_sig += f"data_{i}[i], valid_{i}, "
    
    genop_sig = genop_sig[:-2] + ');\n'
    
    result += genop_sig
    
    # set the output
    result += "\t\tout_data[i] = output.value;\n"
    result += "\t\tout_mask[i] = output.valid;\n"
    
    result += "\t}\n"
    result += "}"
    
    return result


demo_kernel = ''' 
template <typename TypeOut>
__global__
void genop_kernel(cudf::size_type size, cudf::size_type value, TypeOut* out_data) {

	int tid = threadIdx.x;
	int blkid = blockIdx.x;
	int blksz = blockDim.x;
	int gridsz = gridDim.x;
	int start = tid + blkid * blksz;
	int step = blksz * gridsz;

	for (cudf::size_type i=start; i<size; i+=step) {
        out_data[i] = value;
	}
}'''
