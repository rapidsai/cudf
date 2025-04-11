# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from functools import partial

from numba import types
from numba.core import cgutils
from numba.core.extending import lower_builtin
from numba.core.typing import signature as nb_signature
from numba.cuda.cudaimpl import lower as cuda_lower, registry as cuda_registry
from numba import cuda
from numba.core.datamodel import default_manager

from cudf.core.udf.groupby_typing import (
    SUPPORTED_GROUPBY_NUMBA_TYPES,
    GroupView,
    GroupViewType,
    ManagedGroupViewType,
    call_cuda_functions,
    group_size_type,
    index_default_type,
    call_group_sum
)


def group_reduction_impl_basic(context, builder, sig, args, function):
    """
    Instruction boilerplate used for calling a groupby reduction
    __device__ function. Centers around a forward declaration of
    this function and adds the pre/post processing instructions
    necessary for calling it.
    """
    # return type
    retty = sig.return_type

    # a variable logically corresponding to the calling `Group`
    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )

    # what specific (numba) GroupType
    grp_type = sig.args[0]
    group_dataty = grp_type.group_data_type

    # obtain the correct forward declaration from registry
    type_key = (sig.return_type, grp_type.group_scalar_type)
    func = call_cuda_functions[function][type_key]

    # insert the forward declaration and return its result
    # pass it the data pointer and the group's size
    return context.compile_internal(
        builder,
        func,
        nb_signature(retty, group_dataty, grp_type.group_size_type),
        (grp.group_data, grp.size),
    )


def group_corr(context, builder, sig, args):
    """
    Instruction boilerplate used for calling a groupby correlation
    """
    lhs_grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    rhs_grp = cgutils.create_struct_proxy(sig.args[1])(
        context, builder, value=args[1]
    )

    device_func = call_cuda_functions["corr"][
        (
            sig.return_type,
            sig.args[0].group_scalar_type,
            sig.args[1].group_scalar_type,
        )
    ]
    result = context.compile_internal(
        builder,
        device_func,
        nb_signature(
            types.float64,
            types.CPointer(
                sig.args[0].group_scalar_type
            ),  # this group calls corr
            types.CPointer(
                sig.args[1].group_scalar_type
            ),  # this group is passed
            group_size_type,
        ),
        (
            lhs_grp.group_data,
            rhs_grp.group_data,
            lhs_grp.size,
        ),
    )
    return result


@lower_builtin(GroupView, types.Array, group_size_type, types.Array)
def group_constructor(context, builder, sig, args):
    """
    Instruction boilerplate used for instantiating a Group
    struct from a data pointer, an index pointer, and a size
    """
    # a variable logically corresponding to the calling `Group`
    grp = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    grp.group_data = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    ).data
    grp.index = cgutils.create_struct_proxy(sig.args[2])(
        context, builder, value=args[2]
    ).data
    grp.size = args[1]
    return grp._getvalue()


def group_reduction_impl_idx_max_or_min(context, builder, sig, args, function):
    """
    Instruction boilerplate used for calling a groupby reduction
    __device__ function in the case where the function is either
    `idxmax` or `idxmin`. See `group_reduction_impl_basic` for
    details. This lowering differs from other reductions due to
    the presence of the index. This results in the forward
    declaration expecting an extra arg.
    """
    retty = sig.return_type

    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    grp_type = sig.args[0]

    if grp_type.index_type != index_default_type:
        raise TypeError(
            f"Only inputs with default index dtype {index_default_type} "
            "are supported."
        )

    type_key = (index_default_type, grp_type.group_scalar_type)
    func = call_cuda_functions[function][type_key]

    return context.compile_internal(
        builder,
        func,
        nb_signature(
            retty,
            grp_type.group_data_type,
            grp_type.group_index_type,
            grp_type.group_size_type,
        ),
        (grp.group_data, grp.index, grp.size),
    )


cuda_Group_max = partial(group_reduction_impl_basic, function="max")
cuda_Group_min = partial(group_reduction_impl_basic, function="min")
cuda_Group_sum = partial(group_reduction_impl_basic, function="sum")
cuda_Group_mean = partial(group_reduction_impl_basic, function="mean")
cuda_Group_std = partial(group_reduction_impl_basic, function="std")
cuda_Group_var = partial(group_reduction_impl_basic, function="var")

cuda_Group_idxmax = partial(
    group_reduction_impl_idx_max_or_min, function="idxmax"
)
cuda_Group_idxmin = partial(
    group_reduction_impl_idx_max_or_min, function="idxmin"
)


def cuda_Group_size(context, builder, sig, args):
    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    return grp.size


cuda_Group_count = cuda_Group_size


for ty in SUPPORTED_GROUPBY_NUMBA_TYPES:
    cuda_lower("GroupViewType.max", GroupViewType(ty))(cuda_Group_max)
    cuda_lower("GroupViewType.min", GroupViewType(ty))(cuda_Group_min)
    cuda_lower("GroupViewType.sum", GroupViewType(ty))(cuda_Group_sum)
    cuda_lower("GroupViewType.count", GroupViewType(ty))(cuda_Group_count)
    cuda_lower("GroupViewType.size", GroupViewType(ty))(cuda_Group_size)
    cuda_lower("GroupViewType.mean", GroupViewType(ty))(cuda_Group_mean)
    cuda_lower("GroupViewType.std", GroupViewType(ty))(cuda_Group_std)
    cuda_lower("GroupViewType.var", GroupViewType(ty))(cuda_Group_var)
    cuda_lower("GroupViewType.idxmax", GroupViewType(ty, types.int64))(
        cuda_Group_idxmax
    )
    cuda_lower("GroupViewType.idxmin", GroupViewType(ty, types.int64))(
        cuda_Group_idxmin
    )
    cuda_lower("GroupViewType.corr", GroupViewType(ty), GroupViewType(ty))(group_corr)


_print_group_data = cuda.declare_device(
    'print_group_data',
    types.void(types.CPointer(types.int64), types.int64)
)
def call_print_group_data(data_ptr, size):
    _print_group_data(data_ptr, size)

<<<<<<< Updated upstream
_udf_grp_meminfo = cuda.declare_device(
    "meminfo_from_new_udf_group", types.voidptr(types.CPointer(types.int64))
)

def make_udf_grp_meminfo(data_ptr):
    return _udf_grp_meminfo(data_ptr)
=======
_udf_grp_meminfo = cuda.declare_device(                                                                                                                                                                                                  
    "meminfo_from_new_udf_group", types.voidptr(types.CPointer(types.int64))                                                                                                                                                             
)                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                         
def make_udf_grp_meminfo(data_ptr):                                                                                                                                                                                                      
    return _udf_grp_meminfo(data_ptr)                                                                                                                                                                                                    
   
>>>>>>> Stashed changes

import operator
@cuda_lower(operator.add, GroupViewType(types.int64), GroupViewType(types.int64))
def cuda_lower_add(context, builder, sig, args):
    lhs_grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    rhs_grp = cgutils.create_struct_proxy(sig.args[1])(
        context, builder, value=args[1]
    )

    output = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    out_ptr = context.compile_internal(
        builder,
        call_group_sum,
        types.CPointer(types.int64)(
            sig.args[0].group_data_type,
            sig.args[1].group_data_type,
            group_size_type,
        ),
        (
            lhs_grp.group_data,
            rhs_grp.group_data,
            lhs_grp.size,
        ),
    )

    mi = context.compile_internal(                                                                                                                                                                                                        
        builder,                                                                                                                                                                                                                         
        make_udf_grp_meminfo,                                                                                                                                                                                                           
        types.voidptr(                                                                                                                                                                                                                      
            types.CPointer(types.int64),                                                                                                                                                                                   
        ),                                                                                                                                                                                                                               
        (out_ptr,),                                                                                                                                                                                                         
    )  


    out_grp = cgutils.create_struct_proxy(GroupViewType(types.int64))(context, builder)
    out_grp.group_data = out_ptr
    out_grp.size = lhs_grp.size

    output.group_view = out_grp._getvalue()
    output.meminfo = mi
    return output._getvalue()


@cuda_registry.lower_cast(ManagedGroupViewType(GroupViewType(types.int64)), GroupViewType(types.int64))
def cast_managed_group_view_to_group_view(
    context, builder, fromty, toty, val
):

    managed = cgutils.create_struct_proxy(fromty)(
        context, builder, value=val
    )

    result = cgutils.create_struct_proxy(toty)(context, builder, value=managed.group_view)

    return result._getvalue()
