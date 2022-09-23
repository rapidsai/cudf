# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import glob
import math
import os

import cupy as cp
import numba
import numpy as np
from numba import cuda, types
from numba.core import cgutils
from numba.core.extending import (
    lower_builtin,
    make_attribute_wrapper,
    models,
    register_model,
    type_callable,
    typeof_impl,
)
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate, AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_registry
from numba.cuda.cudadrv.devices import get_context
from numba.cuda.cudaimpl import lower as cuda_lower
from numba.np import numpy_support
from numba.types import Record

from cudf.core.column import as_column
from cudf.core.udf.templates import (
    group_initializer_template,
    groupby_apply_kernel_template,
)
from cudf.core.udf.utils import (
    _all_dtypes_from_frame,
    _compile_or_get,
    _get_kernel_groupby_apply,
    _get_udf_return_type,
    _supported_cols_from_frame,
    _supported_dtypes_from_frame,
)
from cudf.utils.utils import _cudf_nvtx_annotate

# Disable occupancy warnings to avoid polluting output when there are few
# groups.
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


class Group(object):
    def __init__(self, group_data, size, index, dtype, index_dtype):
        self.group_data = group_data
        self.size = size
        self.index = index
        self.dtype = dtype
        self.index_dtype = index_dtype


class GroupType(numba.types.Type):
    def __init__(self, group_scalar_type, index_type=types.int64):
        self.group_scalar_type = group_scalar_type
        self.index_type = index_type
        self.group_data_type = types.CPointer(group_scalar_type)
        self.size_type = types.int64
        self.group_index_type = types.CPointer(index_type)
        super().__init__(
            name=f"Group({self.group_scalar_type}, {self.index_type})"
        )


@typeof_impl.register(Group)
def typeof_group(val, c):
    return GroupType(
        numba.np.numpy_support.from_dtype(val.dtype),
        numba.np.numpy_support.from_dtype(val.index_dtype),
    )  # converting from numpy type to numba type


@type_callable(Group)
def type_group(context):
    def typer(group_data, size, index):
        if (
            isinstance(group_data, types.Array)
            and isinstance(size, types.Integer)
            and isinstance(index, types.Array)
        ):
            return GroupType(group_data.dtype, index.dtype)

    return typer


@register_model(GroupType)
class GroupModel(models.StructModel):
    def __init__(
        self, dmm, fe_type
    ):  # fe_type is fully instantiated group type
        members = [
            ("group_data", types.CPointer(fe_type.group_scalar_type)),
            ("size", types.int64),
            ("index", types.CPointer(fe_type.index_type)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


my_max_int32 = cuda.declare_device(
    "BlockMax_int32", "types.int32(types.CPointer(types.int32),types.int64)"
)

my_max_int64 = cuda.declare_device(
    "BlockMax_int64", "types.int64(types.CPointer(types.int64),types.int64)"
)

my_max_float64 = cuda.declare_device(
    "BlockMax_float64",
    "types.float64(types.CPointer(types.float64),types.int64)",
)

my_min_int32 = cuda.declare_device(
    "BlockMin_int32", "types.int32(types.CPointer(types.int32),types.int64)"
)

my_min_int64 = cuda.declare_device(
    "BlockMin_int64", "types.int64(types.CPointer(types.int64),types.int64)"
)

my_min_float64 = cuda.declare_device(
    "BlockMin_float64",
    "types.float64(types.CPointer(types.float64),types.int64)",
)

my_sum_int64 = cuda.declare_device(
    "BlockSum_int64", "types.int64(types.CPointer(types.int64),types.int64)"
)

my_sum_float64 = cuda.declare_device(
    "BlockSum_float64",
    "types.float64(types.CPointer(types.float64),types.int64)",
)

my_mean_int64 = cuda.declare_device(
    "BlockMean_int64",
    "types.float64(types.CPointer(types.int64),types.int64)",
)

my_mean_float64 = cuda.declare_device(
    "BlockMean_float64",
    "types.float64(types.CPointer(types.float64),types.int64)",
)

my_std_int64 = cuda.declare_device(
    "BlockStd_int64",
    "types.float64(types.CPointer(types.int64),types.int64)",
)

my_std_float64 = cuda.declare_device(
    "BlockStd_float64",
    "types.float64(types.CPointer(types.float64),types.int64)",
)

my_var_int64 = cuda.declare_device(
    "BlockVar_int64",
    "types.float64(types.CPointer(types.int64),types.int64)",
)

my_var_float64 = cuda.declare_device(
    "BlockVar_float64",
    "types.float64(types.CPointer(types.float64),types.int64)",
)

my_idxmax_int64 = cuda.declare_device(
    "BlockIdxMax_int64",
    "types.int64(types.CPointer(types.int64),"
    "types.CPointer(types.int64),types.int64)",
)

my_idxmax_float64 = cuda.declare_device(
    "BlockIdxMax_float64",
    "types.int64(types.CPointer(types.float64),"
    "types.CPointer(types.int64),types.int64)",
)

my_idxmin_int64 = cuda.declare_device(
    "BlockIdxMin_int64",
    "types.int64(types.CPointer(types.int64),"
    "types.CPointer(types.int64),types.int64)",
)

my_idxmin_float64 = cuda.declare_device(
    "BlockIdxMin_float64",
    "types.int64(types.CPointer(types.float64),"
    "types.CPointer(types.int64),types.int64)",
)

# Load the highest compute capability file available that is less than
# the current device's.
files = glob.glob(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "function_*.ptx")
)
if len(files) == 0:
    raise RuntimeError(
        "This strings_udf installation is missing the necessary PTX "
        "files. Please file an issue reporting this error and how you "
        "installed cudf and strings_udf."
    )
dev = cuda.get_current_device()
cc = "".join(str(x) for x in dev.compute_capability)
sms = [os.path.basename(f).rstrip(".ptx").lstrip("function_") for f in files]
selected_sm = max(sm for sm in sms if sm < cc)
dev_func_ptx = os.path.join(
    os.path.dirname(__file__), f"function_{selected_sm}.ptx"
)


def call_my_max_int32(data, size):
    return my_max_int32(data, size)


def call_my_max_int64(data, size):
    return my_max_int64(data, size)


def call_my_max_float64(data, size):
    return my_max_float64(data, size)


def call_my_min_int32(data, size):
    return my_min_int32(data, size)


def call_my_min_int64(data, size):
    return my_min_int64(data, size)


def call_my_min_float64(data, size):
    return my_min_float64(data, size)


def call_my_sum_int64(data, size):
    return my_sum_int64(data, size)


def call_my_sum_float64(data, size):
    return my_sum_float64(data, size)


def call_my_mean_int64(data, size):
    return my_mean_int64(data, size)


def call_my_mean_float64(data, size):
    return my_mean_float64(data, size)


def call_my_std_int64(data, size):
    return my_std_int64(data, size)


def call_my_std_float64(data, size):
    return my_std_float64(data, size)


def call_my_var_int64(data, size):
    return my_var_int64(data, size)


def call_my_var_float64(data, size):
    return my_var_float64(data, size)


def call_my_idxmax_int64(data, index, size):
    return my_idxmax_int64(data, index, size)


def call_my_idxmax_float64(data, index, size):
    return my_idxmax_float64(data, index, size)


def call_my_idxmin_int64(data, index, size):
    return my_idxmin_int64(data, index, size)


def call_my_idxmin_float64(data, index, size):
    return my_idxmin_float64(data, index, size)


@lower_builtin(Group, types.Array, types.int64, types.Array)
def group_constructor(context, builder, sig, args):
    group_data, size, index = args

    grp = cgutils.create_struct_proxy(sig.return_type)(context, builder)

    arr_group_data = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=group_data
    )
    group_data_ptr = arr_group_data.data

    arr_index = cgutils.create_struct_proxy(sig.args[2])(
        context, builder, value=index
    )
    index_ptr = arr_index.data

    grp.group_data = group_data_ptr
    grp.index = index_ptr
    grp.size = size

    return grp._getvalue()


make_attribute_wrapper(GroupType, "group_data", "group_data")
make_attribute_wrapper(GroupType, "index", "index")
make_attribute_wrapper(GroupType, "size", "size")


class GroupMax(AbstractTemplate):
    key = "GroupType.max"

    def generic(self, args, kws):
        return nb_signature(self.this.group_scalar_type, recvr=self.this)


class GroupMin(AbstractTemplate):
    key = "GroupType.min"

    def generic(self, args, kws):
        return nb_signature(self.this.group_scalar_type, recvr=self.this)


class GroupSize(AbstractTemplate):
    key = "GroupType.size"

    def generic(self, args, kws):
        return nb_signature(types.int64, recvr=self.this)


class GroupCount(AbstractTemplate):
    key = "GroupType.count"

    def generic(self, args, kws):
        return nb_signature(types.int64, recvr=self.this)


class GroupSum(AbstractTemplate):
    key = "GroupType.sum"

    def generic(self, args, kws):
        return nb_signature(self.this.group_scalar_type, recvr=self.this)


class GroupMean(AbstractTemplate):
    key = "GroupType.mean"

    def generic(self, args, kws):
        return nb_signature(types.float64, recvr=self.this)


class GroupStd(AbstractTemplate):
    key = "GroupType.std"

    def generic(self, args, kws):
        return nb_signature(types.float64, recvr=self.this)


class GroupVar(AbstractTemplate):
    key = "GroupType.var"

    def generic(self, args, kws):
        return nb_signature(types.float64, recvr=self.this)


class GroupIdxMax(AbstractTemplate):
    key = "GroupType.idxmax"

    def generic(self, args, kws):
        return nb_signature(self.this.index_type, recvr=self.this)


class GroupIdxMin(AbstractTemplate):
    key = "GroupType.idxmin"

    def generic(self, args, kws):
        return nb_signature(self.this.index_type, recvr=self.this)


@cuda_registry.register_attr
class GroupAttr(AttributeTemplate):
    key = GroupType

    def resolve_max(self, mod):
        return types.BoundFunction(
            GroupMax, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_min(self, mod):
        return types.BoundFunction(
            GroupMin, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_size(self, mod):
        return types.BoundFunction(
            GroupSize, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_count(self, mod):
        return types.BoundFunction(
            GroupCount, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_sum(self, mod):
        return types.BoundFunction(
            GroupSum, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_mean(self, mod):
        return types.BoundFunction(
            GroupMean, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_std(self, mod):
        return types.BoundFunction(
            GroupStd, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_var(self, mod):
        return types.BoundFunction(
            GroupVar, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_idxmax(self, mod):
        return types.BoundFunction(
            GroupIdxMax, GroupType(mod.group_scalar_type, mod.index_type)
        )

    def resolve_idxmin(self, mod):
        return types.BoundFunction(
            GroupIdxMin, GroupType(mod.group_scalar_type, mod.index_type)
        )


@cuda_lower("GroupType.max", GroupType(types.int32))
@cuda_lower("GroupType.max", GroupType(types.int64))
@cuda_lower("GroupType.max", GroupType(types.float64))
def cuda_Group_max(context, builder, sig, args):
    retty = sig.return_type

    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    grp_type = sig.args[0]

    group_dataty = grp_type.group_data_type
    group_data_ptr = builder.alloca(grp.group_data.type)
    builder.store(grp.group_data, group_data_ptr)

    if grp_type.group_scalar_type == types.int32:
        func = call_my_max_int32
    elif grp_type.group_scalar_type == types.int64:
        func = call_my_max_int64
    elif grp_type.group_scalar_type == types.float64:
        func = call_my_max_float64

    result = context.compile_internal(
        builder,
        func,
        nb_signature(retty, group_dataty, grp_type.size_type),
        (builder.load(group_data_ptr), grp.size),
    )

    return result


@cuda_lower("GroupType.min", GroupType(types.int32))
@cuda_lower("GroupType.min", GroupType(types.int64))
@cuda_lower("GroupType.min", GroupType(types.float64))
def cuda_Group_min(context, builder, sig, args):
    retty = sig.return_type

    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    grp_type = sig.args[0]

    group_dataty = grp_type.group_data_type
    group_data_ptr = builder.alloca(grp.group_data.type)
    builder.store(grp.group_data, group_data_ptr)

    if grp_type.group_scalar_type == types.int32:
        func = call_my_min_int32
    elif grp_type.group_scalar_type == types.int64:
        func = call_my_min_int64
    elif grp_type.group_scalar_type == types.float64:
        func = call_my_min_float64

    result = context.compile_internal(
        builder,
        func,
        nb_signature(retty, group_dataty, grp_type.size_type),
        (builder.load(group_data_ptr), grp.size),
    )
    return result


@cuda_lower("GroupType.size", GroupType(types.int64))
@cuda_lower("GroupType.size", GroupType(types.float64))
def cuda_Group_size(context, builder, sig, args):
    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    result = grp.size
    return result


@cuda_lower("GroupType.count", GroupType(types.int64))
@cuda_lower("GroupType.count", GroupType(types.float64))
def cuda_Group_count(context, builder, sig, args):
    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    result = grp.size
    return result


@cuda_lower("GroupType.sum", GroupType(types.int64))
@cuda_lower("GroupType.sum", GroupType(types.float64))
def cuda_Group_sum(context, builder, sig, args):
    retty = sig.return_type

    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    grp_type = sig.args[0]

    group_dataty = grp_type.group_data_type
    group_data_ptr = builder.alloca(grp.group_data.type)
    builder.store(grp.group_data, group_data_ptr)

    if grp_type.group_scalar_type == types.int64:
        func = call_my_sum_int64
    elif grp_type.group_scalar_type == types.float64:
        func = call_my_sum_float64

    result = context.compile_internal(
        builder,
        func,
        nb_signature(retty, group_dataty, grp_type.size_type),
        (builder.load(group_data_ptr), grp.size),
    )
    return result


@cuda_lower("GroupType.mean", GroupType(types.int64))
@cuda_lower("GroupType.mean", GroupType(types.float64))
def cuda_Group_mean(context, builder, sig, args):
    retty = sig.return_type

    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    grp_type = sig.args[0]

    group_dataty = grp_type.group_data_type
    group_data_ptr = builder.alloca(grp.group_data.type)
    builder.store(grp.group_data, group_data_ptr)

    if grp_type.group_scalar_type == types.int64:
        func = call_my_mean_int64
    elif grp_type.group_scalar_type == types.float64:
        func = call_my_mean_float64

    result = context.compile_internal(
        builder,
        func,
        nb_signature(retty, group_dataty, grp_type.size_type),
        (builder.load(group_data_ptr), grp.size),
    )
    return result


@cuda_lower("GroupType.std", GroupType(types.int64))
@cuda_lower("GroupType.std", GroupType(types.float64))
def cuda_Group_std(context, builder, sig, args):
    retty = sig.return_type

    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    grp_type = sig.args[0]

    group_dataty = grp_type.group_data_type
    group_data_ptr = builder.alloca(grp.group_data.type)
    builder.store(grp.group_data, group_data_ptr)

    if grp_type.group_scalar_type == types.int64:
        func = call_my_std_int64
    elif grp_type.group_scalar_type == types.float64:
        func = call_my_std_float64

    result = context.compile_internal(
        builder,
        func,
        nb_signature(retty, group_dataty, grp_type.size_type),
        (builder.load(group_data_ptr), grp.size),
    )
    return result


@cuda_lower("GroupType.var", GroupType(types.int64))
@cuda_lower("GroupType.var", GroupType(types.float64))
def cuda_Group_var(context, builder, sig, args):
    retty = sig.return_type

    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    grp_type = sig.args[0]

    group_dataty = grp_type.group_data_type
    group_data_ptr = builder.alloca(grp.group_data.type)
    builder.store(grp.group_data, group_data_ptr)

    if grp_type.group_scalar_type == types.int64:
        func = call_my_var_int64
    elif grp_type.group_scalar_type == types.float64:
        func = call_my_var_float64

    result = context.compile_internal(
        builder,
        func,
        nb_signature(retty, group_dataty, grp_type.size_type),
        (builder.load(group_data_ptr), grp.size),
    )
    return result


@cuda_lower("GroupType.idxmax", GroupType(types.int64, types.int64))
@cuda_lower("GroupType.idxmax", GroupType(types.float64, types.int64))
def cuda_Group_idxmax(context, builder, sig, args):
    retty = sig.return_type

    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    grp_type = sig.args[0]

    group_dataty = grp_type.group_data_type
    group_data_ptr = builder.alloca(grp.group_data.type)
    builder.store(grp.group_data, group_data_ptr)

    index_dataty = grp_type.group_index_type
    index_ptr = builder.alloca(grp.index.type)
    builder.store(grp.index, index_ptr)

    if grp_type.group_scalar_type == types.int64:
        func = call_my_idxmax_int64
    elif grp_type.group_scalar_type == types.float64:
        func = call_my_idxmax_float64

    result = context.compile_internal(
        builder,
        func,
        nb_signature(retty, group_dataty, index_dataty, grp_type.size_type),
        (builder.load(group_data_ptr), builder.load(index_ptr), grp.size),
    )
    return result


@cuda_lower("GroupType.idxmin", GroupType(types.int64, types.int64))
@cuda_lower("GroupType.idxmin", GroupType(types.float64, types.int64))
def cuda_Group_idxmin(context, builder, sig, args):
    retty = sig.return_type

    grp = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    grp_type = sig.args[0]

    group_dataty = grp_type.group_data_type
    group_data_ptr = builder.alloca(grp.group_data.type)
    builder.store(grp.group_data, group_data_ptr)

    index_dataty = grp_type.group_index_type
    index_ptr = builder.alloca(grp.index.type)
    builder.store(grp.index, index_ptr)

    if grp_type.group_scalar_type == types.int64:
        func = call_my_idxmin_int64
    elif grp_type.group_scalar_type == types.float64:
        func = call_my_idxmin_float64

    result = context.compile_internal(
        builder,
        func,
        nb_signature(retty, group_dataty, index_dataty, grp_type.size_type),
        (builder.load(group_data_ptr), builder.load(index_ptr), grp.size),
    )
    return result


def _get_frame_groupby_type(dtype, index_dtype):
    """
    Get the numba `Record` type corresponding to a frame.
    Models the column as a dictionary like data structure
    containing GroupTypes.
    Large parts of this function are copied with comments
    from the Numba internals and slightly modified to
    account for validity bools to be present in the final
    struct.
    See numba.np.numpy_support.from_struct_dtype for details.
    """

    # Create the numpy structured type corresponding to the numpy dtype.

    fields = []
    offset = 0

    sizes = [val[0].itemsize for val in dtype.fields.values()]
    for i, (name, info) in enumerate(dtype.fields.items()):
        # *info* consists of the element dtype, its offset from the beginning
        # of the record, and an optional "title" containing metadata.
        # We ignore the offset in info because its value assumes no masking;
        # instead, we compute the correct offset based on the masked type.
        elemdtype = info[0]
        title = info[2] if len(info) == 3 else None
        ty = numpy_support.from_dtype(elemdtype)
        indexty = numpy_support.from_dtype(index_dtype)
        infos = {
            "type": GroupType(ty, indexty),
            "offset": offset,
            "title": title,
        }
        fields.append((name, infos))

        # increment offset by itemsize plus one byte for validity
        offset += 8 + 8 + 8  # group struct size (2 pointers and 1 integer)

        # Align the next member of the struct to be a multiple of the
        # memory access size, per PTX ISA 7.4/5.4.5
        if i < len(sizes) - 1:
            # next_itemsize = sizes[i + 1]
            next_itemsize = 8
            offset = int(math.ceil(offset / next_itemsize) * next_itemsize)

    # Numba requires that structures are aligned for the CUDA target
    _is_aligned_struct = True
    return Record(fields, offset, _is_aligned_struct)


def _groupby_apply_kernel_string_from_template(frame, args):
    """
    Function to write numba kernels for `DataFrame.apply` as a string.
    Workaround until numba supports functions that use `*args`

    Both the number of input columns as well as their nullability and any
    scalar arguments may vary, so the kernels vary significantly. See
    templates.py for the full row kernel template and more details.
    """
    # Create argument list for kernel
    frame = _supported_cols_from_frame(frame)

    input_columns = ", ".join([f"input_col_{i}" for i in range(len(frame))])
    extra_args = ", ".join([f"extra_arg_{i}" for i in range(len(args))])

    # Generate the initializers for each device function argument
    initializers = []
    for i, (colname, col) in enumerate(frame.items()):
        idx = str(i)
        initializers.append(
            group_initializer_template.format(idx=idx, name=colname)
        )

    return groupby_apply_kernel_template.format(
        input_columns=input_columns,
        extra_args=extra_args,
        group_initializers="\n".join(initializers),
    )


def _get_groupby_apply_kernel(frame, func, args):
    dataframe_group_type = _get_frame_groupby_type(
        np.dtype(list(_all_dtypes_from_frame(frame).items())),
        frame.index.dtype,
    )

    return_type = _get_udf_return_type(dataframe_group_type, func, args)

    np_field_types = np.dtype(
        list(_supported_dtypes_from_frame(frame).items())
    )
    dataframe_group_type = _get_frame_groupby_type(
        np_field_types, frame.index.dtype
    )

    # Dict of 'local' variables into which `_kernel` is defined
    global_exec_context = {
        "cuda": cuda,
        "Group": Group,
        "dataframe_group_type": dataframe_group_type,
        "types": types,
    }
    kernel_string = _groupby_apply_kernel_string_from_template(frame, args)

    kernel = _get_kernel_groupby_apply(
        kernel_string, global_exec_context, func, dev_func_ptx
    )

    return kernel, return_type


@_cudf_nvtx_annotate
def jit_groupby_apply(offsets, grouped_values, function, *args, cache=True):
    ngroups = len(offsets) - 1

    if cache is True:
        kernel, return_type = _compile_or_get(
            grouped_values, function, args, _get_groupby_apply_kernel
        )
    else:
        kernel, return_type = _get_groupby_apply_kernel(
            grouped_values, function, args
        )
        return_type = numpy_support.as_dtype(return_type)

    output = cp.empty(ngroups, dtype=return_type)

    launch_args = [
        cp.asarray(offsets),
        output,
        cp.asarray(grouped_values.index),
    ]

    for col in _supported_cols_from_frame(grouped_values).values():
        launch_args.append(cp.asarray(col))

    launch_args += list(args)

    max_group_size = cp.diff(offsets).max()

    if max_group_size >= 1000:
        # if ngroups < 100:
        #     blocklim = 1024
        # else:
        blocklim = 256
    else:
        blocklim = ((max_group_size + 32 - 1) / 32) * 32

    if kernel.specialized:
        specialized = kernel
    else:
        specialized = kernel.specialize(*launch_args)

    # Ask the driver to give a good config
    ctx = get_context()
    # Dispatcher is specialized, so there's only one definition - get
    # it so we can get the cufunc from the code library
    kern_def = next(iter(specialized.overloads.values()))
    kwargs = dict(
        func=kern_def._codelibrary.get_cufunc(),
        b2d_func=0,
        memsize=0,
        blocksizelimit=blocklim,
    )
    _, tpb = ctx.get_max_potential_block_size(**kwargs)

    stream = cuda.default_stream()

    specialized[ngroups, tpb, stream](*launch_args)

    stream.synchronize()

    return as_column(output, dtype=output.dtype)
