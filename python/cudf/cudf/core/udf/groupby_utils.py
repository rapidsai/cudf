# Copyright (c) 2022-2023, NVIDIA CORPORATION.


import cupy as cp
import numpy as np
from numba import cuda, types
from numba.cuda.cudadrv.devices import get_context
from numba.np import numpy_support
from numba.types import Record

import cudf.core.udf.utils
from cudf.core.udf.groupby_typing import (
    SUPPORTED_GROUPBY_NUMPY_TYPES,
    Group,
    GroupType,
)
from cudf.core.udf.templates import (
    group_initializer_template,
    groupby_apply_kernel_template,
)
from cudf.core.udf.utils import (
    _get_extensionty_size,
    _get_kernel,
    _get_udf_return_type,
    _supported_cols_from_frame,
    _supported_dtypes_from_frame,
)
from cudf.utils.utils import _cudf_nvtx_annotate


def _get_frame_groupby_type(dtype, index_dtype):
    """
    Get the numba `Record` type corresponding to a frame.
    Models the column as a dictionary like data structure
    containing GroupTypes.
    See numba.np.numpy_support.from_struct_dtype for details.

    Parameters
    ----------
    level : np.dtype
        A numpy structured array dtype associating field names
        to scalar dtypes
    index_dtype : np.dtype
        A numpy scalar dtype associated with the index of the
        incoming grouped data
    """
    # Create the numpy structured type corresponding to the numpy dtype.
    fields = []
    offset = 0

    sizes = [val[0].itemsize for val in dtype.fields.values()]
    for i, (name, info) in enumerate(dtype.fields.items()):
        elemdtype = info[0]
        title = info[2] if len(info) == 3 else None
        ty = numpy_support.from_dtype(elemdtype)
        indexty = numpy_support.from_dtype(index_dtype)
        groupty = GroupType(ty, indexty)
        infos = {
            "type": groupty,
            "offset": offset,
            "title": title,
        }
        fields.append((name, infos))
        offset += _get_extensionty_size(groupty)

        # Align the next member of the struct to be a multiple of the
        # memory access size, per PTX ISA 7.4/5.4.5
        if i < len(sizes) - 1:
            alignment = offset % 8
            if alignment != 0:
                offset += 8 - alignment

    # Numba requires that structures are aligned for the CUDA target
    _is_aligned_struct = True
    return Record(fields, offset, _is_aligned_struct)


def _groupby_apply_kernel_string_from_template(frame, args):
    """
    Function to write numba kernels for `Groupby.apply` as a string.
    Workaround until numba supports functions that use `*args`
    """
    # Create argument list for kernel
    frame = _supported_cols_from_frame(
        frame, supported_types=SUPPORTED_GROUPBY_NUMPY_TYPES
    )
    input_columns = ", ".join([f"input_col_{i}" for i in range(len(frame))])
    extra_args = ", ".join([f"extra_arg_{i}" for i in range(len(args))])

    # Generate the initializers for each device function argument
    initializers = []
    for i, colname in enumerate(frame.keys()):
        initializers.append(
            group_initializer_template.format(idx=i, name=colname)
        )

    return groupby_apply_kernel_template.format(
        input_columns=input_columns,
        extra_args=extra_args,
        group_initializers="\n".join(initializers),
    )


def _get_groupby_apply_kernel(frame, func, args):
    np_field_types = np.dtype(
        list(
            _supported_dtypes_from_frame(
                frame, supported_types=SUPPORTED_GROUPBY_NUMPY_TYPES
            ).items()
        )
    )
    dataframe_group_type = _get_frame_groupby_type(
        np_field_types, frame.index.dtype
    )
    return_type = _get_udf_return_type(dataframe_group_type, func, args)

    # Dict of 'local' variables into which `_kernel` is defined
    global_exec_context = {
        "cuda": cuda,
        "Group": Group,
        "dataframe_group_type": dataframe_group_type,
        "types": types,
    }
    kernel_string = _groupby_apply_kernel_string_from_template(frame, args)

    kernel = _get_kernel(kernel_string, global_exec_context, None, func)

    return kernel, return_type


@_cudf_nvtx_annotate
def jit_groupby_apply(offsets, grouped_values, function, *args):
    """
    Main entrypoint for JIT Groupby.apply via Numba.

    Parameters
    ----------
    offsets : list
        A list of integers denoting the indices of the group
        boundaries in grouped_values
    grouped_values : DataFrame
        A DataFrame representing the source data
        sorted by group keys
    function : callable
        The user-defined function to execute
    """
    offsets = cp.asarray(offsets)
    ngroups = len(offsets) - 1

    kernel, return_type = _get_groupby_apply_kernel(
        grouped_values, function, args
    )
    return_type = numpy_support.as_dtype(return_type)

    output = cudf.core.column.column_empty(ngroups, dtype=return_type)
    launch_args = [
        offsets,
        output,
        grouped_values.index,
    ]
    launch_args += list(
        _supported_cols_from_frame(
            grouped_values, supported_types=SUPPORTED_GROUPBY_NUMPY_TYPES
        ).values()
    )
    launch_args += list(args)

    max_group_size = cp.diff(offsets).max()

    if max_group_size >= 256:
        blocklim = 256
    else:
        blocklim = ((max_group_size + 32 - 1) // 32) * 32

    if kernel.specialized:
        specialized = kernel
    else:
        specialized = kernel.specialize(*launch_args)

    # Ask the driver to give a good config
    ctx = get_context()
    # Dispatcher is specialized, so there's only one definition - get
    # it so we can get the cufunc from the code library
    (kern_def,) = specialized.overloads.values()
    grid, tpb = ctx.get_max_potential_block_size(
        func=kern_def._codelibrary.get_cufunc(),
        b2d_func=0,
        memsize=0,
        blocksizelimit=int(blocklim),
    )

    # Launch kernel
    specialized[ngroups, tpb](*launch_args)

    return output
