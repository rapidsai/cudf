# Copyright (c) 2020-2025, NVIDIA CORPORATION.

from numba import cuda
from numba.np import numpy_support

from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.masked_typing import MaskedType
from cudf.core.udf.strings_typing import string_view
from cudf.core.udf.templates import (
    masked_input_initializer_template,
    scalar_kernel_template,
    unmasked_input_initializer_template,
)
from cudf.core.udf.udf_kernel_base import ApplyKernelBase
from cudf.core.udf.utils import (
    _mask_get,
)


def _scalar_kernel_string_from_template(sr, args):
    """
    Function to write numba kernels for `Series.apply` as a string.
    Workaround until numba supports functions that use `*args`

    `Series.apply` expects functions of a single variable and possibly
    one or more constants, such as:

    def f(x, c, k):
        return (x + c) / k

    where the `x` are meant to be the values of the series. Since there
    can be only one column, the only thing that varies in the kinds of
    kernels that we want is the number of extra_args. See templates.py
    for the full kernel template.
    """
    extra_args = ", ".join([f"extra_arg_{i}" for i in range(len(args))])

    masked_initializer = (
        masked_input_initializer_template
        if sr._column.mask
        else unmasked_input_initializer_template
    ).format(idx=0)

    return scalar_kernel_template.format(
        extra_args=extra_args, masked_initializer=masked_initializer
    )


class SeriesApplyKernel(ApplyKernelBase):
    @property
    def kernel_type(self):
        return "series_apply"

    def _get_frame_type(self):
        return MaskedType(
            string_view
            if self.frame.dtype == "O"
            else numpy_support.from_dtype(self.frame.dtype)
        )

    def _get_kernel_string(self):
        # This is the kernel string that will be compiled
        # It is generated from the template in templates.py
        # and is specific to the function being compiled
        return _scalar_kernel_string_from_template(self.frame, self.args)

    def _get_kernel_string_exec_context(self):
        # This is the global execution context that will be used
        # to compile the kernel. It contains the function being
        # compiled and the cuda module.
        return {
            "cuda": cuda,
            "Masked": Masked,
            "_mask_get": _mask_get,
            "pack_return": pack_return,
        }
