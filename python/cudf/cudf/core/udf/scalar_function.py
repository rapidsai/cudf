# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from contextlib import nullcontext
from functools import cache

from numba import cuda
from numba.np import numpy_support

from cudf.core.dtype.validators import is_dtype_obj_string
from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.masked_typing import MaskedType
from cudf.core.udf.nrt_utils import CaptureNRTUsage, nrt_enabled
from cudf.core.udf.strings_typing import string_view
from cudf.core.udf.templates import (
    masked_input_initializer_template,
    scalar_kernel_template,
    unmasked_input_initializer_template,
)
from cudf.core.udf.udf_kernel_base import ApplyKernelBase
from cudf.core.udf.utils import (
    DEPRECATED_SM_REGEX,
    _mask_get,
)


def _compile_unmasked_series_apply_kernel(f_, sig, nrt):
    """Compile a source-backed, disk-cacheable Series.apply kernel."""
    ctx = nrt_enabled() if nrt else nullcontext()
    with ctx:
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            warnings.filterwarnings(
                "ignore",
                message=DEPRECATED_SM_REGEX,
                category=UserWarning,
                module=r"^numba\.cuda(\.|$)",
            )

            @cuda.jit(
                sig,
                cache=True,
            )
            def _kernel(retval, size, input_col_0, offset_0):
                i = cuda.grid(1)
                ret_data_arr, ret_mask_arr = retval

                if i < size:
                    masked_0 = Masked(input_col_0[i + offset_0], True)
                    ret_masked = pack_return(f_(masked_0))
                    ret_data_arr[i] = ret_masked.value
                    ret_mask_arr[i] = ret_masked.valid

    return _kernel


def _compile_masked_series_apply_kernel(f_, sig, nrt):
    """Compile a source-backed, disk-cacheable Series.apply kernel."""
    ctx = nrt_enabled() if nrt else nullcontext()
    with ctx:
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            warnings.filterwarnings(
                "ignore",
                message=DEPRECATED_SM_REGEX,
                category=UserWarning,
                module=r"^numba\.cuda(\.|$)",
            )

            @cuda.jit(
                sig,
                cache=True,
            )
            def _kernel(retval, size, input_col_0, offset_0):
                i = cuda.grid(1)
                ret_data_arr, ret_mask_arr = retval

                if i < size:
                    d_0, m_0 = input_col_0
                    masked_0 = Masked(
                        d_0[i + offset_0], _mask_get(m_0, i + offset_0)
                    )
                    ret_masked = pack_return(f_(masked_0))
                    ret_data_arr[i] = ret_masked.value
                    ret_mask_arr[i] = ret_masked.valid

    return _kernel


class SeriesApplyKernel(ApplyKernelBase):
    """
    Class representing a kernel that computes the result of
    a Series.apply operation. Expects that the user passed
    a function that operates on an single element of the Series,
    for example

    def f(x):
        return x + 1
    """

    @property
    def kernel_type(self):
        return "series_apply"

    def _get_frame_type(self):
        return MaskedType(
            string_view
            if is_dtype_obj_string(self.frame.dtype)
            else numpy_support.from_dtype(self.frame.dtype)
        )

    def _get_kernel_string(self):
        extra_args = ", ".join(
            [f"extra_arg_{i}" for i in range(len(self.args))]
        )

        masked_initializer = (
            masked_input_initializer_template
            if self.frame._column.mask
            else unmasked_input_initializer_template
        ).format(idx=0)

        return scalar_kernel_template.format(
            extra_args=extra_args, masked_initializer=masked_initializer
        )

    def compile_kernel(self):
        # The static no-argument kernels below have a real source location, so
        # Numba can persist their compiled specializations. Other signatures
        # keep the existing generated-kernel path for this prototype.
        if self.args or is_dtype_obj_string(self.frame.dtype):
            return super().compile_kernel()

        capture_nrt_usage = CaptureNRTUsage()
        with capture_nrt_usage:
            return_type = self._get_udf_return_type()

        # String allocation uses symbols from UDF_SHIM_FILE, which Numba's
        # on-disk cache cannot serialize. Keep the existing linked path for
        # those specializations.
        if capture_nrt_usage.use_nrt:
            return super().compile_kernel()

        self.sig = self._construct_signature(return_type)
        kernel_factory = (
            _compile_masked_series_apply_kernel
            if self.frame._column.mask is not None
            else _compile_unmasked_series_apply_kernel
        )
        kernel = kernel_factory(
            self.device_func, self.sig, capture_nrt_usage.use_nrt
        )
        return kernel, return_type

    @cache
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
