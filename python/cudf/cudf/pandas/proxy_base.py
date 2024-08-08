# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np


class ProxyNDarrayBase(np.ndarray):
    def __new__(cls, arr):
        if isinstance(arr, cp.ndarray):
            obj = np.asarray(arr.get()).view(cls)
            return obj
        elif isinstance(arr, np.ndarray):
            obj = np.asarray(arr).view(cls)
            return obj
        else:
            raise TypeError(
                "Unsupported array type. Must be numpy.ndarray or cupy.ndarray"
            )

    def __array_finalize__(self, obj):
        self._fsproxy_wrapped = getattr(obj, "_fsproxy_wrapped", None)

    def __array_ufunc__(self, *args, **kwargs):
        args = (args[0], args[1], np.asarray(args[2]), np.asarray(args[3]))
        return super().__array_ufunc__(*args, **kwargs)
