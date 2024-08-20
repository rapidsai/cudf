# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

numpy_asarray = np.asarray


class ProxyNDarrayBase(np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        if isinstance(input_array, np.ndarray):
            obj = input_array
        else:
            obj = super().__new__(
                cls, shape=input_array.shape, dtype=input_array.dtype
            )
        view = numpy_asarray(obj).view(cls)
        view._fsproxy_wrapped = input_array

        return view
