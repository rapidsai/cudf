# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

numpy_asarray = np.asarray


class ProxyNDarrayBase(np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        obj = super().__new__(
            cls, shape=input_array.shape, dtype=input_array.dtype
        )
        view = numpy_asarray(obj, dtype=input_array.dtype).view(cls)

        return view

    def __array_finalize__(self, obj):
        if obj is None:
            return
        for attr in list(self.__dict__.keys()):
            if attr not in {"__new__", "__array_finalize__"}:
                delattr(self, attr)
        self._fsproxy_wrapped = obj
