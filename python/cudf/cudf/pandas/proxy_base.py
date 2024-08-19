# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


class ProxyNDarrayBase(np.ndarray):
    def __new__(cls, input_array, *args, **kwargs):
        obj = super().__new__(cls, shape=(0,))
        obj._fsproxy_wrapped = input_array
        return np.asarray(obj).view(cls)
