# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Utility custom overrides for special methods/properties
from ..fast_slow_proxy import _FastSlowProxy


def array_method(self: _FastSlowProxy, *args, **kwargs):
    return self._xdf_slow.__array__(*args, **kwargs)


def arrow_array_method(self: _FastSlowProxy, *args, **kwargs):
    import pyarrow as pa

    try:
        return self._xdf_fast.to_arrow(*args, **kwargs)
    except Exception:
        return pa.array(self._xdf_slow, *args, **kwargs)


@property  # type: ignore
def cuda_array_interface(self: _FastSlowProxy):
    return self._xdf_fast.__cuda_array_interface__


def custom_iter(self: _FastSlowProxy):
    return iter(self._xdf_slow)
