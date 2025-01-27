# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Utility custom overrides for special methods/properties
from ..fast_slow_proxy import (
    _FastSlowAttribute,
    _FastSlowProxy,
    _maybe_wrap_result,
    _slow_arg,
)


def array_method(self: _FastSlowProxy, *args, **kwargs):
    return self._fsproxy_slow.__array__(*args, **kwargs)


def array_function_method(self, func, types, args, kwargs):
    try:
        return _FastSlowAttribute("__array_function__").__get__(
            self, type(self)
        )(func, types, args, kwargs)
    except Exception:
        # if something went wrong with __array_function__ we
        # attempt to call the function directly on the slow
        # object.  This ensures that the function call is
        # handled in the same way as if the slow object was
        # passed directly to the function.
        slow_args, slow_kwargs = _slow_arg(args), _slow_arg(kwargs)
        return _maybe_wrap_result(
            func(*slow_args, **slow_kwargs), func, *args, **kwargs
        )


def arrow_array_method(self: _FastSlowProxy, *args, **kwargs):
    import pyarrow as pa

    try:
        return self._fsproxy_fast.to_arrow(*args, **kwargs)
    except Exception:
        return pa.array(self._fsproxy_slow, *args, **kwargs)


@property  # type: ignore
def cuda_array_interface(self: _FastSlowProxy):
    return self._fsproxy_fast.__cuda_array_interface__


@property  # type: ignore
def array_interface(self: _FastSlowProxy):
    return self._fsproxy_slow.__array_interface__


def custom_iter(self: _FastSlowProxy):
    """
    Custom iter method to handle the case where only the slow
    object's iter method is used.
    """
    # NOTE: Do not remove this method. This is required to avoid
    # falling back to GPU for iter method.
    return _maybe_wrap_result(
        iter(self._fsproxy_slow),
        None,  # type: ignore
    )
