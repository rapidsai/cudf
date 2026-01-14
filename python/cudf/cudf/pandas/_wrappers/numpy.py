# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cupy
import cupy._core.flags
import numpy
from packaging import version

from cudf.options import _env_get_bool

from ..fast_slow_proxy import (
    _fast_arg,
    _fast_slow_function_call,
    _FastSlowAttribute,
    _maybe_wrap_result,
    _raise_fallback_error,
    _slow_arg,
    is_proxy_object,
    make_final_proxy_type,
    make_intermediate_proxy_type,
)
from ..proxy_base import ProxyNDarrayBase
from .common import (
    array_interface,
    array_method,
    arrow_array_method,
    cuda_array_interface,
    custom_iter,
)

# https://docs.cupy.dev/en/stable/reference/creation.html
_CONSTRUCTORS = frozenset(
    [
        cupy.empty,
        cupy.empty_like,
        cupy.eye,
        cupy.identity,
        cupy.ones,
        cupy.ones_like,
        cupy.zeros,
        cupy.zeros_like,
        cupy.full,
        cupy.full_like,
        cupy.array,
        cupy.asarray,
        cupy.asanyarray,
        cupy.ascontiguousarray,
        cupy.copy,
        cupy.frombuffer,
        cupy.fromfile,
        cupy.fromfunction,
        cupy.fromiter,
        cupy.fromstring,
        cupy.loadtxt,
        cupy.arange,
        cupy.linspace,
        cupy.logspace,
        cupy.meshgrid,
        cupy.diag,
        cupy.diagflat,
        cupy.tri,
        cupy.tril,
        cupy.triu,
        cupy.vander,
    ]
)


def wrap_ndarray(cls, arr: cupy.ndarray | numpy.ndarray, constructor):
    """Wrap an ndarray in a proxy type

    Parameters
    ----------
    cls
        Proxy type for ndarray
    arr
        Concrete result ndarray (cupy or numpy)
    constructor
        Function that was called to construct the concrete array, used
        to check against a denylist to avoid unwrapping.

    Returns
    -------
    The scalar .item() wrapped in its numpy dtype if arr is a
    zero-dimensional cupy array (and wasn't just constructed as such),
    a new proxy type otherwise.

    Notes
    -----
    Axis-reducing operations in numpy return scalar objects but
    zero-dimensional arrays in cupy. This confuses downstream
    libraries when they get a fast (device-based) zero-dim array when
    they were expecting a scalar. To avoid this, if the provided array
    is a cupy array, and its shape is zero, unwrap it.
    """
    if (
        isinstance(arr, cupy.ndarray)
        and arr.shape == ()
        and constructor not in _CONSTRUCTORS
    ):
        value = arr.item()
        if arr.dtype.kind in "mM":
            unit, _ = numpy.datetime_data(arr.dtype)
            return arr.dtype.type(value, unit)
        return arr.dtype.type(value)
    else:
        # Note, this super call means that the constructed ndarray
        # class cannot be subclassed (because then super(cls,
        # cls)._fsproxy_wrap produces an infinite loop). Really this
        # should be super(ndarray, cls), but we don't have access to
        # the ndarray type until after we need to pass this function
        # in. So it works for now since without subclassing,
        # super(ndarray, cls) == super(ndarray, ndarray) == super(cls,
        # cls)
        return super(cls, cls)._fsproxy_wrap(arr, constructor)


def ndarray__array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    result, _ = _fast_slow_function_call(
        getattr(ufunc, method),
        None,
        *inputs,
        **kwargs,
    )
    if isinstance(result, tuple):
        if is_proxy_object(result[0]) and isinstance(
            result[0]._fsproxy_wrapped, numpy.ndarray
        ):
            return tuple(numpy.asarray(x) for x in result)
    elif is_proxy_object(result) and isinstance(
        result._fsproxy_wrapped, numpy.ndarray
    ):
        return numpy.asarray(result)
    return result


def ndarray__reduce__(self):
    # As it stands the custom pickling logic used for all other
    # proxy types is incompatible with our proxy ndarray. The pickle
    # constructor we use to deserialize the other proxy types calls
    # object.__new__(type) which you cannot call on subclasses of
    # numpy arrays because the new array won't be created with numpy's
    # specific memory management logic. Therefore, we have to handle
    # serialization separately for proxy arrays.
    return (
        ndarray.__new__,
        (
            ndarray,
            self._fsproxy_wrapped,
        ),
    )


def _is_cupy_backed_and_non_datetime_array(x) -> bool:
    if is_proxy_object(x):
        x = x._fsproxy_wrapped
    return (
        isinstance(x, cupy.ndarray)
        and x.dtype is not None
        and x.dtype.kind not in ("M", "m")
    )


def ndarray__array_function__(self, func, types, args, kwargs):
    name = func.__name__
    try:
        cupy_func = getattr(cupy, name)
    except AttributeError:
        if getattr(func, "__module__", "").startswith("numpy.linalg"):
            cupy_func = getattr(cupy.linalg, name, None)
        else:
            cupy_func = None

    if cupy_func is not None and all(
        _is_cupy_backed_and_non_datetime_array(a) for a in args
    ):
        fast_args, fast_kwargs = _fast_arg(args), _fast_arg(kwargs)
        if name == "fft":
            cupy_func = cupy_func.fft
        try:
            res = cupy_func(*fast_args, **fast_kwargs)
        except Exception as err:
            slow_args, slow_kwargs = _slow_arg(args), _slow_arg(kwargs)
            if _env_get_bool("CUDF_PANDAS_FAIL_ON_FALLBACK", False):
                _raise_fallback_error(err, slow_args[0].__name__)
            res = func(*slow_args, **slow_kwargs)
        return _maybe_wrap_result(res, func, *args, **kwargs)

    slow_args, slow_kwargs = _slow_arg(args), _slow_arg(kwargs)
    return _maybe_wrap_result(
        func(*slow_args, **slow_kwargs), func, *args, **kwargs
    )


ndarray = make_final_proxy_type(
    "ndarray",
    cupy.ndarray,
    numpy.ndarray,
    fast_to_slow=cupy.ndarray.get,
    slow_to_fast=cupy.asarray,
    bases=(ProxyNDarrayBase,),
    additional_attributes={
        "__array__": array_method,
        "__array_function__": ndarray__array_function__,
        # So that pa.array(wrapped-numpy-array) works
        "__arrow_array__": arrow_array_method,
        "__cuda_array_interface__": cuda_array_interface,
        "__array_interface__": array_interface,
        "__array_ufunc__": ndarray__array_ufunc__,
        "__reduce__": ndarray__reduce__,
        # ndarrays are unhashable
        "__hash__": None,
        # iter(cupy-array) produces an iterable of zero-dim device
        # arrays, which is not usable in many settings (whereas
        # iter(numpy-array) produces an iterable of scalars)
        "__iter__": custom_iter,
        # Special wrapping to handle scalar values
        "_fsproxy_wrap": classmethod(wrap_ndarray),
        "base": _FastSlowAttribute("base", private=True),
        "data": _FastSlowAttribute("data", private=True),
    },
)


flatiter = make_final_proxy_type(
    "flatiter",
    cupy.flatiter,
    numpy.flatiter,
    fast_to_slow=lambda fast: cupy.asnumpy(fast.base).flat,
    slow_to_fast=lambda slow: cupy.asarray(slow).flat,
    additional_attributes={
        "__array__": array_method,
    },
)

if version.parse(numpy.__version__) >= version.parse("2.0"):
    # NumPy 2 introduced `_core` and gives warnings for access to `core`.
    from numpy._core.multiarray import flagsobj as _numpy_flagsobj
else:
    from numpy.core.multiarray import (  # type: ignore[no-redef]
        flagsobj as _numpy_flagsobj,
    )

# Mapping flags between slow and fast types
_ndarray_flags = make_intermediate_proxy_type(
    "_ndarray_flags",
    cupy._core.flags.Flags,
    _numpy_flagsobj,
)
