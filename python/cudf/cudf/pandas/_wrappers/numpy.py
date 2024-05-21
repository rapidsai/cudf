# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cupy
import cupy._core.flags
import numpy
import numpy.core.multiarray

from ..fast_slow_proxy import (
    _FastSlowAttribute,
    make_final_proxy_type,
    make_intermediate_proxy_type,
)
from .common import (
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
        return arr.dtype.type(arr.item())
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


ndarray = make_final_proxy_type(
    "ndarray",
    cupy.ndarray,
    numpy.ndarray,
    fast_to_slow=cupy.ndarray.get,
    slow_to_fast=cupy.asarray,
    additional_attributes={
        "__array__": array_method,
        # So that pa.array(wrapped-numpy-array) works
        "__arrow_array__": arrow_array_method,
        "__cuda_array_interface__": cuda_array_interface,
        # ndarrays are unhashable
        "__hash__": None,
        # iter(cupy-array) produces an iterable of zero-dim device
        # arrays, which is not usable in many settings (whereas
        # iter(numpy-array) produces an iterable of scalars)
        "__iter__": custom_iter,
        # Special wrapping to handle scalar values
        "_fsproxy_wrap": classmethod(wrap_ndarray),
        "base": _FastSlowAttribute("base", private=True),
    },
)

# Mapping flags between slow and fast types
_ndarray_flags = make_intermediate_proxy_type(
    "_ndarray_flags",
    cupy._core.flags.Flags,
    numpy.core.multiarray.flagsobj,
)
