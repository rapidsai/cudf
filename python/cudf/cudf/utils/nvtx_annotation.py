# Copyright (c) 2023, NVIDIA CORPORATION.

import hashlib
from functools import partial

from nvtx import annotate

_NVTX_COLORS = ["green", "blue", "purple", "rapids"]


def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


def _cudf_nvtx_annotate(func, domain="cudf_python"):
    """Decorator for applying nvtx annotations to methods in cudf."""
    return annotate(
        message=func.__qualname__,
        color=_get_color_for_nvtx(func.__qualname__),
        domain=domain,
    )(func)


_dask_cudf_nvtx_annotate = partial(
    _cudf_nvtx_annotate, domain="dask_cudf_python"
)
