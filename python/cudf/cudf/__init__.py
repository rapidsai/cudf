# Copyright (c) 2018-2019, NVIDIA CORPORATION.

from librmm_cffi import librmm as _rmm, librmm_config as _rmm_cfg

from cudf import core, datasets
from cudf._version import get_versions
from cudf.core import DataFrame, Index, MultiIndex, Series, from_pandas, merge
from cudf.core.ops import (
    arccos,
    arcsin,
    arctan,
    cos,
    exp,
    log,
    logical_and,
    logical_not,
    logical_or,
    sin,
    sqrt,
    tan,
)
from cudf.core.reshape import concat, get_dummies, melt
from cudf.io import (
    from_dlpack,
    read_avro,
    read_csv,
    read_feather,
    read_hdf,
    read_json,
    read_orc,
    read_parquet,
)
from cudf.utils.utils import initfunc as _initfunc


def _set_rmm_config(
    use_managed_memory=False,
    use_pool_allocator=False,
    initial_pool_size=None,
    enable_logging=False,
):
    """
    Parameters
    ----------
    use_managed_memory : bool, optional
        If ``True``, use cudaMallocManaged as underlying allocator.
        If ``False`` (default), use  cudaMalloc.
    use_pool_allocator : bool
        If ``True``, enable pool mode.
        If ``False`` (default), disable pool mode.
    initial_pool_size : int, optional
        If ``use_pool_allocator=True``, sets initial pool size.
        If ``None``, us
es 1/2 of total GPU memory.
    enable_logging : bool, optional
        Enable logging (default ``False``).
        Enabling this option will introduce performance overhead.
    """
    _rmm.finalize()
    _rmm_cfg.use_managed_memory = use_managed_memory
    if use_pool_allocator:
        _rmm_cfg.use_pool_allocator = use_pool_allocator
        if initial_pool_size is None:
            initial_pool_size = 0  # 0 means 1/2 GPU memory
        elif initial_pool_size == 0:
            initial_pool_size = 1  # Since "0" is semantic value, use 1 byte
        if not isinstance(initial_pool_size, int):
            raise TypeError("initial_pool_size must be an integer")
        _rmm_cfg.initial_pool_size = initial_pool_size
    _rmm_cfg.enable_logging = enable_logging
    _rmm.initialize()


@_initfunc
def set_allocator(allocator="default", pool=False, initial_pool_size=None):
    """
    allocator : {"default", "managed"}
        ``"default"`` : use default allocator.
        ``"managed"`: use managed memory allocator.
    pool : bool
        Enable memory pool.
    initial_pool_size : int
        Memory pool size in bytes. If ``None`` (default), 1/2 of total
        GPU memory is used. If ``pool=False``, this argument is ignored.
    """
    use_managed_memory = True if allocator == "managed" else False
    _set_rmm_config(use_managed_memory, pool, initial_pool_size)


__version__ = get_versions()["version"]
del get_versions
