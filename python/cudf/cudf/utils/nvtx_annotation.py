# Copyright (c) 2023-2024, NVIDIA CORPORATION.


from functools import partial

from cudf.utils.performance_tracking import _performance_tracking

_cudf_nvtx_annotate = _performance_tracking


_dask_cudf_nvtx_annotate = partial(
    _cudf_nvtx_annotate, domain="dask_cudf_python"
)
