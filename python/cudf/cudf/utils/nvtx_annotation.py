# Copyright (c) 2023-2024, NVIDIA CORPORATION.


from cudf.utils.performance_tracking import (
    _dask_cudf_performance_tracking,
    _performance_tracking,
)

# TODO: will remove this file and use _performance_tracking before merging
_cudf_nvtx_annotate = _performance_tracking
_dask_cudf_nvtx_annotate = _dask_cudf_performance_tracking
