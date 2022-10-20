# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf.api.extensions.accessor import (
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)

__all__ = [
    "register_dataframe_accessor",
    "register_index_accessor",
    "register_series_accessor",
]
