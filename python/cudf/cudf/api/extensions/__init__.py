# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from cudf.api.extensions.accessor import (
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)
from pandas.api.extensions import no_default

__all__ = [
    "no_default",
    "register_dataframe_accessor",
    "register_index_accessor",
    "register_series_accessor",
]
