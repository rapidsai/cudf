# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
from pandas.core.accessor import CachedAccessor


def _register_accessor(name, cls):
    def decorator(accessor):
        if hasattr(cls, name):
            Warning.warning(f"{name} will be overidden in {cls.__name__}")
        cached_accessor = CachedAccessor(name, accessor)
        cls._accessors.add(name)
        setattr(cls, name, cached_accessor)

        return accessor

    return decorator


def register_dataframe_accessor(name):
    return _register_accessor(name, cudf.DataFrame)


def register_index_accessor(name):
    return _register_accessor(name, cudf.Index)


def register_series_accessor(name):
    return _register_accessor(name, cudf.Series)
