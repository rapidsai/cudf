# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf import DataFrame

class accessor_wrapper:
    def __init__(self, accessor_cls):
        self.accessor_cls = accessor_cls
    
    def __get__(self, obj, type=None):
        if self.accessor is None:
            self.accessor = self.accessor_cls(obj)
        return self.accessor

def _register_accessor(name, cls):
    def decorator(accessor):
        if hasattr(cls, name):
            Warning.warning(f"{name} will be overidden in {cls.__name__}")
        wrapper = accessor_wrapper(accessor)
        setattr(cls, name, wrapper)
    
    return decorator

def register_dataframe_accessor(name):
    return _register_accessor(name, DataFrame)
