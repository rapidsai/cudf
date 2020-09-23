# Copyright (c) 2020, NVIDIA CORPORATION.


class AccessorManager:
    def __init__(self, name, accessor_cls):
        self.name = name
        self.accessor_cls = accessor_cls

    def __get__(self, obj, type=None):
        # Accessing accessor on class
        if obj is None:
            return self.accessor_cls

        # First time call, initialize
        accessor_obj = self.accessor_cls(obj)

        # Overwrites obj.accessor with initialized accessor object
        object.__setattr__(obj, self.name, accessor_obj)
        return accessor_obj


def _register_accessor(name, cls):
    def decorator(accessor):
        if hasattr(cls, name):
            Warning.warning(f"{name} will be overidden in {cls.__name__}")
        manager = AccessorManager(name, accessor)
        setattr(cls, name, manager)

    return decorator


def register_dataframe_accessor(name):
    from cudf import DataFrame

    return _register_accessor(name, DataFrame)
