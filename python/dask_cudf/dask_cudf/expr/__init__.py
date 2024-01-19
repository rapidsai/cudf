# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_expr._expr import Expr

from dask.utils import Dispatch

__ext_dispatch_classes = {}  # Track registered "external" dispatch classes


def register_dispatch(cls, meta_type, ext_cls=None):
    """Register an external/custom expression dispatch"""

    def wrapper(ext_cls):
        if cls not in __ext_dispatch_classes:
            __ext_dispatch_classes[cls] = Dispatch(
                f"{cls.__qualname__}_dispatch"
            )
        if isinstance(meta_type, tuple):
            for t in meta_type:
                __ext_dispatch_classes[cls].register(t, ext_cls)
        else:
            __ext_dispatch_classes[cls].register(meta_type, ext_cls)
        return ext_cls

    return wrapper(ext_cls) if ext_cls is not None else wrapper


def _override_new_expr(cls, *args, **kwargs):
    """Override the __new__ method of an Expr class"""
    if args and isinstance(args[0], Expr):
        meta = args[0]._meta
        try:
            use_cls = __ext_dispatch_classes[cls].dispatch(type(meta))
        except (KeyError, TypeError):
            use_cls = None  # Default case
        if use_cls:
            return use_cls(*args, **kwargs)
    return object.__new__(cls)


Expr.register_dispatch = classmethod(register_dispatch)
Expr.__new__ = _override_new_expr

# Make sure custom expressions and collections are defined
import dask_cudf.expr._collection
import dask_cudf.expr._expr
