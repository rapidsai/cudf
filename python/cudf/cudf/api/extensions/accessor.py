# Copyright (c) 2020, NVIDIA CORPORATION.

import warnings

import cudf
from pandas.core.accessor import CachedAccessor
from pandas.util._decorators import doc


def _register_accessor(name, cls):
    def decorator(accessor):
        if hasattr(cls, name):
            msg = f"Attribute {name} will be overidden in {cls.__name__}"
            warnings.warn(msg)
        cached_accessor = CachedAccessor(name, accessor)
        cls._accessors.add(name)
        setattr(cls, name, cached_accessor)

        return accessor

    return decorator


@doc(klass="DataFrame")
def register_dataframe_accessor(name):
    """
    Extends `cudf.DataFrame` with custom defined accessor

    Parameters
    ----------
    name : the name to be registered in {klass} for the custom accessor

    Returns:
    --------
    callable
        A class decorator

    Notes:
    --------
    The {klass} object will be passed to your custom accessor upon first
    invocation. And will be cached for future calls.

    If the data passed to your accessor is of wrong datatype, you should
    raise an `AttributeError` in consistent with other cudf methods.

    Examples:
    --------
    In your library code:

        ..code-block:: python

            import cudf as gd

            @gd.api.extensions.register_dataframe_accessor("point")
            class PointsAccessor:
                def __init__(self, obj):
                    self._validate(obj)
                    self._obj = obj

                @staticmethod
                def _validate(obj):
                    cols = obj.columns
                    if not all([vertex in cols for vertex in ["x", "y"]]):
                        raise AttributeError("Must have vertices 'x', 'y'.")

                @property
                def bounding_box(self):
                    xs, ys = self._obj["x"], self._obj["y"]
                    min_x, min_y = xs.min(), ys.min()
                    max_x, max_y = xs.max(), ys.max()

                    return (min_x, min_y, max_x, max_y)

    Then in user code:

        ..code-block:: ipython
            In [3]: df = gd.DataFrame({'x': [1,2,3,4,5,6], 'y':[7,6,5,4,3,2]})
            In [4]: df.point.bounding_box
            Out[4]: (1, 2, 6, 7)

    """
    return _register_accessor(name, cudf.DataFrame)


@doc(register_dataframe_accessor, klass="Index")
def register_index_accessor(name):
    return _register_accessor(name, cudf.Index)


@doc(register_dataframe_accessor, klass="Series")
def register_series_accessor(name):
    return _register_accessor(name, cudf.Series)
