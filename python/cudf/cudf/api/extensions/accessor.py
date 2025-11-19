# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

from pandas.core.accessor import CachedAccessor

from cudf.core.dataframe import DataFrame
from cudf.core.index import Index
from cudf.core.series import Series
from cudf.utils.docutils import docfmt_partial

_docstring_register_accessor = """
    Extends `cudf.{klass}` with custom defined accessor

    Parameters
    ----------
    name : str
        The name to be registered in `{klass}` for the custom accessor

    Returns
    -------
    decorator : callable
        Decorator function for accessor

    Notes
    -----
    The `{klass}` object will be passed to your custom accessor upon first
    invocation. And will be cached for future calls.

    If the data passed to your accessor is of wrong datatype, you should
    raise an `AttributeError` in consistent with other cudf methods.


    Examples
    --------
    {example}
"""

_dataframe_example = """
    In your library code:

        >>> import cudf
        >>> @cudf.api.extensions.register_dataframe_accessor("point")
        ... class PointsAccessor:
        ...     def __init__(self, obj):
        ...         self._validate(obj)
        ...         self._obj = obj
        ...     @staticmethod
        ...     def _validate(obj):
        ...         cols = obj.columns
        ...         if not all([vertex in cols for vertex in ["x", "y"]]):
        ...             raise AttributeError("Must have vertices 'x', 'y'.")
        ...     @property
        ...     def bounding_box(self):
        ...         xs, ys = self._obj["x"], self._obj["y"]
        ...         min_x, min_y = xs.min(), ys.min()
        ...         max_x, max_y = xs.max(), ys.max()
        ...         return (min_x, min_y, max_x, max_y)

    Then in user code:

        >>> df = cudf.DataFrame({'x': [1,2,3,4,5,6], 'y':[7,6,5,4,3,2]})
        >>> df.point.bounding_box
        (1, 2, 6, 7)

"""

_index_example = """
    In your library code:

        >>> import cudf
        >>> @cudf.api.extensions.register_index_accessor("odd")
        ... class OddRowAccessor:
        ...     def __init__(self, obj):
        ...         self._obj = obj
        ...     def __getitem__(self, i):
        ...         return self._obj[2 * i - 1]

    Then in user code:

        >>> gs = cudf.Index(list(range(0, 50)))
        >>> gs.odd[1]
        1
        >>> gs.odd[2]
        3
        >>> gs.odd[3]
        5

"""

_series_example = """
    In your library code:

        >>> import cudf
        >>> @cudf.api.extensions.register_series_accessor("odd")
        ... class OddRowAccessor:
        ...     def __init__(self, obj):
        ...         self._obj = obj
        ...     def __getitem__(self, i):
        ...         return self._obj[2 * i - 1]

    Then in user code:

        >>> gs = cudf.Series(list(range(0, 50)))
        >>> gs.odd[1]
        1
        >>> gs.odd[2]
        3
        >>> gs.odd[3]
        5

"""


doc_register_dataframe_accessor = docfmt_partial(
    docstring=_docstring_register_accessor.format(
        klass="DataFrame", example=_dataframe_example
    )
)

doc_register_index_accessor = docfmt_partial(
    docstring=_docstring_register_accessor.format(
        klass="Index", example=_index_example
    )
)

doc_register_series_accessor = docfmt_partial(
    docstring=_docstring_register_accessor.format(
        klass="Series", example=_series_example
    )
)


def _register_accessor(name, cls):
    def decorator(accessor):
        if hasattr(cls, name):
            msg = f"Attribute {name} will be overridden in {cls.__name__}"
            warnings.warn(msg)
        cached_accessor = CachedAccessor(name, accessor)
        cls._accessors.add(name)
        setattr(cls, name, cached_accessor)

        return accessor

    return decorator


@doc_register_dataframe_accessor()
def register_dataframe_accessor(name):
    """{docstring}"""
    return _register_accessor(name, DataFrame)


@doc_register_index_accessor()
def register_index_accessor(name):
    """{docstring}"""
    return _register_accessor(name, Index)


@doc_register_series_accessor()
def register_series_accessor(name):
    """{docstring}"""
    return _register_accessor(name, Series)
