# Copyright (c) 2020-2022, NVIDIA CORPORATION.
import warnings

import cupy as cp
import numpy as np

from cudf.core.column import as_column
from cudf.core.index import Index, RangeIndex
from cudf.core.indexed_frame import IndexedFrame
from cudf.core.series import Series


def factorize(values, sort=False, na_sentinel=-1, size_hint=None):
    """Encode the input values as integer labels

    Parameters
    ----------
    values: Series, Index, or CuPy array
        The data to be factorized.
    na_sentinel : number, default -1
        Value to indicate missing category.

    Returns
    -------
    (labels, cats) : (cupy.ndarray, cupy.ndarray or Index)
        - *labels* contains the encoded values
        - *cats* contains the categories in order that the N-th
            item corresponds to the (N-1) code.

    Examples
    --------
    >>> import cudf
    >>> data = cudf.Series(['a', 'c', 'c'])
    >>> codes, uniques = cudf.factorize(data)
    >>> codes
    array([0, 1, 1], dtype=int8)
    >>> uniques
    StringIndex(['a' 'c'], dtype='object')

    See Also
    --------
    cudf.Series.factorize : Encode the input values of Series.

    """
    if sort:
        raise NotImplementedError(
            "Sorting not yet supported during factorization."
        )
    if na_sentinel is None:
        raise NotImplementedError("na_sentinel can not be None.")

    if size_hint:
        warnings.warn("size_hint is not applicable for cudf.factorize")

    return_cupy_array = isinstance(values, cp.ndarray)

    values = Series(values)

    cats = values._column.dropna().unique().astype(values.dtype)

    labels = values._column._label_encoding(
        cats=cats, na_sentinel=na_sentinel
    ).values

    return labels, cats.values if return_cupy_array else Index(cats)


def _linear_interpolation(column, index=None):
    """
    Interpolate over a float column. Implicitly assumes that values are
    evenly spaced with respect to the x-axis, for example the data
    [1.0, NaN, 3.0] will be interpolated assuming the NaN is half way
    between the two valid values, yielding [1.0, 2.0, 3.0]
    """

    index = RangeIndex(start=0, stop=len(column), step=1)
    return _index_or_values_interpolation(column, index=index)


def _index_or_values_interpolation(column, index=None):
    """
    Interpolate over a float column. assumes a linear interpolation
    strategy using the index of the data to denote spacing of the x
    values. For example the data and index [1.0, NaN, 4.0], [1, 3, 4]
    would result in [1.0, 3.0, 4.0]
    """
    # figure out where the nans are
    mask = cp.isnan(column)

    # trivial cases, all nan or no nans
    num_nan = mask.sum()
    if num_nan == 0 or num_nan == len(column):
        return column

    to_interp = IndexedFrame(data={None: column}, index=index)
    known_x_and_y = to_interp._apply_boolean_mask(as_column(~mask))

    known_x = known_x_and_y._index._column.values
    known_y = known_x_and_y._data.columns[0].values

    result = cp.interp(to_interp._index.values, known_x, known_y)

    # find the first nan
    first_nan_idx = (mask == 0).argmax().item()
    result[:first_nan_idx] = np.nan
    return result


def get_column_interpolator(method):
    interpolator = {
        "linear": _linear_interpolation,
        "index": _index_or_values_interpolation,
        "values": _index_or_values_interpolation,
    }.get(method, None)
    if not interpolator:
        raise ValueError(f"Interpolation method `{method}` not found")
    return interpolator
