# Copyright (c) 2020, NVIDIA CORPORATION.
from warnings import warn
import numpy as np
import cupy as cp

from cudf.core.series import Index, Series
from cudf.core.column import as_column

def factorize(values, sort=False, na_sentinel=-1, size_hint=None):
    """Encode the input values as integer labels

    Parameters
    ----------
    values: Series, Index, or CuPy array
        The data to be factorized.
    na_sentinel : number, default -1
        Value to indicate missing category.

    Returns
    --------
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
    cudf.core.series.Series.factorize : Encode the input values of Series.

    """
    if sort:
        raise NotImplementedError(
            "Sorting not yet supported during factorization."
        )
    if na_sentinel is None:
        raise NotImplementedError("na_sentinel can not be None.")

    if size_hint:
        warn("size_hint is not applicable for cudf.factorize")

    return_cupy_array = isinstance(values, cp.ndarray)

    values = Series(values)

    cats = values._column.dropna().unique().astype(values.dtype)

    name = values.name  # label_encoding mutates self.name
    labels = values.label_encoding(cats=cats, na_sentinel=na_sentinel).values
    values.name = name

    return labels, cats.values if return_cupy_array else Index(cats)

def linear_interpolation(col, xax):
    # fill all NAs with NaNs
    col = col.astype('float64').fillna(np.nan)

    # figure out where the nans are
    not_nan_mask = ~cp.isnan(col)

    # find the first nan
    first_nan_idx = as_column(not_nan_mask).find_first_value(1)

    known_x = cp.asarray(xax.apply_boolean_mask(not_nan_mask))
    known_y = cp.asarray(col.apply_boolean_mask(not_nan_mask)).astype(np.dtype('float64'))

    result = cp.interp(
        cp.asarray(xax), 
        known_x, 
        known_y
    )

    result[:first_nan_idx] = np.nan

    return result

def get_column_interpolator(method):
    if method == 'linear':
        return linear_interpolation
    else:
        raise ValueError(
            f"Interpolation method `{method}` not found"
        )        
