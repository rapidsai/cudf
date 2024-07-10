# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from cudf.core.column import as_column
from cudf.core.index import RangeIndex, ensure_index
from cudf.core.scalar import Scalar
from cudf.options import get_option
from cudf.utils.dtypes import can_convert_to_column

if TYPE_CHECKING:
    from cudf.core.column.column import ColumnBase
    from cudf.core.index import BaseIndex


def factorize(values, sort=False, use_na_sentinel=True, size_hint=None):
    """Encode the input values as integer labels

    Parameters
    ----------
    values: Series, Index, or CuPy array
        The data to be factorized.
    sort : bool, default True
        Sort uniques and shuffle codes to maintain the relationship.
    use_na_sentinel : bool, default True
        If True, the sentinel -1 will be used for NA values.
        If False, NA values will be encoded as non-negative
        integers and will not drop the NA from the uniques
        of the values.

    Returns
    -------
    (labels, cats) : (cupy.ndarray, cupy.ndarray or Index)
        - *labels* contains the encoded values
        - *cats* contains the categories in order that the N-th
            item corresponds to the (N-1) code.

    See Also
    --------
    cudf.Series.factorize : Encode the input values of Series.

    Examples
    --------
    >>> import cudf
    >>> import numpy as np
    >>> data = cudf.Series(['a', 'c', 'c'])
    >>> codes, uniques = cudf.factorize(data)
    >>> codes
    array([0, 1, 1], dtype=int8)
    >>> uniques
    Index(['a' 'c'], dtype='object')

    When ``use_na_sentinel=True`` (the default), missing values are indicated
    in the `codes` with the sentinel value ``-1`` and missing values are not
    included in `uniques`.

    >>> codes, uniques = cudf.factorize(['b', None, 'a', 'c', 'b'])
    >>> codes
    array([ 1, -1,  0,  2,  1], dtype=int8)
    >>> uniques
    Index(['a', 'b', 'c'], dtype='object')

    If NA is in the values, and we want to include NA in the uniques of the
    values, it can be achieved by setting ``use_na_sentinel=False``.

    >>> values = np.array([1, 2, 1, np.nan])
    >>> codes, uniques = cudf.factorize(values)
    >>> codes
    array([ 0,  1,  0, -1], dtype=int8)
    >>> uniques
    Index([1.0, 2.0], dtype='float64')
    >>> codes, uniques = cudf.factorize(values, use_na_sentinel=False)
    >>> codes
    array([1, 2, 1, 0], dtype=int8)
    >>> uniques
    Index([<NA>, 1.0, 2.0], dtype='float64')
    """

    return_cupy_array = isinstance(values, cp.ndarray)

    if not can_convert_to_column(values):
        raise TypeError(
            "'values' can only be a Series, Index, or CuPy array, "
            f"got {type(values)}"
        )

    values = as_column(values)

    if size_hint:
        warnings.warn("size_hint is not applicable for cudf.factorize")

    if use_na_sentinel:
        na_sentinel = Scalar(-1)
        cats = values.dropna()
    else:
        na_sentinel = Scalar(None, dtype=values.dtype)
        cats = values

    cats = cats.unique().astype(values.dtype)

    if sort:
        cats = cats.sort_values()

    labels = values._label_encoding(
        cats=cats,
        na_sentinel=na_sentinel,
        dtype="int64" if get_option("mode.pandas_compatible") else None,
    ).values

    return labels, cats.values if return_cupy_array else ensure_index(cats)


def _interpolation(column: ColumnBase, index: BaseIndex) -> ColumnBase:
    """
    Interpolate over a float column. assumes a linear interpolation
    strategy using the index of the data to denote spacing of the x
    values. For example the data and index [1.0, NaN, 4.0], [1, 3, 4]
    would result in [1.0, 3.0, 4.0].
    """
    # figure out where the nans are
    mask = column.isnull()

    # trivial cases, all nan or no nans
    if not mask.any() or mask.all():
        return column.copy()

    valid_locs = ~mask
    if isinstance(index, RangeIndex):
        # Each point is evenly spaced, index values don't matter
        known_x = cp.flatnonzero(valid_locs.values)
    else:
        known_x = index._column.apply_boolean_mask(valid_locs).values  # type: ignore[attr-defined]
    known_y = column.apply_boolean_mask(valid_locs).values

    result = cp.interp(index.to_cupy(), known_x, known_y)

    # find the first nan
    first_nan_idx = valid_locs.values.argmax().item()
    result[:first_nan_idx] = np.nan
    return as_column(result)
