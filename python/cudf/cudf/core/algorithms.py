# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

import cudf
from cudf.core.column import as_column
from cudf.core.index import Index, RangeIndex
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

    return labels, cats.values if return_cupy_array else Index._from_column(
        cats
    )


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


def unique(values):
    """
    Return unique values from array-like

    Parameters
    ----------
    values : 1d array-like

    Returns
    -------
    cudf.Series,

        The return can be:

        * Index : when the input is an Index
        * cudf.Series : when the input is a Series
        * cupy.ndarray : when the input is a cupy.ndarray

        Return cudf.Series, cudf.Index, or cupy.ndarray.

    See Also
    --------
    Index.unique : Return unique values from an Index.
    Series.unique : Return unique values of Series object.

    Examples
    --------
    >>> cudf.unique(cudf.Series([2, 1, 3, 3]))
    0    2
    1    1
    2    3
    dtype: int64

    >>> cudf.unique(cudf.Series([2] + [1] * 5))
    0    2
    1    1
    dtype: int64

    >>> cudf.unique(cudf.Series([pd.Timestamp("20160101"), pd.Timestamp("20160101")]))
    0   2016-01-01
    dtype: datetime64[ns]

    >>> cudf.unique(
    ...     cudf.Series(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160103", tz="US/Eastern"),
    ...         ]
    ...     )
    ... )
    0   2016-01-01 00:00:00-05:00
    1   2016-01-03 00:00:00-05:00
    dtype: datetime64[ns, US/Eastern]

    >>> cudf.unique(
    ...     cudf.Index(
    ...         [
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160101", tz="US/Eastern"),
    ...             pd.Timestamp("20160103", tz="US/Eastern"),
    ...         ]
    ...     )
    ... )
    DatetimeIndex(['2016-01-01 00:00:00-05:00', '2016-01-03 00:00:00-05:00'],dtype='datetime64[ns, US/Eastern]')

    An unordered Categorical will return categories in the
    order of appearance.

    >>> cudf.unique(cudf.Series(pd.Categorical(list("baabc"))))
    0    b
    1    a
    2    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']

    >>> cudf.unique(cudf.Series(pd.Categorical(list("baabc"), categories=list("abc"))))
    0    b
    1    a
    2    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']

    An ordered Categorical preserves the category ordering.

    >>> pd.unique(
    ...     pd.Series(
    ...         pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ...     )
    ... )
    0    b
    1    a
    2    c
    dtype: category
    Categories (3, object): ['a' < 'b' < 'c']

    An array of tuples

    >>> pd.unique(pd.Series([("a", "b"), ("b", "a"), ("a", "c"), ("b", "a")]).values)
    array([('a', 'b'), ('b', 'a'), ('a', 'c')], dtype=object)
    """
    if not isinstance(values, (cudf.Series, cudf.Index, cp.ndarray)):
        raise ValueError(
            "Must pass cudf.Series, cudf.Index, or cupy.ndarray object"
        )
    if isinstance(values, cp.ndarray):
        # pandas.unique will not sort the values in the result
        # while cupy.unique documents it will, so we pass cupy.ndarray
        # through cudf.Index to maintain the original order.
        return cp.asarray(cudf.Index(values).unique())
    if isinstance(values, cudf.Series):
        if get_option("mode.pandas_compatible"):
            if isinstance(values.dtype, cudf.CategoricalDtype):
                raise NotImplementedError(
                    "cudf.Categorical is not implemented"
                )
            else:
                return cp.asarray(values.unique())
    return values.unique()
