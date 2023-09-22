# Copyright (c) 2020-2023, NVIDIA CORPORATION.
import warnings

import cupy as cp
import numpy as np

from cudf.core.column import as_column
from cudf.core.copy_types import BooleanMask
from cudf.core.index import RangeIndex, as_index
from cudf.core.indexed_frame import IndexedFrame
from cudf.core.scalar import Scalar
from cudf.options import get_option
from cudf.utils.dtypes import can_convert_to_column


def factorize(
    values, sort=False, na_sentinel=None, use_na_sentinel=None, size_hint=None
):
    """Encode the input values as integer labels

    Parameters
    ----------
    values: Series, Index, or CuPy array
        The data to be factorized.
    sort : bool, default True
        Sort uniques and shuffle codes to maintain the relationship.
    na_sentinel : number, default -1
        Value to indicate missing category.

        .. deprecated:: 23.04

           The na_sentinel argument is deprecated and will be removed in
           a future version of cudf. Specify use_na_sentinel as
           either True or False.
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
    StringIndex(['a' 'c'], dtype='object')

    When ``use_na_sentinel=True`` (the default), missing values are indicated
    in the `codes` with the sentinel value ``-1`` and missing values are not
    included in `uniques`.

    >>> codes, uniques = cudf.factorize(['b', None, 'a', 'c', 'b'])
    >>> codes
    array([ 1, -1,  0,  2,  1], dtype=int8)
    >>> uniques
    StringIndex(['a' 'b' 'c'], dtype='object')

    If NA is in the values, and we want to include NA in the uniques of the
    values, it can be achieved by setting ``use_na_sentinel=False``.

    >>> values = np.array([1, 2, 1, np.nan])
    >>> codes, uniques = cudf.factorize(values)
    >>> codes
    array([ 0,  1,  0, -1], dtype=int8)
    >>> uniques
    Float64Index([1.0, 2.0], dtype='float64')
    >>> codes, uniques = cudf.factorize(values, use_na_sentinel=False)
    >>> codes
    array([1, 2, 1, 0], dtype=int8)
    >>> uniques
    Float64Index([<NA>, 1.0, 2.0], dtype='float64')
    """
    # TODO: Drop `na_sentinel` in the next release immediately after
    # pandas 2.0 upgrade.
    if na_sentinel is not None and use_na_sentinel is not None:
        raise ValueError(
            "Cannot specify both `na_sentinel` and `use_na_sentile`; "
            f"got `na_sentinel={na_sentinel}` and "
            f"`use_na_sentinel={use_na_sentinel}`"
        )

    return_cupy_array = isinstance(values, cp.ndarray)

    if not can_convert_to_column(values):
        raise TypeError(
            "'values' can only be a Series, Index, or CuPy array, "
            f"got {type(values)}"
        )

    values = as_column(values)

    if na_sentinel is None:
        na_sentinel = (
            -1
            if use_na_sentinel is None or use_na_sentinel
            else Scalar(None, dtype=values.dtype)
        )
    else:
        if na_sentinel is None:
            msg = (
                "Specifying `na_sentinel=None` is deprecated, specify "
                "`use_na_sentinel=False` instead."
            )
        elif na_sentinel == -1:
            msg = (
                "Specifying `na_sentinel=-1` is deprecated, specify "
                "`use_na_sentinel=True` instead."
            )
        else:
            msg = (
                "Specifying the specific value to use for `na_sentinel` is "
                "deprecated and will be removed in a future version of cudf. "
                "Specify `use_na_sentinel=True` to use the sentinel value -1, "
                "and `use_na_sentinel=False` to encode NA values.",
            )
        # Do not remove until pandas 2.0 support is added.
        warnings.warn(msg, FutureWarning)

    if size_hint:
        warnings.warn("size_hint is not applicable for cudf.factorize")

    if use_na_sentinel is None or use_na_sentinel:
        cats = values.dropna()
    else:
        cats = values

    cats = cats.unique().astype(values.dtype)

    if sort:
        cats = cats.sort_values()

    labels = values._label_encoding(
        cats=cats,
        na_sentinel=Scalar(na_sentinel),
        dtype="int64" if get_option("mode.pandas_compatible") else None,
    ).values

    return labels, cats.values if return_cupy_array else as_index(cats)


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
    known_x_and_y = to_interp._apply_boolean_mask(
        BooleanMask(~mask, len(to_interp))
    )

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
