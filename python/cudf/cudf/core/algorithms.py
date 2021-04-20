# Copyright (c) 2020, NVIDIA CORPORATION.
from warnings import warn

import cupy as cp

from cudf.core.series import Index, Series


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
    (labels, cats) : (Series, Series)
        - *labels* contains the encoded values
        - *cats* contains the categories in order that the N-th
            item corresponds to the (N-1) code.

    Examples
    --------
    >>> import cudf
    >>> data = cudf.Series(['a', 'c', 'c'])
    >>> codes, uniques = cudf.factorize(data)
    >>> codes
    0    0
    1    1
    2    1
    dtype: int8
    >>> uniques
    0    a
    1    c
    dtype: object

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

    return_cupy_array = isinstance(values, cp.core.core.ndarray)

    values = Series(values)

    cats = values._column.dropna().unique().astype(values.dtype)

    name = values.name  # label_encoding mutates self.name
    labels = values.label_encoding(cats=cats, na_sentinel=na_sentinel).values
    values.name = name

    return labels, cats.values if return_cupy_array else Index(cats)
