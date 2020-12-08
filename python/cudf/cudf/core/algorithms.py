# Copyright (c) 2020, NVIDIA CORPORATION.
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
    cudf.Series.factorize

    """
    if sort:
        raise NotImplementedError(
            "Sorting not yet supported during factorization."
        )
    if size_hint:
        raise NotImplementedError(
            "size_hint is not applicable for cudf.factorize"
        )
    if not na_sentinel:
        raise NotImplementedError("na_sentinel can not be None.")

    values = Series(values)

    cats = values.dropna().unique().astype(values.dtype)

    name = values.name  # label_encoding mutates self.name
    labels = values.label_encoding(cats=cats, na_sentinel=na_sentinel)
    values.name = name

    return labels, cats
