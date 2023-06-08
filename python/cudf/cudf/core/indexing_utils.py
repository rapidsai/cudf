# Copyright (c) 2023, NVIDIA CORPORATION.

from __future__ import annotations

import enum
import itertools
from typing import TYPE_CHECKING, Any, List, Tuple, Union

import numpy as np
from typing_extensions import TypeAlias

import cudf
import cudf._lib as libcudf
from cudf.api.types import (
    _is_scalar_or_zero_d_array,
    is_bool_dtype,
    is_integer,
    is_integer_dtype,
)

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase


class IndexTag(enum.IntEnum):
    SLICE = enum.auto()
    MASK = enum.auto()
    MAP = enum.auto()
    SCALAR = enum.auto()


IndexSpec: TypeAlias = Tuple[IndexTag, Union[slice, "ColumnBase"]]
ColumnLabels: TypeAlias = List[str]


def unpack_iloc_key(
    key: Any, frame: cudf.DataFrame | cudf.Series, n: int
) -> List[Any]:
    """Unpack a user-level key to iloc.__getitem__

    Parameters
    ----------
    key
        Key to unpack
    frame
        DataFrame or Series to provide context
    n
        Number of dimensions we're expecting to unpack to.

    Returns
    -------
    tuple
        Tuple of row and (for dataframes) column keys

    Raises
    ------
    IndexError
        If provided a structurally invalid key
    """
    # This is more consistent than pandas, using a fixed point
    # iteration to remove all callables.
    # See https://github.com/pandas-dev/pandas/issues/53533
    if callable(key):
        return unpack_iloc_key(key(frame), frame, n)
    if isinstance(key, tuple):
        indexers = tuple(
            itertools.chain(key, itertools.repeat(slice(None), n - len(key)))
        )
        if (ni := len(indexers)) > n:
            raise IndexError(f"Too many indexers: got {ni} expected {n}")
        if any(isinstance(k, tuple) for k in indexers):
            # Only one level of tuple-nesting allowed
            raise IndexError(
                "Too many indexers: can't have nested tuples for iloc"
            )
        # Hack, do this better
        return list(unpack_iloc_key(k, frame, n - 1)[0] for k in indexers)
    # No special-casing, key gets rows, and if a dataframe second part
    # gets all columns
    return [key, slice(None)][:n]


def unpack_dataframe_iloc_indexer(
    key: Any, frame: cudf.DataFrame
) -> Tuple[Any, Tuple[bool, ColumnLabels]]:
    """Unpack and index key for DataFrame iloc getitem.

    Parameters
    ----------
    key
        Key to unpack
    frame
        DataFrame for unpacking context

    Returns
    -------
    tuple
        2-tuple of a key for the rows and tuple of
        (scalar_column_index, column_names) for the columns

    Raises
    ------
    TypeError
        If the column indexer is invalid
    """
    rows, cols = unpack_iloc_key(key, frame, len(frame.shape))
    if cols is Ellipsis:
        cols = slice(None)
    scalar = is_integer(cols)
    try:
        column_names: ColumnLabels = list(
            frame._data.get_labels_by_index(cols)
        )
        if len(set(column_names)) != len(column_names):
            raise NotImplementedError(
                "cudf DataFrames do not support repeated column names"
            )
    except TypeError:
        raise TypeError(
            "Column indices must be integers, slices, "
            "or list-like of integers"
        )
    return (rows, (scalar, column_names))


def unpack_series_iloc_indexer(key: Any, frame: cudf.Series) -> Any:
    """Unpack an index key for Series iloc getitem.

    Parameters
    ----------
    key
        Key to unpack
    frame
        Series for unpacking context

    Returns
    -------
    Single key that will index the rows
    """
    (rows,) = unpack_iloc_key(key, frame, len(frame.shape))
    return rows


def normalize_row_iloc_indexer(
    key: Any, n: int, check_bounds=False
) -> IndexSpec:
    """
    Normalize and produce structured information about a row indexer

    Given a row indexer that has already been normalized by
    :func:`unpack_iloc_key`, inspect further and produce structured
    information for indexing operations to act upon.

    Parameters
    ----------
    key
        Suitably normalized key for row indexing
    n
        Length of frame to index
    check_bounds
        If True, perform bounds checking of the key if it is a gather
        map.

    Returns
    -------
    IndexSpec
        Structured data for indexing. The first entry is a
        :class:`Indexer` tag, the second entry is normalized
        arguments to the tag-specific indexing routine.

    Raises
    ------
    IndexError
        If a valid type of indexer is provided, but it is out of
        bounds
    TypeError
        If the indexing key is otherwise invalid.
    """
    if key is Ellipsis:
        key = slice(None)
    if isinstance(key, slice):
        return (IndexTag.SLICE, key)
    else:
        if _is_scalar_or_zero_d_array(key):
            key = np.asarray(key)
            if not is_integer_dtype(key.dtype):
                raise TypeError(
                    "Cannot index by location with non-integer key"
                )
            if key < 0:
                key += n
            if not 0 <= key < n:
                raise IndexError("Positional indexer is out-of-bounds")
            # We make a column here because we'll do _gather then
            # element_indexing if appropriate.
            return (
                IndexTag.SCALAR,
                cudf.core.column.as_column(key.astype(np.int32)),
            )
        key = cudf.core.column.as_column(key)
        if isinstance(key, cudf.core.column.CategoricalColumn):
            key = key.as_numerical_column(key.codes.dtype)
        if is_bool_dtype(key.dtype):
            if (kn := len(key)) != n:
                raise IndexError(
                    f"Invalid length for boolean mask (got {kn}, need {n})"
                )
            return (IndexTag.MASK, key)
        elif is_integer_dtype(key.dtype) or len(key) == 0:
            if len(key):
                if check_bounds:
                    mi, ma = libcudf.reduce.minmax(key)
                    # Faster than going through cudf.Scalar binop machinery.
                    if mi.value < -n or ma.value >= n:
                        raise IndexError("Gather map index is out of bounds.")
            else:
                key = cudf.core.column.column_empty(
                    0, dtype=libcudf.types.size_type_dtype
                )
            return (IndexTag.MAP, key)
        else:
            raise TypeError(
                "Cannot index by location "
                f"with non-integer key of type {type(key)}"
            )
