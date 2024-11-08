# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import cudf
from cudf.api.types import _is_scalar_or_zero_d_array, is_integer
from cudf.core.copy_types import BooleanMask, GatherMap


class EmptyIndexer:
    """An indexer that will produce an empty result."""

    pass


@dataclass
class MapIndexer:
    """An indexer for a gather map."""

    key: GatherMap


@dataclass
class MaskIndexer:
    """An indexer for a boolean mask."""

    key: BooleanMask


@dataclass
class SliceIndexer:
    """An indexer for a slice."""

    key: slice


@dataclass
class ScalarIndexer:
    """An indexer for a scalar value."""

    key: GatherMap


IndexingSpec: TypeAlias = (
    EmptyIndexer | MapIndexer | MaskIndexer | ScalarIndexer | SliceIndexer
)

ColumnLabels: TypeAlias = list[str]


def destructure_iloc_key(
    key: Any, frame: cudf.Series | cudf.DataFrame
) -> tuple[Any, ...]:
    """
    Destructure a potentially tuple-typed key into row and column indexers.

    Tuple arguments to iloc indexing are treated specially. They are
    picked apart into indexers for the row and column. If the number
    of entries is less than the number of modes of the frame, missing
    entries are slice-expanded.

    If the user-provided key is not a tuple, it is treated as if it
    were a singleton tuple, and then slice-expanded.

    Once this destructuring has occurred, any entries that are
    callables are then called with the indexed frame. This should
    return a valid indexing object for the rows (respectively
    columns), namely one of:

    - A boolean mask of the same length as the frame in the given
      dimension
    - A scalar integer that indexes the frame
    - An array-like of integers that index the frame
    - A slice that indexes the frame

    Integer and slice-based indexing follows usual Python conventions.

    Parameters
    ----------
    key
        The key to destructure
    frame
        DataFrame or Series to provide context

    Returns
    -------
    tuple
        Indexers with length equal to the dimension of the frame

    Raises
    ------
    IndexError
        If there are too many indexers, or any individual indexer is a tuple.
    """
    n = len(frame.shape)
    if isinstance(key, tuple):
        # Key potentially indexes rows and columns, slice-expand to
        # shape of frame
        indexers = key + (slice(None),) * (n - len(key))
        if len(indexers) > n:
            raise IndexError(
                f"Too many indexers: got {len(indexers)} expected {n}"
            )
    else:
        # Key indexes rows, slice-expand to shape of frame
        indexers = (key, *(slice(None),) * (n - 1))
    indexers = tuple(k(frame) if callable(k) else k for k in indexers)
    if any(isinstance(k, tuple) for k in indexers):
        raise IndexError(
            "Too many indexers: can't have nested tuples in iloc indexing"
        )
    return indexers


def destructure_dataframe_iloc_indexer(
    key: Any, frame: cudf.DataFrame
) -> tuple[Any, tuple[bool, ColumnLabels]]:
    """Destructure an index key for DataFrame iloc getitem.

    Parameters
    ----------
    key
        Key to destructure
    frame
        DataFrame to provide context context

    Returns
    -------
    tuple
        2-tuple of a key for the rows and tuple of
        (column_index_is_scalar, column_names) for the columns

    Raises
    ------
    TypeError
        If the column indexer is invalid
    IndexError
        If the provided key does not destructure correctly
    NotImplementedError
        If the requested column indexer repeats columns
    """
    rows, cols = destructure_iloc_key(key, frame)
    if cols is Ellipsis:
        cols = slice(None)
    scalar = is_integer(cols)
    try:
        column_names: ColumnLabels = list(
            frame._data.get_labels_by_index(cols)
        )
    except TypeError:
        raise TypeError(
            "Column indices must be integers, slices, "
            "or list-like of integers"
        )
    if scalar:
        assert (
            len(column_names) == 1
        ), "Scalar column indexer should not produce more than one column"

    return rows, (scalar, column_names)


def destructure_series_iloc_indexer(key: Any, frame: cudf.Series) -> Any:
    """Destructure an index key for Series iloc getitem.

    Parameters
    ----------
    key
        Key to destructure
    frame
        Series for unpacking context

    Returns
    -------
    Single key that will index the rows
    """
    (rows,) = destructure_iloc_key(key, frame)
    return rows


def parse_row_iloc_indexer(key: Any, n: int) -> IndexingSpec:
    """
    Normalize and produce structured information about a row indexer.

    Given a row indexer that has already been destructured by
    :func:`destructure_iloc_key`, inspect further and produce structured
    information for indexing operations to act upon.

    Parameters
    ----------
    key
        Suitably destructured key for row indexing
    n
        Length of frame to index

    Returns
    -------
    IndexingSpec
        Structured data for indexing. A tag + parsed data.

    Raises
    ------
    IndexError
        If a valid type of indexer is provided, but it is out of
        bounds
    TypeError
        If the indexing key is otherwise invalid.
    """
    if key is Ellipsis:
        return SliceIndexer(slice(None))
    elif isinstance(key, slice):
        return SliceIndexer(key)
    elif _is_scalar_or_zero_d_array(key):
        return ScalarIndexer(GatherMap(key, n, nullify=False))
    else:
        key = cudf.core.column.as_column(key)
        if isinstance(key, cudf.core.column.CategoricalColumn):
            key = key.astype(key.codes.dtype)
        if key.dtype.kind == "b":
            return MaskIndexer(BooleanMask(key, n))
        elif len(key) == 0:
            return EmptyIndexer()
        elif key.dtype.kind in "iu":
            return MapIndexer(GatherMap(key, n, nullify=False))
        else:
            raise TypeError(
                "Cannot index by location "
                f"with non-integer key of type {type(key)}"
            )
