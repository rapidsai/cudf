# Copyright (c) 2023, NVIDIA CORPORATION.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Callable, Tuple, Union

import cupy as cp
import numpy as np
import pandas as pd
from typing_extensions import TypeAlias

import cudf
import cudf._lib as libcudf
from cudf._lib.types import size_type_dtype
from cudf.api.types import (
    _is_scalar_or_zero_d_array,
    is_bool_dtype,
    is_integer,
    is_integer_dtype,
    is_scalar,
)
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.copy_types import BooleanMask, GatherMap

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase


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


IndexingSpec: TypeAlias = Union[
    EmptyIndexer, MapIndexer, MaskIndexer, ScalarIndexer, SliceIndexer
]


# Helpers for code-sharing between loc and iloc paths
def expand_key(key, frame):
    """Slice-expand key into a tuple of length frame.dim

    Also apply callables on each piece.
    """
    dim = len(frame.shape)
    if isinstance(key, tuple):
        # Key potentially indexes rows and columns, slice-expand to
        # shape of frame
        indexers = key + (slice(None),) * (dim - len(key))
        if len(indexers) > dim:
            raise IndexError(
                f"Too many indexers: got {len(indexers)} expected {dim}"
            )
    else:
        # Key indexes rows, slice-expand to shape of frame
        indexers = (key, *(slice(None),) * (dim - 1))
    return tuple(k(frame) if callable(k) else k for k in indexers)


def destructure_dataframe_indexer(
    key: Any,
    frame: cudf.DataFrame,
    destructure: Callable[[Any, cudf.DataFrame], tuple[Any, ...]],
    is_scalar: Callable[[Any, ColumnAccessor], bool],
    get_ca: str,
):
    rows, cols = destructure(key, frame)
    if cols is Ellipsis:
        cols = slice(None)
    try:
        ca = getattr(frame._data, get_ca)(cols)
    except TypeError as e:
        raise TypeError(
            "Column indices must be names, slices, "
            "list-like of names, or boolean mask"
        ) from e
    scalar = is_scalar(cols, ca)
    if scalar:
        assert (
            len(ca) == 1
        ), "Scalar column indexer should not produce more than one column"
    return rows, (scalar, ca)


def destructure_iloc_key(
    key: Any, frame: Union[cudf.Series, cudf.DataFrame]
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
    indexers = expand_key(key, frame)
    if any(isinstance(k, tuple) for k in indexers):
        raise IndexError(
            "Too many indexers: can't have nested tuples in iloc indexing"
        )
    return indexers


def destructure_dataframe_iloc_indexer(
    key: Any, frame: cudf.DataFrame
) -> Tuple[Any, Tuple[bool, ColumnAccessor]]:
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
        (column_index_is_scalar, ColumnAccessor) for the columns

    Raises
    ------
    TypeError
        If the column indexer is invalid
    IndexError
        If the provided key does not destructure correctly
    NotImplementedError
        If the requested column indexer repeats columns
    """
    return destructure_dataframe_indexer(
        key,
        frame,
        destructure_iloc_key,
        lambda col, _ca: is_integer(col),
        "select_by_index",
    )


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
            key = key.as_numerical_column(key.codes.dtype)
        if is_bool_dtype(key.dtype):
            return MaskIndexer(BooleanMask(key, n))
        elif len(key) == 0:
            return EmptyIndexer()
        elif is_integer_dtype(key.dtype):
            return MapIndexer(GatherMap(key, n, nullify=False))
        else:
            raise TypeError(
                "Cannot index by location "
                f"with non-integer key of type {type(key)}"
            )


def destructure_loc_key(
    key: Any, frame: cudf.Series | cudf.DataFrame
) -> tuple[Any, ...]:
    """
    Destructure a potentially tuple-typed key into row and column indexers

    Tuple arguments to loc indexing are treated specially. They are
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
    - A scalar label looked up in the index
    - A scalar integer that indexes the frame
    - An array-like of labels looked up in the index
    - A slice of the index
    - For multiindices, a tuple of per level indexers

    Slice-based indexing is on the closed interval [start, end], rather
    than the semi-open interval [start, end)

    Parameters
    ----------
    key
        The key to destructure
    frame
        DataFrame or Series to provide context

    Returns
    -------
    tuple of indexers with length equal to the dimension of the frame

    Raises
    ------
    IndexError
        If there are too many indexers.
    """
    if (
        isinstance(frame.index, cudf.MultiIndex)
        and len(frame.shape) == 2
        and isinstance(key, tuple)
        and all(map(is_scalar, key))
    ):
        # This is "best-effort" and ambiguous
        if len(key) == 2:
            if key[1] in frame.index._columns[1]:
                # key just indexes the rows
                key = (key,)
            elif key[1] in frame._data:
                # key indexes rows and columns
                key = key
            else:
                # key indexes rows and we will raise a keyerror
                key = (key,)
        else:
            # key just indexes rows
            key = (key,)
    return expand_key(key, frame)


def destructure_dataframe_loc_indexer(
    key: Any, frame: cudf.DataFrame
) -> Tuple[Any, Tuple[bool, ColumnAccessor]]:
    """Destructure an index key for DataFrame loc getitem.

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
        (column_index_is_scalar, ColumnAccessor) for the columns

    Raises
    ------
    TypeError
        If the column indexer is invalid
    IndexError
        If the provided key does not destructure correctly
    NotImplementedError
        If the requested column indexer repeats columns
    """

    def is_scalar(name, ca):
        try:
            return name in ca
        except TypeError:
            return False

    return destructure_dataframe_indexer(
        key, frame, destructure_loc_key, is_scalar, "select_by_label"
    )


def destructure_series_loc_indexer(key: Any, frame: cudf.Series) -> Any:
    """Destructure an index key for Series loc getitem.

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
    (rows,) = destructure_loc_key(key, frame)
    return rows


def ordered_find(needles: "ColumnBase", haystack: "ColumnBase") -> GatherMap:
    """Find locations of needles in a haystack preserving order

    Parameters
    ----------
    needles
        Labels to look for
    haystack
        Haystack to search in

    Returns
    -------
    NumericalColumn
        Integer gather map of locations needles were found in haystack

    Raises
    ------
    KeyError
        If not all needles were found in the haystack.
        If needles cannot be converted to the dtype of haystack.

    Notes
    -----
    This sorts the gather map so that the result comes back in the
    order the needles were specified (and are found in the haystack).
    """
    # Pre-process to match dtypes
    needle_kind = needles.dtype.kind
    haystack_kind = haystack.dtype.kind
    if haystack_kind == "O":
        try:
            needles = needles.astype(haystack.dtype)
        except ValueError:
            # Pandas raise KeyError here
            raise KeyError("Dtype mismatch in label lookup")
    elif needle_kind == haystack_kind or {
        haystack_kind,
        needle_kind,
    }.issubset({"i", "u", "f"}):
        needles = needles.astype(haystack.dtype)
    elif needles.dtype != haystack.dtype:
        # Pandas raise KeyError here
        raise KeyError("Dtype mismatch in label lookup")
    # Can't always do an inner join because then we can't check if we
    # had missing keys (can't check the length because the entries in
    # the needle might appear multiple times in the haystack).
    lgather, rgather = libcudf.join.join([needles], [haystack], how="left")
    (left_order,) = libcudf.copying.gather(
        [cudf.core.column.arange(len(needles), dtype=size_type_dtype)],
        lgather,
        nullify=False,
    )
    (right_order,) = libcudf.copying.gather(
        [cudf.core.column.arange(len(haystack), dtype=size_type_dtype)],
        rgather,
        nullify=True,
    )
    if right_order.null_count > 0:
        raise KeyError("Not all keys in index")
    (rgather,) = libcudf.sort.sort_by_key(
        [rgather],
        [left_order, right_order],
        [True, True],
        ["last", "last"],
        stable=True,
    )
    return GatherMap.from_column_unchecked(
        rgather, len(haystack), nullify=False
    )


def find_label_range_or_mask(
    key: slice, index: cudf.BaseIndex
) -> EmptyIndexer | MapIndexer | MaskIndexer | SliceIndexer:
    """
    Convert a slice of labels into a slice of positions

    Parameters
    ----------
    key
        Slice to convert
    index
        Index to look up in

    Returns
    -------
    IndexingSpec
        Structured data for indexing (but never a :class:`ScalarIndexer`)

    Raises
    ------
    KeyError
        If the index is unsorted and not a DatetimeIndex

    Notes
    -----
    Until Pandas 2, looking up slices in an unsorted DatetimeIndex
    constructs a mask by checking which dates fall in the range.

    From Pandas 2, slice lookup in DatetimeIndexes will behave
    identically to other index types and fail with a KeyError for
    an unsorted index if either of the slice endpoints are not unique
    in the index or are not in the index at all.
    """
    if (
        not (key.start is None and key.stop is None)
        and isinstance(index, cudf.core.index.DatetimeIndex)
        and not index.is_monotonic_increasing
    ):
        # TODO: datetime index must only be handled specially until pandas 2
        start = pd.to_datetime(key.start)
        stop = pd.to_datetime(key.stop)
        mask = []
        if start is not None:
            mask.append(index >= start)
        if stop is not None:
            mask.append(index <= stop)
        bool_mask = reduce(partial(cp.logical_and, out=mask[0]), mask)
        if key.step is None or key.step == 1:
            return MaskIndexer(BooleanMask(bool_mask, len(index)))
        else:
            (map_,) = bool_mask.nonzero()
            return MapIndexer(
                GatherMap.from_column_unchecked(
                    cudf.core.column.as_column(map_[:: key.step]),
                    len(index),
                    nullify=False,
                )
            )
    else:
        parsed_key = index.find_label_range(key)
        if len(range(len(index))[parsed_key]) == 0:
            return EmptyIndexer()
        else:
            return SliceIndexer(parsed_key)


def parse_single_row_loc_key(
    key: Any,
    index: cudf.BaseIndex,
) -> IndexingSpec:
    """
    Turn a single label-based row indexer into structured information.

    This converts label-based lookups into structured positional
    lookups.

    Valid values for the key are
    - a slice (endpoints are looked up)
    - a scalar label
    - a boolean mask of the same length as the index
    - a column of labels to look up (may be empty)

    Parameters
    ----------
    key
        Key for label-based row indexing
    index
        Index to act as haystack for labels

    Returns
    -------
    IndexingSpec
        Structured information for indexing

    Raises
    ------
    KeyError
        If any label is not found
    ValueError
        If labels cannot be coerced to index dtype
    """
    n = len(index)
    if isinstance(key, slice):
        return find_label_range_or_mask(key, index)
    else:
        is_scalar = _is_scalar_or_zero_d_array(key)
        if is_scalar and isinstance(key, np.ndarray):
            key = cudf.core.column.as_column(key.item(), dtype=key.dtype)
        else:
            key = cudf.core.column.as_column(key)
        if (
            isinstance(key, cudf.core.column.CategoricalColumn)
            and index.dtype != key.dtype
        ):
            # TODO: is this right?
            key = key._get_decategorized_column()
        if is_bool_dtype(key.dtype):
            # The only easy one.
            return MaskIndexer(BooleanMask(key, n))
        elif len(key) == 0:
            return EmptyIndexer()
        else:
            # TODO: promote to Index objects, so this can handle
            # categoricals correctly?
            (haystack,) = index._columns
            if isinstance(index, cudf.DatetimeIndex):
                # Try to turn strings into datetimes
                key = cudf.core.column.as_column(key, dtype=index.dtype)
            gather_map = ordered_find(key, haystack)
            if is_scalar and len(gather_map.column) == 1:
                return ScalarIndexer(gather_map)
            else:
                return MapIndexer(gather_map)


def parse_row_loc_indexer(key: Any, index: cudf.BaseIndex) -> IndexingSpec:
    """
    Normalize to return structured information for a label-based row indexer.

    Given a label-based row indexer that has already been destructured by
    :func:`destructure_loc_key`, inspect further and produce structured
    information for indexing operations to act upon.

    Parameters
    ----------
    key
        Suitably destructured key for row indexing
    index
        Index to provide context

    Returns
    -------
    IndexingSpec
        Structured data for indexing. A tag + parsed data.

    Raises
    ------
    KeyError
        If a valid type of indexer is provided, but not all keys are
        found
    TypeError
        If the indexing key is otherwise invalid.
    """
    # TODO: multiindices need to be treated separately
    if key is Ellipsis:
        # Ellipsis is handled here because multiindex level-based
        # indices don't handle ellipsis in pandas.
        return SliceIndexer(slice(None))
    else:
        return parse_single_row_loc_key(key, index)
