# Copyright (c) 2023, NVIDIA CORPORATION.
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import cudf
import cudf._lib as libcudf
from cudf._lib.types import size_type_dtype

if TYPE_CHECKING:
    from cudf.core.column import NumericalColumn


@dataclass
class GatherMap:
    """A witness to the validity of a given column as a gather map.

    This object witnesses that the column it carries is suitable as a
    gather map for an object with the specified number of rows.

    It is used to provide a safe API for calling the internal
    Frame._gather method without needing to (expensively) verify that
    the provided column is valid for gathering.
    """

    #: The gather map
    column: "NumericalColumn"
    #: The number of rows the gather map has been validated for
    nrows: int
    #: Was the validation for nullify=True?
    nullify: bool


@dataclass
class BooleanMask:
    """A witness to the validity of a given column as a boolean mask.

    This object witnesses that the column it carries is suitable as a
    boolean mask for an object with number of rows equal to the mask's
    length.
    """

    column: "NumericalColumn"

    @cached_property
    def nrows(self):
        return len(self.column)


def as_gather_map(
    column: Any,
    nrows: int,
    *,
    nullify: bool,
    check_bounds: bool,
) -> GatherMap:
    """Turn a column into a gather map

    This augments the column with the information that it is valid as
    a gather map for the specified number of rows with the given
    nullification flag.

    Parameters
    ----------
    column
        The column to verify
    nrows
        The number of rows to verify against
    nullify
        Will this gather map be used nullifying out of bounds accesses
    check_bounds
        Actually check whether the map is in bounds. Set to False if
        you know by construction that the map is in bounds.

    Returns
    -------
    GatherMap
        New object wrapping the column bearing witness to its
        suitability as a gather map for columns with nrows.

    Raises
    ------
    IndexError
        If the column is of unsuitable dtype, or the map is not in bounds.
    """
    column = cudf.core.column.as_column(column)
    if len(column) == 0:
        # Any empty column is valid as a gather map
        # This is necessary because as_column([]) defaults to float64
        # TODO: we should fix this further up.
        # Alternately we can have an Optional[Column] and handle None
        # specially in _gather.
        return GatherMap(
            cast("NumericalColumn", column.astype(size_type_dtype)),
            nrows,
            nullify,
        )
    if column.dtype.kind not in {"i", "u"}:
        raise IndexError("Gather map must have integer dtype")
    if not nullify and check_bounds:
        lo, hi = libcudf.reduce.minmax(column)
        if lo.value < -nrows or hi.value >= nrows:
            raise IndexError(f"Gather map is out of bounds for [0, {nrows})")
    return GatherMap(cast("NumericalColumn", column), nrows, nullify)


def as_boolean_mask(column: Any, nrows: int) -> BooleanMask:
    """Turn a column into a boolean mask

    This augments the column with information that it is valid as a
    boolean mask for columns with a given number of rows

    Parameters
    ----------
    column
        The column to verify
    nrows
        the number of rows to verify against

    Returns
    -------
    BooleanMask
        New object wrapping the column bearing witness to its
        suitability as a boolean mask for columns with matching row
        count.

    Raises
    ------
    IndexError
        If the column is of unsuitable dtype.
    """
    column = cudf.core.column.as_column(column)
    if column.dtype.kind != "b":
        raise IndexError("Boolean mask must have bool dtype")
    if (n := len(column)) != nrows:
        raise IndexError(
            f"Column with {n} rows not suitable "
            f"as a boolean mask for {nrows} rows"
        )
    return BooleanMask(cast("NumericalColumn", column))
