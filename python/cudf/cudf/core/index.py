# Copyright (c) 2018-2023, NVIDIA CORPORATION.

from __future__ import annotations

import math
import pickle
import warnings
from functools import cached_property
from numbers import Number
from typing import (
    Any,
    Dict,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import cupy
import numpy as np
import pandas as pd
from pandas._config import get_option

import cudf
from cudf._lib.datetime import extract_quarter, is_leap_year
from cudf._lib.filling import sequence
from cudf._lib.search import search_sorted
from cudf.api.types import (
    _is_non_decimal_numeric_dtype,
    is_categorical_dtype,
    is_dtype_equal,
    is_interval_dtype,
    is_list_like,
    is_scalar,
    is_string_dtype,
)
from cudf.core._base_index import BaseIndex, _index_astype_docstring
from cudf.core.column import (
    CategoricalColumn,
    ColumnBase,
    DatetimeColumn,
    IntervalColumn,
    NumericalColumn,
    StringColumn,
    StructColumn,
    TimeDeltaColumn,
    arange,
    column,
)
from cudf.core.column.column import as_column, concat_columns
from cudf.core.column.string import StringMethods as StringMethods
from cudf.core.dtypes import IntervalDtype
from cudf.core.frame import Frame
from cudf.core.mixins import BinaryOperand
from cudf.core.single_column_frame import SingleColumnFrame
from cudf.utils.docutils import copy_docstring, doc_apply
from cudf.utils.dtypes import (
    _maybe_convert_to_default_type,
    find_common_type,
    is_mixed_with_object_dtype,
    numeric_normalize_types,
)
from cudf.utils.utils import _cudf_nvtx_annotate, search_range

T = TypeVar("T", bound="Frame")


def _lexsorted_equal_range(
    idx: Union[GenericIndex, cudf.MultiIndex],
    key_as_table: Frame,
    is_sorted: bool,
) -> Tuple[int, int, Optional[ColumnBase]]:
    """Get equal range for key in lexicographically sorted index. If index
    is not sorted when called, a sort will take place and `sort_inds` is
    returned. Otherwise `None` is returned in that position.
    """
    if not is_sorted:
        sort_inds = idx._get_sorted_inds()
        sort_vals = idx._gather(sort_inds)
    else:
        sort_inds = None
        sort_vals = idx
    lower_bound = search_sorted(
        [*sort_vals._data.columns], [*key_as_table._columns], side="left"
    ).element_indexing(0)
    upper_bound = search_sorted(
        [*sort_vals._data.columns], [*key_as_table._columns], side="right"
    ).element_indexing(0)

    return lower_bound, upper_bound, sort_inds


def _index_from_data(data: MutableMapping, name: Any = None):
    """Construct an index of the appropriate type from some data."""

    if len(data) == 0:
        raise ValueError("Cannot construct Index from any empty Table")
    if len(data) == 1:
        values = next(iter(data.values()))

        if isinstance(values, NumericalColumn):
            try:
                index_class_type: Type[
                    Union[GenericIndex, cudf.MultiIndex]
                ] = _dtype_to_index[values.dtype.type]
            except KeyError:
                index_class_type = GenericIndex
        elif isinstance(values, DatetimeColumn):
            index_class_type = DatetimeIndex
        elif isinstance(values, TimeDeltaColumn):
            index_class_type = TimedeltaIndex
        elif isinstance(values, StringColumn):
            index_class_type = StringIndex
        elif isinstance(values, CategoricalColumn):
            index_class_type = CategoricalIndex
        elif isinstance(values, (IntervalColumn, StructColumn)):
            index_class_type = IntervalIndex
        else:
            raise NotImplementedError(
                "Unsupported column type passed to "
                f"create an Index: {type(values)}"
            )
    else:
        index_class_type = cudf.MultiIndex
    return index_class_type._from_data(data, name)


def _index_from_columns(
    columns: List[cudf.core.column.ColumnBase], name: Any = None
):
    """Construct an index from ``columns``, with levels named 0, 1, 2..."""
    return _index_from_data(dict(zip(range(len(columns)), columns)), name=name)


class RangeIndex(BaseIndex, BinaryOperand):
    """
    Immutable Index implementing a monotonic integer range.

    This is the default index type used by DataFrame and Series
    when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), or other range instance
    stop : int (default: 0)
    step : int (default: 1)
    name : object, optional
        Name to be stored in the index.
    dtype : numpy dtype
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.

    Returns
    -------
    RangeIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.RangeIndex(0, 10, 1, name="a")
    RangeIndex(start=0, stop=10, step=1, name='a')

    >>> cudf.RangeIndex(range(1, 10, 1), name="a")
    RangeIndex(start=1, stop=10, step=1, name='a')
    """

    _VALID_BINARY_OPERATIONS = BinaryOperand._SUPPORTED_BINARY_OPERATIONS

    _range: range

    @_cudf_nvtx_annotate
    def __init__(
        self, start, stop=None, step=1, dtype=None, copy=False, name=None
    ):
        if step == 0:
            raise ValueError("Step must not be zero.")

        if isinstance(start, range):
            therange = start
            start = therange.start
            stop = therange.stop
            step = therange.step
        if stop is None:
            start, stop = 0, start
        self._start = int(start)
        self._stop = int(stop)
        self._step = int(step) if step is not None else 1
        self._index = None
        self._name = name
        self._range = range(self._start, self._stop, self._step)
        # _end is the actual last element of RangeIndex,
        # whereas _stop is an upper bound.
        self._end = self._start + self._step * (len(self._range) - 1)

    def _copy_type_metadata(
        self: RangeIndex, other: RangeIndex, *, override_dtypes=None
    ) -> RangeIndex:
        # There is no metadata to be copied for RangeIndex since it does not
        # have an underlying column.
        return self

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def name(self):
        """
        Returns the name of the Index.
        """
        return self._name

    @name.setter  # type: ignore
    @_cudf_nvtx_annotate
    def name(self, value):
        self._name = value

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def start(self):
        """
        The value of the `start` parameter (0 if this was not supplied).
        """
        return self._start

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def stop(self):
        """
        The value of the stop parameter.
        """
        return self._stop

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def step(self):
        """
        The value of the step parameter.
        """
        return self._step

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def _num_rows(self):
        return len(self)

    @cached_property  # type: ignore
    @_cudf_nvtx_annotate
    def _values(self):
        if len(self) > 0:
            return column.arange(
                self._start, self._stop, self._step, dtype=self.dtype
            )
        else:
            return column.column_empty(0, masked=False, dtype=self.dtype)

    def _clean_nulls_from_index(self):
        return self

    def _is_numeric(self):
        return True

    def _is_boolean(self):
        return False

    def _is_integer(self):
        return True

    def _is_floating(self):
        return False

    def _is_object(self):
        return False

    def _is_categorical(self):
        return False

    def _is_interval(self):
        return False

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def hasnans(self):
        return False

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def _data(self):
        return cudf.core.column_accessor.ColumnAccessor(
            {self.name: self._values}
        )

    @_cudf_nvtx_annotate
    def __contains__(self, item):
        if not isinstance(
            item, tuple(np.sctypes["int"] + np.sctypes["float"] + [int, float])
        ):
            return False
        if not item % 1 == 0:
            return False
        return item in range(self._start, self._stop, self._step)

    @_cudf_nvtx_annotate
    def copy(self, name=None, deep=False):
        """
        Make a copy of this object.

        Parameters
        ----------
        name : object optional (default: None), name of index
        deep : Bool (default: False)
            Ignored for RangeIndex

        Returns
        -------
        New RangeIndex instance with same range
        """

        name = self.name if name is None else name

        return RangeIndex(
            start=self._start, stop=self._stop, step=self._step, name=name
        )

    @_cudf_nvtx_annotate
    @doc_apply(_index_astype_docstring)
    def astype(self, dtype, copy: bool = True):
        if is_dtype_equal(dtype, self.dtype):
            return self
        return self._as_int_index().astype(dtype, copy=copy)

    @_cudf_nvtx_annotate
    def drop_duplicates(self, keep="first"):
        return self

    @_cudf_nvtx_annotate
    def duplicated(self, keep="first"):
        return cupy.zeros(len(self), dtype=bool)

    @_cudf_nvtx_annotate
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(start={self._start}, stop={self._stop}"
            f", step={self._step}"
            + (
                f", name={pd.io.formats.printing.default_pprint(self.name)}"
                if self.name is not None
                else ""
            )
            + ")"
        )

    @_cudf_nvtx_annotate
    def __len__(self):
        return len(range(self._start, self._stop, self._step))

    @_cudf_nvtx_annotate
    def __getitem__(self, index):
        if isinstance(index, slice):
            sl_start, sl_stop, sl_step = index.indices(len(self))

            lo = self._start + sl_start * self._step
            hi = self._start + sl_stop * self._step
            st = self._step * sl_step
            return RangeIndex(start=lo, stop=hi, step=st, name=self._name)

        elif isinstance(index, Number):
            len_self = len(self)
            if index < 0:
                index += len_self
            if not (0 <= index < len_self):
                raise IndexError("Index out of bounds")
            return self._start + index * self._step
        return self._as_int_index()[index]

    @_cudf_nvtx_annotate
    def equals(self, other):
        if isinstance(other, RangeIndex):
            if (self._start, self._stop, self._step) == (
                other._start,
                other._stop,
                other._step,
            ):
                return True
        return self._as_int_index().equals(other)

    @_cudf_nvtx_annotate
    def serialize(self):
        header = {}
        header["index_column"] = {}

        # store metadata values of index separately
        # We don't need to store the GPU buffer for RangeIndexes
        # cuDF only needs to store start/stop and rehydrate
        # during de-serialization
        header["index_column"]["start"] = self._start
        header["index_column"]["stop"] = self._stop
        header["index_column"]["step"] = self._step
        frames = []

        header["name"] = pickle.dumps(self.name)
        header["dtype"] = pickle.dumps(self.dtype)
        header["type-serialized"] = pickle.dumps(type(self))
        header["frame_count"] = 0
        return header, frames

    @classmethod
    @_cudf_nvtx_annotate
    def deserialize(cls, header, frames):
        h = header["index_column"]
        name = pickle.loads(header["name"])
        start = h["start"]
        stop = h["stop"]
        step = h.get("step", 1)
        return RangeIndex(start=start, stop=stop, step=step, name=name)

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def dtype(self):
        """
        `dtype` of the range of values in RangeIndex.

        By default the dtype is 64 bit signed integer. This is configurable
        via `default_integer_bitwidth` as 32 bit in `cudf.options`
        """
        dtype = np.dtype(np.int64)
        return _maybe_convert_to_default_type(dtype)

    @_cudf_nvtx_annotate
    def find_label_range(self, first=None, last=None):
        """Find subrange in the ``RangeIndex``, marked by their positions, that
        starts greater or equal to ``first`` and ends less or equal to ``last``

        The range returned is assumed to be monotonically increasing. In cases
        where there is no such range that suffice the constraint, an exception
        will be raised.

        Parameters
        ----------
        first, last : int, optional, Default None
            The "start" and "stop" values of the subrange. If None, will use
            ``self._start`` as first, ``self._stop`` as last.

        Returns
        -------
        begin, end : 2-tuple of int
            The starting index and the ending index.
            The `last` value occurs at ``end - 1`` position.
        """

        first = self._start if first is None else first
        last = self._stop if last is None else last

        if self._step < 0:
            first = -first
            last = -last
            start = -self._start
            step = -self._step
        else:
            start = self._start
            step = self._step

        stop = start + len(self) * step
        begin = search_range(start, stop, first, step, side="left")
        end = search_range(start, stop, last, step, side="right")

        return begin, end

    @_cudf_nvtx_annotate
    def to_pandas(self, nullable=False):
        return pd.RangeIndex(
            start=self._start,
            stop=self._stop,
            step=self._step,
            dtype=self.dtype,
            name=self.name,
        )

    @property
    def is_unique(self):
        """
        Return if the index has unique values.
        """
        return True

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def is_monotonic_increasing(self):
        return self._step > 0 or len(self) <= 1

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def is_monotonic_decreasing(self):
        return self._step < 0 or len(self) <= 1

    @_cudf_nvtx_annotate
    def get_slice_bound(self, label, side):
        """
        Calculate slice bound that corresponds to given label.
        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.

        Parameters
        ----------
        label : int
            A valid value in the ``RangeIndex``
        side : {'left', 'right'}

        Returns
        -------
        int
            Index of label.
        """
        if side not in {"left", "right"}:
            raise ValueError(f"Unrecognized side parameter: {side}")

        if self._step < 0:
            label = -label
            start = -self._start
            step = -self._step
        else:
            start = self._start
            step = self._step

        stop = start + len(self) * step
        pos = search_range(start, stop, label, step, side=side)
        return pos

    @_cudf_nvtx_annotate
    def memory_usage(self, deep=False):
        if deep:
            warnings.warn(
                "The deep parameter is ignored and is only included "
                "for pandas compatibility."
            )
        return 0

    def unique(self):
        # RangeIndex always has unique values
        return self

    @_cudf_nvtx_annotate
    def __mul__(self, other):
        # Multiplication by raw ints must return a RangeIndex to match pandas.
        if isinstance(other, cudf.Scalar) and other.dtype.kind in "iu":
            other = other.value
        elif (
            isinstance(other, (np.ndarray, cupy.ndarray))
            and other.ndim == 0
            and other.dtype.kind in "iu"
        ):
            other = other.item()
        if isinstance(other, (int, np.integer)):
            return RangeIndex(
                self.start * other, self.stop * other, self.step * other
            )
        return self._as_int_index().__mul__(other)

    @_cudf_nvtx_annotate
    def __rmul__(self, other):
        # Multiplication is commutative.
        return self.__mul__(other)

    @_cudf_nvtx_annotate
    def _as_int_index(self):
        # Convert self to an integer index. This method is used to perform ops
        # that are not defined directly on RangeIndex.
        return _dtype_to_index[self.dtype.type]._from_data(self._data)

    @_cudf_nvtx_annotate
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self._as_int_index().__array_ufunc__(
            ufunc, method, *inputs, **kwargs
        )

    @_cudf_nvtx_annotate
    def get_loc(self, key, method=None, tolerance=None):
        # We should not actually remove this code until we have implemented the
        # get_indexers method as an alternative, see
        # https://github.com/rapidsai/cudf/issues/12312
        if method is not None:
            warnings.warn(
                f"Passing method to {self.__class__.__name__}.get_loc is "
                "deprecated and will raise in a future version.",
                FutureWarning,
            )

        # Given an actual integer,
        idx = (key - self._start) / self._step
        idx_int_upper_bound = (self._stop - self._start) // self._step
        if method is None:
            if tolerance is not None:
                raise ValueError(
                    "tolerance argument only valid if using pad, "
                    "backfill or nearest lookups"
                )

            if idx > idx_int_upper_bound or idx < 0:
                raise KeyError(key)

            idx_int = (key - self._start) // self._step
            if idx_int != idx:
                raise KeyError(key)
            return idx_int

        if (method == "ffill" and idx < 0) or (
            method == "bfill" and idx > idx_int_upper_bound
        ):
            raise KeyError(key)

        round_method = {
            "ffill": math.floor,
            "bfill": math.ceil,
            "nearest": round,
        }[method]
        if tolerance is not None and (abs(idx) * self._step > tolerance):
            raise KeyError(key)
        return np.clip(round_method(idx), 0, idx_int_upper_bound, dtype=int)

    @_cudf_nvtx_annotate
    def _union(self, other, sort=None):
        if isinstance(other, RangeIndex):
            # Variable suffixes are of the
            # following notation: *_o -> other, *_s -> self,
            # and *_r -> result
            start_s, step_s = self.start, self.step
            end_s = self._end
            start_o, step_o = other.start, other.step
            end_o = other._end
            if self.step < 0:
                start_s, step_s, end_s = end_s, -step_s, start_s
            if other.step < 0:
                start_o, step_o, end_o = end_o, -step_o, start_o
            if len(self) == 1 and len(other) == 1:
                step_s = step_o = abs(self.start - other.start)
            elif len(self) == 1:
                step_s = step_o
            elif len(other) == 1:
                step_o = step_s

            # Determine minimum start value of the result.
            start_r = min(start_s, start_o)
            # Determine maximum end value of the result.
            end_r = max(end_s, end_o)
            result = None
            min_step = min(step_o, step_s)

            if ((start_s - start_o) % min_step) == 0:
                # Checking to determine other is a subset of self with
                # equal step size.
                if (
                    step_o == step_s
                    and (start_s - end_o) <= step_s
                    and (start_o - end_s) <= step_s
                ):
                    result = type(self)(start_r, end_r + step_s, step_s)
                # Checking if self is a subset of other with unequal
                # step sizes.
                elif (
                    step_o % step_s == 0
                    and (start_o + step_s >= start_s)
                    and (end_o - step_s <= end_s)
                ):
                    result = type(self)(start_r, end_r + step_s, step_s)
                # Checking if other is a subset of self with unequal
                # step sizes.
                elif (
                    step_s % step_o == 0
                    and (start_s + step_o >= start_o)
                    and (end_s - step_o <= end_o)
                ):
                    result = type(self)(start_r, end_r + step_o, step_o)
            # Checking to determine when the steps are even but one of
            # the inputs spans across is near half or less then half
            # the other input. This case needs manipulation to step
            # size.
            elif (
                step_o == step_s
                and (step_s % 2 == 0)
                and (abs(start_s - start_o) <= step_s / 2)
                and (abs(end_s - end_o) <= step_s / 2)
            ):
                result = type(self)(start_r, end_r + step_s / 2, step_s / 2)
            if result is not None:
                if sort is None and not result.is_monotonic_increasing:
                    return result.sort_values()
                else:
                    return result

        # If all the above optimizations don't cater to the inputs,
        # we materialize RangeIndexes into integer indexes and
        # then perform `union`.
        return self._as_int_index()._union(other, sort=sort)

    @_cudf_nvtx_annotate
    def _intersection(self, other, sort=False):
        if not isinstance(other, RangeIndex):
            return super()._intersection(other, sort=sort)

        if not len(self) or not len(other):
            return RangeIndex(0)

        first = self._range[::-1] if self.step < 0 else self._range
        second = other._range[::-1] if other.step < 0 else other._range

        # check whether intervals intersect
        # deals with in- and decreasing ranges
        int_low = max(first.start, second.start)
        int_high = min(first.stop, second.stop)
        if int_high <= int_low:
            return RangeIndex(0)

        # Method hint: linear Diophantine equation
        # solve intersection problem
        # performance hint: for identical step sizes, could use
        # cheaper alternative
        gcd, s, _ = _extended_gcd(first.step, second.step)

        # check whether element sets intersect
        if (first.start - second.start) % gcd:
            return RangeIndex(0)

        # calculate parameters for the RangeIndex describing the
        # intersection disregarding the lower bounds
        tmp_start = (
            first.start + (second.start - first.start) * first.step // gcd * s
        )
        new_step = first.step * second.step // gcd
        no_steps = -(-(int_low - tmp_start) // abs(new_step))
        new_start = tmp_start + abs(new_step) * no_steps
        new_range = range(new_start, int_high, new_step)
        new_index = RangeIndex(new_range)

        if (self.step < 0 and other.step < 0) is not (new_index.step < 0):
            new_index = new_index[::-1]
        if sort is None:
            new_index = new_index.sort_values()

        return new_index

    def sort_values(
        self,
        return_indexer=False,
        ascending=True,
        na_position="last",
        key=None,
    ):
        if key is not None:
            raise NotImplementedError("key parameter is not yet implemented.")
        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")

        sorted_index = self
        indexer = RangeIndex(range(len(self)))

        sorted_index = self
        if ascending:
            if self.step < 0:
                sorted_index = self[::-1]
                indexer = indexer[::-1]
        else:
            if self.step > 0:
                sorted_index = self[::-1]
                indexer = indexer = indexer[::-1]

        if return_indexer:
            return sorted_index, indexer
        else:
            return sorted_index

    @_cudf_nvtx_annotate
    def _gather(self, gather_map, nullify=False, check_bounds=True):
        gather_map = cudf.core.column.as_column(gather_map)
        return _dtype_to_index[self.dtype.type]._from_columns(
            [self._values.take(gather_map, nullify, check_bounds)], [self.name]
        )

    @_cudf_nvtx_annotate
    def _apply_boolean_mask(self, boolean_mask):
        return _dtype_to_index[self.dtype.type]._from_columns(
            [self._values.apply_boolean_mask(boolean_mask)], [self.name]
        )

    def repeat(self, repeats, axis=None):
        return self._as_int_index().repeat(repeats, axis)

    def _split(self, splits):
        return _dtype_to_index[self.dtype.type]._from_columns(
            [self._as_int_index()._split(splits)], [self.name]
        )

    def _binaryop(self, other, op: str):
        # TODO: certain binops don't require materializing range index and
        # could use some optimization.
        return self._as_int_index()._binaryop(other, op=op)

    def join(
        self, other, how="left", level=None, return_indexers=False, sort=False
    ):
        # TODO: pandas supports directly merging RangeIndex objects and can
        # intelligently create RangeIndex outputs depending on the type of
        # join. We need to implement that for the supported special cases.
        return self._as_int_index().join(
            other, how, level, return_indexers, sort
        )

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def _column(self):
        return self._as_int_index()._column

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def _columns(self):
        return self._as_int_index()._columns

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def values_host(self):
        return self.to_pandas().values

    @_cudf_nvtx_annotate
    def argsort(
        self,
        ascending=True,
        na_position="last",
    ):
        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")

        indices = cupy.arange(0, len(self))
        if (ascending and self._step < 0) or (
            not ascending and self._step > 0
        ):
            indices = indices[::-1]
        return indices

    @_cudf_nvtx_annotate
    def where(self, cond, other=None, inplace=False):
        return self._as_int_index().where(cond, other, inplace)

    @_cudf_nvtx_annotate
    def to_numpy(self):
        return self.values_host

    @_cudf_nvtx_annotate
    def to_arrow(self):
        return self._as_int_index().to_arrow()

    def __array__(self, dtype=None):
        raise TypeError(
            "Implicit conversion to a host NumPy array via __array__ is not "
            "allowed, To explicitly construct a GPU matrix, consider using "
            ".to_cupy()\nTo explicitly construct a host matrix, consider "
            "using .to_numpy()."
        )

    @_cudf_nvtx_annotate
    def nunique(self):
        return len(self)

    @_cudf_nvtx_annotate
    def isna(self):
        return cupy.zeros(len(self), dtype=bool)

    isnull = isna

    @_cudf_nvtx_annotate
    def notna(self):
        return cupy.ones(len(self), dtype=bool)

    notnull = isna

    @_cudf_nvtx_annotate
    def _minmax(self, meth: str):
        no_steps = len(self) - 1
        if no_steps == -1:
            return np.nan
        elif (meth == "min" and self.step > 0) or (
            meth == "max" and self.step < 0
        ):
            return self.start

        return self.start + self.step * no_steps

    def min(self):
        return self._minmax("min")

    def max(self):
        return self._minmax("max")

    @property
    def values(self):
        return cupy.arange(self.start, self.stop, self.step)

    def any(self):
        return any(self._range)

    def append(self, other):
        return self._as_int_index().append(other)

    def isin(self, values):
        if is_scalar(values):
            raise TypeError(
                "only list-like objects are allowed to be passed "
                f"to isin(), you passed a {type(values).__name__}"
            )

        return self._values.isin(values).values

    def __neg__(self):
        return -self._as_int_index()

    def __pos__(self):
        return +self._as_int_index()

    def __abs__(self):
        return abs(self._as_int_index())


class GenericIndex(SingleColumnFrame, BaseIndex):
    """
    An array of orderable values that represent the indices of another Column

    Attributes
    ----------
    _values: A Column object
    name: A string

    Parameters
    ----------
    data : Column
        The Column of data for this index
    name : str optional
        The name of the Index. If not provided, the Index adopts the value
        Column's name. Otherwise if this name is different from the value
        Column's, the data Column will be cloned to adopt this name.
    """

    @_cudf_nvtx_annotate
    def __init__(self, data, **kwargs):
        kwargs = _setdefault_name(data, **kwargs)

        # normalize the input
        if isinstance(data, cudf.Series):
            data = data._column
        elif isinstance(data, column.ColumnBase):
            data = data
        else:
            if isinstance(data, (list, tuple)):
                if len(data) == 0:
                    data = np.asarray([], dtype="int64")
                else:
                    data = np.asarray(data)
            data = column.as_column(data)
            assert isinstance(data, (NumericalColumn, StringColumn))

        name = kwargs.get("name")
        super().__init__({name: data})

    @_cudf_nvtx_annotate
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ret = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

        if ret is not None:
            return ret

        # Attempt to dispatch all other functions to cupy.
        cupy_func = getattr(cupy, ufunc.__name__)
        if cupy_func:
            if ufunc.nin == 2:
                other = inputs[self is inputs[0]]
                inputs = self._make_operands_for_binop(other)
            else:
                inputs = {
                    name: (col, None, False, None)
                    for name, col in self._data.items()
                }

            data = self._apply_cupy_ufunc_to_operands(
                ufunc, cupy_func, inputs, **kwargs
            )

            out = [_index_from_data(out) for out in data]

            # pandas returns numpy arrays when the outputs are boolean.
            for i, o in enumerate(out):
                # We explicitly _do not_ use isinstance here: we want only
                # boolean GenericIndexes, not dtype-specific subclasses.
                if type(o) is GenericIndex and o.dtype.kind == "b":
                    out[i] = o.values

            return out[0] if ufunc.nout == 1 else tuple(out)

        return NotImplemented

    @classmethod
    @_cudf_nvtx_annotate
    def _from_data(
        cls, data: MutableMapping, name: Any = None
    ) -> GenericIndex:
        out = super()._from_data(data=data)
        if name is not None:
            out.name = name
        return out

    def _binaryop(
        self,
        other: T,
        op: str,
        fill_value: Any = None,
        *args,
        **kwargs,
    ) -> SingleColumnFrame:
        reflect, op = self._check_reflected_op(op)
        operands = self._make_operands_for_binop(other, fill_value, reflect)
        if operands is NotImplemented:
            return NotImplemented
        ret = _index_from_data(self._colwise_binop(operands, op))

        # pandas returns numpy arrays when the outputs are boolean. We
        # explicitly _do not_ use isinstance here: we want only boolean
        # GenericIndexes, not dtype-specific subclasses.
        if type(ret) is GenericIndex and ret.dtype.kind == "b":
            return ret.values
        return ret

    # Override just to make mypy happy.
    @_cudf_nvtx_annotate
    def _copy_type_metadata(
        self: GenericIndex, other: GenericIndex, *, override_dtypes=None
    ) -> GenericIndex:
        return super()._copy_type_metadata(
            other, override_dtypes=override_dtypes
        )

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def _values(self):
        return self._column

    @classmethod
    @_cudf_nvtx_annotate
    def _concat(cls, objs):
        if all(isinstance(obj, RangeIndex) for obj in objs):
            result = _concat_range_index(objs)
        else:
            data = concat_columns([o._values for o in objs])
            result = as_index(data)

        names = {obj.name for obj in objs}
        if len(names) == 1:
            [name] = names
        else:
            name = None

        result.name = name
        return result

    @_cudf_nvtx_annotate
    def memory_usage(self, deep=False):
        return self._column.memory_usage

    @_cudf_nvtx_annotate
    def equals(self, other, **kwargs):
        """
        Determine if two Index objects contain the same elements.

        Returns
        -------
        out: bool
            True if "other" is an Index and it has the same elements
            as calling index; False otherwise.
        """
        if (
            other is None
            or not isinstance(other, BaseIndex)
            or len(self) != len(other)
        ):
            return False

        check_dtypes = False

        self_is_categorical = isinstance(self, CategoricalIndex)
        other_is_categorical = isinstance(other, CategoricalIndex)
        if self_is_categorical and not other_is_categorical:
            other = other.astype(self.dtype)
            check_dtypes = True
        elif other_is_categorical and not self_is_categorical:
            self = self.astype(other.dtype)
            check_dtypes = True

        try:
            return self._column.equals(
                other._column, check_dtypes=check_dtypes
            )
        except TypeError:
            return False

    @_cudf_nvtx_annotate
    def copy(self, name=None, deep=False):
        """
        Make a copy of this object.

        Parameters
        ----------
        name : object, default None
            Name of index, use original name when None
        deep : bool, default True
            Make a deep copy of the data.
            With ``deep=False`` the original data is used

        Returns
        -------
        New index instance.
        """

        name = self.name if name is None else name

        return _index_from_data(
            {name: self._values.copy(True) if deep else self._values}
        )

    @_cudf_nvtx_annotate
    @doc_apply(_index_astype_docstring)
    def astype(self, dtype, copy: bool = True):
        return _index_from_data(super().astype({self.name: dtype}, copy))

    @_cudf_nvtx_annotate
    def get_loc(self, key, method=None, tolerance=None):
        """Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label
        method : {None, 'pad'/'fill', 'backfill'/'bfill', 'nearest'}, optional
            - default: exact matches only.
            - pad / ffill: find the PREVIOUS index value if no exact match.
            - backfill / bfill: use NEXT index value if no exact match.
            - nearest: use the NEAREST index value if no exact match. Tied
              distances are broken by preferring the larger index
              value.
        tolerance : int or float, optional
            Maximum distance from index value for inexact matches. The value
            of the index at the matching location must satisfy the equation
            ``abs(index[loc] - key) <= tolerance``.

        Returns
        -------
        int or slice or boolean mask
            - If result is unique, return integer index
            - If index is monotonic, loc is returned as a slice object
            - Otherwise, a boolean mask is returned

        Examples
        --------
        >>> unique_index = cudf.Index(list('abc'))
        >>> unique_index.get_loc('b')
        1
        >>> monotonic_index = cudf.Index(list('abbc'))
        >>> monotonic_index.get_loc('b')
        slice(1, 3, None)
        >>> non_monotonic_index = cudf.Index(list('abcb'))
        >>> non_monotonic_index.get_loc('b')
        array([False,  True, False,  True])
        >>> numeric_unique_index = cudf.Index([1, 2, 3])
        >>> numeric_unique_index.get_loc(3)
        2
        """
        # We should not actually remove this code until we have implemented the
        # get_indexers method as an alternative, see
        # https://github.com/rapidsai/cudf/issues/12312
        if method is not None:
            warnings.warn(
                f"Passing method to {self.__class__.__name__}.get_loc is "
                "deprecated and will raise in a future version.",
                FutureWarning,
            )
        if tolerance is not None:
            raise NotImplementedError(
                "Parameter tolerance is not supported yet."
            )
        if method not in {
            None,
            "ffill",
            "bfill",
            "pad",
            "backfill",
            "nearest",
        }:
            raise ValueError(
                f"Invalid fill method. Expecting pad (ffill), backfill (bfill)"
                f" or nearest. Got {method}"
            )

        is_sorted = (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        )

        if not is_sorted and method is not None:
            raise ValueError(
                "index must be monotonic increasing or decreasing if `method`"
                "is specified."
            )

        key_as_table = cudf.core.frame.Frame(
            {"None": as_column(key, length=1)}
        )
        lower_bound, upper_bound, sort_inds = _lexsorted_equal_range(
            self, key_as_table, is_sorted
        )

        if lower_bound == upper_bound:
            # Key not found, apply method
            if method in ("pad", "ffill"):
                if lower_bound == 0:
                    raise KeyError(key)
                return lower_bound - 1
            elif method in ("backfill", "bfill"):
                if lower_bound == self._data.nrows:
                    raise KeyError(key)
                return lower_bound
            elif method == "nearest":
                if lower_bound == self._data.nrows:
                    return lower_bound - 1
                elif lower_bound == 0:
                    return 0
                lower_val = self._column.element_indexing(lower_bound - 1)
                upper_val = self._column.element_indexing(lower_bound)
                return (
                    lower_bound - 1
                    if abs(lower_val - key) < abs(upper_val - key)
                    else lower_bound
                )
            else:
                raise KeyError(key)

        if lower_bound + 1 == upper_bound:
            # Search result is unique, return int.
            return (
                lower_bound
                if is_sorted
                else sort_inds.element_indexing(lower_bound)
            )

        if is_sorted:
            # In monotonic index, lex search result is continuous. A slice for
            # the range is returned.
            return slice(lower_bound, upper_bound)

        # Not sorted and not unique. Return a boolean mask
        mask = cupy.full(self._data.nrows, False)
        true_inds = sort_inds.slice(lower_bound, upper_bound).values
        mask[true_inds] = True
        return mask

    @_cudf_nvtx_annotate
    def __repr__(self):
        max_seq_items = get_option("max_seq_items") or len(self)
        mr = 0
        if 2 * max_seq_items < len(self):
            mr = max_seq_items + 1

        if len(self) > mr and mr != 0:
            top = self[0:mr]
            bottom = self[-1 * mr :]

            preprocess = cudf.concat([top, bottom])
        else:
            preprocess = self

        # TODO: Change below usages accordingly to
        # utilize `Index.to_string` once it is implemented
        # related issue : https://github.com/pandas-dev/pandas/issues/35389
        if isinstance(preprocess, CategoricalIndex):
            if preprocess.categories.dtype.kind == "f":
                output = repr(
                    preprocess.astype("str")
                    .to_pandas()
                    .astype(
                        dtype=pd.CategoricalDtype(
                            categories=preprocess.dtype.categories.astype(
                                "str"
                            ).to_pandas(),
                            ordered=preprocess.dtype.ordered,
                        )
                    )
                )
                break_idx = output.find("ordered=")
                output = (
                    output[:break_idx].replace("'", "") + output[break_idx:]
                )
            else:
                output = repr(preprocess.to_pandas())

            output = output.replace("nan", cudf._NA_REP)
        elif preprocess._values.nullable:
            output = repr(self._clean_nulls_from_index().to_pandas())

            if not isinstance(self, StringIndex):
                # We should remove all the single quotes
                # from the output due to the type-cast to
                # object dtype happening above.
                # Note : The replacing of single quotes has
                # to happen only in case of non-StringIndex types,
                # as we want to preserve single quotes in case
                # of StringIndex and it is valid to have them.
                output = output.replace("'", "")
        else:
            output = repr(preprocess.to_pandas())

        # Fix and correct the class name of the output
        # string by finding first occurrence of "(" in the output
        index_class_split_index = output.find("(")
        output = self.__class__.__name__ + output[index_class_split_index:]

        lines = output.split("\n")

        tmp_meta = lines[-1]
        dtype_index = tmp_meta.rfind(" dtype=")
        prior_to_dtype = tmp_meta[:dtype_index]
        lines = lines[:-1]
        lines.append(prior_to_dtype + " dtype='%s'" % self.dtype)
        if self.name is not None:
            lines[-1] = lines[-1] + ", name='%s'" % self.name
        if "length" in tmp_meta:
            lines[-1] = lines[-1] + ", length=%d)" % len(self)
        else:
            lines[-1] = lines[-1] + ")"

        return "\n".join(lines)

    @_cudf_nvtx_annotate
    def __getitem__(self, index):
        res = self._get_elements_from_column(index)
        if not isinstance(index, int):
            res = as_index(res)
            res.name = self.name
        return res

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def dtype(self):
        """
        `dtype` of the underlying values in GenericIndex.
        """
        return self._values.dtype

    @_cudf_nvtx_annotate
    def find_label_range(self, first, last):
        """Find range that starts with *first* and ends with *last*,
        inclusively.

        Returns
        -------
        begin, end : 2-tuple of int
            The starting index and the ending index.
            The *last* value occurs at ``end - 1`` position.
        """
        col = self._values
        begin, end = None, None
        if first is not None:
            begin = col.find_first_value(first, closest=True)
        if last is not None:
            end = col.find_last_value(last, closest=True)
            end += 1
        return begin, end

    @_cudf_nvtx_annotate
    def isna(self):
        return self._column.isnull().values

    isnull = isna

    @_cudf_nvtx_annotate
    def notna(self):
        return self._column.notnull().values

    notnull = notna

    @_cudf_nvtx_annotate
    def get_slice_bound(self, label, side):
        return self._values.get_slice_bound(label, side)

    def _is_numeric(self):
        return False

    def _is_boolean(self):
        return True

    def _is_integer(self):
        return False

    def _is_floating(self):
        return False

    def _is_object(self):
        return False

    def _is_categorical(self):
        return False

    def _is_interval(self):
        return False

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def hasnans(self):
        return self._column.has_nulls(include_nan=True)

    @_cudf_nvtx_annotate
    def argsort(
        self,
        axis=0,
        kind="quicksort",
        order=None,
        ascending=True,
        na_position="last",
    ):
        """Return the integer indices that would sort the Series values.

        Parameters
        ----------
        axis : {0 or "index"}
            Has no effect but is accepted for compatibility with numpy.
        kind : {'mergesort', 'quicksort', 'heapsort', 'stable'}, default 'quicksort'
            Choice of sorting algorithm. See :func:`numpy.sort` for more
            information. 'mergesort' and 'stable' are the only stable
            algorithms. Only quicksort is supported in cuDF.
        order : None
            Has no effect but is accepted for compatibility with numpy.
        ascending : bool or list of bool, default True
            If True, sort values in ascending order, otherwise descending.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs
            at the end.

        Returns
        -------
        cupy.ndarray: The indices sorted based on input.
        """  # noqa: E501
        return super().argsort(
            axis=axis,
            kind=kind,
            order=order,
            ascending=ascending,
            na_position=na_position,
        )

    def repeat(self, repeats, axis=None):
        return self._from_columns_like_self(
            Frame._repeat([*self._columns], repeats, axis), self._column_names
        )

    @_cudf_nvtx_annotate
    def where(self, cond, other=None, inplace=False):
        result_col = super().where(cond, other, inplace)
        return self._mimic_inplace(
            _index_from_data({self.name: result_col}),
            inplace=inplace,
        )

    @property
    def values(self):
        return self._column.values

    def __contains__(self, item):
        return item in self._values

    def _clean_nulls_from_index(self):
        if self._values.has_nulls():
            return cudf.Index(
                self._values.astype("str").fillna(cudf._NA_REP), name=self.name
            )

        return self

    def any(self):
        return self._values.any()

    def to_pandas(self, nullable=False):
        return pd.Index(
            self._values.to_pandas(nullable=nullable), name=self.name
        )

    def append(self, other):
        if is_list_like(other):
            to_concat = [self]
            to_concat.extend(other)
        else:
            this = self
            if len(other) == 0:
                # short-circuit and return a copy
                to_concat = [self]

            other = cudf.Index(other)

            if len(self) == 0:
                to_concat = [other]

            if len(self) and len(other):
                if is_mixed_with_object_dtype(this, other):
                    got_dtype = (
                        other.dtype
                        if this.dtype == cudf.dtype("object")
                        else this.dtype
                    )
                    raise TypeError(
                        f"cudf does not support appending an Index of "
                        f"dtype `{cudf.dtype('object')}` with an Index "
                        f"of dtype `{got_dtype}`, please type-cast "
                        f"either one of them to same dtypes."
                    )

                if isinstance(self._values, cudf.core.column.NumericalColumn):
                    if self.dtype != other.dtype:
                        this, other = numeric_normalize_types(self, other)
                to_concat = [this, other]

        for obj in to_concat:
            if not isinstance(obj, BaseIndex):
                raise TypeError("all inputs must be Index")

        return self._concat(to_concat)

    def unique(self):
        return cudf.core.index._index_from_data(
            {self.name: self._values.unique()}, name=self.name
        )

    def isin(self, values):
        if is_scalar(values):
            raise TypeError(
                "only list-like objects are allowed to be passed "
                f"to isin(), you passed a {type(values).__name__}"
            )

        return self._values.isin(values).values


class NumericIndex(GenericIndex):
    """Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    Index
    """

    # Subclasses must define the dtype they are associated with.
    _dtype: Union[None, Type[np.number]] = None

    @_cudf_nvtx_annotate
    def __init__(self, data=None, dtype=None, copy=False, name=None):
        warnings.warn(
            f"cudf.{self.__class__.__name__} is deprecated and will be "
            "removed from cudf in a future version. Use cudf.Index with the "
            "appropriate dtype instead.",
            FutureWarning,
        )

        dtype = type(self)._dtype
        if copy:
            data = column.as_column(data, dtype=dtype).copy()

        kwargs = _setdefault_name(data, name=name)

        data = column.as_column(data, dtype=dtype)

        super().__init__(data, **kwargs)

    def _is_numeric(self):
        return True

    def _is_boolean(self):
        return False

    def _is_integer(self):
        return True

    def _is_floating(self):
        return False

    def _is_object(self):
        return False

    def _is_categorical(self):
        return False

    def _is_interval(self):
        return False


class Int8Index(NumericIndex):
    """
    Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.
    Int8Index is a special case of Index with purely
    integer(``int8``) labels.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    Int8Index
    """

    _dtype = np.int8


class Int16Index(NumericIndex):
    """
    Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.
    Int16Index is a special case of Index with purely
    integer(``int16``) labels.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    Int16Index
    """

    _dtype = np.int16


class Int32Index(NumericIndex):
    """
    Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.
    Int32Index is a special case of Index with purely
    integer(``int32``) labels.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    Int32Index
    """

    _dtype = np.int32


class Int64Index(NumericIndex):
    """
    Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.
    Int64Index is a special case of Index with purely
    integer(``int64``) labels.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    Int64Index
    """

    _dtype = np.int64


class UInt8Index(NumericIndex):
    """
    Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.
    UInt8Index is a special case of Index with purely
    integer(``uint64``) labels.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    UInt8Index
    """

    _dtype = np.uint8


class UInt16Index(NumericIndex):
    """
    Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.
    UInt16Index is a special case of Index with purely
    integer(``uint16``) labels.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    UInt16Index
    """

    _dtype = np.uint16


class UInt32Index(NumericIndex):
    """
    Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.
    UInt32Index is a special case of Index with purely
    integer(``uint32``) labels.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    UInt32Index
    """

    _dtype = np.uint32


class UInt64Index(NumericIndex):
    """
    Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.
    UInt64Index is a special case of Index with purely
    integer(``uint64``) labels.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    UInt64Index
    """

    _dtype = np.uint64


class Float32Index(NumericIndex):
    """
    Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.
    Float32Index is a special case of Index with purely
    float(``float32``) labels.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    Float32Index
    """

    _dtype = np.float32

    def _is_integer(self):
        return False

    def _is_floating(self):
        return True


class Float64Index(NumericIndex):
    """
    Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.
    Float64Index is a special case of Index with purely
    float(``float64``) labels.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    Float64Index
    """

    _dtype = np.float64

    def _is_integer(self):
        return False

    def _is_floating(self):
        return True


class DatetimeIndex(GenericIndex):
    """
    Immutable , ordered and sliceable sequence of datetime64 data,
    represented internally as int64.

    Parameters
    ----------
    data : array-like (1-dimensional), optional
        Optional datetime-like data to construct index with.
    copy : bool
        Make a copy of input.
    freq : str, optional
        This is not yet supported
    tz : pytz.timezone or dateutil.tz.tzfile
        This is not yet supported
    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
        This is not yet supported
    name : object
        Name to be stored in the index.
    dayfirst : bool, default False
        If True, parse dates in data with the day first order.
        This is not yet supported
    yearfirst : bool, default False
        If True parse dates in data with the year first order.
        This is not yet supported

    Returns
    -------
    DatetimeIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.DatetimeIndex([1, 2, 3, 4], name="a")
    DatetimeIndex(['1970-01-01 00:00:00.000000001',
                   '1970-01-01 00:00:00.000000002',
                   '1970-01-01 00:00:00.000000003',
                   '1970-01-01 00:00:00.000000004'],
                  dtype='datetime64[ns]', name='a')
    """

    @_cudf_nvtx_annotate
    def __init__(
        self,
        data=None,
        freq=None,
        tz=None,
        normalize=False,
        closed=None,
        ambiguous="raise",
        dayfirst=False,
        yearfirst=False,
        dtype=None,
        copy=False,
        name=None,
    ):
        # we should be more strict on what we accept here but
        # we'd have to go and figure out all the semantics around
        # pandas dtindex creation first which.  For now
        # just make sure we handle np.datetime64 arrays
        # and then just dispatch upstream
        if freq is not None:
            raise NotImplementedError("Freq is not yet supported")
        if tz is not None:
            raise NotImplementedError("tz is not yet supported")
        if normalize is not False:
            raise NotImplementedError("normalize == True is not yet supported")
        if closed is not None:
            raise NotImplementedError("closed is not yet supported")
        if ambiguous != "raise":
            raise NotImplementedError("ambiguous is not yet supported")
        if dayfirst is not False:
            raise NotImplementedError("dayfirst == True is not yet supported")
        if yearfirst is not False:
            raise NotImplementedError("yearfirst == True is not yet supported")

        valid_dtypes = tuple(
            f"datetime64[{res}]" for res in ("s", "ms", "us", "ns")
        )
        if dtype is None:
            # nanosecond default matches pandas
            dtype = "datetime64[ns]"
        elif dtype not in valid_dtypes:
            raise TypeError("Invalid dtype")

        kwargs = _setdefault_name(data, name=name)
        data = column.as_column(data, dtype=dtype)

        if copy:
            data = data.copy()

        super().__init__(data, **kwargs)

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def year(self):
        """
        The year of the datetime.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="Y"))
        >>> datetime_index
        DatetimeIndex(['2000-12-31', '2001-12-31', '2002-12-31'], dtype='datetime64[ns]')
        >>> datetime_index.year
        Int16Index([2000, 2001, 2002], dtype='int16')
        """  # noqa: E501
        return self._get_dt_field("year")

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def month(self):
        """
        The month as January=1, December=12.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="M"))
        >>> datetime_index
        DatetimeIndex(['2000-01-31', '2000-02-29', '2000-03-31'], dtype='datetime64[ns]')
        >>> datetime_index.month
        Int16Index([1, 2, 3], dtype='int16')
        """  # noqa: E501
        return self._get_dt_field("month")

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def day(self):
        """
        The day of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="D"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-03'], dtype='datetime64[ns]')
        >>> datetime_index.day
        Int16Index([1, 2, 3], dtype='int16')
        """  # noqa: E501
        return self._get_dt_field("day")

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def hour(self):
        """
        The hours of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="h"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01 00:00:00', '2000-01-01 01:00:00',
                    '2000-01-01 02:00:00'],
                    dtype='datetime64[ns]')
        >>> datetime_index.hour
        Int16Index([0, 1, 2], dtype='int16')
        """
        return self._get_dt_field("hour")

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def minute(self):
        """
        The minutes of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="T"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01 00:00:00', '2000-01-01 00:01:00',
                    '2000-01-01 00:02:00'],
                    dtype='datetime64[ns]')
        >>> datetime_index.minute
        Int16Index([0, 1, 2], dtype='int16')
        """
        return self._get_dt_field("minute")

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def second(self):
        """
        The seconds of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="s"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01 00:00:00', '2000-01-01 00:00:01',
                    '2000-01-01 00:00:02'],
                    dtype='datetime64[ns]')
        >>> datetime_index.second
        Int16Index([0, 1, 2], dtype='int16')
        """
        return self._get_dt_field("second")

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def microsecond(self):
        """
        The microseconds of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="us"))
        >>> datetime_index
        DatetimeIndex([       '2000-01-01 00:00:00', '2000-01-01 00:00:00.000001',
               '2000-01-01 00:00:00.000002'],
              dtype='datetime64[ns]')
        >>> datetime_index.microsecond
        Int32Index([0, 1, 2], dtype='int32')
        """  # noqa: E501
        return as_index(
            (
                # Need to manually promote column to int32 because
                # pandas-matching binop behaviour requires that this
                # __mul__ returns an int16 column.
                self._values.get_dt_field("millisecond").astype("int32")
                * cudf.Scalar(1000, dtype="int32")
            )
            + self._values.get_dt_field("microsecond"),
            name=self.name,
        )

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def nanosecond(self):
        """
        The nanoseconds of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="ns"))
        >>> datetime_index
        DatetimeIndex([          '2000-01-01 00:00:00',
                       '2000-01-01 00:00:00.000000001',
                       '2000-01-01 00:00:00.000000002'],
                      dtype='datetime64[ns]')
        >>> datetime_index.nanosecond
        Int16Index([0, 1, 2], dtype='int16')
        """
        return self._get_dt_field("nanosecond")

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def weekday(self):
        """
        The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[ns]')
        >>> datetime_index.weekday
        Int16Index([5, 6, 0, 1, 2, 3, 4, 5, 6], dtype='int16')
        """
        return self._get_dt_field("weekday")

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def dayofweek(self):
        """
        The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[ns]')
        >>> datetime_index.dayofweek
        Int16Index([5, 6, 0, 1, 2, 3, 4, 5, 6], dtype='int16')
        """
        return self._get_dt_field("weekday")

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def dayofyear(self):
        """
        The day of the year, from 1-365 in non-leap years and
        from 1-366 in leap years.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[ns]')
        >>> datetime_index.dayofyear
        Int16Index([366, 1, 2, 3, 4, 5, 6, 7, 8], dtype='int16')
        """
        return self._get_dt_field("day_of_year")

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def day_of_year(self):
        """
        The day of the year, from 1-365 in non-leap years and
        from 1-366 in leap years.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[ns]')
        >>> datetime_index.day_of_year
        Int16Index([366, 1, 2, 3, 4, 5, 6, 7, 8], dtype='int16')
        """
        return self._get_dt_field("day_of_year")

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def is_leap_year(self):
        """
        Boolean indicator if the date belongs to a leap year.

        A leap year is a year, which has 366 days (instead of 365) including
        29th of February as an intercalary day. Leap years are years which are
        multiples of four with the exception of years divisible by 100 but not
        by 400.

        Returns
        -------
        ndarray
        Booleans indicating if dates belong to a leap year.
        """
        res = is_leap_year(self._values).fillna(False)
        return cupy.asarray(res)

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def quarter(self):
        """
        Integer indicator for which quarter of the year the date belongs in.

        There are 4 quarters in a year. With the first quarter being from
        January - March, second quarter being April - June, third quarter
        being July - September and fourth quarter being October - December.

        Returns
        -------
        Int8Index
        Integer indicating which quarter the date belongs to.

        Examples
        --------
        >>> import cudf
        >>> gIndex = cudf.DatetimeIndex(["2020-05-31 08:00:00",
        ...    "1999-12-31 18:40:00"])
        >>> gIndex.quarter
        Int8Index([2, 4], dtype='int8')
        """
        res = extract_quarter(self._values)
        return Index(res, dtype="int8")

    @_cudf_nvtx_annotate
    def isocalendar(self):
        """
        Returns a DataFrame with the year, week, and day
        calculated according to the ISO 8601 standard.

        Returns
        -------
        DataFrame
        with columns year, week and day

        Examples
        --------
        >>> gIndex = cudf.DatetimeIndex(["2020-05-31 08:00:00",
        ...    "1999-12-31 18:40:00"])
        >>> gIndex.isocalendar()
                             year  week  day
        2020-05-31 08:00:00  2020    22    7
        1999-12-31 18:40:00  1999    52    5
        """
        return cudf.core.tools.datetimes._to_iso_calendar(self)

    @_cudf_nvtx_annotate
    def to_pandas(self, nullable=False):
        nanos = self._values.astype("datetime64[ns]")
        return pd.DatetimeIndex(nanos.to_pandas(), name=self.name)

    @_cudf_nvtx_annotate
    def _get_dt_field(self, field):
        out_column = self._values.get_dt_field(field)
        # column.column_empty_like always returns a Column object
        # but we need a NumericalColumn for GenericIndex..
        # how should this be handled?
        out_column = column.build_column(
            data=out_column.base_data,
            dtype=out_column.dtype,
            mask=out_column.base_mask,
            offset=out_column.offset,
        )
        return as_index(out_column, name=self.name)

    def _is_boolean(self):
        return False

    @_cudf_nvtx_annotate
    def ceil(self, freq):
        """
        Perform ceil operation on the data to the specified freq.

        Parameters
        ----------
        freq : str
            One of ["D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"].
            Must be a fixed frequency like 'S' (second) not 'ME' (month end).
            See `frequency aliases <https://pandas.pydata.org/docs/\
                user_guide/timeseries.html#timeseries-offset-aliases>`__
            for more details on these aliases.

        Returns
        -------
        DatetimeIndex
            Index of the same type for a DatetimeIndex

        Examples
        --------
        >>> import cudf
        >>> gIndex = cudf.DatetimeIndex([
        ...     "2020-05-31 08:05:42",
        ...     "1999-12-31 18:40:30",
        ... ])
        >>> gIndex.ceil("T")
        DatetimeIndex(['2020-05-31 08:06:00', '1999-12-31 18:41:00'], dtype='datetime64[ns]')
        """  # noqa: E501
        out_column = self._values.ceil(freq)

        return self.__class__._from_data({self.name: out_column})

    @_cudf_nvtx_annotate
    def floor(self, freq):
        """
        Perform floor operation on the data to the specified freq.

        Parameters
        ----------
        freq : str
            One of ["D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"].
            Must be a fixed frequency like 'S' (second) not 'ME' (month end).
            See `frequency aliases <https://pandas.pydata.org/docs/\
                user_guide/timeseries.html#timeseries-offset-aliases>`__
            for more details on these aliases.

        Returns
        -------
        DatetimeIndex
            Index of the same type for a DatetimeIndex

        Examples
        --------
        >>> import cudf
        >>> gIndex = cudf.DatetimeIndex([
        ...     "2020-05-31 08:59:59",
        ...     "1999-12-31 18:44:59",
        ... ])
        >>> gIndex.floor("T")
        DatetimeIndex(['2020-05-31 08:59:00', '1999-12-31 18:44:00'], dtype='datetime64[ns]')
        """  # noqa: E501
        out_column = self._values.floor(freq)

        return self.__class__._from_data({self.name: out_column})

    @_cudf_nvtx_annotate
    def round(self, freq):
        """
        Perform round operation on the data to the specified freq.

        Parameters
        ----------
        freq : str
            One of ["D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"].
            Must be a fixed frequency like 'S' (second) not 'ME' (month end).
            See `frequency aliases <https://pandas.pydata.org/docs/\
                user_guide/timeseries.html#timeseries-offset-aliases>`__
            for more details on these aliases.

        Returns
        -------
        DatetimeIndex
            Index containing rounded datetimes.

        Examples
        --------
        >>> import cudf
        >>> dt_idx = cudf.Index([
        ...     "2001-01-01 00:04:45",
        ...     "2001-01-01 00:04:58",
        ...     "2001-01-01 00:05:04",
        ... ], dtype="datetime64[ns]")
        >>> dt_idx
        DatetimeIndex(['2001-01-01 00:04:45', '2001-01-01 00:04:58',
                       '2001-01-01 00:05:04'],
                      dtype='datetime64[ns]')
        >>> dt_idx.round('H')
        DatetimeIndex(['2001-01-01', '2001-01-01', '2001-01-01'], dtype='datetime64[ns]')
        >>> dt_idx.round('T')
        DatetimeIndex(['2001-01-01 00:05:00', '2001-01-01 00:05:00', '2001-01-01 00:05:00'], dtype='datetime64[ns]')
        """  # noqa: E501
        out_column = self._values.round(freq)

        return self.__class__._from_data({self.name: out_column})


class TimedeltaIndex(GenericIndex):
    """
    Immutable, ordered and sliceable sequence of timedelta64 data,
    represented internally as int64.

    Parameters
    ----------
    data : array-like (1-dimensional), optional
        Optional datetime-like data to construct index with.
    unit : str, optional
        This is not yet supported
    copy : bool
        Make a copy of input.
    freq : str, optional
        This is not yet supported
    closed : str, optional
        This is not yet supported
    dtype : str or :class:`numpy.dtype`, optional
        Data type for the output Index. If not specified, the
        default dtype will be ``timedelta64[ns]``.
    name : object
        Name to be stored in the index.

    Returns
    -------
    TimedeltaIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.TimedeltaIndex([1132223, 2023232, 342234324, 4234324],
    ...     dtype="timedelta64[ns]")
    TimedeltaIndex(['0 days 00:00:00.001132223', '0 days 00:00:00.002023232',
                    '0 days 00:00:00.342234324', '0 days 00:00:00.004234324'],
                  dtype='timedelta64[ns]')
    >>> cudf.TimedeltaIndex([1, 2, 3, 4], dtype="timedelta64[s]",
    ...     name="delta-index")
    TimedeltaIndex(['0 days 00:00:01', '0 days 00:00:02', '0 days 00:00:03',
                    '0 days 00:00:04'],
                  dtype='timedelta64[s]', name='delta-index')
    """

    @_cudf_nvtx_annotate
    def __init__(
        self,
        data=None,
        unit=None,
        freq=None,
        closed=None,
        dtype="timedelta64[ns]",
        copy=False,
        name=None,
    ):

        if freq is not None:
            raise NotImplementedError("freq is not yet supported")

        if unit is not None:
            raise NotImplementedError(
                "unit is not yet supported, alternatively "
                "dtype parameter is supported"
            )

        valid_dtypes = tuple(
            f"timedelta64[{res}]" for res in ("s", "ms", "us", "ns")
        )
        if dtype not in valid_dtypes:
            raise TypeError("Invalid dtype")

        kwargs = _setdefault_name(data, name=name)
        data = column.as_column(data, dtype=dtype)

        if copy:
            data = data.copy()

        super().__init__(data, **kwargs)

    @_cudf_nvtx_annotate
    def to_pandas(self, nullable=False):
        return pd.TimedeltaIndex(
            self._values.to_pandas(),
            name=self.name,
            unit=self._values.time_unit,
        )

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def days(self):
        """
        Number of days for each element.
        """
        return as_index(arbitrary=self._values.days, name=self.name)

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def seconds(self):
        """
        Number of seconds (>= 0 and less than 1 day) for each element.
        """
        return as_index(arbitrary=self._values.seconds, name=self.name)

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def microseconds(self):
        """
        Number of microseconds (>= 0 and less than 1 second) for each element.
        """
        return as_index(arbitrary=self._values.microseconds, name=self.name)

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def nanoseconds(self):
        """
        Number of nanoseconds (>= 0 and less than 1 microsecond) for each
        element.
        """
        return as_index(arbitrary=self._values.nanoseconds, name=self.name)

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def components(self):
        """
        Return a dataframe of the components (days, hours, minutes,
        seconds, milliseconds, microseconds, nanoseconds) of the Timedeltas.
        """
        return self._values.components()

    @property
    def inferred_freq(self):
        """
        Infers frequency of TimedeltaIndex.

        Notes
        -----
        This property is currently not supported.
        """
        raise NotImplementedError("inferred_freq is not yet supported")

    def _is_boolean(self):
        return False


class CategoricalIndex(GenericIndex):
    """
    A categorical of orderable values that represent the indices of another
    Column

    Parameters
    ----------
    data : array-like (1-dimensional)
        The values of the categorical. If categories are given,
        values not in categories will be replaced with None/NaN.
    categories : list-like, optional
        The categories for the categorical. Items need to be unique.
        If the categories are not given here (and also not in dtype),
        they will be inferred from the data.
    ordered : bool, optional
        Whether or not this categorical is treated as an ordered categorical.
        If not given here or in dtype, the resulting categorical will be
        unordered.
    dtype : CategoricalDtype or "category", optional
        If CategoricalDtype, cannot be used together with categories or
        ordered.
    copy : bool, default False
        Make a copy of input.
    name : object, optional
        Name to be stored in the index.

    Returns
    -------
    CategoricalIndex

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> cudf.CategoricalIndex(
    ... data=[1, 2, 3, 4], categories=[1, 2], ordered=False, name="a")
    CategoricalIndex([1, 2, <NA>, <NA>], categories=[1, 2], ordered=False, dtype='category', name='a')

    >>> cudf.CategoricalIndex(
    ... data=[1, 2, 3, 4], dtype=pd.CategoricalDtype([1, 2, 3]), name="a")
    CategoricalIndex([1, 2, 3, <NA>], categories=[1, 2, 3], ordered=False, dtype='category', name='a')
    """  # noqa: E501

    @_cudf_nvtx_annotate
    def __init__(
        self,
        data=None,
        categories=None,
        ordered=None,
        dtype=None,
        copy=False,
        name=None,
    ):
        if isinstance(dtype, (pd.CategoricalDtype, cudf.CategoricalDtype)):
            if categories is not None or ordered is not None:
                raise ValueError(
                    "Cannot specify `categories` or "
                    "`ordered` together with `dtype`."
                )
        if copy:
            data = column.as_column(data, dtype=dtype).copy(deep=True)
        kwargs = _setdefault_name(data, name=name)
        if isinstance(data, CategoricalColumn):
            data = data
        elif isinstance(data, pd.Series) and (
            is_categorical_dtype(data.dtype)
        ):
            codes_data = column.as_column(data.cat.codes.values)
            data = column.build_categorical_column(
                categories=data.cat.categories,
                codes=codes_data,
                ordered=data.cat.ordered,
            )
        elif isinstance(data, (pd.Categorical, pd.CategoricalIndex)):
            codes_data = column.as_column(data.codes)
            data = column.build_categorical_column(
                categories=data.categories,
                codes=codes_data,
                ordered=data.ordered,
            )
        else:
            data = column.as_column(
                data, dtype="category" if dtype is None else dtype
            )
            # dtype has already been taken care
            dtype = None

        if categories is not None:
            data = data.set_categories(categories, ordered=ordered)
        elif isinstance(dtype, (pd.CategoricalDtype, cudf.CategoricalDtype)):
            data = data.set_categories(dtype.categories, ordered=ordered)
        elif ordered is True and data.ordered is False:
            data = data.as_ordered()
        elif ordered is False and data.ordered is True:
            data = data.as_unordered()

        super().__init__(data, **kwargs)

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def codes(self):
        """
        The category codes of this categorical.
        """
        return as_index(self._values.codes)

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def categories(self):
        """
        The categories of this categorical.
        """
        return as_index(self._values.categories)

    def _is_boolean(self):
        return False

    def _is_categorical(self):
        return True


@_cudf_nvtx_annotate
def interval_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    name=None,
    closed="right",
) -> "IntervalIndex":
    """
    Returns a fixed frequency IntervalIndex.

    Parameters
    ----------
    start : numeric, default None
        Left bound for generating intervals.
    end : numeric , default None
        Right bound for generating intervals.
    periods : int, default None
        Number of periods to generate
    freq : numeric, default None
        The length of each interval. Must be consistent
        with the type of start and end
    name : str, default None
        Name of the resulting IntervalIndex.
    closed : {"left", "right", "both", "neither"}, default "right"
        Whether the intervals are closed on the left-side, right-side,
        both or neither.

    Returns
    -------
    IntervalIndex

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> cudf.interval_range(start=0,end=5)
    IntervalIndex([(0, 0], (1, 1], (2, 2], (3, 3], (4, 4], (5, 5]],
    ...closed='right',dtype='interval')
    >>> cudf.interval_range(start=0,end=10, freq=2,closed='left')
    IntervalIndex([[0, 2), [2, 4), [4, 6), [6, 8), [8, 10)],
    ...closed='left',dtype='interval')
    >>> cudf.interval_range(start=0,end=10, periods=3,closed='left')
    ...IntervalIndex([[0.0, 3.3333333333333335),
            [3.3333333333333335, 6.666666666666667),
            [6.666666666666667, 10.0)],
            closed='left',
            dtype='interval')
    """
    if freq and periods and start and end:
        raise ValueError(
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )
    args = [
        cudf.Scalar(x) if x is not None else None
        for x in (start, end, freq, periods)
    ]
    if any(
        not _is_non_decimal_numeric_dtype(x.dtype) if x is not None else False
        for x in args
    ):
        raise ValueError("start, end, periods, freq must be numeric values.")
    *rargs, periods = args
    common_dtype = find_common_type([x.dtype for x in rargs if x])
    start, end, freq = rargs
    periods = periods.astype("int64") if periods is not None else None

    if periods and not freq:
        # if statement for mypy to pass
        if end is not None and start is not None:
            # divmod only supported on host side scalars
            quotient, remainder = divmod((end - start).value, periods.value)
            if remainder:
                freq_step = cudf.Scalar((end - start) / periods)
            else:
                freq_step = cudf.Scalar(quotient)
            if start.dtype != freq_step.dtype:
                start = start.astype(freq_step.dtype)
            bin_edges = sequence(
                size=periods + 1,
                init=start.device_value,
                step=freq_step.device_value,
            )
            left_col = bin_edges.slice(0, len(bin_edges) - 1)
            right_col = bin_edges.slice(1, len(bin_edges))
    elif freq and periods:
        if end:
            start = end - (freq * periods)
        if start:
            end = freq * periods + start
        if end is not None and start is not None:
            left_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
            end = end + 1
            start = start + freq
            right_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
    elif freq and not periods:
        if end is not None and start is not None:
            end = end - freq + 1
            left_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
            end = end + freq + 1
            start = start + freq
            right_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
    elif start is not None and end is not None:
        # if statements for mypy to pass
        if freq:
            left_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
        else:
            left_col = arange(start.value, end.value, dtype=common_dtype)
        start = start + 1
        end = end + 1
        if freq:
            right_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
        else:
            right_col = arange(start.value, end.value, dtype=common_dtype)
    else:
        raise ValueError(
            "Of the four parameters: start, end, periods, and "
            "freq, at least two must be specified"
        )
    if len(right_col) == 0 or len(left_col) == 0:
        dtype = IntervalDtype("int64", closed)
        data = column.column_empty_like_same_mask(left_col, dtype)
        return IntervalIndex(data, closed=closed)

    interval_col = column.build_interval_column(
        left_col, right_col, closed=closed
    )
    return IntervalIndex(interval_col)


class IntervalIndex(GenericIndex):
    """
    Immutable index of intervals that are closed on the same side.

    Parameters
    ----------
    data : array-like (1-dimensional)
        Array-like containing Interval objects from which to build the
        IntervalIndex.
    closed : {"left", "right", "both", "neither"}, default "right"
        Whether the intervals are closed on the left-side, right-side,
        both or neither.
    dtype : dtype or None, default None
        If None, dtype will be inferred.
    copy : bool, default False
        Copy the input data.
    name : object, optional
        Name to be stored in the index.

    Returns
    -------
    IntervalIndex
    """

    @_cudf_nvtx_annotate
    def __init__(
        self,
        data,
        closed=None,
        dtype=None,
        copy=False,
        name=None,
    ):
        if copy:
            data = column.as_column(data, dtype=dtype).copy()
        kwargs = _setdefault_name(data, name=name)
        if isinstance(data, IntervalColumn):
            data = data
        elif isinstance(data, pd.Series) and (is_interval_dtype(data.dtype)):
            data = column.as_column(data, data.dtype)
        elif isinstance(data, (pd._libs.interval.Interval, pd.IntervalIndex)):
            data = column.as_column(
                data,
                dtype=dtype,
            )
        elif not data:
            dtype = IntervalDtype("int64", closed)
            data = column.column_empty_like_same_mask(
                column.as_column(data), dtype
            )
        else:
            data = column.as_column(data)
            data.dtype.closed = closed

        self.closed = closed
        super().__init__(data, **kwargs)

    @_cudf_nvtx_annotate
    def from_breaks(breaks, closed="right", name=None, copy=False, dtype=None):
        """
        Construct an IntervalIndex from an array of splits.

        Parameters
        ----------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {"left", "right", "both", "neither"}, default "right"
            Whether the intervals are closed on the left-side, right-side,
            both or neither.
        copy : bool, default False
            Copy the input data.
        name : object, optional
            Name to be stored in the index.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        IntervalIndex

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> cudf.IntervalIndex.from_breaks([0, 1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]], dtype='interval')
        """
        if copy:
            breaks = column.as_column(breaks, dtype=dtype).copy()
        left_col = breaks[:-1:]
        right_col = breaks[+1::]

        interval_col = column.build_interval_column(
            left_col, right_col, closed=closed
        )

        return IntervalIndex(interval_col, name=name)

    def __getitem__(self, index):
        raise NotImplementedError(
            "Getting a scalar from an IntervalIndex is not yet supported"
        )

    def _is_interval(self):
        return True

    def _is_boolean(self):
        return False


class StringIndex(GenericIndex):
    """String defined indices into another Column

    Attributes
    ----------
    _values: A StringColumn object or NDArray of strings
    name: A string
    """

    @_cudf_nvtx_annotate
    def __init__(self, values, copy=False, **kwargs):
        kwargs = _setdefault_name(values, **kwargs)
        if isinstance(values, StringColumn):
            values = values.copy(deep=copy)
        elif isinstance(values, StringIndex):
            values = values._values.copy(deep=copy)
        else:
            values = column.as_column(values, dtype="str")
            if not is_string_dtype(values.dtype):
                raise ValueError(
                    "Couldn't create StringIndex from passed in object"
                )

        super().__init__(values, **kwargs)

    @_cudf_nvtx_annotate
    def to_pandas(self, nullable=False):
        return pd.Index(
            self.to_numpy(na_value=None),
            name=self.name,
            dtype=pd.StringDtype() if nullable else "object",
        )

    @_cudf_nvtx_annotate
    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self._values.values_host},"
            f" dtype='object'"
            + (
                f", name={pd.io.formats.printing.default_pprint(self.name)}"
                if self.name is not None
                else ""
            )
            + ")"
        )

    @copy_docstring(StringMethods)  # type: ignore
    @property
    @_cudf_nvtx_annotate
    def str(self):
        return StringMethods(parent=self)

    def _clean_nulls_from_index(self):
        if self._values.has_nulls():
            return self.fillna(cudf._NA_REP)
        else:
            return self

    def _is_boolean(self):
        return False

    def _is_object(self):
        return True


@_cudf_nvtx_annotate
def as_index(arbitrary, nan_as_null=None, **kwargs) -> BaseIndex:
    """Create an Index from an arbitrary object

    Currently supported inputs are:

    * ``Column``
    * ``Buffer``
    * ``Series``
    * ``Index``
    * numba device array
    * numpy array
    * pyarrow array
    * pandas.Categorical

    Returns
    -------
    result : subclass of Index
        - CategoricalIndex for Categorical input.
        - DatetimeIndex for Datetime input.
        - GenericIndex for all other inputs.
    """
    kwargs = _setdefault_name(arbitrary, **kwargs)
    if isinstance(arbitrary, cudf.MultiIndex):
        return arbitrary
    elif isinstance(arbitrary, BaseIndex):
        if arbitrary.name == kwargs["name"]:
            return arbitrary
        idx = arbitrary.copy(deep=False)
        idx.rename(kwargs["name"], inplace=True)
        return idx
    elif isinstance(arbitrary, ColumnBase):
        res = _index_from_data({kwargs.get("name", None): arbitrary})
        if (dtype := kwargs.get("dtype")) is not None:
            res = res.astype(dtype)
        return res
    elif isinstance(arbitrary, cudf.Series):
        return as_index(arbitrary._column, nan_as_null=nan_as_null, **kwargs)
    elif isinstance(arbitrary, (pd.RangeIndex, range)):
        return RangeIndex(
            start=arbitrary.start,
            stop=arbitrary.stop,
            step=arbitrary.step,
            **kwargs,
        )
    elif isinstance(arbitrary, pd.MultiIndex):
        return cudf.MultiIndex.from_pandas(arbitrary, nan_as_null=nan_as_null)
    elif isinstance(arbitrary, cudf.DataFrame):
        return cudf.MultiIndex.from_frame(arbitrary)
    return as_index(
        column.as_column(
            arbitrary, dtype=kwargs.get("dtype", None), nan_as_null=nan_as_null
        ),
        **kwargs,
    )


_dtype_to_index: Dict[Any, Type[NumericIndex]] = {
    np.int8: Int8Index,
    np.int16: Int16Index,
    np.int32: Int32Index,
    np.int64: Int64Index,
    np.uint8: UInt8Index,
    np.uint16: UInt16Index,
    np.uint32: UInt32Index,
    np.uint64: UInt64Index,
    np.float32: Float32Index,
    np.float64: Float64Index,
}


def _setdefault_name(values, **kwargs):
    if kwargs.get("name") is None:
        kwargs["name"] = getattr(values, "name", None)
    return kwargs


class IndexMeta(type):
    """Custom metaclass for Index that overrides instance/subclass tests."""

    def __instancecheck__(self, instance):
        return isinstance(instance, BaseIndex)

    def __subclasscheck__(self, subclass):
        return issubclass(subclass, BaseIndex)


class Index(BaseIndex, metaclass=IndexMeta):
    """The basic object storing row labels for all cuDF objects.

    Parameters
    ----------
    data : array-like (1-dimensional)/ DataFrame
        If it is a DataFrame, it will return a MultiIndex
    dtype : NumPy dtype (default: object)
        If dtype is None, we find the dtype that best fits the data.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.
    tupleize_cols : bool (default: True)
        When True, attempt to create a MultiIndex if possible.
        tupleize_cols == False is not yet supported.
    nan_as_null : bool, Default True
        If ``None``/``True``, converts ``np.nan`` values to
        ``null`` values.
        If ``False``, leaves ``np.nan`` values as is.

    Returns
    -------
    Index
        cudf Index

    Warnings
    --------
    This class should not be subclassed. It is designed as a factory for
    different subclasses of :class:`BaseIndex` depending on the provided input.
    If you absolutely must, and if you're intimately familiar with the
    internals of cuDF, subclass :class:`BaseIndex` instead.

    Examples
    --------
    >>> import cudf
    >>> cudf.Index([1, 2, 3], dtype="uint64", name="a")
    UInt64Index([1, 2, 3], dtype='uint64', name='a')

    >>> cudf.Index(cudf.DataFrame({"a":[1, 2], "b":[2, 3]}))
    MultiIndex([(1, 2),
                (2, 3)],
                names=['a', 'b'])
    """

    @_cudf_nvtx_annotate
    def __new__(
        cls,
        data=None,
        dtype=None,
        copy=False,
        name=None,
        tupleize_cols=True,
        nan_as_null=True,
        **kwargs,
    ):
        assert (
            cls is Index
        ), "Index cannot be subclassed, extend BaseIndex instead."
        if tupleize_cols is not True:
            raise NotImplementedError(
                "tupleize_cols != True is not yet supported"
            )

        return as_index(
            data,
            copy=copy,
            dtype=dtype,
            name=name,
            nan_as_null=nan_as_null,
            **kwargs,
        )

    @classmethod
    @_cudf_nvtx_annotate
    def from_arrow(cls, obj):
        try:
            return cls(ColumnBase.from_arrow(obj))
        except TypeError:
            # Try interpreting object as a MultiIndex before failing.
            return cudf.MultiIndex.from_arrow(obj)


@_cudf_nvtx_annotate
def _concat_range_index(indexes: List[RangeIndex]) -> BaseIndex:
    """
    An internal Utility function to concat RangeIndex objects.
    """
    start = step = next_ = None

    # Filter the empty indexes
    non_empty_indexes = [obj for obj in indexes if len(obj)]

    if not non_empty_indexes:
        # Here all "indexes" had 0 length, i.e. were empty.
        # In this case return an empty range index.
        return RangeIndex(0, 0)

    for obj in non_empty_indexes:
        if start is None:
            # This is set by the first non-empty index
            start = obj.start
            if step is None and len(obj) > 1:
                step = obj.step
        elif step is None:
            # First non-empty index had only one element
            if obj.start == start:
                result = as_index(concat_columns([x._values for x in indexes]))
                return result
            step = obj.start - start

        non_consecutive = (step != obj.step and len(obj) > 1) or (
            next_ is not None and obj.start != next_
        )
        if non_consecutive:
            result = as_index(concat_columns([x._values for x in indexes]))
            return result
        if step is not None:
            next_ = obj[-1] + step

    stop = non_empty_indexes[-1].stop if next_ is None else next_
    return RangeIndex(start, stop, step)


@_cudf_nvtx_annotate
def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean algorithms to solve Bezout's identity:
       a*x + b*y = gcd(x, y)
    Finds one particular solution for x, y: s, t
    Returns: gcd, s, t
    """
    s, old_s = 0, 1
    t, old_t = 1, 0
    r, old_r = b, a
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return old_r, old_s, old_t
