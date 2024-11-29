# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from __future__ import annotations

import operator
import pickle
import warnings
from collections.abc import Hashable, MutableMapping
from functools import cache, cached_property
from numbers import Number
from typing import TYPE_CHECKING, Any, Literal, cast

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
from typing_extensions import Self

import pylibcudf as plc

import cudf
from cudf import _lib as libcudf
from cudf._lib.filling import sequence
from cudf._lib.types import size_type_dtype
from cudf.api.extensions import no_default
from cudf.api.types import (
    _is_non_decimal_numeric_dtype,
    is_dtype_equal,
    is_integer,
    is_list_like,
    is_scalar,
    is_string_dtype,
)
from cudf.core._base_index import BaseIndex, _return_get_indexer_result
from cudf.core._compat import PANDAS_LT_300
from cudf.core._internals.search import search_sorted
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column import (
    CategoricalColumn,
    ColumnBase,
    DatetimeColumn,
    IntervalColumn,
    NumericalColumn,
    StringColumn,
    StructColumn,
    TimeDeltaColumn,
    column,
)
from cudf.core.column.column import as_column, concat_columns
from cudf.core.column.string import StringMethods as StringMethods
from cudf.core.dtypes import IntervalDtype
from cudf.core.join._join_helpers import _match_join_keys
from cudf.core.mixins import BinaryOperand
from cudf.core.single_column_frame import SingleColumnFrame
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    _maybe_convert_to_default_type,
    find_common_type,
    is_mixed_with_object_dtype,
)
from cudf.utils.performance_tracking import _performance_tracking
from cudf.utils.utils import _warn_no_dask_cudf, search_range

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from datetime import tzinfo

    from cudf.core.frame import Frame


def ensure_index(index_like: Any) -> BaseIndex:
    """
    Ensure an Index is returned.

    Avoids a shallow copy compared to calling cudf.Index(...)
    """
    if not isinstance(index_like, BaseIndex):
        return cudf.Index(index_like)
    return index_like


class IndexMeta(type):
    """Custom metaclass for Index that overrides instance/subclass tests."""

    def __call__(cls, data, *args, **kwargs):
        if kwargs.get("tupleize_cols", True) is not True:
            raise NotImplementedError(
                "tupleize_cols is currently not supported."
            )

        if cls is Index:
            return as_index(
                arbitrary=data,
                *args,
                **kwargs,
            )
        return super().__call__(data, *args, **kwargs)

    def __instancecheck__(self, instance):
        if self is cudf.Index:
            return isinstance(instance, BaseIndex)
        else:
            return type.__instancecheck__(self, instance)

    def __subclasscheck__(self, subclass):
        if self is cudf.Index:
            return issubclass(subclass, BaseIndex)
        else:
            return type.__subclasscheck__(self, subclass)


def _lexsorted_equal_range(
    idx: Index | cudf.MultiIndex,
    keys: list[ColumnBase],
    is_sorted: bool,
) -> tuple[int, int, ColumnBase | None]:
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
        list(sort_vals._columns),
        keys,
        side="left",
        ascending=sort_vals.is_monotonic_increasing,
    ).element_indexing(0)
    upper_bound = search_sorted(
        list(sort_vals._columns),
        keys,
        side="right",
        ascending=sort_vals.is_monotonic_increasing,
    ).element_indexing(0)

    return lower_bound, upper_bound, sort_inds


def _index_from_data(data: MutableMapping, name: Any = no_default):
    """Construct an index of the appropriate type from some data."""

    if len(data) == 0:
        raise ValueError("Cannot construct Index from any empty Table")
    if len(data) == 1:
        values = next(iter(data.values()))

        if isinstance(values, NumericalColumn):
            index_class_type = Index
        elif isinstance(values, DatetimeColumn):
            index_class_type = DatetimeIndex
        elif isinstance(values, TimeDeltaColumn):
            index_class_type = TimedeltaIndex
        elif isinstance(values, StringColumn):
            index_class_type = Index
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


def validate_range_arg(arg, arg_name: Literal["start", "stop", "step"]) -> int:
    """Validate start/stop/step argument in RangeIndex.__init__"""
    if not is_integer(arg):
        raise TypeError(
            f"{arg_name} must be an integer, not {type(arg).__name__}"
        )
    return int(arg)


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

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    to_numpy
    to_arrow

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

    @_performance_tracking
    def __init__(
        self, start, stop=None, step=1, dtype=None, copy=False, name=None
    ):
        if not cudf.api.types.is_hashable(name):
            raise ValueError("Name must be a hashable value.")
        self._name = name
        if dtype is not None and cudf.dtype(dtype).kind != "i":
            raise ValueError(f"{dtype=} must be a signed integer type")

        if isinstance(start, range):
            self._range = start
        else:
            if stop is None:
                start, stop = 0, start
            start = validate_range_arg(start, "start")
            stop = validate_range_arg(stop, "stop")
            if step is not None:
                step = validate_range_arg(step, "step")
            else:
                step = 1
            try:
                self._range = range(start, stop, step)
            except ValueError as err:
                if step == 0:
                    raise ValueError("Step must not be zero.") from err
                raise

    def _copy_type_metadata(self: Self, other: Self) -> Self:
        # There is no metadata to be copied for RangeIndex since it does not
        # have an underlying column.
        return self

    def searchsorted(
        self,
        value: int,
        side: Literal["left", "right"] = "left",
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ):
        assert (len(self) <= 1) or (
            ascending == (self.step > 0)
        ), "Invalid ascending flag"
        return search_range(value, self._range, side=side)

    def factorize(
        self, sort: bool = False, use_na_sentinel: bool = True
    ) -> tuple[cupy.ndarray, Self]:
        if sort and self.step < 0:
            codes = cupy.arange(len(self) - 1, -1, -1)
            uniques = self[::-1]
        else:
            codes = cupy.arange(len(self), dtype=np.intp)
            uniques = self
        return codes, uniques

    @property  # type: ignore
    @_performance_tracking
    def name(self):
        return self._name

    @name.setter  # type: ignore
    @_performance_tracking
    def name(self, value):
        self._name = value

    @property
    @_performance_tracking
    def _column_names(self) -> tuple[Any]:
        return (self.name,)

    @property
    @_performance_tracking
    def _columns(self) -> tuple[ColumnBase]:
        return (self._values,)

    @property
    def _column_labels_and_values(self) -> Iterable:
        return zip(self._column_names, self._columns)

    @property  # type: ignore
    @_performance_tracking
    def start(self) -> int:
        """
        The value of the `start` parameter (0 if this was not supplied).
        """
        return self._range.start

    @property  # type: ignore
    @_performance_tracking
    def stop(self) -> int:
        """
        The value of the stop parameter.
        """
        return self._range.stop

    @property  # type: ignore
    @_performance_tracking
    def step(self) -> int:
        """
        The value of the step parameter.
        """
        return self._range.step

    @property  # type: ignore
    @_performance_tracking
    def _num_rows(self) -> int:
        return len(self)

    @cached_property  # type: ignore
    @_performance_tracking
    def _values(self) -> ColumnBase:
        if len(self) > 0:
            return column.as_column(self._range, dtype=self.dtype)
        else:
            return column.column_empty(0, masked=False, dtype=self.dtype)

    def _clean_nulls_from_index(self) -> Self:
        return self

    def _is_numeric(self) -> bool:
        return True

    def _is_boolean(self) -> bool:
        return False

    def _is_integer(self) -> bool:
        return True

    def _is_floating(self) -> bool:
        return False

    def _is_object(self) -> bool:
        return False

    def _is_categorical(self) -> bool:
        return False

    def _is_interval(self) -> bool:
        return False

    @property  # type: ignore
    @_performance_tracking
    def hasnans(self) -> bool:
        return False

    @property  # type: ignore
    @_performance_tracking
    def _data(self):
        return cudf.core.column_accessor.ColumnAccessor(
            {self.name: self._values}, verify=False
        )

    @_performance_tracking
    def __contains__(self, item):
        hash(item)
        if not isinstance(item, (np.floating, np.integer, int, float)):
            return False
        elif isinstance(item, (np.timedelta64, np.datetime64, bool)):
            # Cases that would pass the above check
            return False
        try:
            int_item = int(item)
            return int_item == item and int_item in self._range
        except (ValueError, OverflowError):
            return False

    @_performance_tracking
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

        return RangeIndex(self._range, name=name)

    @_performance_tracking
    def astype(self, dtype, copy: bool = True):
        if is_dtype_equal(dtype, self.dtype):
            return self
        return self._as_int_index().astype(dtype, copy=copy)

    def fillna(self, value, downcast=None):
        return self.copy()

    @_performance_tracking
    def drop_duplicates(self, keep="first"):
        return self

    @_performance_tracking
    def duplicated(self, keep="first") -> cupy.ndarray:
        return cupy.zeros(len(self), dtype=bool)

    @_performance_tracking
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(start={self.start}, stop={self.stop}"
            f", step={self.step}"
            + (
                f", name={pd.io.formats.printing.default_pprint(self.name)}"
                if self.name is not None
                else ""
            )
            + ")"
        )

    @property
    @_performance_tracking
    def size(self) -> int:
        return len(self)

    @_performance_tracking
    def __len__(self):
        return len(self._range)

    @_performance_tracking
    def __getitem__(self, index):
        if isinstance(index, slice):
            sl_start, sl_stop, sl_step = index.indices(len(self))

            lo = self.start + sl_start * self.step
            hi = self.start + sl_stop * self.step
            st = self.step * sl_step
            return RangeIndex(start=lo, stop=hi, step=st, name=self._name)

        elif isinstance(index, Number):
            len_self = len(self)
            if index < 0:
                index += len_self
            if not (0 <= index < len_self):
                raise IndexError("Index out of bounds")
            return self.start + index * self.step
        return self._as_int_index()[index]

    def _get_columns_by_label(self, labels) -> Index:
        # used in .sort_values
        if isinstance(labels, Hashable):
            if labels == self.name:
                return self._as_int_index()
        elif is_list_like(labels):
            if list(self.names) == list(labels):
                return self._as_int_index()
        raise KeyError(labels)

    @_performance_tracking
    def equals(self, other) -> bool:
        if isinstance(other, RangeIndex):
            return self._range == other._range
        return self._as_int_index().equals(other)

    @_performance_tracking
    def serialize(self):
        header = {}
        header["index_column"] = {}

        # store metadata values of index separately
        # We don't need to store the GPU buffer for RangeIndexes
        # cuDF only needs to store start/stop and rehydrate
        # during de-serialization
        header["index_column"]["start"] = self.start
        header["index_column"]["stop"] = self.stop
        header["index_column"]["step"] = self.step
        frames = []

        header["name"] = pickle.dumps(self.name)
        header["dtype"] = pickle.dumps(self.dtype)
        header["type-serialized"] = pickle.dumps(type(self))
        header["frame_count"] = 0
        return header, frames

    @classmethod
    @_performance_tracking
    def deserialize(cls, header, frames):
        h = header["index_column"]
        name = pickle.loads(header["name"])
        start = h["start"]
        stop = h["stop"]
        step = h.get("step", 1)
        return RangeIndex(start=start, stop=stop, step=step, name=name)

    @property  # type: ignore
    @_performance_tracking
    def dtype(self):
        """
        `dtype` of the range of values in RangeIndex.

        By default the dtype is 64 bit signed integer. This is configurable
        via `default_integer_bitwidth` as 32 bit in `cudf.options`
        """
        dtype = np.dtype(np.int64)
        return _maybe_convert_to_default_type(dtype)

    @property
    def _dtypes(self) -> Iterable:
        return [(self.name, self.dtype)]

    @_performance_tracking
    def to_pandas(
        self, *, nullable: bool = False, arrow_type: bool = False
    ) -> pd.RangeIndex:
        if nullable:
            raise NotImplementedError(f"{nullable=} is not implemented.")
        elif arrow_type:
            raise NotImplementedError(f"{arrow_type=} is not implemented.")
        return pd.RangeIndex(
            start=self.start,
            stop=self.stop,
            step=self.step,
            dtype=self.dtype,
            name=self.name,
        )

    def to_frame(
        self, index: bool = True, name: Hashable = no_default
    ) -> cudf.DataFrame:
        return self._as_int_index().to_frame(index=index, name=name)

    @property
    def is_unique(self) -> bool:
        return True

    @cached_property  # type: ignore
    @_performance_tracking
    def is_monotonic_increasing(self) -> bool:
        return self.step > 0 or len(self) <= 1

    @cached_property  # type: ignore
    @_performance_tracking
    def is_monotonic_decreasing(self):
        return self.step < 0 or len(self) <= 1

    @_performance_tracking
    def memory_usage(self, deep: bool = False) -> int:
        if deep:
            warnings.warn(
                "The deep parameter is ignored and is only included "
                "for pandas compatibility."
            )
        return 0

    def unique(self, level: int | None = None) -> Self:
        # RangeIndex always has unique values
        if level is not None and level > 0:
            raise IndexError(
                f"Too many levels: Index has only 1 level, not {level + 1}"
            )
        return self.copy()

    @_performance_tracking
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

    @_performance_tracking
    def __rmul__(self, other):
        # Multiplication is commutative.
        return self.__mul__(other)

    @_performance_tracking
    def _as_int_index(self) -> Index:
        # Convert self to an integer index. This method is used to perform ops
        # that are not defined directly on RangeIndex.
        return cudf.Index._from_data(self._data)

    @_performance_tracking
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self._as_int_index().__array_ufunc__(
            ufunc, method, *inputs, **kwargs
        )

    @_performance_tracking
    def get_indexer(self, target, limit=None, method=None, tolerance=None):
        target_col = cudf.core.column.as_column(target)
        if method is not None or not isinstance(
            target_col, cudf.core.column.NumericalColumn
        ):
            # TODO: See if we can implement this without converting to
            # Integer index.
            return self._as_int_index().get_indexer(
                target=target, limit=limit, method=method, tolerance=tolerance
            )

        if self.step > 0:
            start, stop, step = self.start, self.stop, self.step
        else:
            # Reversed
            reverse = self._range[::-1]
            start, stop, step = reverse.start, reverse.stop, reverse.step

        target_array = target_col.values
        locs = target_array - start
        valid = (locs % step == 0) & (locs >= 0) & (target_array < stop)
        locs[~valid] = -1
        locs[valid] = locs[valid] / step

        if step != self.step:
            # Reversed
            locs[valid] = len(self) - 1 - locs[valid]
        return locs

    @_performance_tracking
    def get_loc(self, key):
        if not is_scalar(key):
            raise TypeError("Should be a scalar-like")
        idx = (key - self.start) / self.step
        idx_int_upper_bound = (self.stop - self.start) // self.step
        if idx > idx_int_upper_bound or idx < 0:
            raise KeyError(key)

        idx_int = (key - self.start) // self.step
        if idx_int != idx:
            raise KeyError(key)
        return idx_int

    @_performance_tracking
    def _union(self, other, sort=None):
        if isinstance(other, RangeIndex):
            # Variable suffixes are of the
            # following notation: *_o -> other, *_s -> self,
            # and *_r -> result
            start_s, step_s = self.start, self.step
            end_s = self.start + self.step * (len(self) - 1)
            start_o, step_o = other.start, other.step
            end_o = other.start + other.step * (len(other) - 1)
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
                if sort in {None, True} and not result.is_monotonic_increasing:
                    return result.sort_values()
                else:
                    return result

        # If all the above optimizations don't cater to the inputs,
        # we materialize RangeIndexes into integer indexes and
        # then perform `union`.
        return self._try_reconstruct_range_index(
            self._as_int_index()._union(other, sort=sort)
        )

    @_performance_tracking
    def _intersection(self, other, sort=None):
        if not isinstance(other, RangeIndex):
            return self._try_reconstruct_range_index(
                super()._intersection(other, sort=sort)
            )

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
        if sort in {None, True}:
            new_index = new_index.sort_values()

        return self._try_reconstruct_range_index(new_index)

    @_performance_tracking
    def difference(self, other, sort=None):
        if isinstance(other, RangeIndex) and self.equals(other):
            return self[:0]._get_reconciled_name_object(other)

        return self._try_reconstruct_range_index(
            super().difference(other, sort=sort)
        )

    def _try_reconstruct_range_index(
        self, index: BaseIndex
    ) -> Self | BaseIndex:
        if isinstance(index, RangeIndex) or index.dtype.kind not in "iu":
            return index
        # Evenly spaced values can return a
        # RangeIndex instead of a materialized Index.
        if not index._column.has_nulls():  # type: ignore[attr-defined]
            uniques = cupy.unique(cupy.diff(index.values))
            if len(uniques) == 1 and (diff := uniques[0].get()) != 0:
                new_range = range(index[0], index[-1] + diff, diff)
                return type(self)(new_range, name=index.name)
        return index

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

    @_performance_tracking
    def _gather(self, gather_map, nullify=False, check_bounds=True):
        gather_map = cudf.core.column.as_column(gather_map)
        return cudf.Index._from_column(
            self._column.take(gather_map, nullify, check_bounds),
            name=self.name,
        )

    @_performance_tracking
    def _apply_boolean_mask(self, boolean_mask):
        return cudf.Index._from_column(
            self._column.apply_boolean_mask(boolean_mask), name=self.name
        )

    def repeat(self, repeats, axis=None):
        return self._as_int_index().repeat(repeats, axis)

    def _split(self, splits):
        return cudf.Index._from_column(
            self._as_int_index()._split(splits), name=self.name
        )

    def _binaryop(self, other, op: str):
        # TODO: certain binops don't require materializing range index and
        # could use some optimization.
        return self._as_int_index()._binaryop(other, op=op)

    def join(
        self, other, how="left", level=None, return_indexers=False, sort=False
    ):
        if how in {"left", "right"} or self.equals(other):
            # pandas supports directly merging RangeIndex objects and can
            # intelligently create RangeIndex outputs depending on the type of
            # join. Hence falling back to performing a merge on pd.RangeIndex
            # since the conversion is cheap.
            if isinstance(other, RangeIndex):
                result = self.to_pandas().join(
                    other.to_pandas(),
                    how=how,
                    level=level,
                    return_indexers=return_indexers,
                    sort=sort,
                )
                if return_indexers:
                    return tuple(
                        cudf.from_pandas(result[0]), result[1], result[2]
                    )
                else:
                    return cudf.from_pandas(result)
        return self._as_int_index().join(
            other, how, level, return_indexers, sort
        )

    @property  # type: ignore
    @_performance_tracking
    def _column(self) -> ColumnBase:
        return self._as_int_index()._column

    @property  # type: ignore
    @_performance_tracking
    def _columns(self) -> list[ColumnBase]:
        return self._as_int_index()._columns

    @property  # type: ignore
    @_performance_tracking
    def values_host(self) -> np.ndarray:
        return np.arange(start=self.start, stop=self.stop, step=self.step)

    @_performance_tracking
    def argsort(
        self,
        ascending=True,
        na_position="last",
    ) -> cupy.ndarray:
        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")
        if (ascending and self.step < 0) or (not ascending and self.step > 0):
            return cupy.arange(len(self) - 1, -1, -1)
        else:
            return cupy.arange(len(self))

    @_performance_tracking
    def where(self, cond, other=None, inplace=False):
        return self._as_int_index().where(cond, other, inplace)

    @_performance_tracking
    def to_numpy(self) -> np.ndarray:
        return self.values_host

    @_performance_tracking
    def to_cupy(self) -> cupy.ndarray:
        return self.values

    @_performance_tracking
    def to_arrow(self) -> pa.Array:
        return pa.array(self._range, type=pa.from_numpy_dtype(self.dtype))

    def __array__(self, dtype=None):
        raise TypeError(
            "Implicit conversion to a host NumPy array via __array__ is not "
            "allowed, To explicitly construct a GPU matrix, consider using "
            ".to_cupy()\nTo explicitly construct a host matrix, consider "
            "using .to_numpy()."
        )

    @_performance_tracking
    def nunique(self, dropna: bool = True) -> int:
        return len(self)

    @_performance_tracking
    def isna(self) -> cupy.ndarray:
        return cupy.zeros(len(self), dtype=bool)

    isnull = isna

    @_performance_tracking
    def notna(self) -> cupy.ndarray:
        return cupy.ones(len(self), dtype=bool)

    notnull = isna

    @_performance_tracking
    def _minmax(self, meth: str) -> int | float:
        no_steps = len(self) - 1
        if no_steps == -1:
            return np.nan
        elif (meth == "min" and self.step > 0) or (
            meth == "max" and self.step < 0
        ):
            return self.start

        return self.start + self.step * no_steps

    def min(self) -> int | float:
        return self._minmax("min")

    def max(self) -> int | float:
        return self._minmax("max")

    @property
    def values(self) -> cupy.ndarray:
        return cupy.arange(self.start, self.stop, self.step)

    def any(self) -> bool:
        return any(self._range)

    def all(self) -> bool:
        return 0 not in self._range

    def append(self, other):
        result = self._as_int_index().append(other)
        return self._try_reconstruct_range_index(result)

    def _indices_of(self, value) -> cudf.core.column.NumericalColumn:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"Cannot use {type(value).__name__} to get an index of a "
                f"{type(self).__name__}."
            )
        try:
            i = [self._range.index(value)]
        except ValueError:
            i = []
        return as_column(i, dtype=size_type_dtype)

    def isin(self, values, level=None):
        if level is not None and level > 0:
            raise IndexError(
                f"Too many levels: Index has only 1 level, not {level + 1}"
            )
        if is_scalar(values):
            raise TypeError(
                "only list-like objects are allowed to be passed "
                f"to isin(), you passed a {type(values).__name__}"
            )

        return self._values.isin(values).values

    def __pos__(self) -> Self:
        return self.copy()

    def __neg__(self) -> Self:
        rng = range(-self.start, -self.stop, -self.step)
        return type(self)(rng, name=self.name)

    def __abs__(self) -> Self | Index:
        if len(self) == 0 or self.min() >= 0:
            return self.copy()
        elif self.max() <= 0:
            return -self
        else:
            return abs(self._as_int_index())

    def _columns_for_reset_index(
        self, levels: tuple | None
    ) -> Generator[tuple[Any, ColumnBase], None, None]:
        """Return the columns and column names for .reset_index"""
        # We need to explicitly materialize the RangeIndex to a column
        yield "index" if self.name is None else self.name, as_column(self)

    @_warn_no_dask_cudf
    def __dask_tokenize__(self):
        return (type(self), self.start, self.stop, self.step)


class Index(SingleColumnFrame, BaseIndex, metaclass=IndexMeta):
    """
    Immutable sequence used for indexing and alignment.

    The basic object storing axis labels for all pandas objects.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Index. If not specified, this will be
        inferred from `data`.
    copy : bool, default False
        Copy input data.
    name : object
        Name to be stored in the index.
    tupleize_cols : bool (default: True)
        When True, attempt to create a MultiIndex if possible.
        Currently not supported.
    """

    @_performance_tracking
    def __init__(self, data, **kwargs):
        name = _getdefault_name(data, name=kwargs.get("name"))
        super().__init__({name: data})

    @_performance_tracking
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
                    for name, col in self._column_labels_and_values
                }

            data = self._apply_cupy_ufunc_to_operands(
                ufunc, cupy_func, inputs, **kwargs
            )

            out = [_index_from_data(out) for out in data]

            # pandas returns numpy arrays when the outputs are boolean.
            for i, o in enumerate(out):
                # We explicitly _do not_ use isinstance here: we want only
                # boolean Indexes, not dtype-specific subclasses.
                if type(o) is Index and o.dtype.kind == "b":
                    out[i] = o.values

            return out[0] if ufunc.nout == 1 else tuple(out)

        return NotImplemented

    @classmethod
    @_performance_tracking
    def _from_column(
        cls, column: ColumnBase, *, name: Hashable = None
    ) -> Self:
        if cls is Index:
            ca = cudf.core.column_accessor.ColumnAccessor(
                {name: column}, verify=False
            )
            return _index_from_data(ca)
        else:
            return super()._from_column(column, name=name)

    @classmethod
    @_performance_tracking
    def _from_data(cls, data: MutableMapping, name: Any = no_default) -> Self:
        out = super()._from_data(data=data)
        if name is not no_default:
            out.name = name
        return out

    @classmethod
    @_performance_tracking
    def _from_data_like_self(
        cls, data: MutableMapping, name: Any = no_default
    ) -> Self:
        out = _index_from_data(data, name)
        if name is not no_default:
            out.name = name
        return out

    @classmethod
    @_performance_tracking
    def from_arrow(cls, obj) -> Index | cudf.MultiIndex:
        """Create from PyArrow Array/ChunkedArray.

        Parameters
        ----------
        array : PyArrow Array/ChunkedArray
            PyArrow Object which has to be converted.

        Raises
        ------
        TypeError for invalid input type.

        Returns
        -------
        SingleColumnFrame

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> cudf.Index.from_arrow(pa.array(["a", "b", None]))
        Index(['a', 'b', <NA>], dtype='object')
        """
        try:
            return cls._from_column(ColumnBase.from_arrow(obj))
        except TypeError:
            # Try interpreting object as a MultiIndex before failing.
            return cudf.MultiIndex.from_arrow(obj)

    @cached_property
    def is_monotonic_increasing(self) -> bool:
        return super().is_monotonic_increasing

    @cached_property
    def is_monotonic_decreasing(self) -> bool:
        return super().is_monotonic_decreasing

    def _binaryop(
        self,
        other: Frame,
        op: str,
        fill_value: Any = None,
        *args,
        **kwargs,
    ) -> SingleColumnFrame:
        reflect, op = self._check_reflected_op(op)
        operands = self._make_operands_for_binop(other, fill_value, reflect)
        if operands is NotImplemented:
            return NotImplemented
        binop_result = self._colwise_binop(operands, op)

        if isinstance(other, cudf.Series):
            ret = other._from_data_like_self(binop_result)
            other_name = other.name
        else:
            ret = _index_from_data(binop_result)
            other_name = getattr(other, "name", self.name)

        ret.name = (
            self.name
            if cudf.utils.utils._is_same_name(self.name, other_name)
            else None
        )

        # pandas returns numpy arrays when the outputs are boolean. We
        # explicitly _do not_ use isinstance here: we want only boolean
        # Indexes, not dtype-specific subclasses.
        if isinstance(ret, (Index, cudf.Series)) and ret.dtype.kind == "b":
            if ret._column.has_nulls():
                ret = ret.fillna(op == "__ne__")

            return ret.values
        return ret

    @property  # type: ignore
    @_performance_tracking
    def _values(self) -> ColumnBase:
        return self._column

    @classmethod
    @_performance_tracking
    def _concat(cls, objs):
        non_empties = [index for index in objs if len(index)]
        if len(objs) != len(non_empties):
            # Do not remove until pandas-3.0 support is added.
            assert (
                PANDAS_LT_300
            ), "Need to drop after pandas-3.0 support is added."
            warning_msg = (
                "The behavior of array concatenation with empty entries is "
                "deprecated. In a future version, this will no longer exclude "
                "empty items when determining the result dtype. "
                "To retain the old behavior, exclude the empty entries before "
                "the concat operation."
            )
            # Warn only if the type might _actually_ change
            if len(non_empties) == 0:
                if not all(objs[0].dtype == index.dtype for index in objs[1:]):
                    warnings.warn(warning_msg, FutureWarning)
            else:
                common_all_type = find_common_type(
                    [index.dtype for index in objs]
                )
                common_non_empty_type = find_common_type(
                    [index.dtype for index in non_empties]
                )
                if common_all_type != common_non_empty_type:
                    warnings.warn(warning_msg, FutureWarning)
        if all(isinstance(obj, RangeIndex) for obj in non_empties):
            result = _concat_range_index(non_empties)
        else:
            data = concat_columns([o._column for o in non_empties])
            result = Index._from_column(data)

        names = {obj.name for obj in objs}
        if len(names) == 1:
            name = names.pop()
        else:
            name = None

        result.name = name
        return result

    @_performance_tracking
    def memory_usage(self, deep: bool = False) -> int:
        return self._column.memory_usage

    @cached_property  # type: ignore
    @_performance_tracking
    def is_unique(self) -> bool:
        return self._column.is_unique

    @_performance_tracking
    def equals(self, other) -> bool:
        if not isinstance(other, BaseIndex) or len(self) != len(other):
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

    @_performance_tracking
    def copy(self, name: Hashable = None, deep: bool = False) -> Self:
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
        col = self._column.copy(deep=True) if deep else self._column
        return type(self)._from_column(col, name=name)

    @_performance_tracking
    def astype(self, dtype, copy: bool = True) -> Index:
        return super().astype({self.name: dtype}, copy)

    @_performance_tracking
    def get_indexer(self, target, method=None, limit=None, tolerance=None):
        if is_scalar(target):
            raise TypeError("Should be a sequence")

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

        if not self.is_unique:
            raise ValueError("Cannot get index for a non-unique Index.")

        is_sorted = (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        )

        if not is_sorted and method is not None:
            raise ValueError(
                "index must be monotonic increasing or decreasing if `method`"
                "is specified."
            )

        needle = as_column(target)
        result = as_column(
            -1,
            length=len(needle),
            dtype=libcudf.types.size_type_dtype,
        )

        if not len(self):
            return _return_get_indexer_result(result.values)
        try:
            lcol, rcol = _match_join_keys(needle, self._column, "inner")
        except ValueError:
            return _return_get_indexer_result(result.values)

        with acquire_spill_lock():
            left_plc, right_plc = plc.join.inner_join(
                plc.Table([lcol.to_pylibcudf(mode="read")]),
                plc.Table([rcol.to_pylibcudf(mode="read")]),
                plc.types.NullEquality.EQUAL,
            )
            scatter_map = libcudf.column.Column.from_pylibcudf(left_plc)
            indices = libcudf.column.Column.from_pylibcudf(right_plc)
        result = libcudf.copying.scatter([indices], scatter_map, [result])[0]
        result_series = cudf.Series._from_column(result)

        if method in {"ffill", "bfill", "pad", "backfill"}:
            result_series = _get_indexer_basic(
                index=self,
                positions=result_series,
                method=method,
                target_col=cudf.Series._from_column(needle),
                tolerance=tolerance,
            )
        elif method == "nearest":
            result_series = _get_nearest_indexer(
                index=self,
                positions=result_series,
                target_col=cudf.Series._from_column(needle),
                tolerance=tolerance,
            )
        elif method is not None:
            raise ValueError(
                f"{method=} is unsupported, only supported values are: "
                "{['ffill'/'pad', 'bfill'/'backfill', 'nearest', None]}"
            )

        return _return_get_indexer_result(result_series.to_cupy())

    @_performance_tracking
    def get_loc(self, key) -> int | slice | cupy.ndarray:
        if not is_scalar(key):
            raise TypeError("Should be a scalar-like")

        is_sorted = (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        )

        lower_bound, upper_bound, sort_inds = _lexsorted_equal_range(
            self, [as_column([key])], is_sorted
        )

        if lower_bound == upper_bound:
            raise KeyError(key)

        if lower_bound + 1 == upper_bound:
            # Search result is unique, return int.
            return (
                lower_bound
                if is_sorted
                else sort_inds.element_indexing(lower_bound)  # type: ignore[union-attr]
            )

        if is_sorted:
            # In monotonic index, lex search result is continuous. A slice for
            # the range is returned.
            return slice(lower_bound, upper_bound)

        # Not sorted and not unique. Return a boolean mask
        mask = cupy.full(len(self), False)
        true_inds = sort_inds.slice(lower_bound, upper_bound).values  # type: ignore[union-attr]
        mask[true_inds] = True
        return mask

    @_performance_tracking
    def __repr__(self) -> str:
        max_seq_items = pd.get_option("max_seq_items") or len(self)
        mr = 0
        if 2 * max_seq_items < len(self):
            mr = max_seq_items + 1

        if len(self) > mr and mr != 0:
            top = self[0:mr]
            bottom = self[-1 * mr :]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
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
                # Too many non-unique categories will cause
                # the output to take too long. In this case, we
                # split the categories into data and categories
                # and generate the repr separately and
                # merge them.
                pd_cats = pd.Categorical(
                    preprocess.astype(preprocess.categories.dtype).to_pandas()
                )
                pd_preprocess = pd.CategoricalIndex(pd_cats)
                data_repr = repr(pd_preprocess).split("\n")
                pd_preprocess.dtype._categories = (
                    preprocess.categories.to_pandas()
                )
                pd_preprocess.dtype._ordered = preprocess.dtype.ordered
                cats_repr = repr(pd_preprocess).split("\n")
                output = "\n".join(data_repr[:-1] + cats_repr[-1:])

            output = output.replace("nan", str(cudf.NA))
        elif preprocess._values.nullable:
            if isinstance(self._values, StringColumn):
                output = repr(self.to_pandas(nullable=True))
            else:
                output = repr(self._clean_nulls_from_index().to_pandas())
                # We should remove all the single quotes
                # from the output due to the type-cast to
                # object dtype happening above.
                # Note : The replacing of single quotes has
                # to happen only in case of non-Index[string] types,
                # as we want to preserve single quotes in case
                # of Index[string] and it is valid to have them.
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
        keywords = [f"dtype='{self.dtype}'"]
        if self.name is not None:
            keywords.append(f"name={self.name!r}")
        if "length" in tmp_meta:
            keywords.append(f"length={len(self)}")
        if (
            "freq" in tmp_meta
            and isinstance(self, DatetimeIndex)
            and self._freq is not None
        ):
            keywords.append(
                f"freq={self._freq._maybe_as_fast_pandas_offset().freqstr!r}"
            )
        joined_keywords = ", ".join(keywords)
        lines.append(f"{prior_to_dtype} {joined_keywords})")
        return "\n".join(lines)

    @_performance_tracking
    def __getitem__(self, index):
        res = self._get_elements_from_column(index)
        if isinstance(res, ColumnBase):
            res = Index._from_column(res, name=self.name)
        return res

    @property  # type: ignore
    @_performance_tracking
    def dtype(self):
        """
        `dtype` of the underlying values in Index.
        """
        return self._column.dtype

    @_performance_tracking
    def isna(self) -> cupy.ndarray:
        return self._column.isnull().values

    isnull = isna

    @_performance_tracking
    def notna(self) -> cupy.ndarray:
        return self._column.notnull().values

    notnull = notna

    def _is_numeric(self) -> bool:
        return (
            isinstance(self._values, cudf.core.column.NumericalColumn)
            and self.dtype.kind != "b"
        )

    def _is_boolean(self) -> bool:
        return self.dtype.kind == "b"

    def _is_integer(self) -> bool:
        return self.dtype.kind in "iu"

    def _is_floating(self) -> bool:
        return self.dtype.kind == "f"

    def _is_object(self) -> bool:
        return isinstance(self._column, cudf.core.column.StringColumn)

    def _is_categorical(self) -> bool:
        return False

    def _is_interval(self) -> bool:
        return False

    @property  # type: ignore
    @_performance_tracking
    def hasnans(self) -> bool:
        return self._column.has_nulls(include_nan=True)

    @_performance_tracking
    def argsort(
        self,
        axis=0,
        kind="quicksort",
        order=None,
        ascending=True,
        na_position="last",
    ) -> cupy.ndarray:
        """Return the integer indices that would sort the index.

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

    def repeat(self, repeats, axis=None) -> Self:
        result = super()._repeat([self._column], repeats, axis)[0]
        result = result._with_type_metadata(self.dtype)
        return type(self)._from_column(result, name=self.name)

    @_performance_tracking
    def where(self, cond, other=None, inplace=False) -> Index:
        result_col = super().where(cond, other, inplace)
        return self._mimic_inplace(
            _index_from_data({self.name: result_col}),
            inplace=inplace,
        )

    @property
    def values(self) -> cupy.ndarray:
        return self._column.values

    def __contains__(self, item) -> bool:
        hash(item)
        return item in self._column

    def _clean_nulls_from_index(self) -> Index:
        if self._values.has_nulls():
            fill_value = (
                str(cudf.NaT)
                if isinstance(self, (DatetimeIndex, TimedeltaIndex))
                else str(cudf.NA)
            )
            return cudf.Index._from_column(
                self._column.astype("str").fillna(fill_value),
                name=self.name,
            )

        return self

    def any(self) -> bool:
        return self._column.any()

    def to_pandas(
        self, *, nullable: bool = False, arrow_type: bool = False
    ) -> pd.Index:
        result = self._column.to_pandas(
            nullable=nullable, arrow_type=arrow_type
        )
        result.name = self.name
        return result

    def to_frame(
        self, index: bool = True, name: Hashable = no_default
    ) -> cudf.DataFrame:
        """Create a DataFrame with a column containing this Index

        Parameters
        ----------
        index : boolean, default True
            Set the index of the returned DataFrame as the original Index
        name : object, defaults to index.name
            The passed name should substitute for the index name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame containing the original Index data.

        See Also
        --------
        Index.to_series : Convert an Index to a Series.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index(['Ant', 'Bear', 'Cow'], name='animal')
        >>> idx.to_frame()
               animal
        animal
        Ant       Ant
        Bear     Bear
        Cow       Cow

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
            animal
        0   Ant
        1  Bear
        2   Cow

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(index=False, name='zoo')
            zoo
        0   Ant
        1  Bear
        2   Cow
        """
        return self._to_frame(name=name, index=self if index else None)

    def append(self, other):
        if is_list_like(other):
            to_concat = [self]
            for obj in other:
                if not isinstance(obj, BaseIndex):
                    raise TypeError("all inputs must be Index")
                to_concat.append(obj)
        else:
            this = self
            other = ensure_index(other)

            if len(this) == 0 or len(other) == 0:
                # we'll filter out empties later in ._concat
                to_concat = [this, other]
            else:
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

                if (
                    isinstance(self._column, cudf.core.column.NumericalColumn)
                    and self.dtype != other.dtype
                ):
                    common_type = find_common_type((self.dtype, other.dtype))
                    this = this.astype(common_type)
                    other = other.astype(common_type)
                to_concat = [this, other]

        return self._concat(to_concat)

    def unique(self, level: int | None = None) -> Self:
        if level is not None and level > 0:
            raise IndexError(
                f"Too many levels: Index has only 1 level, not {level + 1}"
            )
        return type(self)._from_column(self._column.unique(), name=self.name)

    def isin(self, values, level=None) -> cupy.ndarray:
        if level is not None and level > 0:
            raise IndexError(
                f"Too many levels: Index has only 1 level, not {level + 1}"
            )
        if is_scalar(values):
            raise TypeError(
                "only list-like objects are allowed to be passed "
                f"to isin(), you passed a {type(values).__name__}"
            )

        return self._column.isin(values).values

    @copy_docstring(StringMethods)  # type: ignore
    @property
    @_performance_tracking
    def str(self):
        if is_string_dtype(self.dtype):
            return StringMethods(parent=self)
        else:
            raise AttributeError(
                "Can only use .str accessor with string values!"
            )

    @cache
    @_warn_no_dask_cudf
    def __dask_tokenize__(self):
        # We can use caching, because an index is immutable
        return super().__dask_tokenize__()


class DatetimeIndex(Index):
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
        Frequency of the DatetimeIndex
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

    Attributes
    ----------
    year
    month
    day
    hour
    minute
    second
    microsecond
    nanosecond
    date
    time
    dayofyear
    day_of_year
    weekday
    quarter
    freq

    Methods
    -------
    ceil
    floor
    round
    tz_convert
    tz_localize

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

    @_performance_tracking
    def __init__(
        self,
        data=None,
        freq=None,
        tz=None,
        normalize: bool = False,
        closed=None,
        ambiguous: Literal["raise"] = "raise",
        dayfirst: bool = False,
        yearfirst: bool = False,
        dtype=None,
        copy: bool = False,
        name=None,
    ):
        # we should be more strict on what we accept here but
        # we'd have to go and figure out all the semantics around
        # pandas dtindex creation first which.  For now
        # just make sure we handle np.datetime64 arrays
        # and then just dispatch upstream
        if tz is not None:
            raise NotImplementedError("tz is not yet supported")
        if normalize is not False:
            warnings.warn(
                "The 'normalize' keyword is "
                "deprecated and will be removed in a future version. ",
                FutureWarning,
            )
            raise NotImplementedError("normalize == True is not yet supported")
        if closed is not None:
            warnings.warn(
                "The 'closed' keyword is "
                "deprecated and will be removed in a future version. ",
                FutureWarning,
            )
            raise NotImplementedError("closed is not yet supported")
        if ambiguous != "raise":
            raise NotImplementedError("ambiguous is not yet supported")
        if dayfirst is not False:
            raise NotImplementedError("dayfirst == True is not yet supported")
        if yearfirst is not False:
            raise NotImplementedError("yearfirst == True is not yet supported")

        self._freq = _validate_freq(freq)

        if dtype is None:
            # nanosecond default matches pandas
            dtype = "datetime64[ns]"
        dtype = cudf.dtype(dtype)
        if dtype.kind != "M":
            raise TypeError("dtype must be a datetime type")

        name = _getdefault_name(data, name=name)
        data = column.as_column(data)

        # TODO: if data.dtype.kind == "M" (i.e. data is already datetime type)
        # We probably shouldn't always astype to datetime64[ns]
        if not isinstance(data.dtype, pd.DatetimeTZDtype):
            data = data.astype(dtype)

        if copy:
            data = data.copy()

        super().__init__(data, name=name)

        if self._freq is not None:
            unique_vals = self.to_series().diff().unique()
            if len(unique_vals) > 2 or (
                len(unique_vals) == 2
                and unique_vals[1] != self._freq._maybe_as_fast_pandas_offset()
            ):
                raise ValueError("No unique frequency found")

    @_performance_tracking
    def _copy_type_metadata(self: Self, other: Self) -> Self:
        super()._copy_type_metadata(other)
        self._freq = _validate_freq(other._freq)
        return self

    @classmethod
    def _from_data(
        cls, data: MutableMapping, name: Any = no_default, freq: Any = None
    ):
        result = super()._from_data(data, name)
        result._freq = _validate_freq(freq)
        return result

    @classmethod
    @_performance_tracking
    def _from_column(
        cls, column: ColumnBase, *, name: Hashable = None, freq: Any = None
    ) -> Self:
        if column.dtype.kind != "M":
            raise ValueError("column must have a datetime type.")
        result = super()._from_column(column, name=name)
        result._freq = _validate_freq(freq)
        return result

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if cudf.get_option("mode.pandas_compatible") and isinstance(
            value, np.datetime64
        ):
            return pd.Timestamp(value)
        return value

    @_performance_tracking
    def copy(self, name=None, deep=False):
        idx_copy = super().copy(name=name, deep=deep)
        return idx_copy._copy_type_metadata(self)

    def searchsorted(
        self,
        value,
        side: Literal["left", "right"] = "left",
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ):
        value = self.dtype.type(value)
        return super().searchsorted(
            value, side=side, ascending=ascending, na_position=na_position
        )

    def as_unit(self, unit: str, round_ok: bool = True) -> Self:
        """
        Convert to a dtype with the given unit resolution.

        Currently not implemented.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}
        round_ok : bool, default True
            If False and the conversion requires rounding, raise ValueError.
        """
        raise NotImplementedError("as_unit is currently not implemented")

    def mean(self, *, skipna: bool = True, axis: int | None = 0):
        return self._column.mean(skipna=skipna)

    def std(self, *, skipna: bool = True, axis: int | None = 0, ddof: int = 1):
        return self._column.std(skipna=skipna, ddof=ddof)

    def strftime(self, date_format: str) -> Index:
        """
        Convert to Index using specified date_format.

        Return an Index of formatted strings specified by date_format, which
        supports the same string format as the python standard library.

        Parameters
        ----------
        date_format : str
            Date format string (e.g. "%Y-%m-%d").
        """
        return Index._from_column(
            self._column.strftime(date_format), name=self.name
        )

    @property
    def asi8(self) -> cupy.ndarray:
        return self._column.astype("int64").values

    @property
    def inferred_freq(self) -> cudf.DateOffset | None:
        raise NotImplementedError("inferred_freq is currently not implemented")

    @property
    def freq(self) -> cudf.DateOffset | None:
        return self._freq

    @freq.setter
    def freq(self) -> None:
        raise NotImplementedError("Setting freq is currently not supported.")

    @property
    def freqstr(self) -> str:
        raise NotImplementedError("freqstr is currently not implemented")

    @property
    def resolution(self) -> str:
        """
        Returns day, hour, minute, second, millisecond or microsecond
        """
        raise NotImplementedError("resolution is currently not implemented")

    @property
    def unit(self) -> str:
        return self._column.time_unit

    @property
    def tz(self) -> tzinfo | None:
        """
        Return the timezone.

        Returns
        -------
        datetime.tzinfo or None
            Returns None when the array is tz-naive.
        """
        return getattr(self.dtype, "tz", None)

    @property
    def tzinfo(self) -> tzinfo | None:
        """
        Alias for tz attribute
        """
        return self.tz

    def to_pydatetime(self) -> np.ndarray:
        """
        Return an ndarray of ``datetime.datetime`` objects.

        Returns
        -------
        numpy.ndarray
            An ndarray of ``datetime.datetime`` objects.
        """
        return self.to_pandas().to_pydatetime()

    def to_julian_date(self) -> Index:
        return Index._from_column(
            self._column.to_julian_date(), name=self.name
        )

    def to_period(self, freq) -> pd.PeriodIndex:
        return self.to_pandas().to_period(freq=freq)

    def normalize(self) -> Self:
        """
        Convert times to midnight.

        Currently not implemented.
        """
        return type(self)._from_column(
            self._column.normalize(), name=self.name
        )

    @property
    def time(self) -> np.ndarray:
        """
        Returns numpy array of ``datetime.time`` objects.

        The time part of the Timestamps.
        """
        return self.to_pandas().time

    @property
    def timetz(self) -> np.ndarray:
        """
        Returns numpy array of ``datetime.time`` objects with timezones.

        The time part of the Timestamps.
        """
        return self.to_pandas().timetz

    @property
    def date(self) -> np.ndarray:
        """
        Returns numpy array of python ``datetime.date`` objects.

        Namely, the date part of Timestamps without time and
        timezone information.
        """
        return self.to_pandas().date

    @property
    def is_month_start(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the first day of the month.
        """
        return self._column.is_month_start.values

    @property
    def is_month_end(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the last day of the month.
        """
        return self._column.is_month_end.values

    @property
    def is_quarter_end(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the last day of the quarter.
        """
        return self._column.is_quarter_end.values

    @property
    def is_quarter_start(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the start day of the quarter.
        """
        return self._column.is_quarter_start.values

    @property
    def is_year_end(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the last day of the year.
        """
        return self._column.is_year_end.values

    @property
    def is_year_start(self) -> cupy.ndarray:
        """
        Booleans indicating if dates are the first day of the year.
        """
        return self._column.is_year_start.values

    @property
    def is_normalized(self) -> bool:
        """
        Returns True if all of the dates are at midnight ("no time")
        """
        return self._column.is_normalized

    @property
    def days_in_month(self) -> Index:
        """
        Get the total number of days in the month that the date falls on.
        """
        return Index._from_column(self._column.days_in_month, name=self.name)

    daysinmonth = days_in_month

    @property
    def day_of_week(self) -> Index:
        """
        Get the day of week that the date falls on.
        """
        return Index._from_column(self._column.day_of_week, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def year(self) -> Index:
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
        Index([2000, 2001, 2002], dtype='int16')
        """  # noqa: E501
        return Index._from_column(self._column.year, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def month(self) -> Index:
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
        Index([1, 2, 3], dtype='int16')
        """  # noqa: E501
        return Index._from_column(self._column.month, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def day(self) -> Index:
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
        Index([1, 2, 3], dtype='int16')
        """  # noqa: E501
        return Index._from_column(self._column.day, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def hour(self) -> Index:
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
        Index([0, 1, 2], dtype='int16')
        """
        return Index._from_column(self._column.hour, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def minute(self) -> Index:
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
        Index([0, 1, 2], dtype='int16')
        """
        return Index._from_column(self._column.minute, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def second(self) -> Index:
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
        Index([0, 1, 2], dtype='int16')
        """
        return Index._from_column(self._column.second, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def microsecond(self) -> Index:
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
        Index([0, 1, 2], dtype='int32')
        """  # noqa: E501
        return Index._from_column(
            (
                # Need to manually promote column to int32 because
                # pandas-matching binop behaviour requires that this
                # __mul__ returns an int16 column.
                self._column.millisecond.astype("int32")
                * cudf.Scalar(1000, dtype="int32")
            )
            + self._column.microsecond,
            name=self.name,
        )

    @property  # type: ignore
    @_performance_tracking
    def nanosecond(self) -> Index:
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
        Index([0, 1, 2], dtype='int16')
        """
        return Index._from_column(self._column.nanosecond, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def weekday(self) -> Index:
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
        Index([5, 6, 0, 1, 2, 3, 4, 5, 6], dtype='int16')
        """
        return Index._from_column(self._column.weekday, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def dayofweek(self) -> Index:
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
        Index([5, 6, 0, 1, 2, 3, 4, 5, 6], dtype='int16')
        """
        return Index._from_column(self._column.weekday, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def dayofyear(self) -> Index:
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
        Index([366, 1, 2, 3, 4, 5, 6, 7, 8], dtype='int16')
        """
        return Index._from_column(self._column.day_of_year, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def day_of_year(self) -> Index:
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
        Index([366, 1, 2, 3, 4, 5, 6, 7, 8], dtype='int16')
        """
        return Index._from_column(self._column.day_of_year, name=self.name)

    @property  # type: ignore
    @_performance_tracking
    def is_leap_year(self) -> cupy.ndarray:
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
        res = self._column.is_leap_year.fillna(False)
        return cupy.asarray(res)

    @property  # type: ignore
    @_performance_tracking
    def quarter(self) -> Index:
        """
        Integer indicator for which quarter of the year the date belongs in.

        There are 4 quarters in a year. With the first quarter being from
        January - March, second quarter being April - June, third quarter
        being July - September and fourth quarter being October - December.

        Returns
        -------
        Index
        Integer indicating which quarter the date belongs to.

        Examples
        --------
        >>> import cudf
        >>> gIndex = cudf.DatetimeIndex(["2020-05-31 08:00:00",
        ...    "1999-12-31 18:40:00"])
        >>> gIndex.quarter
        Index([2, 4], dtype='int8')
        """
        return Index._from_column(self._column.quarter.astype("int8"))

    @_performance_tracking
    def day_name(self, locale: str | None = None) -> Index:
        """
        Return the day names. Currently supports English locale only.

        Examples
        --------
        >>> import cudf
        >>> datetime_index = cudf.date_range("2016-12-31", "2017-01-08", freq="D")
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                       '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                       '2017-01-08'],
                      dtype='datetime64[ns]', freq='D')
        >>> datetime_index.day_name()
        Index(['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
               'Friday', 'Saturday', 'Sunday'],
              dtype='object')
        """
        day_names = self._column.get_day_names(locale)
        return Index._from_column(day_names, name=self.name)

    @_performance_tracking
    def month_name(self, locale: str | None = None) -> Index:
        """
        Return the month names. Currently supports English locale only.

        Examples
        --------
        >>> import cudf
        >>> datetime_index = cudf.date_range("2017-12-30", periods=6, freq='W')
        >>> datetime_index
        DatetimeIndex(['2017-12-30', '2018-01-06', '2018-01-13', '2018-01-20',
                    '2018-01-27', '2018-02-03'],
                      dtype='datetime64[ns]', freq='7D')
        >>> datetime_index.month_name()
        Index(['December', 'January', 'January', 'January', 'January', 'February'], dtype='object')
        """
        month_names = self._column.get_month_names(locale)
        return Index._from_column(month_names, name=self.name)

    @_performance_tracking
    def isocalendar(self) -> cudf.DataFrame:
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
        ca = cudf.core.column_accessor.ColumnAccessor(
            self._column.isocalendar(), verify=False
        )
        return cudf.DataFrame._from_data(ca, index=self)

    @_performance_tracking
    def to_pandas(
        self, *, nullable: bool = False, arrow_type: bool = False
    ) -> pd.DatetimeIndex:
        result = super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        if not arrow_type and self._freq is not None:
            result.freq = self._freq._maybe_as_fast_pandas_offset()
        return result

    def _is_boolean(self) -> bool:
        return False

    @_performance_tracking
    def ceil(self, freq: str) -> Self:
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
        return type(self)._from_column(self._column.ceil(freq), name=self.name)

    @_performance_tracking
    def floor(self, freq: str) -> Self:
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
        return type(self)._from_column(
            self._column.floor(freq), name=self.name
        )

    @_performance_tracking
    def round(self, freq: str) -> Self:
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
        return type(self)._from_column(
            self._column.round(freq), name=self.name
        )

    def tz_localize(
        self,
        tz: str | None,
        ambiguous: Literal["NaT"] = "NaT",
        nonexistent: Literal["NaT"] = "NaT",
    ) -> Self:
        """
        Localize timezone-naive data to timezone-aware data.

        Parameters
        ----------
        tz : str
            Timezone to convert timestamps to.

        Returns
        -------
        DatetimeIndex containing timezone aware timestamps.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> tz_naive = cudf.date_range('2018-03-01 09:00', periods=3, freq='D')
        >>> tz_aware = tz_naive.tz_localize("America/New_York")
        >>> tz_aware
        DatetimeIndex(['2018-03-01 09:00:00-05:00', '2018-03-02 09:00:00-05:00',
                       '2018-03-03 09:00:00-05:00'],
                      dtype='datetime64[ns, America/New_York]', freq='D')

        Ambiguous or nonexistent datetimes are converted to NaT.

        >>> s = cudf.to_datetime(cudf.Series(['2018-10-28 01:20:00',
        ...                                   '2018-10-28 02:36:00',
        ...                                   '2018-10-28 03:46:00']))
        >>> s.dt.tz_localize("CET")
        0    2018-10-28 01:20:00.000000000
        1                              NaT
        2    2018-10-28 03:46:00.000000000
        dtype: datetime64[ns, CET]

        Notes
        -----
        'NaT' is currently the only supported option for the
        ``ambiguous`` and ``nonexistent`` arguments. Any
        ambiguous or nonexistent timestamps are converted
        to 'NaT'.
        """  # noqa: E501
        result_col = self._column.tz_localize(tz, ambiguous, nonexistent)
        return DatetimeIndex._from_column(
            result_col, name=self.name, freq=self._freq
        )

    def tz_convert(self, tz: str | None) -> Self:
        """
        Convert tz-aware datetimes from one time zone to another.

        Parameters
        ----------
        tz : str
            Time zone for time. Corresponding timestamps would be converted
            to this time zone of the Datetime Array/Index.
            A `tz` of None will convert to UTC and remove the timezone
            information.

        Returns
        -------
        DatetimeIndex containing timestamps corresponding to the timezone
        `tz`.

        Examples
        --------
        >>> import cudf
        >>> dti = cudf.date_range('2018-03-01 09:00', periods=3, freq='D')
        >>> dti = dti.tz_localize("America/New_York")
        >>> dti
        DatetimeIndex(['2018-03-01 09:00:00-05:00', '2018-03-02 09:00:00-05:00',
                       '2018-03-03 09:00:00-05:00'],
                      dtype='datetime64[ns, America/New_York]', freq='D')
        >>> dti.tz_convert("Europe/London")
        DatetimeIndex(['2018-03-01 14:00:00+00:00',
                       '2018-03-02 14:00:00+00:00',
                       '2018-03-03 14:00:00+00:00'],
                      dtype='datetime64[ns, Europe/London]')
        """  # noqa: E501
        result_col = self._column.tz_convert(tz)
        return DatetimeIndex._from_column(result_col, name=self.name)

    def repeat(self, repeats, axis=None) -> Self:
        res = super().repeat(repeats, axis=axis)
        res._freq = None
        return res


class TimedeltaIndex(Index):
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

    Attributes
    ----------
    days
    seconds
    microseconds
    nanoseconds
    components
    inferred_freq

    Methods
    -------
    None

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

    @_performance_tracking
    def __init__(
        self,
        data=None,
        unit=None,
        freq=None,
        closed=None,
        dtype=None,
        copy: bool = False,
        name=None,
    ):
        if freq is not None:
            raise NotImplementedError("freq is not yet supported")

        if closed is not None:
            warnings.warn(
                "The 'closed' keyword is "
                "deprecated and will be removed in a future version. ",
                FutureWarning,
            )
            raise NotImplementedError("closed is not yet supported")

        if unit is not None:
            warnings.warn(
                "The 'unit' keyword is "
                "deprecated and will be removed in a future version. ",
                FutureWarning,
            )
            raise NotImplementedError(
                "unit is not yet supported, alternatively "
                "dtype parameter is supported"
            )

        if dtype is None:
            dtype = "timedelta64[ns]"
        dtype = cudf.dtype(dtype)
        if dtype.kind != "m":
            raise TypeError("dtype must be a timedelta type")

        name = _getdefault_name(data, name=name)
        data = column.as_column(data, dtype=dtype)

        if copy:
            data = data.copy()

        super().__init__(data, name=name)

    @classmethod
    @_performance_tracking
    def _from_column(
        cls, column: ColumnBase, *, name: Hashable = None, freq: Any = None
    ) -> Self:
        if column.dtype.kind != "m":
            raise ValueError("column must have a timedelta type.")
        return super()._from_column(column, name=name)

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if cudf.get_option("mode.pandas_compatible") and isinstance(
            value, np.timedelta64
        ):
            return pd.Timedelta(value)
        return value

    def as_unit(self, unit: str, round_ok: bool = True) -> Self:
        """
        Convert to a dtype with the given unit resolution.

        Currently not implemented.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}
        round_ok : bool, default True
            If False and the conversion requires rounding, raise ValueError.
        """
        raise NotImplementedError("as_unit is currently not implemented")

    @property
    def freq(self) -> cudf.DateOffset | None:
        raise NotImplementedError("freq is currently not implemented")

    @property
    def freqstr(self) -> str:
        raise NotImplementedError("freqstr is currently not implemented")

    @property
    def resolution(self) -> str:
        """
        Returns day, hour, minute, second, millisecond or microsecond
        """
        raise NotImplementedError("resolution is currently not implemented")

    @property
    def unit(self) -> str:
        return self._column.time_unit

    def to_pytimedelta(self) -> np.ndarray:
        """
        Return an ndarray of ``datetime.timedelta`` objects.

        Returns
        -------
        numpy.ndarray
            An ndarray of ``datetime.timedelta`` objects.
        """
        return self.to_pandas().to_pytimedelta()

    @property
    def asi8(self) -> cupy.ndarray:
        return self._column.astype("int64").values

    def sum(self, *, skipna: bool = True, axis: int | None = 0):
        return self._column.sum(skipna=skipna)

    def mean(self, *, skipna: bool = True, axis: int | None = 0):
        return self._column.mean(skipna=skipna)

    def median(self, *, skipna: bool = True, axis: int | None = 0):
        return self._column.median(skipna=skipna)

    def std(self, *, skipna: bool = True, axis: int | None = 0, ddof: int = 1):
        return self._column.std(skipna=skipna, ddof=ddof)

    def total_seconds(self) -> cupy.ndarray:
        """
        Return total duration of each element expressed in seconds.

        This method is currently not implemented.
        """
        return self._column.total_seconds().values

    def ceil(self, freq: str) -> Self:
        """
        Ceil to the specified resolution.

        This method is currently not implemented.
        """
        return type(self)._from_column(self._column.ceil(freq), name=self.name)

    def floor(self, freq: str) -> Self:
        """
        Floor to the specified resolution.

        This method is currently not implemented.
        """
        return type(self)._from_column(
            self._column.floor(freq), name=self.name
        )

    def round(self, freq: str) -> Self:
        """
        Round to the specified resolution.

        This method is currently not implemented.
        """
        return type(self)._from_column(
            self._column.round(freq), name=self.name
        )

    @property  # type: ignore
    @_performance_tracking
    def days(self) -> cudf.Index:
        """
        Number of days for each element.
        """
        # Need to specifically return `int64` to avoid overflow.
        return Index._from_column(
            self._column.days.astype("int64"), name=self.name
        )

    @property  # type: ignore
    @_performance_tracking
    def seconds(self) -> cudf.Index:
        """
        Number of seconds (>= 0 and less than 1 day) for each element.
        """
        return Index._from_column(
            self._column.seconds.astype("int32"), name=self.name
        )

    @property  # type: ignore
    @_performance_tracking
    def microseconds(self) -> cudf.Index:
        """
        Number of microseconds (>= 0 and less than 1 second) for each element.
        """
        return Index._from_column(
            self._column.microseconds.astype("int32"), name=self.name
        )

    @property  # type: ignore
    @_performance_tracking
    def nanoseconds(self) -> cudf.Index:
        """
        Number of nanoseconds (>= 0 and less than 1 microsecond) for each
        element.
        """
        return Index._from_column(
            self._column.nanoseconds.astype("int32"), name=self.name
        )

    @property  # type: ignore
    @_performance_tracking
    def components(self) -> cudf.DataFrame:
        """
        Return a dataframe of the components (days, hours, minutes,
        seconds, milliseconds, microseconds, nanoseconds) of the Timedeltas.
        """
        ca = cudf.core.column_accessor.ColumnAccessor(
            self._column.components(), verify=False
        )
        return cudf.DataFrame._from_data(ca)

    @property
    def inferred_freq(self):
        """
        Infers frequency of TimedeltaIndex.

        Notes
        -----
        This property is currently not supported.
        """
        raise NotImplementedError("inferred_freq is not yet supported")

    def _is_boolean(self) -> bool:
        return False


class CategoricalIndex(Index):
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

    Attributes
    ----------
    codes
    categories

    Methods
    -------
    equals

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

    @_performance_tracking
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
        name = _getdefault_name(data, name=name)
        if isinstance(data, CategoricalColumn):
            data = data
        elif isinstance(getattr(data, "dtype", None), pd.CategoricalDtype):
            data = column.as_column(data)
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
            data = data.as_ordered(ordered=True)
        elif ordered is False and data.ordered is True:
            data = data.as_ordered(ordered=False)
        super().__init__(data, name=name)

    @classmethod
    @_performance_tracking
    def _from_column(
        cls, column: ColumnBase, *, name: Hashable = None, freq: Any = None
    ) -> Self:
        if not isinstance(column.dtype, cudf.CategoricalDtype):
            raise ValueError("column must have a categorial type.")
        return super()._from_column(column, name=name)

    @property
    def ordered(self) -> bool:
        return self._column.ordered

    @property  # type: ignore
    @_performance_tracking
    def codes(self) -> cudf.Index:
        """
        The category codes of this categorical.
        """
        return Index._from_column(self._column.codes)

    @property  # type: ignore
    @_performance_tracking
    def categories(self) -> cudf.Index:
        """
        The categories of this categorical.
        """
        return self.dtype.categories

    def _is_boolean(self) -> bool:
        return False

    def _is_categorical(self) -> bool:
        return True

    def add_categories(self, new_categories) -> Self:
        """
        Add new categories.

        `new_categories` will be included at the last/highest place in the
        categories and will be unused directly after this call.
        """
        return type(self)._from_column(
            self._column.add_categories(new_categories), name=self.name
        )

    def as_ordered(self) -> Self:
        """
        Set the Categorical to be ordered.
        """
        return type(self)._from_column(
            self._column.as_ordered(ordered=True), name=self.name
        )

    def as_unordered(self) -> Self:
        """
        Set the Categorical to be unordered.
        """
        return type(self)._from_column(
            self._column.as_ordered(ordered=False), name=self.name
        )

    def remove_categories(self, removals) -> Self:
        """
        Remove the specified categories.

        `removals` must be included in the old categories.

        Parameters
        ----------
        removals : category or list of categories
           The categories which should be removed.
        """
        return type(self)._from_column(
            self._column.remove_categories(removals), name=self.name
        )

    def remove_unused_categories(self) -> Self:
        """
        Remove categories which are not used.

        This method is currently not supported.
        """
        return type(self)._from_column(
            self._column.remove_unused_categories(), name=self.name
        )

    def rename_categories(self, new_categories) -> Self:
        """
        Rename categories.

        This method is currently not supported.
        """
        return type(self)._from_column(
            self._column.rename_categories(new_categories), name=self.name
        )

    def reorder_categories(self, new_categories, ordered=None) -> Self:
        """
        Reorder categories as specified in new_categories.

        ``new_categories`` need to include all old categories and no new category
        items.

        Parameters
        ----------
        new_categories : Index-like
           The categories in new order.
        ordered : bool, optional
           Whether or not the categorical is treated as a ordered categorical.
           If not given, do not change the ordered information.
        """
        return type(self)._from_column(
            self._column.reorder_categories(new_categories, ordered=ordered),
            name=self.name,
        )

    def set_categories(
        self, new_categories, ordered=None, rename: bool = False
    ) -> Self:
        """
        Set the categories to the specified new_categories.

        Parameters
        ----------
        new_categories : list-like
            The categories in new order.
        ordered : bool, default None
            Whether or not the categorical is treated as
            a ordered categorical. If not given, do
            not change the ordered information.
        rename : bool, default False
            Whether or not the `new_categories` should be
            considered as a rename of the old categories
            or as reordered categories.
        """
        return type(self)._from_column(
            self._column.set_categories(
                new_categories, ordered=ordered, rename=rename
            ),
            name=self.name,
        )


@_performance_tracking
def interval_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    name=None,
    closed="right",
) -> IntervalIndex:
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
    nargs = sum(_ is not None for _ in (start, end, periods, freq))

    # we need at least three of (start, end, periods, freq)
    if nargs == 2 and freq is None:
        freq = 1
        nargs += 1

    if nargs != 3:
        raise ValueError(
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )

    start = cudf.Scalar(start) if start is not None else start
    end = cudf.Scalar(end) if end is not None else end
    if periods is not None and not cudf.api.types.is_integer(periods):
        warnings.warn(
            "Non-integer 'periods' in cudf.date_range, and cudf.interval_range"
            " are deprecated and will raise in a future version.",
            FutureWarning,
        )
    periods = cudf.Scalar(int(periods)) if periods is not None else periods
    freq = cudf.Scalar(freq) if freq is not None else freq

    if start is None:
        start = end - freq * periods
    elif freq is None:
        quotient, remainder = divmod((end - start).value, periods.value)
        if remainder:
            freq = (end - start) / periods
        else:
            freq = cudf.Scalar(int(quotient))
    elif periods is None:
        periods = cudf.Scalar(int((end - start) / freq))
    elif end is None:
        end = start + periods * freq

    if any(
        not _is_non_decimal_numeric_dtype(x.dtype)
        for x in (start, periods, freq, end)
    ):
        raise ValueError("start, end, periods, freq must be numeric values.")

    periods = periods.astype("int64")
    common_dtype = find_common_type((start.dtype, freq.dtype, end.dtype))
    start = start.astype(common_dtype)
    freq = freq.astype(common_dtype)

    bin_edges = sequence(
        size=periods + 1,
        init=start.device_value,
        step=freq.device_value,
    )
    return IntervalIndex.from_breaks(bin_edges, closed=closed, name=name)


class IntervalIndex(Index):
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

    Attributes
    ----------
    values

    Methods
    -------
    from_breaks
    get_loc

    Returns
    -------
    IntervalIndex
    """

    @_performance_tracking
    def __init__(
        self,
        data,
        closed: Literal["left", "right", "neither", "both"] | None = None,
        dtype=None,
        copy: bool = False,
        name=None,
        verify_integrity: bool = True,
    ):
        name = _getdefault_name(data, name=name)

        if dtype is not None:
            dtype = cudf.dtype(dtype)
            if not isinstance(dtype, IntervalDtype):
                raise TypeError("dtype must be an IntervalDtype")
            if closed is not None and closed != dtype.closed:
                raise ValueError("closed keyword does not match dtype.closed")
            closed = dtype.closed

        if closed is None and isinstance(dtype, IntervalDtype):
            closed = dtype.closed

        closed = closed or "right"

        if len(data) == 0:
            if not hasattr(data, "dtype"):
                data = np.array([], dtype=np.int64)
            elif isinstance(data.dtype, (pd.IntervalDtype, IntervalDtype)):
                data = np.array([], dtype=data.dtype.subtype)
            interval_col = IntervalColumn(
                None,
                dtype=IntervalDtype(data.dtype, closed),
                size=len(data),
                children=(as_column(data), as_column(data)),
            )
        else:
            col = as_column(data)
            if not isinstance(col, IntervalColumn):
                raise TypeError("data must be an iterable of Interval data")
            if copy:
                col = col.copy()
            interval_col = IntervalColumn(
                data=None,
                dtype=IntervalDtype(col.dtype.subtype, closed),
                mask=col.mask,
                size=col.size,
                offset=col.offset,
                null_count=col.null_count,
                children=col.children,  # type: ignore[arg-type]
            )

        if dtype:
            interval_col = interval_col.astype(dtype)  # type: ignore[assignment]

        super().__init__(interval_col, name=name)

    @property
    def closed(self) -> Literal["left", "right", "neither", "both"]:
        return self.dtype.closed

    @classmethod
    @_performance_tracking
    def _from_column(
        cls, column: ColumnBase, *, name: Hashable = None, freq: Any = None
    ) -> Self:
        if not isinstance(column.dtype, cudf.IntervalDtype):
            raise ValueError("column must have a interval type.")
        return super()._from_column(column, name=name)

    @classmethod
    @_performance_tracking
    def from_breaks(
        cls,
        breaks,
        closed: Literal["left", "right", "neither", "both"] | None = "right",
        name=None,
        copy: bool = False,
        dtype=None,
    ) -> Self:
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
        IntervalIndex([(0, 1], (1, 2], (2, 3]], dtype='interval[int64, right]')
        """
        breaks = as_column(breaks, dtype=dtype)
        if copy:
            breaks = breaks.copy()
        left_col = breaks.slice(0, len(breaks) - 1)
        right_col = breaks.slice(1, len(breaks))
        # For indexing, children should both have 0 offset
        right_col = type(right_col)(
            data=right_col.data,
            dtype=right_col.dtype,
            size=right_col.size,
            mask=right_col.mask,
            offset=0,
            null_count=right_col.null_count,
            children=right_col.children,
        )

        interval_col = IntervalColumn(
            data=None,
            dtype=IntervalDtype(left_col.dtype, closed),
            size=len(left_col),
            children=(left_col, right_col),
        )
        return IntervalIndex._from_column(interval_col, name=name)

    @classmethod
    def from_arrays(
        cls,
        left,
        right,
        closed: Literal["left", "right", "both", "neither"] = "right",
        copy: bool = False,
        dtype=None,
    ) -> Self:
        raise NotImplementedError("from_arrays is currently not supported.")

    @classmethod
    def from_tuples(
        cls,
        data,
        closed: Literal["left", "right", "both", "neither"] = "right",
        name=None,
        copy: bool = False,
        dtype=None,
    ) -> Self:
        piidx = pd.IntervalIndex.from_tuples(
            data, closed=closed, name=name, copy=copy, dtype=dtype
        )
        return cls.from_pandas(piidx)

    def __getitem__(self, index):
        raise NotImplementedError(
            "Getting a scalar from an IntervalIndex is not yet supported"
        )

    def _is_interval(self) -> bool:
        return True

    def _is_boolean(self) -> bool:
        return False

    def _clean_nulls_from_index(self) -> Self:
        return self

    @property
    def is_empty(self) -> cupy.ndarray:
        """
        Indicates if an interval is empty, meaning it contains no points.
        """
        return self._column.is_empty.values

    @property
    def is_non_overlapping_monotonic(self) -> bool:
        """
        Return a True if the IntervalIndex is non-overlapping and monotonic.
        """
        return self._column.is_non_overlapping_monotonic

    @property
    def is_overlapping(self) -> bool:
        """
        Return True if the IntervalIndex has overlapping intervals, else False.

        Currently not implemented
        """
        return self._column.is_overlapping

    @property
    def length(self) -> Index:
        """
        Return an Index with entries denoting the length of each Interval.
        """
        return _index_from_data({None: self._column.length})

    @property
    def left(self) -> Index:
        """
        Return left bounds of the intervals in the IntervalIndex.

        The left bounds of each interval in the IntervalIndex are
        returned as an Index. The datatype of the left bounds is the
        same as the datatype of the endpoints of the intervals.
        """
        return _index_from_data({None: self._column.left})

    @property
    def mid(self) -> Index:
        """
        Return the midpoint of each interval in the IntervalIndex as an Index.

        Each midpoint is calculated as the average of the left and right bounds
        of each interval.
        """
        return _index_from_data({None: self._column.mid})

    @property
    def right(self) -> Index:
        """
        Return right bounds of the intervals in the IntervalIndex.

        The right bounds of each interval in the IntervalIndex are
        returned as an Index. The datatype of the right bounds is the
        same as the datatype of the endpoints of the intervals.
        """
        return _index_from_data({None: self._column.right})

    def overlaps(self, other) -> cupy.ndarray:
        """
        Check elementwise if an Interval overlaps the values in the IntervalIndex.

        Currently not supported.
        """
        return self._column.overlaps(other).values

    def set_closed(
        self, closed: Literal["left", "right", "both", "neither"]
    ) -> Self:
        """
        Return an identical IntervalArray closed on the specified side.

        Parameters
        ----------
        closed : {'left', 'right', 'both', 'neither'}
            Whether the intervals are closed on the left-side, right-side, both
            or neither.
        """
        return type(self)._from_column(
            self._column.set_closed(closed), name=self.name
        )

    def to_tuples(self, na_tuple: bool = True) -> pd.Index:
        """
        Return an Index of tuples of the form (left, right).

        Parameters
        ----------
        na_tuple : bool, default True
            If ``True``, return ``NA`` as a tuple ``(nan, nan)``. If ``False``,
            just return ``NA`` as ``nan``.
        """
        return self.to_pandas().to_tuples(na_tuple=na_tuple)


@_performance_tracking
def as_index(
    arbitrary, nan_as_null=no_default, copy=False, name=no_default, dtype=None
) -> BaseIndex:
    """Create an Index from an arbitrary object

    Parameters
    ----------
    arbitrary : object
        Object to construct the Index from. See *Notes*.
    nan_as_null : bool, optional, default None
        If None (default), treats NaN values in arbitrary as null.
        If True, combines the mask and NaNs to
        form a new validity mask. If False, leaves NaN values as is.
    copy : bool, default False
        If True, Make copies of `arbitrary` if possible and create an
        Index out of it.
        If False, `arbitrary` will be shallow-copied if it is a
        device-object to construct an Index.
    name : object, optional
        Name of the index being created, by default it is `None`.
    dtype : optional
        Optionally typecast the constructed Index to the given
        dtype.

    Returns
    -------
    result : subclass of Index
        - CategoricalIndex for Categorical input.
        - DatetimeIndex for Datetime input.
        - Index for all other inputs.

    Notes
    -----
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
        - Index for all other inputs.
    """
    if nan_as_null is no_default:
        nan_as_null = (
            False if cudf.get_option("mode.pandas_compatible") else None
        )

    if name is no_default:
        name = getattr(arbitrary, "name", None)

    if isinstance(arbitrary, cudf.MultiIndex):
        if dtype is not None:
            raise TypeError(
                "dtype must be `None` for inputs of type: "
                f"{type(arbitrary).__name__}, found {dtype=} "
            )
        return arbitrary.copy(deep=copy)
    elif isinstance(arbitrary, BaseIndex):
        idx = arbitrary.copy(deep=copy).rename(name)
    elif isinstance(arbitrary, ColumnBase):
        raise ValueError("Use cudf.Index._from_column instead.")
    elif isinstance(arbitrary, (pd.RangeIndex, range)):
        idx = RangeIndex(
            start=arbitrary.start,
            stop=arbitrary.stop,
            step=arbitrary.step,
            name=name,
        )
    elif isinstance(arbitrary, pd.MultiIndex):
        if dtype is not None:
            raise TypeError(
                "dtype must be `None` for inputs of type: "
                f"{type(arbitrary).__name__}, found {dtype=} "
            )
        return cudf.MultiIndex.from_pandas(
            arbitrary.copy(deep=copy), nan_as_null=nan_as_null
        )
    elif isinstance(arbitrary, cudf.DataFrame) or is_scalar(arbitrary):
        raise ValueError("Index data must be 1-dimensional and list-like")
    else:
        return Index._from_column(
            column.as_column(arbitrary, dtype=dtype, nan_as_null=nan_as_null),
            name=name,
        )
    if dtype is not None:
        idx = idx.astype(dtype)
    return idx


def _getdefault_name(values, name):
    if name is None:
        return getattr(values, "name", None)
    return name


@_performance_tracking
def _concat_range_index(indexes: list[RangeIndex]) -> BaseIndex:
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
                result = Index._from_column(
                    concat_columns([x._column for x in indexes])
                )
                return result
            step = obj.start - start

        non_consecutive = (step != obj.step and len(obj) > 1) or (
            next_ is not None and obj.start != next_
        )
        if non_consecutive:
            result = Index._from_column(
                concat_columns([x._column for x in indexes])
            )
            return result
        if step is not None:
            next_ = obj[-1] + step

    stop = non_empty_indexes[-1].stop if next_ is None else next_
    return RangeIndex(start, stop, step)


@_performance_tracking
def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
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


def _get_indexer_basic(index, positions, method, target_col, tolerance):
    # `positions` will be modified in-place, so it is the
    # responsibility of the caller to decide whether or not
    # to make a copy of it before passing it to this method.
    nonexact = positions == -1
    positions[nonexact] = index.searchsorted(
        target_col[nonexact],
        side="left" if method in {"pad", "ffill"} else "right",
    )
    if method in {"pad", "ffill"}:
        # searchsorted returns "indices into a sorted array such that,
        # if the corresponding elements in v were inserted before the
        # indices, the order of a would be preserved".
        # Thus, we need to subtract 1 to find values to the left.
        positions[nonexact] -= 1
        # This also mapped not found values (values of 0 from
        # np.searchsorted) to -1, which conveniently is also our
        # sentinel for missing values
    else:
        # Mark indices to the right of the largest value as not found
        positions[positions == len(index)] = np.int32(-1)

    if tolerance is not None:
        distance = abs(index[positions] - target_col)
        return positions.where(distance <= tolerance, -1)
    return positions


def _get_nearest_indexer(
    index: Index,
    positions: cudf.Series,
    target_col: cudf.core.column.ColumnBase,
    tolerance: int | float,
):
    """
    Get the indexer for the nearest index labels; requires an index with
    values that can be subtracted from each other.
    """
    left_indexer = _get_indexer_basic(
        index=index,
        positions=positions.copy(deep=True),
        method="pad",
        target_col=target_col,
        tolerance=tolerance,
    )
    right_indexer = _get_indexer_basic(
        index=index,
        # positions no longer used so don't copy
        positions=positions,
        method="backfill",
        target_col=target_col,
        tolerance=tolerance,
    )

    left_distances = abs(index[left_indexer] - target_col)
    right_distances = abs(index[right_indexer] - target_col)

    op = operator.lt if index.is_monotonic_increasing else operator.le
    indexer = left_indexer.where(
        op(left_distances, right_distances) | (right_indexer == -1),
        right_indexer,
    )

    if tolerance is not None:
        distance = abs(index[indexer] - target_col)
        return indexer.where(distance <= tolerance, -1)
    return indexer


def _validate_freq(freq: Any) -> cudf.DateOffset | None:
    if isinstance(freq, str):
        return cudf.DateOffset._from_freqstr(freq)
    elif freq is None:
        return freq
    elif freq is not None and not isinstance(freq, cudf.DateOffset):
        raise ValueError(f"Invalid frequency: {freq}")
    return cast(cudf.DateOffset, freq)
