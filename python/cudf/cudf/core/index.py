# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division, print_function

import functools
import pickle

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
from cudf.core.column import (
    CategoricalColumn,
    ColumnBase,
    DatetimeColumn,
    NumericalColumn,
    StringColumn,
    column,
)
from cudf.core.frame import Frame
from cudf.utils import ioutils, utils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import is_categorical_dtype, is_scalar, min_signed_type
from cudf.utils.utils import cached_property


def _to_frame(this_index, index=True, name=None):
    """Create a DataFrame with a column containing this Index

    Parameters
    ----------
    index : boolean, default True
        Set the index of the returned DataFrame as the original Index
    name : str, default None
        Name to be used for the column

    Returns
    -------
    DataFrame
        cudf DataFrame
    """

    from cudf import DataFrame

    if name is not None:
        col_name = name
    elif this_index.name is None:
        col_name = 0
    else:
        col_name = this_index.name

    return DataFrame(
        {col_name: this_index._values}, index=this_index if index else None
    )


class Index(Frame):
    """The root interface for all Series indexes.
    """

    def serialize(self):
        """Serialize into pickle format suitable for file storage or network
        transmission.
        """
        header = {}
        header["index_column"] = {}
        # store metadata values of index separately
        # Indexes: Numerical/DateTime/String are often GPU backed
        header["index_column"], frames = self._values.serialize()

        header["name"] = pickle.dumps(self.name)
        header["dtype"] = pickle.dumps(self.dtype)
        header["type-serialized"] = pickle.dumps(type(self))
        header["frame_count"] = len(frames)
        return header, frames

    def __contains__(self, item):
        return item in self._values

    def get_level_values(self, level):
        if level == self.name:
            return self
        elif pd.api.types.is_integer(level):
            if level != 0:
                raise IndexError(
                    f"Cannot get level: {level} " f"for index with 1 level"
                )
            return self
        else:
            raise KeyError(f"Requested level with name {level} " "not found")

    @classmethod
    def deserialize(cls, header, frames):
        """
        """
        h = header["index_column"]
        idx_typ = pickle.loads(header["type-serialized"])
        name = pickle.loads(header["name"])

        col_typ = pickle.loads(h["type-serialized"])
        index = col_typ.deserialize(h, frames[: header["frame_count"]])
        return idx_typ(index, name=name)

    @property
    def names(self):
        return (self.name,)

    @names.setter
    def names(self, values):
        if not pd.api.types.is_list_like(values):
            raise ValueError("Names must be a list-like")

        num_values = len(values)
        if num_values > 1:
            raise ValueError(
                "Length of new names must be 1, got %d" % num_values
            )

        self.name = values[0]

    @property
    def name(self):
        return next(iter(self._data.names))

    @name.setter
    def name(self, value):
        col = self._data.pop(self.name)
        self._data[value] = col

    def dropna(self):
        """
        Return a Series with null values removed.
        """
        return super().dropna(subset=[self.name])

    def take(self, indices):
        """Gather only the specific subset of indices

        Parameters
        ---
        indices: An array-like that maps to values contained in this Index.
        """
        return self[indices]

    def argsort(self, ascending=True):
        indices = self._values.argsort(ascending=ascending)
        indices.name = self.name
        return indices

    @property
    def values(self):
        if is_categorical_dtype(self.dtype) or np.issubdtype(
            self.dtype, np.dtype("object")
        ):
            raise TypeError("Data must be numeric")
        if len(self) == 0:
            return cupy.asarray([], dtype=self.dtype)
        if self._values.null_count > 0:
            raise ValueError("Column must have no nulls.")

        return cupy.asarray(self._values.data_array_view)

    def to_pandas(self):
        return pd.Index(self._values.to_pandas(), name=self.name)

    def to_arrow(self):
        return self._values.to_arrow()

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""
        import cudf.io.dlpack as dlpack

        return dlpack.to_dlpack(self)

    @property
    def gpu_values(self):
        return self._values.data_array_view

    def min(self):
        return self._values.min()

    def max(self):
        return self._values.max()

    def sum(self):
        return self._values.sum()

    @classmethod
    def _concat(cls, objs):
        data = ColumnBase._concat([o._values for o in objs])
        names = {obj.name for obj in objs}
        if len(names) == 1:
            [name] = names
        else:
            name = None
        result = as_index(data)
        result.name = name
        return result

    def _apply_op(self, fn, other=None):
        from cudf.core.series import Series

        idx_series = Series(self, name=self.name)
        op = getattr(idx_series, fn)
        if other is not None:
            return as_index(op(other))
        else:
            return as_index(op())

    def unique(self):
        return as_index(self._values.unique(), name=self.name)

    def __add__(self, other):
        return self._apply_op("__add__", other)

    def __radd__(self, other):
        return self._apply_op("__radd__", other)

    def __sub__(self, other):
        return self._apply_op("__sub__", other)

    def __rsub__(self, other):
        return self._apply_op("__rsub__", other)

    def __mul__(self, other):
        return self._apply_op("__mul__", other)

    def __rmul__(self, other):
        return self._apply_op("__rmul__", other)

    def __mod__(self, other):
        return self._apply_op("__mod__", other)

    def __rmod__(self, other):
        return self._apply_op("__rmod__", other)

    def __pow__(self, other):
        return self._apply_op("__pow__", other)

    def __floordiv__(self, other):
        return self._apply_op("__floordiv__", other)

    def __rfloordiv__(self, other):
        return self._apply_op("__rfloordiv__", other)

    def __truediv__(self, other):
        return self._apply_op("__truediv__", other)

    def __rtruediv__(self, other):
        return self._apply_op("__rtruediv__", other)

    __div__ = __truediv__

    def __and__(self, other):
        return self._apply_op("__and__", other)

    def __or__(self, other):
        return self._apply_op("__or__", other)

    def __xor__(self, other):
        return self._apply_op("__xor__", other)

    def __eq__(self, other):
        return self._apply_op("__eq__", other)

    def __ne__(self, other):
        return self._apply_op("__ne__", other)

    def __lt__(self, other):
        return self._apply_op("__lt__", other)

    def __le__(self, other):
        return self._apply_op("__le__", other)

    def __gt__(self, other):
        return self._apply_op("__gt__", other)

    def __ge__(self, other):
        return self._apply_op("__ge__", other)

    def equals(self, other):
        if self is other:
            return True
        if len(self) != len(other):
            return False
        elif len(self) == 1:
            val = self[0] == other[0]
            # when self is multiindex we need to checkall
            if isinstance(val, np.ndarray):
                return val.all()
            return bool(val)
        else:
            result = self == other
            if isinstance(result, bool):
                return result
            else:
                return result._values.all()

    def join(self, other, method, how="left", return_indexers=False):
        column_join_res = self._values.join(
            other._values,
            how=how,
            return_indexers=return_indexers,
            method=method,
        )
        if return_indexers:
            joined_col, indexers = column_join_res
            joined_index = as_index(joined_col)
            return joined_index, indexers
        else:
            return column_join_res

    def rename(self, name, inplace=False):
        """
        Alter Index name.

        Defaults to returning new index.

        Parameters
        ----------
        name : label
            Name(s) to set.

        Returns
        -------
        Index

        """
        if inplace is True:
            self.name = name
            return None
        else:
            out = self.copy(deep=False)
            out.name = name
            return out.copy(deep=True)

    def astype(self, dtype):
        """Convert to the given ``dtype``.

        Returns
        -------
        If the dtype changed, a new ``Index`` is returned by casting each
        values to the given dtype.
        If the dtype is not changed, ``self`` is returned.
        """
        if pd.api.types.is_dtype_equal(dtype, self.dtype):
            return self

        return as_index(self._values.astype(dtype), name=self.name)

    def to_array(self, fillna=None):
        """Get a dense numpy array for the data.

        Parameters
        ----------
        fillna : str or None
            Defaults to None, which will skip null values.
            If it equals "pandas", null values are filled with NaNs.
            Non integral dtype is promoted to np.float64.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        return self._values.to_array(fillna=fillna)

    def to_series(self):
        from cudf.core.series import Series

        return Series(self._values)

    @property
    @property
    def is_unique(self):
        raise (NotImplementedError)

    @property
    def is_monotonic(self):
        return self.is_monotonic_increasing

    @property
    def is_monotonic_increasing(self):
        raise (NotImplementedError)

    @property
    def is_monotonic_decreasing(self):
        raise (NotImplementedError)

    def get_slice_bound(self, label, side, kind):
        raise (NotImplementedError)

    def __array_function__(self, func, types, args, kwargs):
        from cudf.core.series import Series

        # check if the function is implemented for the current type
        cudf_index_module = type(self)
        for submodule in func.__module__.split(".")[1:]:
            # point cudf_index_module to the correct submodule
            if hasattr(cudf_index_module, submodule):
                cudf_index_module = getattr(cudf_index_module, submodule)
            else:
                return NotImplemented

        fname = func.__name__

        handled_types = [Index, Series]

        # check if  we don't handle any of the types (including sub-class)
        for t in types:
            if not any(
                issubclass(t, handled_type) for handled_type in handled_types
            ):
                return NotImplemented

        if hasattr(cudf_index_module, fname):
            cudf_func = getattr(cudf_index_module, fname)
            # Handle case if cudf_func is same as numpy function
            if cudf_func is func:
                return NotImplemented
            else:
                return cudf_func(*args, **kwargs)

        else:
            return NotImplemented

    def isin(self, values):
        return self.to_series().isin(values)

    @property
    def __cuda_array_interface__(self):
        raise (NotImplementedError)

    def memory_usage(self, deep=False):
        return self._values._memory_usage(deep=deep)

    @classmethod
    def from_pandas(cls, index):
        if not isinstance(index, pd.Index):
            raise TypeError("not a pandas.Index")

        ind = as_index(pa.Array.from_pandas(index))
        ind.name = index.name
        return ind

    @classmethod
    def _from_table(cls, table, names=None):
        if not isinstance(table, RangeIndex):
            if table._num_columns == 0:
                raise ValueError("Cannot construct Index from any empty Table")
            if names is None:
                names = table._data.names
            if table._num_columns == 1:
                return as_index(table._data.columns[0], name=names[0])
            else:
                return cudf.MultiIndex._from_table(table, names=names)
        else:
            return as_index(table)


class RangeIndex(Index):
    """An iterable integer index defined by a starting value and ending value.
    Can be sliced and indexed arbitrarily without allocating memory for the
    complete structure.

    Properties
    ---
    _start: The first value
    _stop: The last value
    name: Name of the index
    """

    def __init__(self, start, stop=None, name=None):
        """RangeIndex(size), RangeIndex(start, stop)

        Parameters
        ----------
        start, stop: int
        name: string
        """
        if isinstance(start, range):
            therange = start
            start = therange.start
            stop = therange.stop
        if stop is None:
            start, stop = 0, start
        self._start = int(start)
        self._stop = int(stop)
        self._cached_values = None
        self._index = None
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def _num_columns(self):
        return 1

    @property
    def _num_rows(self):
        return len(self)

    @cached_property
    def _values(self):
        if len(self) > 0:
            vals = cupy.arange(self._start, self._stop, dtype=self.dtype)
            return column.as_column(vals)
        else:
            return column.column_empty(0, masked=False, dtype=self.dtype)

    @property
    def _data(self):
        from cudf.core.column_accessor import ColumnAccessor

        return ColumnAccessor({self.name: self._values})

    def __contains__(self, item):
        if not isinstance(
            item, tuple(np.sctypes["int"] + np.sctypes["float"] + [int, float])
        ):
            return False
        if not item % 1 == 0:
            return False
        if self._start <= item < self._stop:
            return True
        else:
            return False

    def copy(self, deep=True):
        return RangeIndex(start=self._start, stop=self._stop, name=self.name)

    def __repr__(self):
        return (
            "{}(start={}, stop={}".format(
                self.__class__.__name__, self._start, self._stop
            )
            + (
                ", name='{}'".format(str(self.name))
                if self.name is not None
                else ""
            )
            + ")"
        )

    def __len__(self):
        return max(0, self._stop - self._start)

    def __getitem__(self, index):
        from numbers import Number

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            sln = (stop - start) // step
            sln = max(0, sln)
            start += self._start
            stop += self._start
            if sln == 0:
                return RangeIndex(0, None, self.name)
            elif step == 1:
                return RangeIndex(start, stop, self.name)
            else:
                return index_from_range(start, stop, step)

        elif isinstance(index, Number):
            index = utils.normalize_index(index, len(self))
            index += self._start
            return index
        else:
            if is_scalar(index):
                index = min_signed_type(index)(index)
            index = column.as_column(index)

        return as_index(self._values[index], name=self.name)

    def __eq__(self, other):
        return super(type(self), self).__eq__(other)

    def __reduce__(self):
        return (RangeIndex, (self._start, self._stop, self.name))

    def equals(self, other):
        if self is other:
            return True
        if len(self) != len(other):
            return False
        if isinstance(other, cudf.core.index.RangeIndex):
            return self._start == other._start and self._stop == other._stop
        else:
            return (self == other)._values.all()

    def serialize(self):
        """Serialize Index file storage or network transmission.
        """
        header = {}
        header["index_column"] = {}

        # store metadata values of index separately
        # We don't need to store the GPU buffer for RangeIndexes
        # cuDF only needs to store start/stop and rehydrate
        # during de-serialization
        header["index_column"]["start"] = self._start
        header["index_column"]["stop"] = self._stop
        frames = []

        header["name"] = pickle.dumps(self.name)
        header["dtype"] = pickle.dumps(self.dtype)
        header["type-serialized"] = pickle.dumps(type(self))
        header["frame_count"] = 0
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        """
        """
        h = header["index_column"]
        name = pickle.loads(header["name"])
        start = h["start"]
        stop = h["stop"]
        return RangeIndex(start=start, stop=stop, name=name)

    @property
    def dtype(self):
        return np.dtype(np.int64)

    @property
    def is_contiguous(self):
        return True

    @property
    def size(self):
        return max(0, self._stop - self._start)

    def find_label_range(self, first, last):
        # clip first to range
        if first is None or first < self._start:
            begin = self._start
        elif first < self._stop:
            begin = first
        else:
            begin = self._stop
        # clip last to range
        if last is None:
            end = self._stop
        elif last < self._start:
            end = begin
        elif last < self._stop:
            end = last + 1
        else:
            end = self._stop
        # shift to index
        return begin - self._start, end - self._start

    @copy_docstring(_to_frame)
    def to_frame(self, index=True, name=None):
        return _to_frame(self, index, name)

    def to_gpu_array(self):
        return self._values.to_gpu_array()

    def to_pandas(self):
        return pd.RangeIndex(
            start=self._start,
            stop=self._stop,
            dtype=self.dtype,
            name=self.name,
        )

    @property
    def is_unique(self):
        return True

    @property
    def is_monotonic_increasing(self):
        return self._start <= self._stop

    @property
    def is_monotonic_decreasing(self):
        return self._start >= self._stop

    def get_slice_bound(self, label, side, kind):
        if label < self._start:
            return 0
        elif label >= self._stop:
            return len(self)
        else:
            if side == "left":
                return label - self._start
            elif side == "right":
                return (label - self._start) + 1

    @property
    def __cuda_array_interface__(self):
        return self._values.__cuda_array_interface__

    def memory_usage(self, **kwargs):
        return 0

    def unique(self):
        # RangeIndex always has unique values
        return self


def index_from_range(start, stop=None, step=None):
    vals = cupy.arange(start, stop, step, dtype=np.int64)
    return as_index(vals)


class GenericIndex(Index):
    """An array of orderable values that represent the indices of another Column

    Attributes
    ---
    _values: A Column object
    name: A string
    """

    def __init__(self, values, **kwargs):
        """
        Parameters
        ----------
        values : Column
            The Column of values for this index
        name : str optional
            The name of the Index. If not provided, the Index adopts the value
            Column's name. Otherwise if this name is different from the value
            Column's, the values Column will be cloned to adopt this name.
        """
        from cudf.core.series import Series

        kwargs = _setdefault_name(values, kwargs)

        # normalize the input
        if isinstance(values, Series):
            values = values._column
        elif isinstance(values, column.ColumnBase):
            values = values
        else:
            if isinstance(values, (list, tuple)):
                if len(values) == 0:
                    values = np.asarray([], dtype="int64")
                else:
                    values = np.asarray(values)
            values = column.as_column(values)
            assert isinstance(values, (NumericalColumn, StringColumn))

        name = kwargs.get("name")
        super().__init__({name: values})

    @property
    def _values(self):
        return next(iter(self._data.columns))

    def copy(self, deep=True):
        result = as_index(self._values.copy(deep=deep))
        result.name = self.name
        return result

    def __sizeof__(self):
        return self._values.__sizeof__()

    def __reduce__(self):
        _maker = functools.partial(
            self.__class__, self._values, name=self.name
        )

        return _maker, ()

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        vals = [self._values[i] for i in range(min(len(self), 10))]
        return (
            "{}({}, dtype={}".format(
                self.__class__.__name__, vals, self._values.dtype
            )
            + (
                ", name='{}'".format(self.name)
                if self.name is not None
                else ""
            )
            + ")"
        )

    def __getitem__(self, index):
        res = self._values[index]
        if not isinstance(index, int):
            res = as_index(res)
            res.name = self.name
            return res
        else:
            return res

    @copy_docstring(_to_frame)
    def to_frame(self, index=True, name=None):
        return _to_frame(self, index, name)

    @property
    def dtype(self):
        return self._values.dtype

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

    @property
    def is_unique(self):
        return self._values.is_unique

    @property
    def is_monotonic(self):
        return self._values.is_monotonic

    @property
    def is_monotonic_increasing(self):
        return self._values.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        return self._values.is_monotonic_decreasing

    def get_slice_bound(self, label, side, kind):
        return self._values.get_slice_bound(label, side, kind)

    @property
    def __cuda_array_interface__(self):
        return self._values.__cuda_array_interface__


class DatetimeIndex(GenericIndex):
    # TODO this constructor should take a timezone or something to be
    # consistent with pandas
    def __init__(self, values, **kwargs):
        # we should be more strict on what we accept here but
        # we'd have to go and figure out all the semantics around
        # pandas dtindex creation first which.  For now
        # just make sure we handle np.datetime64 arrays
        # and then just dispatch upstream
        kwargs = _setdefault_name(values, kwargs)
        if isinstance(values, np.ndarray) and values.dtype.kind == "M":
            values = column.as_column(values)
        elif isinstance(values, pd.DatetimeIndex):
            values = column.as_column(values.values)
        elif isinstance(values, (list, tuple)):
            values = column.as_column(np.array(values, dtype="<M8[ms]"))
        super(DatetimeIndex, self).__init__(values, **kwargs)

    @property
    def year(self):
        return self.get_dt_field("year")

    @property
    def month(self):
        return self.get_dt_field("month")

    @property
    def day(self):
        return self.get_dt_field("day")

    @property
    def hour(self):
        return self.get_dt_field("hour")

    @property
    def minute(self):
        return self.get_dt_field("minute")

    @property
    def second(self):
        return self.get_dt_field("second")

    @property
    def weekday(self):
        return self.get_dt_field("weekday")

    def to_pandas(self):
        nanos = self._values.astype("datetime64[ns]")
        return pd.DatetimeIndex(nanos.to_pandas(), name=self.name)

    def get_dt_field(self, field):
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


class CategoricalIndex(GenericIndex):
    """An categorical of orderable values that represent the indices of another
    Column

    Attributes
    ---
    _values: A CategoricalColumn object
    name: A string
    """

    def __init__(self, values, **kwargs):
        kwargs = _setdefault_name(values, kwargs)
        if isinstance(values, CategoricalColumn):
            values = values
        elif isinstance(values, pd.Series) and (
            is_categorical_dtype(values.dtype)
        ):
            codes_data = column.as_column(values.cat.codes.values)
            values = column.build_categorical_column(
                categories=values.cat.categories,
                codes=codes_data,
                ordered=values.cat.ordered,
            )
        elif isinstance(values, (pd.Categorical, pd.CategoricalIndex)):
            codes_data = column.as_column(values.codes)
            values = column.build_categorical_column(
                categories=values.categories,
                codes=codes_data,
                ordered=values.ordered,
            )
        elif isinstance(values, (list, tuple)):
            values = column.as_column(
                pd.Categorical(values, categories=values)
            )
        super(CategoricalIndex, self).__init__(values, **kwargs)

    @property
    def codes(self):
        return self._values.cat().codes

    @property
    def categories(self):
        return self._values.cat().categories


class StringIndex(GenericIndex):
    """String defined indices into another Column

    Attributes
    ---
    _values: A StringColumn object or NDArray of strings
    name: A string
    """

    def __init__(self, values, **kwargs):
        kwargs = _setdefault_name(values, kwargs)
        if isinstance(values, StringColumn):
            values = values.copy()
        elif isinstance(values, StringIndex):
            values = values._values.copy()
        else:
            values = column.as_column(values, dtype="str")
            if not pd.api.types.is_string_dtype(values.dtype):
                raise ValueError(
                    "Couldn't create StringIndex from passed in object"
                )
        super(StringIndex, self).__init__(values, **kwargs)

    def to_pandas(self):
        return pd.Index(self.to_array(), name=self.name, dtype="object")

    def take(self, indices):
        return self._values[indices]

    def __repr__(self):
        return (
            "{}({}, dtype='object'".format(
                self.__class__.__name__, self._values.to_array()
            )
            + (
                ", name='{}'".format(self.name)
                if self.name is not None
                else ""
            )
            + ")"
        )


def as_index(arbitrary, **kwargs):
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

    kwargs = _setdefault_name(arbitrary, kwargs)

    if isinstance(arbitrary, cudf.MultiIndex):
        return arbitrary
    elif isinstance(arbitrary, Index):
        idx = arbitrary.copy(deep=False)
        idx.rename(**kwargs, inplace=True)
        return idx
    elif isinstance(arbitrary, NumericalColumn):
        return GenericIndex(arbitrary, **kwargs)
    elif isinstance(arbitrary, StringColumn):
        return StringIndex(arbitrary, **kwargs)
    elif isinstance(arbitrary, DatetimeColumn):
        return DatetimeIndex(arbitrary, **kwargs)
    elif isinstance(arbitrary, CategoricalColumn):
        return CategoricalIndex(arbitrary, **kwargs)
    elif isinstance(arbitrary, cudf.Series):
        return as_index(arbitrary._column, **kwargs)
    elif isinstance(arbitrary, pd.RangeIndex):
        return RangeIndex(start=arbitrary.start, stop=arbitrary.stop, **kwargs)
    elif isinstance(arbitrary, pd.MultiIndex):
        return cudf.MultiIndex.from_pandas(arbitrary)
    elif isinstance(arbitrary, range):
        if arbitrary.step == 1:
            return RangeIndex(arbitrary.start, arbitrary.stop, **kwargs)
    return as_index(column.as_column(arbitrary), **kwargs)


def _setdefault_name(values, kwargs):
    if "name" not in kwargs:
        if not hasattr(values, "name"):
            kwargs.setdefault("name", None)
        else:
            kwargs.setdefault("name", values.name)
    return kwargs
