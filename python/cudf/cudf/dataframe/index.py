# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division, print_function

import pickle
from copy import copy, deepcopy

import numpy as np
import pandas as pd
from numba.cuda.cudadrv.devicearray import DeviceNDArray

import nvstrings
from librmm_cffi import librmm as rmm

import cudf
import cudf.bindings.copying as cpp_copying
from cudf.comm.serialize import register_distributed_serializer
from cudf.dataframe import columnops
from cudf.dataframe.buffer import Buffer
from cudf.dataframe.categorical import CategoricalColumn
from cudf.dataframe.column import Column
from cudf.dataframe.datetime import DatetimeColumn
from cudf.dataframe.numerical import NumericalColumn
from cudf.dataframe.string import StringColumn
from cudf.indexing import _IndexLocIndexer
from cudf.utils import cudautils, ioutils, utils


class Index(object):
    """The root interface for all Series indexes.
    """

    is_monotonic = None
    is_monotonic_increasing = None
    is_monotonic_decreasing = None

    def serialize(self, serialize):
        """Serialize into pickle format suitable for file storage or network
        transmission.

        Parameters
        ---
        serialize:  A function provided by register_distributed_serializer
        middleware.
        """
        header = {}
        header["payload"], frames = serialize(pickle.dumps(self))
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        """Convert from pickle format into Index

        Parameters
        ---
        deserialize:  A function provided by register_distributed_serializer
        middleware.
        header: The data header produced by the serialize function.
        frames: The serialized data
        """
        payload = deserialize(
            header["payload"], frames[: header["frame_count"]]
        )
        return pickle.loads(payload)

    def take(self, indices):
        """Gather only the specific subset of indices

        Parameters
        ---
        indices: An array-like that maps to values contained in this Index.
        """
        assert indices.dtype.kind in "iu"
        if indices.size == 0:
            # Empty indices
            return RangeIndex(indices.size)
        else:
            # Gather
            index = cpp_copying.apply_gather_array(self.gpu_values, indices)
            col = self.as_column().replace(data=index.data)
            new_index = as_index(col)
            new_index.name = self.name
            return new_index

    def argsort(self, ascending=True):
        return self.as_column().argsort(ascending=ascending)

    @property
    def values(self):
        return np.asarray([i for i in self.as_column()])

    def to_pandas(self):
        return pd.Index(self.as_column().to_pandas(), name=self.name)

    def to_arrow(self):
        return self.as_column().to_arrow()

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""
        import cudf.io.dlpack as dlpack

        return dlpack.to_dlpack(self)

    @property
    def gpu_values(self):
        return self.as_column().to_gpu_array()

    def min(self):
        return self.as_column().min()

    def max(self):
        return self.as_column().max()

    def sum(self):
        return self.as_column().sum()

    def find_segments(self):
        """Return the beginning index for segments

        Returns
        -------
        result : NumericalColumn
        """
        segments, _ = self._find_segments()
        return segments

    def _find_segments(self):
        seg, markers = cudautils.find_segments(self.gpu_values)
        return NumericalColumn(data=Buffer(seg), dtype=seg.dtype), markers

    @classmethod
    def _concat(cls, objs):
        data = Column._concat([o.as_column() for o in objs])
        return as_index(data)

    def _apply_op(self, fn, other=None):
        from cudf.dataframe.series import Series

        idx_series = Series(self, name=self.name)
        op = getattr(idx_series, fn)
        if other is not None:
            return as_index(op(other))
        else:
            return as_index(op())

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
        column_join_res = self.as_column().join(
            other.as_column(),
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

    def rename(self, name):
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

        Difference from pandas:
          * Not supporting: inplace
        """
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
        if dtype == self.dtype:
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
        from cudf.dataframe.series import Series

        return Series(self._values)

    @property
    def loc(self):
        return _IndexLocIndexer(self)


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
        self.name = name

    def copy(self, deep=True):
        if deep:
            result = deepcopy(self)
        else:
            result = copy(self)
        result.name = self.name
        return result

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
        if isinstance(index, slice):
            start, stop, step, sln = utils.standard_python_slice(
                len(self), index
            )
            start += self._start
            stop += self._start
            if sln == 0:
                return RangeIndex(0)
            else:
                return index_from_range(start, stop, step)

        elif isinstance(index, int):
            index = utils.normalize_index(index, len(self))
            index += self._start
            return index
        elif isinstance(index, (list, np.ndarray)):
            index = np.array(index)
            index = rmm.to_device(index)

        if isinstance(index, (DeviceNDArray)):
            return self.take(index)
        else:
            raise ValueError(index)

    def __eq__(self, other):
        return super(type(self), self).__eq__(other)

    def equals(self, other):
        if isinstance(other, cudf.dataframe.index.RangeIndex):
            return self._start == other._start and self._stop == other._stop
        else:
            return (self == other)._values.all()

    @property
    def dtype(self):
        return np.dtype(np.int64)

    @property
    def _values(self):
        return self.as_column()

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

    def as_column(self):
        if len(self) > 0:
            vals = cudautils.arange(self._start, self._stop, dtype=self.dtype)
        else:
            vals = rmm.device_array(0, dtype=self.dtype)
        return NumericalColumn(data=Buffer(vals), dtype=vals.dtype)

    def to_gpu_array(self):
        return self.as_column().to_gpu_array()

    def to_pandas(self):
        return pd.RangeIndex(
            start=self._start,
            stop=self._stop,
            dtype=self.dtype,
            name=self.name,
        )


def index_from_range(start, stop=None, step=None):
    vals = cudautils.arange(start, stop, step, dtype=np.int64)
    return as_index(vals)


class GenericIndex(Index):
    """An array of orderable values that represent the indices of another Column

    Attributes
    ---
    _values: A Column object
    name: A string
    """

    def __init__(self, values, name=None):
        from cudf.dataframe.series import Series

        # normalize the input
        if isinstance(values, Series):
            name = values.name
            values = values._column
        elif isinstance(values, columnops.TypedColumnBase):
            values = values
        else:
            if isinstance(values, (list, tuple)):
                if len(values) == 0:
                    values = np.asarray([], dtype="int64")
                else:
                    values = np.asarray(values)
            values = columnops.as_column(values)
            assert isinstance(values, (NumericalColumn, StringColumn))

        assert isinstance(values, columnops.TypedColumnBase), type(values)

        self._values = values
        self.name = name

    def copy(self, deep=True):
        if deep:
            result = deepcopy(self)
        else:
            result = copy(self)
        result._values = self._values.copy(deep)
        result.name = self.name
        return result

    def serialize(self, serialize):
        header = {}
        header["payload"], frames = serialize(self._values)
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        payload = deserialize(
            header["payload"], frames[: header["frame_count"]]
        )
        return cls(payload)

    def __sizeof__(self):
        return self._values.__sizeof__()

    def __reduce__(self):
        return GenericIndex, tuple([self._values])

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
            return as_index(res)
        else:
            return res

    def as_column(self):
        """Convert the index as a Series.
        """
        return self._values

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
            begin = col.find_first_value(first)
        if last is not None:
            end = col.find_last_value(last)
            end += 1
        return begin, end


class DatetimeIndex(GenericIndex):
    # TODO this constructor should take a timezone or something to be
    # consistent with pandas
    def __init__(self, values, name=None):
        # we should be more strict on what we accept here but
        # we'd have to go and figure out all the semantics around
        # pandas dtindex creation first which.  For now
        # just make sure we handle np.datetime64 arrays
        # and then just dispatch upstream
        if name is None and hasattr(values, "name"):
            name = values.name
        if isinstance(values, np.ndarray) and values.dtype.kind == "M":
            values = DatetimeColumn.from_numpy(values)
        elif isinstance(values, pd.DatetimeIndex):
            values = DatetimeColumn.from_numpy(values.values)
        elif isinstance(values, (list, tuple)):
            values = DatetimeColumn.from_numpy(
                np.array(values, dtype="<M8[ms]")
            )

        assert values.null_count == 0
        self._values = values
        self.name = name

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

    def get_dt_field(self, field):
        out_column = self._values.get_dt_field(field)
        # columnops.column_empty_like always returns a Column object
        # but we need a NumericalColumn for GenericIndex..
        # how should this be handled?
        out_column = NumericalColumn(
            data=out_column.data,
            mask=out_column.mask,
            null_count=out_column.null_count,
            dtype=out_column.dtype,
            name=self.name,
        )
        return as_index(out_column)


class CategoricalIndex(GenericIndex):
    """An categorical of orderable values that represent the indices of another
    Column

    Attributes
    ---
    _values: A CategoricalColumn object
    name: A string
    """

    def __init__(self, values, name=None):
        if isinstance(values, CategoricalColumn):
            values = values
        elif isinstance(values, pd.Series) and \
                pd.api.types.is_categorical_dtype(values.dtype):
            values = CategoricalColumn(
                data=Buffer(values.cat.codes.values),
                categories=values.cat.categories,
                ordered=values.cat.ordered
            )
        elif isinstance(values, (pd.Categorical, pd.CategoricalIndex)):
            values = CategoricalColumn(
                data=Buffer(values.codes),
                categories=values.categories,
                ordered=values.ordered
            )
        elif isinstance(values, (list, tuple)):
            values = columnops.as_column(
                pd.Categorical(values, categories=values)
            )

        assert values.null_count == 0
        self._values = values
        self.name = name
        self.names = [name]

    @property
    def codes(self):
        return self._values.codes

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

    def __init__(self, values, name=None):
        if isinstance(values, StringColumn):
            self._values = values.copy()
        elif isinstance(values, StringIndex):
            if name is None:
                name = values.name
            self._values = values._values.copy()
        else:
            self._values = columnops.build_column(
                nvstrings.to_device(values), dtype="object"
            )
        assert self._values.null_count == 0
        self.name = name

    @property
    def codes(self):
        return self._values.codes

    @property
    def categories(self):
        return self._values.categories

    def to_pandas(self):
        result = pd.Index(self.values, name=self.name, dtype="object")
        return result

    def take(self, indices):
        return columnops.as_column(self._values).element_indexing(indices)

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


def as_index(arbitrary, name=None):
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
    # This function should probably be moved to Index.__new__
    if hasattr(arbitrary, "name") and name is None:
        name = arbitrary.name

    if isinstance(arbitrary, Index):
        return arbitrary.rename(name=name)
    elif isinstance(arbitrary, NumericalColumn):
        return GenericIndex(arbitrary, name=name)
    elif isinstance(arbitrary, StringColumn):
        return StringIndex(arbitrary, name=name)
    elif isinstance(arbitrary, DatetimeColumn):
        return DatetimeIndex(arbitrary, name=name)
    elif isinstance(arbitrary, CategoricalColumn):
        return CategoricalIndex(arbitrary, name=name)
    else:
        return as_index(columnops.as_column(arbitrary), name=name)


register_distributed_serializer(RangeIndex)
register_distributed_serializer(GenericIndex)
register_distributed_serializer(DatetimeIndex)
register_distributed_serializer(CategoricalIndex)
