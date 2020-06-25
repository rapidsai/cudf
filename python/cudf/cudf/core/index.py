# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import division, print_function

import pickle

import cupy
import numpy as np
import pandas as pd

import cudf
from cudf._lib.nvtx import annotate
from cudf.core.abc import Serializable
from cudf.core.column import (
    CategoricalColumn,
    ColumnBase,
    DatetimeColumn,
    NumericalColumn,
    StringColumn,
    column,
)
from cudf.core.column.string import StringMethods as StringMethods
from cudf.core.frame import Frame
from cudf.utils import ioutils, utils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import is_categorical_dtype, is_scalar
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


class Index(Frame, Serializable):
    def __new__(
        cls,
        data=None,
        dtype=None,
        copy=False,
        name=None,
        tupleize_cols=True,
        **kwargs,
    ):
        if tupleize_cols is not True:
            raise NotImplementedError(
                "tupleize_cols != True is not yet supported"
            )

        return as_index(data, copy=copy, dtype=dtype, name=name, **kwargs)

    def __init__(
        self,
        data=None,
        dtype=None,
        copy=False,
        name=None,
        tupleize_cols=True,
        **kwargs,
    ):
        """Immutable, ordered and sliceable sequence of integer labels.
        The basic object storing row labels for all cuDF objects.

        Parameters:
        -----------
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

        Returns
        -------
        Index
            cudf Index

        Examples
        --------
        >>> import cudf
        >>> cudf.Index([1, 2, 3], dtype="uint64", name="a")
        UInt64Index([1, 2, 3], dtype='uint64', name='a')

        >>> cudf.Index(cudf.DataFrame({"a":[1, 2], "b":[2, 3]}))
        MultiIndex(levels=[0    1
        1    2
        dtype: int64, 0    2
        1    3
        dtype: int64],
        codes=   a  b
        0  0  0
        1  1  1)
        """
        pass

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
        """
        Return an Index of values for requested level.

        This is primarily useful to get an individual level of values from a
        MultiIndex, but is provided on Index as well for compatibility.

        Parameters
        ----------
        level : int or str
            It is either the integer position or the name of the level.

        Returns
        -------
        Index
            Calling object, as there is only one level in the Index.

        See Also
        --------
        cudf.core.multiindex.get_level_values : Get values for a level
            of a MultiIndex.

        Notes
        -----
        For Index, level should be 0, since there are no multiple levels.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.core.index.StringIndex(["a","b","c"])
        >>> idx.get_level_values(0)
        StringIndex(['a' 'b' 'c'], dtype='object')
        """

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

    def _mimic_inplace(self, other, inplace=False):
        if inplace is True:
            col = self._data[self.name]
            col._mimic_inplace(other._data[other.name], inplace=True)
        else:
            return other

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
    def ndim(self):
        """Dimension of the data. Apart from MultiIndex ndim is always 1.
        """
        return 1

    @property
    def names(self):
        """
        Returns a tuple containing the name of the Index.
        """
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
        """
        Returns the name of the Index.
        """
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
        ----------
        indices: An array-like that maps to values contained in this Index.
        """
        return self[indices]

    def argsort(self, ascending=True):
        """
        Return the integer indices that would sort the index.

        Parameters
        ----------
        ascending : bool, default True
            If True, returns the indices for ascending order.
            If False, returns the indices for descending order.

        Returns
        -------
        array : A cupy array containing Integer indices that
            would sort the index if used as an indexer.
        """
        indices = self._values.argsort(ascending=ascending)
        return cupy.asarray(indices)

    @property
    def values(self):
        """
        Return an array representing the data in the Index.

        Returns
        -------
        array : A cupy array of data in the Index.
        """
        if is_categorical_dtype(self.dtype) or np.issubdtype(
            self.dtype, np.dtype("object")
        ):
            raise TypeError("Data must be numeric")
        if len(self) == 0:
            return cupy.asarray([], dtype=self.dtype)
        if self._values.null_count > 0:
            raise ValueError("Column must have no nulls.")

        return cupy.asarray(self._values.data_array_view)

    def any(self):
        """
        Return whether any elements is True in Index.
        """
        return self._values.any()

    def to_pandas(self):
        """
        Convert to a Pandas Index.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.core.index.as_index([-3, 10, 15, 20])
        >>> idx
        Int64Index([-3, 10, 15, 20], dtype='int64')
        >>> idx.to_pandas()
        Int64Index([-3, 10, 15, 20], dtype='int64')
        >>> type(idx.to_pandas())
        <class 'pandas.core.indexes.numeric.Int64Index'>
        >>> type(idx)
        <class 'cudf.core.index.GenericIndex'>
        """
        return pd.Index(self._values.to_pandas(), name=self.name)

    def to_arrow(self):
        """
        Convert Index to a PyArrow Array.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.core.index.as_index([-3, 10, 15, 20])
        >>> idx.to_arrow()
        <pyarrow.lib.Int64Array object at 0x7fcaa6f53440>
        [
        -3,
        10,
        15,
        20
        ]
        """
        return self._values.to_arrow()

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""
        import cudf.io.dlpack as dlpack

        return dlpack.to_dlpack(self)

    @property
    def gpu_values(self):
        """
        View the data as a numba device array object
        """
        return self._values.data_array_view

    def min(self):
        """
        Return the minimum value of the Index.

        Returns
        -------
        scalar
            Minimum value.

        See Also
        --------
        Index.max : Return the maximum value in an Index.
        cudf.core.series.Series.min : Return the minimum value in a Series.
        cudf.core.dataframe.DataFrame.min : Return the minimum values in
            a DataFrame.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.core.index.as_index([3, 2, 1])
        >>> idx.min()
        1
        """
        return self._values.min()

    def max(self):
        """
        Return the maximum value of the Index.

        Returns
        -------
        scalar
            Maximum value.

        See Also
        --------
        Index.min : Return the minimum value in an Index.
        cudf.core.series.Series.max : Return the maximum value in a Series.
        cudf.core.dataframe.Dataframe.max : Return the maximum values in
            a DataFrame.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.core.index.as_index([3, 2, 1])
        >>> idx.max()
        3
        """
        return self._values.max()

    def sum(self):
        """
        Return the sum of all values of the Index.

        Returns
        -------
        scalar
            Sum of all values.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.core.index.as_index([3, 2, 1])
        >>> idx.sum()
        6
        """
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

        idx_series = cudf.Series(self, name=self.name)
        op = getattr(idx_series, fn)
        if other is not None:
            return as_index(op(other))
        else:
            return as_index(op())

    def unique(self):
        """
        Return unique values in the index.

        Returns
        -------
        Index without duplicates
        """
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

    @annotate("INDEX_EQUALS", color="green", domain="cudf_python")
    def equals(self, other):
        """
        Determine if two Index objects contain the same elements.

        Returns
        -------
        out: bool
            True if “other” is an Index and it has the same elements
            as calling index; False otherwise.
        """
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

    def join(
        self, other, how="left", level=None, return_indexers=False, sort=False
    ):
        """
        Compute join_index and indexers to conform data structures
        to the new index.

        Parameters
        ----------
        other : Index.
        how : {'left', 'right', 'inner', 'outer'}
        return_indexers : bool, default False
        sort : bool, default False
            Sort the join keys lexicographically in the result Index. If False,
            the order of the join keys depends on the join type (how keyword).

        Returns: index

        Examples
        --------
        >>> import cudf
        >>> lhs = cudf.DataFrame(
        ...     {"a":[2, 3, 1], "b":[3, 4, 2]}).set_index(['a', 'b']
        ... ).index
        >>> rhs = cudf.DataFrame({"a":[1, 4, 3]}).set_index('a').index
        >>> lhs.join(rhs, how='inner')
        MultiIndex(levels=[0    1
        1    3
        dtype: int64, 0    2
        1    4
        dtype: int64],
        codes=   a  b
        0  1  1
        1  0  0)
        """

        if isinstance(self, cudf.MultiIndex) and isinstance(
            other, cudf.MultiIndex
        ):
            raise TypeError(
                "Join on level between two MultiIndex objects is ambiguous"
            )

        if level is not None and not is_scalar(level):
            raise ValueError("level should be an int or a label only")

        if isinstance(other, cudf.MultiIndex):
            if how == "left":
                how = "right"
            elif how == "right":
                how = "left"
            rhs = self.copy(deep=False)
            lhs = other.copy(deep=False)
        else:
            lhs = self.copy(deep=False)
            rhs = other.copy(deep=False)

        on = level
        # In case of MultiIndex, it will be None as
        # we don't need to update name
        left_names = lhs.names
        right_names = rhs.names
        # There should be no `None` values in Joined indices,
        # so essentially it would be `left/right` or 'inner'
        # in case of MultiIndex
        if isinstance(lhs, cudf.MultiIndex):
            if level is not None and isinstance(level, int):
                on = lhs._data.get_by_index(level).names[0]
            right_names = (on,) or right_names
            on = right_names[0]
            if how == "outer":
                how = "left"
            elif how == "right":
                how = "inner"
        else:
            # Both are nomal indices
            right_names = left_names
            on = right_names[0]

        lhs.names = left_names
        rhs.names = right_names

        output = lhs._merge(rhs, how=how, on=on, sort=sort)

        return output

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

    def astype(self, dtype, copy=False):
        """
        Create an Index with values cast to dtypes. The class of a new Index
        is determined by dtype. When conversion is impossible, a ValueError
        exception is raised.

        Parameters
        ----------
        dtype : numpy dtype
            Use a numpy.dtype to cast entire Index object to.
        copy : bool, default False
            By default, astype always returns a newly allocated object.
            If copy is set to False and internal requirements on dtype are
            satisfied, the original data is used to create a new Index
            or the original Index is returned.

        Returns
        -------
        Index
            Index with values cast to specified dtype.
        """
        if pd.api.types.is_dtype_equal(dtype, self.dtype):
            return self.copy(deep=copy)

        return as_index(
            self.copy(deep=copy)._values.astype(dtype), name=self.name
        )

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

    def to_series(self, index=None, name=None):
        """
        Create a Series with both index and values equal to the index keys.
        Useful with map for returning an indexer based on an index.

        Parameters
        ----------
        index : Index, optional
            Index of resulting Series. If None, defaults to original index.
        name : str, optional
            Dame of resulting Series. If None, defaults to name of original
            index.

        Returns
        -------
        Series
            The dtype will be based on the type of the Index values.
        """

        return cudf.Series(
            self._values,
            index=self.copy(deep=False) if index is None else index,
            name=self.name if name is None else name,
        )

    @property
    def is_unique(self):
        """
        Return if the index has unique values.
        """
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
        """
        Calculate slice bound that corresponds to given label.
        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}
        kind : {'ix', 'loc', 'getitem'}

        Returns
        -------
        int
            Index of label.
        """
        raise (NotImplementedError)

    def __array_function__(self, func, types, args, kwargs):

        # check if the function is implemented for the current type
        cudf_index_module = type(self)
        for submodule in func.__module__.split(".")[1:]:
            # point cudf_index_module to the correct submodule
            if hasattr(cudf_index_module, submodule):
                cudf_index_module = getattr(cudf_index_module, submodule)
            else:
                return NotImplemented

        fname = func.__name__

        handled_types = [Index, cudf.Series]

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
        """Return a boolean array where the index values are in values.

        Compute boolean array of whether each index value is found in
        the passed set of values. The length of the returned boolean
        array matches the length of the index.

        Parameters
        ----------
        values : set, list-like, Index
            Sought values.

        Returns
        -------
        is_contained : cupy array
            CuPy array of boolean values.

        """

        result = self.to_series().isin(values).values

        return result

    def where(self, cond, other=None):
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : bool array-like with the same length as self
            Where cond is True, keep the original value.
            Where False, replace with corresponding value from other.
            Callables are not supported.
        other: scalar, or array-like
            Entries where cond is False are replaced with
            corresponding value from other. Callables are not
            supported. Default is None.

        Returns
        -------
        Same type as caller
        """
        return super().where(cond=cond, other=other)

    @property
    def __cuda_array_interface__(self):
        raise (NotImplementedError)

    def memory_usage(self, deep=False):
        """
        Memory usage of the values.

        Parameters
        ----------
            deep : bool
                Introspect the data deeply,
                interrogate `object` dtypes for system-level
                memory consumption.

        Returns
        -------
            bytes used
        """
        return self._values._memory_usage(deep=deep)

    @classmethod
    def from_pandas(cls, index, nan_as_null=None):
        """
        Convert from a Pandas Index.

        Parameters
        ----------
        index : Pandas Index object
            A Pandas Index object which has to be converted
            to cuDF Index.
        nan_as_null : bool, Default None
            If ``None``/``True``, converts ``np.nan`` values
            to ``null`` values.
            If ``False``, leaves ``np.nan`` values as is.

        Raises
        ------
        TypeError for invalid input type.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = [10, 20, 30, np.nan]
        >>> pdi = pd.Index(data)
        >>> cudf.core.index.Index.from_pandas(pdi)
        Index(['10.0', '20.0', '30.0', 'null'], dtype='object')
        >>> cudf.core.index.Index.from_pandas(pdi, nan_as_null=False)
        Float64Index([10.0, 20.0, 30.0, nan], dtype='float64')
        """
        if not isinstance(index, pd.Index):
            raise TypeError("not a pandas.Index")

        ind = as_index(column.as_column(index, nan_as_null=nan_as_null))
        ind.name = index.name
        return ind

    @classmethod
    def _from_table(cls, table):
        if not isinstance(table, RangeIndex):
            if table._num_columns == 0:
                raise ValueError("Cannot construct Index from any empty Table")
            if table._num_columns == 1:
                values = next(iter(table._data.values()))

                if isinstance(values, NumericalColumn):
                    try:
                        index_class_type = _dtype_to_index[values.dtype.type]
                    except KeyError:
                        index_class_type = GenericIndex
                    out = super(Index, index_class_type).__new__(
                        index_class_type
                    )
                elif isinstance(values, DatetimeColumn):
                    out = super(Index, DatetimeIndex).__new__(DatetimeIndex)
                elif isinstance(values, StringColumn):
                    out = super(Index, StringIndex).__new__(StringIndex)
                elif isinstance(values, CategoricalColumn):
                    out = super(Index, CategoricalIndex).__new__(
                        CategoricalIndex
                    )
                out._data = table._data
                out._index = None
                return out
            else:
                return cudf.MultiIndex._from_table(
                    table, names=table._data.names
                )
        else:
            return as_index(table)


class RangeIndex(Index):
    """
    Immutable Index implementing a monotonic integer range.

    This is the default index type used by DataFrame and Series
    when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), or other range instance
    stop : int (default: 0)
    step : int (default: 1)
        Not yet supported
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
    >>> cudf.RangeIndex(0, 10, name="a")
    RangeIndex(start=0, stop=10, name='a')

    >>> cudf.RangeIndex(range(1, 10), name="a")
    RangeIndex(start=1, stop=10, name='a')
    """

    def __new__(
        cls, start, stop=None, step=None, dtype=None, copy=False, name=None
    ) -> "RangeIndex":

        if step is not None:
            raise NotImplementedError("step is not yet supported")

        out = Frame.__new__(cls)
        if isinstance(start, range):
            therange = start
            start = therange.start
            stop = therange.stop
        if stop is None:
            start, stop = 0, start
        out._start = int(start)
        out._stop = int(stop)
        out._cached_values = None
        out._index = None
        out._name = name

        return out

    @property
    def name(self):
        """
        Returns the name of the Index.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def start(self):
        """
        The value of the `start` parameter (0 if this was not supplied).
        """
        return self._start

    @property
    def stop(self):
        """
        The value of the stop parameter.
        """
        return self._stop

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
        """
        Make a copy of this object.
        """
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
                return RangeIndex(0, stop=None, name=self.name)
            elif step == 1:
                return RangeIndex(start, stop=stop, name=self.name)
            else:
                return index_from_range(start, stop, step)

        elif isinstance(index, Number):
            index = utils.normalize_index(index, len(self))
            index += self._start
            return index
        else:
            if is_scalar(index):
                index = np.min_scalar_type(index).type(index)
            index = column.as_column(index)

        return as_index(self._values[index], name=self.name)

    def __eq__(self, other):
        return super(type(self), self).__eq__(other)

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
        """
        `dtype` of the range of values in RangeIndex.
        """
        return np.dtype(np.int64)

    @property
    def is_contiguous(self):
        """
        Returns if the index is contiguous. `True` incase of RangeIndex.
        """
        return True

    @property
    def size(self):
        """
        Return the number of elements in the underlying data.
        """
        return max(0, self._stop - self._start)

    def find_label_range(self, first, last):
        """Find range that starts with `first` and ends with `last`,
        inclusively.

        Returns
        -------
        begin, end : 2-tuple of int
            The starting index and the ending index.
            The `last` value occurs at ``end - 1`` position.
        """
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

    def to_gpu_array(self, fillna=None):
        """Get a dense numba device array for the data.

        Parameters
        ----------
        fillna : str or None
            Replacement value to fill in place of nulls.

        Notes
        -----
        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        return self._values.to_gpu_array(fillna=fillna)

    def to_pandas(self):
        return pd.RangeIndex(
            start=self._start,
            stop=self._stop,
            dtype=self.dtype,
            name=self.name,
        )

    @property
    def is_unique(self):
        """
        Return if the index has unique values.
        """
        return True

    @property
    def is_monotonic_increasing(self):
        """
        Return if the index is monotonic increasing
        (only equal or increasing) values.
        """
        return self._start <= self._stop

    @property
    def is_monotonic_decreasing(self):
        """
        Return if the index is monotonic decreasing
        (only equal or decreasing) values.
        """
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

    def __new__(cls, values, **kwargs):
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
        out = Frame.__new__(cls)
        out._initialize(values, **kwargs)

        return out

    def _initialize(self, values, **kwargs):

        kwargs = _setdefault_name(values, **kwargs)

        # normalize the input
        if isinstance(values, cudf.Series):
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
        super(Index, self).__init__({name: values})

    @property
    def _values(self):
        return next(iter(self._data.columns))

    def copy(self, deep=True):
        """
        Make a copy of this object.

        Parameters
        ----------
        deep : bool, default True
            Make a deep copy of the data.
            With ``deep=False`` the is not copied.

        Returns
        -------
        copy : Index
        """
        result = as_index(self._values.copy(deep=deep))
        result.name = self.name
        return result

    def __sizeof__(self):
        return self._values.__sizeof__()

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        from pandas._config import get_option

        max_seq_items = get_option("max_seq_items") or len(self)
        mr = 0
        if 2 * max_seq_items < len(self):
            mr = max_seq_items + 1

        if len(self) > mr and mr != 0:
            top = self[0:mr]
            bottom = self[-1 * mr :]
            from cudf import concat

            preprocess = concat([top, bottom])
        else:
            preprocess = self
        if preprocess._values.nullable:
            output = (
                self.__class__(preprocess._values.astype("O").fillna("null"))
                .to_pandas()
                .__repr__()
            )
        else:
            output = preprocess.to_pandas().__repr__()

        lines = output.split("\n")
        if len(lines) > 1:
            tmp_meta = lines[-1]
            prior_to_dtype = lines[-1].split("dtype")[0]
            lines = lines[:-1]
            lines.append(prior_to_dtype + "dtype='%s'" % self.dtype)
            if self.name is not None:
                lines[-1] = lines[-1] + ", name='%s'" % self.name
            if "length" in tmp_meta:
                lines[-1] = lines[-1] + ", length=%d)" % len(self)
            else:
                lines[-1] = lines[-1] + ")"

        return "\n".join(lines)

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
        """
        `dtype` of the underlying values in GenericIndex.
        """
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
        """
        Return if the index has unique values.
        """
        return self._values.is_unique

    @property
    def is_monotonic(self):
        """
        Alias for is_monotonic_increasing.
        """
        return self._values.is_monotonic

    @property
    def is_monotonic_increasing(self):
        """
        Return if the index is monotonic increasing
        (only equal or increasing) values.
        """
        return self._values.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        """
        Return if the index is monotonic decreasing
        (only equal or decreasing) values.
        """
        return self._values.is_monotonic_decreasing

    def get_slice_bound(self, label, side, kind):
        return self._values.get_slice_bound(label, side, kind)

    @property
    def __cuda_array_interface__(self):
        return self._values.__cuda_array_interface__


class NumericIndex(GenericIndex):
    """Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.

    Parameters:
    -----------
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

    def __new__(cls, data=None, dtype=None, copy=False, name=None):

        out = Frame.__new__(cls)
        dtype = _index_to_dtype[cls]
        if copy:
            data = column.as_column(data, dtype=dtype).copy()

        kwargs = _setdefault_name(data, name=name)

        data = column.as_column(data, dtype=dtype)

        out._initialize(data, **kwargs)

        return out


class Int8Index(NumericIndex):
    def __init__(cls, data=None, dtype=None, copy=False, name=None):
        pass


class Int16Index(NumericIndex):
    def __init__(cls, data=None, dtype=None, copy=False, name=None):
        pass


class Int32Index(NumericIndex):
    def __init__(cls, data=None, dtype=None, copy=False, name=None):
        pass


class Int64Index(NumericIndex):
    def __init__(self, data=None, dtype=None, copy=False, name=None):
        pass


class UInt8Index(NumericIndex):
    def __init__(self, data=None, dtype=None, copy=False, name=None):
        pass


class UInt16Index(NumericIndex):
    def __init__(cls, data=None, dtype=None, copy=False, name=None):
        pass


class UInt32Index(NumericIndex):
    def __init__(cls, data=None, dtype=None, copy=False, name=None):
        pass


class UInt64Index(NumericIndex):
    def __init__(cls, data=None, dtype=None, copy=False, name=None):
        pass


class Float32Index(NumericIndex):
    def __init__(cls, data=None, dtype=None, copy=False, name=None):
        pass


class Float64Index(NumericIndex):
    def __init__(cls, data=None, dtype=None, copy=False, name=None):
        pass


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
    ambiguous : ‘infer’, bool-ndarray, ‘NaT’, default ‘raise’
        This is not yet supported
    name : object
        Name to be stored in the index.
    dayfirst : bool, default False
        If True, parse dates in data with the day first order.
        This is not yet supported
    yearfirst : bool, default False
        If True parse dates in data with the year first order.
        This is not yet supported

    Returns:
    --------
    DatetimeIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.DatetimeIndex([1, 2, 3, 4], name="a")
    DatetimeIndex(['1970-01-01 00:00:00.001000', '1970-01-01 00:00:00.002000',
                   '1970-01-01 00:00:00.003000', '1970-01-01 00:00:00.004000'],
                  dtype='datetime64[ms]', name='a')
    """

    def __new__(
        cls,
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
    ) -> "DatetimeIndex":
        # we should be more strict on what we accept here but
        # we'd have to go and figure out all the semantics around
        # pandas dtindex creation first which.  For now
        # just make sure we handle np.datetime64 arrays
        # and then just dispatch upstream
        out = Frame.__new__(cls)

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

        if copy:
            data = column.as_column(data).copy()
        kwargs = _setdefault_name(data, name=name)
        if isinstance(data, np.ndarray) and data.dtype.kind == "M":
            data = column.as_column(data)
        elif isinstance(data, pd.DatetimeIndex):
            data = column.as_column(data.values)
        elif isinstance(data, (list, tuple)):
            data = column.as_column(np.array(data, dtype="<M8[ms]"))
        out._initialize(data, **kwargs)
        return out

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
    dtype : CategoricalDtype or “category”, optional
        If CategoricalDtype, cannot be used together with categories or
        ordered.
    copy : bool, default False
        Make a copy of input.
    name : object, optional
        Name to be stored in the index.

    Return
    ------
    CategoricalIndex

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> cudf.CategoricalIndex(
    ... data=[1, 2, 3, 4], categories=[1, 2], ordered=False, name="a")
    CategoricalIndex(['1', '2', 'null', 'null'],
            categories=['1', '2', 'null'],
            ordered=False,
            dtype='category')

    >>> cudf.CategoricalIndex(
    ... data=[1, 2, 3, 4], dtype=pd.CategoricalDtype([1, 2, 3]), name="a")
    CategoricalIndex(['1', '2', '3', 'null'],
            categories=['1', '2', '3', 'null'],
            ordered=False,
            dtype='category')
    """

    def __new__(
        cls,
        data=None,
        categories=None,
        ordered=None,
        dtype=None,
        copy=False,
        name=None,
    ) -> "CategoricalIndex":
        if isinstance(dtype, (pd.CategoricalDtype, cudf.CategoricalDtype)):
            if categories is not None or ordered is not None:
                raise ValueError(
                    "Cannot specify `categories` or \
                        `ordered` together with `dtype`."
                )

        if copy:
            data = column.as_column(data, dtype=dtype).copy()
        out = Frame.__new__(cls)
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
            data.cat().set_categories(
                categories, ordered=ordered, inplace=True
            )
        elif isinstance(dtype, (pd.CategoricalDtype, cudf.CategoricalDtype)):
            data.cat().set_categories(
                dtype.categories, ordered=ordered, inplace=True
            )
        elif ordered is True and data.ordered is False:
            data.cat().as_ordered(inplace=True)
        elif ordered is False and data.ordered is True:
            data.cat().as_unordered(inplace=True)

        out._initialize(data, **kwargs)

        return out

    @property
    def codes(self):
        """
        The category codes of this categorical.
        """
        return self._values.cat().codes

    @property
    def categories(self):
        """
        The categories of this categorical.
        """
        return self._values.cat().categories


class StringIndex(GenericIndex):
    """String defined indices into another Column

    Attributes
    ----------
    _values: A StringColumn object or NDArray of strings
    name: A string
    """

    def __new__(cls, values, **kwargs):
        out = Frame.__new__(cls)
        kwargs = _setdefault_name(values, **kwargs)
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

        out._initialize(values, **kwargs)
        return out

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

    @copy_docstring(StringMethods.__init__)
    @property
    def str(self):
        return self._values.str(parent=self)

    @property
    def _constructor_expanddim(self):
        return cudf.MultiIndex


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

    kwargs = _setdefault_name(arbitrary, **kwargs)

    if isinstance(arbitrary, cudf.MultiIndex):
        return arbitrary
    elif isinstance(arbitrary, Index):
        idx = arbitrary.copy(deep=False)
        idx.rename(**kwargs, inplace=True)
        return idx
    elif isinstance(arbitrary, NumericalColumn):
        try:
            return _dtype_to_index[arbitrary.dtype.type](arbitrary, **kwargs)
        except KeyError:
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
    elif isinstance(arbitrary, cudf.DataFrame):
        return cudf.MultiIndex(source_data=arbitrary)
    elif isinstance(arbitrary, range):
        if arbitrary.step == 1:
            return RangeIndex(arbitrary.start, arbitrary.stop, **kwargs)
    return as_index(
        column.as_column(arbitrary, dtype=kwargs.get("dtype", None)), **kwargs
    )


_dtype_to_index = {
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

_index_to_dtype = {
    Int8Index: np.int8,
    Int16Index: np.int16,
    Int32Index: np.int32,
    Int64Index: np.int64,
    UInt8Index: np.uint8,
    UInt16Index: np.uint16,
    UInt32Index: np.uint32,
    UInt64Index: np.uint64,
    Float32Index: np.float32,
    Float64Index: np.float64,
}


def _setdefault_name(values, **kwargs):
    if "name" not in kwargs or kwargs["name"] is None:
        if not hasattr(values, "name"):
            kwargs.update({"name": None})
        else:
            kwargs.update({"name": values.name})
    return kwargs
