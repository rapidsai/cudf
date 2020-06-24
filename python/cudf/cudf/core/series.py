# Copyright (c) 2018, NVIDIA CORPORATION.
import pickle
import warnings
from numbers import Number

import cupy
import numpy as np
import pandas as pd
from pandas.api.types import is_dict_like

import cudf
import cudf._lib as libcudf
from cudf._lib.nvtx import annotate
from cudf._lib.transform import bools_to_mask
from cudf.core.abc import Serializable
from cudf.core.column import (
    ColumnBase,
    DatetimeColumn,
    as_column,
    column,
    column_empty_like,
)
from cudf.core.column.categorical import (
    CategoricalAccessor as CategoricalAccessor,
)
from cudf.core.column.string import StringMethods as StringMethods
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.frame import Frame
from cudf.core.groupby.groupby import SeriesGroupBy
from cudf.core.index import Index, RangeIndex, as_index
from cudf.core.indexing import _SeriesIlocIndexer, _SeriesLocIndexer
from cudf.core.window import Rolling
from cudf.utils import cudautils, ioutils, utils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    can_convert_to_column,
    is_categorical_dtype,
    is_datetime_dtype,
    is_list_like,
    is_scalar,
    is_string_dtype,
    min_scalar_type,
)


class Series(Frame, Serializable):
    @property
    def _constructor(self):
        return Series

    @property
    def _constructor_sliced(self):
        raise NotImplementedError(
            "_constructor_sliced not supported for Series!"
        )

    @property
    def _constructor_expanddim(self):
        from cudf import DataFrame

        return DataFrame

    @classmethod
    def from_categorical(cls, categorical, codes=None):
        """Creates from a pandas.Categorical

        If ``codes`` is defined, use it instead of ``categorical.codes``
        """
        from cudf.core.column.categorical import pandas_categorical_as_column

        col = pandas_categorical_as_column(categorical, codes=codes)
        return Series(data=col)

    @classmethod
    def from_masked_array(cls, data, mask, null_count=None):
        """Create a Series with null-mask.
        This is equivalent to:

            Series(data).set_mask(mask, null_count=null_count)

        Parameters
        ----------
        data : 1D array-like
            The values.  Null values must not be skipped.  They can appear
            as garbage values.
        mask : 1D array-like
            The null-mask.  Valid values are marked as ``1``; otherwise ``0``.
            The mask bit given the data index ``idx`` is computed as::

                (mask[idx // 8] >> (idx % 8)) & 1
        null_count : int, optional
            The number of null values.
            If None, it is calculated automatically.
        """
        col = column.as_column(data).set_mask(mask)
        return cls(data=col)

    def __init__(
        self, data=None, index=None, dtype=None, name=None, nan_as_null=True,
    ):
        """
        One-dimensional GPU array (including time series).

        Labels need not be unique but must be a hashable type. The object
        supports both integer- and label-based indexing and provides a
        host of methods for performing operations involving the index.
        Statistical methods from ndarray have been overridden to
        automatically exclude missing data (currently represented
        as null/NaN).

        Operations between Series (+, -, /, , *) align values based on their
        associated index values– they need not be the same length. The
        result index will be the sorted union of the two indexes.

        ``Series`` objects are used as columns of ``DataFrame``.

        Parameters
        ----------
        data : array-like, Iterable, dict, or scalar value
            Contains data stored in Series.

        index : array-like or Index (1d)
            Values must be hashable and have the same length
            as data. Non-unique index values are allowed. Will
            default to RangeIndex (0, 1, 2, …, n) if not provided.
            If both a dict and index sequence are used, the index will
            override the keys found in the dict.

        dtype : str, numpy.dtype, or ExtensionDtype, optional
            Data type for the output Series. If not specified,
            this will be inferred from data.

        name : str, optional
            The name to give to the Series.

        nan_as_null : bool, Default True
            If ``None``/``True``, converts ``np.nan`` values to
            ``null`` values.
            If ``False``, leaves ``np.nan`` values as is.
        """
        if isinstance(data, pd.Series):
            if name is None:
                name = data.name
            if isinstance(data.index, pd.MultiIndex):
                import cudf

                index = cudf.from_pandas(data.index)
            else:
                index = as_index(data.index)
        elif isinstance(data, pd.Index):
            name = data.name
            data = data.values
        elif isinstance(data, Index):
            name = data.name
            data = data._values
            if dtype is not None:
                data = data.astype(dtype)
        elif isinstance(data, ColumnAccessor):
            name, data = data.names[0], data.columns[0]

        if isinstance(data, Series):
            index = data._index if index is None else index
            if name is None:
                name = data.name
            data = data._column
            if dtype is not None:
                data = data.astype(dtype)

        if data is None:
            data = {}

        if not isinstance(data, column.ColumnBase):
            data = column.as_column(data, nan_as_null=nan_as_null, dtype=dtype)

        if index is not None and not isinstance(index, Index):
            index = as_index(index)

        assert isinstance(data, column.ColumnBase)

        super().__init__({name: data})
        self._index = RangeIndex(len(data)) if index is None else index

    @classmethod
    def _from_table(cls, table, index=None):
        name = next(iter(table._data.keys()))
        data = next(iter(table._data.values()))
        if index is None:
            if table._index is not None:
                index = Index._from_table(table._index)
        return cls(data=data, index=index, name=name)

    @property
    def _column(self):
        return self._data[self.name]

    @_column.setter
    def _column(self, value):
        self._data[self.name] = value

    def __contains__(self, item):
        return item in self._index

    @classmethod
    def from_pandas(cls, s, nan_as_null=None):
        """
        Convert from a Pandas Series.

        Parameters
        ----------
        s : Pandas Series object
            A Pandas Series object which has to be converted
            to cuDF Series.
        nan_as_null : bool, Default None
            If ``None``/``True``, converts ``np.nan`` values to
            ``null`` values.
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
        >>> pds = pd.Series(data)
        >>> cudf.Series.from_pandas(pds)
        0    10.0
        1    20.0
        2    30.0
        3    null
        dtype: float64
        >>> cudf.Series.from_pandas(pds, nan_as_null=False)
        0    10.0
        1    20.0
        2    30.0
        3     NaN
        dtype: float64
        """
        return cls(s, nan_as_null=nan_as_null)

    @property
    def values(self):
        """
        Return a CuPy representation of the Series.

        Only the values in the Series will be returned.

        Returns
        -------
        out : cupy.ndarray
            The values of the Series.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, -10, 100, 20])
        >>> ser.values
        array([  1, -10, 100,  20])
        >>> type(ser.values)
        <class 'cupy.core.core.ndarray'>
        """

        if is_categorical_dtype(self.dtype) or np.issubdtype(
            self.dtype, np.dtype("object")
        ):
            raise TypeError("Data must be numeric")

        if len(self) == 0:
            return cupy.asarray([], dtype=self.dtype)

        if self.has_nulls:
            raise ValueError("Column must have no nulls.")

        return cupy.asarray(self._column.data_array_view)

    @property
    def values_host(self):
        """
        Return a numpy representation of the Series.

        Only the values in the Series will be returned.

        Returns
        -------
        out : numpy.ndarray
            The values of the Series.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, -10, 100, 20])
        >>> ser.values_host
        array([  1, -10, 100,  20])
        >>> type(ser.values)
        <class 'numpy.ndarray'>
        """
        if self.dtype == np.dtype("object"):
            return self._column.to_array()
        elif is_categorical_dtype(self.dtype):
            return self._column.to_pandas().values
        else:
            return self._column.data_array_view.copy_to_host()

    @classmethod
    def from_arrow(cls, s):
        """Convert from a PyArrow Array.

        Parameters
        ----------
        s : PyArrow Object
            PyArrow Object which has to be converted to cudf Series.

        Raises
        ------
        TypeError for invalid input type.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import cudf
        >>> import pyarrow as pa
        >>> data = pa.array([1, 2, 3])
        >>> data
        <pyarrow.lib.Int64Array object at 0x7f67007e07c0>
        [
        1,
        2,
        3
        ]
        >>> cudf.Series.from_arrow(data)
        0    1
        1    2
        2    3
        dtype: int64
        """
        return cls(s)

    def serialize(self):
        header = {}
        frames = []
        header["index"], index_frames = self._index.serialize()
        header["name"] = pickle.dumps(self.name)
        frames.extend(index_frames)
        header["index_frame_count"] = len(index_frames)
        header["column"], column_frames = self._column.serialize()
        header["type-serialized"] = pickle.dumps(type(self))
        frames.extend(column_frames)
        header["column_frame_count"] = len(column_frames)

        return header, frames

    @property
    def shape(self):
        """Returns a tuple representing the dimensionality of the Series.
        """
        return (len(self),)

    @property
    def dt(self):
        """
        Accessor object for datetimelike properties of the Series values.

        Examples
        --------
        >>> s.dt.hour
        >>> s.dt.second
        >>> s.dt.day

        Returns
        -------
            A Series indexed like the original Series.

        Raises
        ------
            TypeError if the Series does not contain datetimelike values.
        """
        if isinstance(self._column, DatetimeColumn):
            return DatetimeProperties(self)
        else:
            raise AttributeError(
                "Can only use .dt accessor with datetimelike values"
            )

    @property
    def ndim(self):
        """Dimension of the data. Series ndim is always 1.
        """
        return 1

    @property
    def name(self):
        """Returns name of the Series.
        """
        return self._data.names[0]

    @name.setter
    def name(self, value):
        col = self._data.pop(self.name)
        self._data[value] = col

    @classmethod
    def deserialize(cls, header, frames):

        index_nframes = header["index_frame_count"]
        idx_typ = pickle.loads(header["index"]["type-serialized"])
        index = idx_typ.deserialize(header["index"], frames[:index_nframes])
        name = pickle.loads(header["name"])

        frames = frames[index_nframes:]

        column_nframes = header["column_frame_count"]
        col_typ = pickle.loads(header["column"]["type-serialized"])
        column = col_typ.deserialize(header["column"], frames[:column_nframes])

        return Series(column, index=index, name=name)

    def _copy_construct_defaults(self):
        return dict(data=self._column, index=self._index, name=self.name)

    def _copy_construct(self, **kwargs):
        """Shallow copy this object by replacing certain ctor args.
        """
        params = self._copy_construct_defaults()
        cls = type(self)
        params.update(kwargs)
        return cls(**params)

    def copy(self, deep=True):
        """
        Make a copy of this object's indices and data.

        When ``deep=True`` (default), a new object will be created with a
        copy of the calling object's data and indices. Modifications to
        the data or indices of the copy will not be reflected in the
        original object (see notes below).
        When ``deep=False``, a new object will be created without copying
        the calling object's data or index (only references to the data
        and index are copied). Any changes to the data of the original
        will be reflected in the shallow copy (and vice versa).

        Parameters
        ----------
        deep : bool, default True
            Make a deep copy, including a copy of the data and the indices.
            With ``deep=False`` neither the indices nor the data are copied.

        Returns
        -------
        copy : Series or DataFrame
            Object type matches caller.


        Examples
        --------
        >>> s = cudf.Series([1, 2], index=["a", "b"])
        >>> s
        a    1
        b    2
        dtype: int64
        >>> s_copy = s.copy()
        >>> s_copy
        a    1
        b    2
        dtype: int64

        **Shallow copy versus default (deep) copy:**

        >>> s = cudf.Series([1, 2], index=["a", "b"])
        >>> deep = s.copy()
        >>> shallow = s.copy(deep=False)

        Shallow copy shares data and index with original.

        >>> s is shallow
        False
        >>> s._column is shallow._column and s.index is shallow.index
        True

        Deep copy has own copy of data and index.

        >>> s is deep
        False
        >>> s.values is deep.values or s.index is deep.index
        False

        Updates to the data shared by shallow copy and original is reflected
        in both; deep copy remains unchanged.

        >>> s['a'] = 3
        >>> shallow['b'] = 4
        >>> s
        a    3
        b    4
        dtype: int64
        >>> shallow
        a    3
        b    4
        dtype: int64
        >>> deep
        a    1
        b    2
        dtype: int64
        """
        result = self._copy_construct()
        if deep:
            result._column = self._column.copy(deep)
            result.index = self.index.copy(deep)
        return result

    def __copy__(self, deep=True):
        return self.copy(deep)

    def __deepcopy__(self):
        return self.copy()

    def append(self, other, ignore_index=False):
        """Append values from another ``Series`` or array-like object.
        If ``ignore_index=True``, the index is reset.

        Parameters
        ----------
        other : ``Series`` or array-like object
        ignore_index : boolean, default False. If true, the index is reset.

        Returns
        -------
        A new Series equivalent to self concatenated with other
        """
        this = self
        other = Series(other)

        from cudf.core.column import numerical
        from cudf.utils.dtypes import numeric_normalize_types

        if isinstance(this._column, numerical.NumericalColumn):
            if self.dtype != other.dtype:
                this, other = numeric_normalize_types(this, other)

        if ignore_index:
            index = None
        else:
            index = True

        return Series._concat([this, other], index=index)

    def reindex(self, index=None, copy=True):
        """Return a Series that conforms to a new index

        Parameters
        ----------
        index : Index, Series-convertible, default None
        copy : boolean, default True

        Returns
        -------
        A new Series that conforms to the supplied index
        """
        name = self.name or 0
        idx = self._index if index is None else index
        return self.to_frame(name).reindex(idx, copy=copy)[name]

    def reset_index(self, drop=False, inplace=False):
        """ Reset index to RangeIndex """
        if not drop:
            if inplace is True:
                raise TypeError(
                    "Cannot reset_index inplace on a Series "
                    "to create a DataFrame"
                )
            return self.to_frame().reset_index(drop=drop)
        else:
            if inplace is True:
                self._index = RangeIndex(len(self))
            else:
                return self._copy_construct(index=RangeIndex(len(self)))

    def set_index(self, index):
        """Returns a new Series with a different index.

        Parameters
        ----------
        index : Index, Series-convertible
            the new index or values for the new index
        """
        index = index if isinstance(index, Index) else as_index(index)
        return self._copy_construct(index=index)

    def as_index(self):
        """Returns a new Series with a RangeIndex.

        Examples
        ----------
        >>> s = cudf.Series([1,2,3], index=['a','b','c'])
        >>> s
        a    1
        b    2
        c    3
        dtype: int64
        >>> s.as_index()
        0    1
        1    2
        2    3
        dtype: int64
        """
        return self.set_index(RangeIndex(len(self)))

    def to_frame(self, name=None):
        """Convert Series into a DataFrame

        Parameters
        ----------
        name : str, default None
            Name to be used for the column

        Returns
        -------
        DataFrame
            cudf DataFrame
        """

        from cudf import DataFrame

        if name is not None:
            col = name
        elif self.name is None:
            col = 0
        else:
            col = self.name

        return DataFrame({col: self._column}, index=self.index)

    def set_mask(self, mask, null_count=None):
        """Create new Series by setting a mask array.

        This will override the existing mask.  The returned Series will
        reference the same data buffer as this Series.

        Parameters
        ----------
        mask : 1D array-like
            The null-mask.  Valid values are marked as ``1``; otherwise ``0``.
            The mask bit given the data index ``idx`` is computed as::

                (mask[idx // 8] >> (idx % 8)) & 1
        null_count : int, optional
            The number of null values.
            If None, it is calculated automatically.
        """
        col = self._column.set_mask(mask)
        return self._copy_construct(data=col)

    def __sizeof__(self):
        return self._column.__sizeof__() + self._index.__sizeof__()

    def memory_usage(self, index=True, deep=False):
        """
        Return the memory usage of the Series.

        The memory usage can optionally include the contribution of
        the index and of elements of `object` dtype.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the Series index.
        deep : bool, default False
            If True, introspect the data deeply by interrogating
            `object` dtypes for system-level memory consumption, and include
            it in the returned value.

        Returns
        -------
        int
            Bytes of memory consumed.

        See Also
        --------
        cudf.DataFrame.memory_usage : Bytes consumed by a DataFrame.

        Examples
        --------
        >>> s = cudf.Series(range(3), index=['a','b','c'])
        >>> s.memory_usage()
        48

        Not including the index gives the size of the rest of the data, which
        is necessarily smaller:

        >>> s.memory_usage(index=False)
        24
        """
        n = self._column._memory_usage(deep=deep)
        if index:
            n += self._index.memory_usage(deep=deep)
        return n

    def __len__(self):
        """Returns the size of the ``Series`` including null values.
        """
        return len(self._column)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        import cudf

        if method == "__call__" and hasattr(cudf, ufunc.__name__):
            func = getattr(cudf, ufunc.__name__)
            return func(*inputs)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):

        cudf_series_module = Series
        for submodule in func.__module__.split(".")[1:]:
            # point cudf to the correct submodule
            if hasattr(cudf_series_module, submodule):
                cudf_series_module = getattr(cudf_series_module, submodule)
            else:
                return NotImplemented

        fname = func.__name__

        handled_types = [cudf_series_module]
        for t in types:
            if t not in handled_types:
                return NotImplemented

        if hasattr(cudf_series_module, fname):
            cudf_func = getattr(cudf_series_module, fname)
            # Handle case if cudf_func is same as numpy function
            if cudf_func is func:
                return NotImplemented
            else:
                return cudf_func(*args, **kwargs)

        else:
            return NotImplemented

    @property
    def empty(self):
        """
        Indicator whether Series is empty.

        True if Series is entirely empty (no items).

        Returns
        -------
        out : bool
            If Series is empty, return True, if not return False.
        """
        return not len(self)

    def __getitem__(self, arg):
        if isinstance(arg, slice):
            return self.iloc[arg]
        else:
            return self.loc[arg]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self.iloc[key] = value
        else:
            self.loc[key] = value

    def take(self, indices, keep_index=True):
        """Return Series by taking values from the corresponding *indices*.
        """
        if keep_index is True or is_scalar(indices):
            return self.iloc[indices]
        else:
            col_inds = as_column(indices)
            data = self._column.take(col_inds, keep_index=False)
            return self._copy_construct(data=data)

    def __bool__(self):
        """Always raise TypeError when converting a Series
        into a boolean.
        """
        raise TypeError("can't compute boolean for {!r}".format(type(self)))

    def values_to_string(self, nrows=None):
        """Returns a list of string for each element.
        """
        values = self[:nrows]
        if self.dtype == np.dtype("object"):
            out = [str(v) for v in values]
        else:
            out = ["" if v is None else str(v) for v in values]
        return out

    def tolist(self):
        """
        Return a list type from series data.

        Returns
        -------
        list
        """
        return self.to_arrow().to_pylist()

    def head(self, n=5):
        """
        Return the first `n` rows.
        This function returns the first `n` rows for the object based
        on position. It is useful for quickly testing if your object
        has the right type of data in it.
        For negative values of `n`, this function returns all rows except
        the last `n` rows, equivalent to ``df[:-n]``.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        same type as caller
            The first `n` rows of the caller object.

        See Also
        --------
        Series.tail: Returns the last `n` rows.

        Examples
        --------
        >>> ser = cudf.Series(['alligator', 'bee', 'falcon', 'lion', 'monkey', 'parrot', 'shark', 'whale', 'zebra'])        # noqa E501
        >>> ser
        0    alligator
        1          bee
        2       falcon
        3         lion
        4       monkey
        5       parrot
        6        shark
        7        whale
        8        zebra
        dtype: object

        Viewing the first 5 lines

        >>> ser.head()
        0    alligator
        1          bee
        2       falcon
        3         lion
        4       monkey
        dtype: object

        Viewing the first `n` lines (three in this case)

        >>> ser.head(3)
        0    alligator
        1          bee
        2       falcon
        dtype: object

        For negative values of `n`

        >>> ser.head(-3)
        0    alligator
        1          bee
        2       falcon
        3         lion
        4       monkey
        5       parrot
        dtype: object
        """
        return self.iloc[:n]

    def tail(self, n=5):
        """
        Returns the last n rows as a new Series

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> print(ser.tail(2))
        3    1
        4    0
        """
        if n == 0:
            return self.iloc[0:0]

        return self.iloc[-n:]

    def to_string(self):
        """Convert to string

        Uses Pandas formatting internals to produce output identical to Pandas.
        Use the Pandas formatting settings directly in Pandas to control cuDF
        output.
        """
        return self.__repr__()

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        mr = pd.options.display.max_rows
        if len(self) > mr and mr != 0:
            top = self.head(int(mr / 2 + 1))
            bottom = self.tail(int(mr / 2 + 1))
            from cudf import concat

            preprocess = concat([top, bottom])
        else:
            preprocess = self
        if (
            preprocess.nullable
            and not preprocess.dtype == "O"
            and not is_categorical_dtype(preprocess.dtype)
            and not is_datetime_dtype(preprocess.dtype)
        ):
            output = (
                preprocess.astype("O").fillna("null").to_pandas().__repr__()
            )
        else:
            output = preprocess.to_pandas().__repr__()
        lines = output.split("\n")
        if is_categorical_dtype(preprocess.dtype):
            for idx, value in enumerate(preprocess):
                if value is None:
                    lines[idx] = lines[idx].replace(" NaN", "null")
        if is_datetime_dtype(preprocess.dtype):
            for idx, value in enumerate(preprocess):
                if value is None:
                    lines[idx] = lines[idx].replace(" NaT", "null")
        if is_categorical_dtype(preprocess.dtype):
            category_memory = lines[-1]
            lines = lines[:-1]
        if len(lines) > 1:
            if lines[-1].startswith("Name: "):
                lines = lines[:-1]
                lines.append("Name: %s" % str(self.name))
                if len(self) > len(preprocess):
                    lines[-1] = lines[-1] + ", Length: %d" % len(self)
                lines[-1] = lines[-1] + ", "
            elif lines[-1].startswith("Length: "):
                lines = lines[:-1]
                lines.append("Length: %d" % len(self))
                lines[-1] = lines[-1] + ", "
            else:
                lines = lines[:-1]
                lines[-1] = lines[-1] + "\n"
            lines[-1] = lines[-1] + "dtype: %s" % self.dtype
        else:
            lines = output.split(",")
            return lines[0] + ", dtype: %s)" % self.dtype
        if is_categorical_dtype(preprocess.dtype):
            lines.append(category_memory)
        return "\n".join(lines)

    @annotate("BINARY_OP", color="orange", domain="cudf_python")
    def _binaryop(self, other, fn, fill_value=None, reflect=False):
        """
        Internal util to call a binary operator *fn* on operands *self*
        and *other*.  Return the output Series.  The output dtype is
        determined by the input operands.

        If ``reflect`` is ``True``, swap the order of the operands.
        """
        from cudf import DataFrame

        if isinstance(other, DataFrame):
            # TODO: fn is not the same as arg expected by _apply_op
            # e.g. for fn = 'and', _apply_op equivalent is '__and__'
            return other._apply_op(self, fn)

        result_name = utils.get_result_name(self, other)
        if isinstance(other, Series):
            lhs, rhs = _align_indices([self, other], allow_non_unique=True)
        else:
            lhs, rhs = self, other
        rhs = self._normalize_binop_value(rhs)

        if fn == "truediv":
            if str(lhs.dtype) in truediv_int_dtype_corrections:
                truediv_type = truediv_int_dtype_corrections[str(lhs.dtype)]
                lhs = lhs.astype(truediv_type)

        if fill_value is not None:
            if is_scalar(rhs):
                lhs = lhs.fillna(fill_value)
            else:
                if lhs.nullable and rhs.nullable:
                    lmask = Series(data=lhs.nullmask)
                    rmask = Series(data=rhs.nullmask)
                    mask = (lmask | rmask).data
                    lhs = lhs.fillna(fill_value)
                    rhs = rhs.fillna(fill_value)
                    result = lhs._binaryop(rhs, fn=fn, reflect=reflect)
                    data = column.build_column(
                        data=result.data, dtype=result.dtype, mask=mask
                    )
                    return lhs._copy_construct(data=data)
                elif lhs.nullable:
                    lhs = lhs.fillna(fill_value)
                elif rhs.nullable:
                    rhs = rhs.fillna(fill_value)

        outcol = lhs._column.binary_operator(fn, rhs, reflect=reflect)
        result = lhs._copy_construct(data=outcol, name=result_name)
        return result

    def add(self, other, fill_value=None, axis=0):
        """Addition of series and other, element-wise
        (binary operator add).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "add", fill_value)

    def __add__(self, other):
        return self._binaryop(other, "add")

    def radd(self, other, fill_value=None, axis=0):
        """Addition of series and other, element-wise
        (binary operator radd).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(
            other, "add", fill_value=fill_value, reflect=True
        )

    def __radd__(self, other):
        return self._binaryop(other, "add", reflect=True)

    def sub(self, other, fill_value=None, axis=0):
        """Subtraction of series and other, element-wise
        (binary operator sub).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "sub", fill_value)

    def __sub__(self, other):
        return self._binaryop(other, "sub")

    def rsub(self, other, fill_value=None, axis=0):
        """Subtraction of series and other, element-wise
        (binary operator rsub).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "sub", fill_value, reflect=True)

    def __rsub__(self, other):
        return self._binaryop(other, "sub", reflect=True)

    def mul(self, other, fill_value=None, axis=0):
        """Multiplication of series and other, element-wise
        (binary operator mul).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "mul", fill_value=fill_value)

    def __mul__(self, other):
        return self._binaryop(other, "mul")

    def rmul(self, other, fill_value=None, axis=0):
        """Multiplication of series and other, element-wise
        (binary operator rmul).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "mul", fill_value, True)

    def __rmul__(self, other):
        return self._binaryop(other, "mul", reflect=True)

    def mod(self, other, fill_value=None, axis=0):
        """Modulo of series and other, element-wise
        (binary operator mod).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "mod", fill_value)

    def __mod__(self, other):
        return self._binaryop(other, "mod")

    def rmod(self, other, fill_value=None, axis=0):
        """Modulo of series and other, element-wise
        (binary operator rmod).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "mod", fill_value, True)

    def __rmod__(self, other):
        return self._binaryop(other, "mod", reflect=True)

    def pow(self, other, fill_value=None, axis=0):
        """Exponential power of series and other, element-wise
        (binary operator pow).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "pow", fill_value)

    def __pow__(self, other):
        return self._binaryop(other, "pow")

    def rpow(self, other, fill_value=None, axis=0):
        """Exponential power of series and other, element-wise
        (binary operator rpow).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "pow", fill_value, True)

    def __rpow__(self, other):
        return self._binaryop(other, "pow", reflect=True)

    def floordiv(self, other, fill_value=None, axis=0):
        """Integer division of series and other, element-wise
        (binary operator floordiv).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "floordiv", fill_value)

    def __floordiv__(self, other):
        return self._binaryop(other, "floordiv")

    def rfloordiv(self, other, fill_value=None, axis=0):
        """Integer division of series and other, element-wise
        (binary operator rfloordiv).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "floordiv", fill_value, True)

    def __rfloordiv__(self, other):
        return self._binaryop(other, "floordiv", reflect=True)

    def truediv(self, other, fill_value=None, axis=0):
        """Floating division of series and other, element-wise
        (binary operator truediv).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "truediv", fill_value)

    def __truediv__(self, other):
        return self._binaryop(other, "truediv")

    def rtruediv(self, other, fill_value=None, axis=0):
        """Floating division of series and other, element-wise
        (binary operator rtruediv).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "truediv", fill_value, True)

    def __rtruediv__(self, other):
        return self._binaryop(other, "truediv", reflect=True)

    __div__ = __truediv__

    def _bitwise_binop(self, other, op):
        if (
            np.issubdtype(self.dtype, np.bool_)
            or np.issubdtype(self.dtype, np.integer)
        ) and (
            np.issubdtype(other.dtype, np.bool_)
            or np.issubdtype(other.dtype, np.integer)
        ):
            # TODO: This doesn't work on Series (op) DataFrame
            # because dataframe doesn't have dtype
            ser = self._binaryop(other, op)
            if np.issubdtype(self.dtype, np.bool_) or np.issubdtype(
                other.dtype, np.bool_
            ):
                ser = ser.astype(np.bool_)
            return ser
        else:
            raise TypeError(
                f"Operation 'bitwise {op}' not supported between "
                f"{self.dtype.type.__name__} and {other.dtype.type.__name__}"
            )

    def __and__(self, other):
        """Performs vectorized bitwise and (&) on corresponding elements of two
        series.
        """
        return self._bitwise_binop(other, "and")

    def __or__(self, other):
        """Performs vectorized bitwise or (|) on corresponding elements of two
        series.
        """
        return self._bitwise_binop(other, "or")

    def __xor__(self, other):
        """Performs vectorized bitwise xor (^) on corresponding elements of two
        series.
        """
        return self._bitwise_binop(other, "xor")

    def logical_and(self, other):
        ser = self._binaryop(other, "l_and")
        return ser.astype(np.bool_)

    def remainder(self, other):
        ser = self._binaryop(other, "mod")
        return ser

    def logical_or(self, other):
        ser = self._binaryop(other, "l_or")
        return ser.astype(np.bool_)

    def logical_not(self):
        return self._unaryop("not")

    def _normalize_binop_value(self, other):
        """Returns a *column* (not a Series) or scalar for performing
        binary operations with self._column.
        """
        if isinstance(other, ColumnBase):
            return other
        if isinstance(other, Series):
            return other._column
        elif isinstance(other, Index):
            return Series(other)._column
        else:
            return self._column.normalize_binop_value(other)

    def eq(self, other, fill_value=None, axis=0):
        """Equal to of series and other, element-wise
        (binary operator eq).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "eq", fill_value)

    def __eq__(self, other):
        return self._binaryop(other, "eq")

    def equals(self, other):
        """
        Test whether two objects contain the same elements.
        This function allows two Series or DataFrames to be compared against
        each other to see if they have the same shape and elements. NaNs in
        the same location are considered equal. The column headers do not
        need to have the same type.

        Parameters
        ----------
        other : Series or DataFrame
            The other Series or DataFrame to be compared with the first.

        Returns
        -------
        bool
            True if all elements are the same in both objects, False
            otherwise.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([1, 2, 3])
        >>> other = cudf.Series([1, 2, 3])
        >>> s.equals(other)
        True
        >>> different = cudf.Series([1.5, 2, 3])
        >>> s.equals(different)
        False
        """
        if self is other:
            return True
        if other is None or len(self) != len(other):
            return False
        return self._binaryop(other, "eq").min()

    def ne(self, other, fill_value=None, axis=0):
        """Not equal to of series and other, element-wise
        (binary operator ne).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "ne", fill_value)

    def __ne__(self, other):
        return self._binaryop(other, "ne")

    def lt(self, other, fill_value=None, axis=0):
        """Less than of series and other, element-wise
        (binary operator lt).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "lt", fill_value)

    def __lt__(self, other):
        return self._binaryop(other, "lt")

    def le(self, other, fill_value=None, axis=0):
        """Less than or equal to of series and other, element-wise
        (binary operator le).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "le", fill_value)

    def __le__(self, other):
        return self._binaryop(other, "le")

    def gt(self, other, fill_value=None, axis=0):
        """Greater than of series and other, element-wise
        (binary operator gt).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "gt", fill_value)

    def __gt__(self, other):
        return self._binaryop(other, "gt")

    def ge(self, other, fill_value=None, axis=0):
        """Greater than or equal to of series and other, element-wise
        (binary operator ge).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return self._binaryop(other, "ge", fill_value)

    def __ge__(self, other):
        return self._binaryop(other, "ge")

    def __invert__(self):
        """Bitwise invert (~) for each element.
        Logical NOT if dtype is bool

        Returns a new Series.
        """
        if np.issubdtype(self.dtype, np.integer):
            return self._unaryop("invert")
        elif np.issubdtype(self.dtype, np.bool_):
            return self._unaryop("not")
        else:
            raise TypeError(
                f"Operation `~` not supported on {self.dtype.type.__name__}"
            )

    def __neg__(self):
        """Negated value (-) for each element

        Returns a new Series.
        """
        return self.__mul__(-1)

    @copy_docstring(CategoricalAccessor.__init__)
    @property
    def cat(self):
        return self._column.cat(parent=self)

    @copy_docstring(StringMethods.__init__)
    @property
    def str(self):
        return self._column.str(parent=self)

    @property
    def dtype(self):
        """dtype of the Series"""
        return self._column.dtype

    @classmethod
    def _concat(cls, objs, axis=0, index=True):
        # Concatenate index if not provided
        if index is True:
            from cudf.core.multiindex import MultiIndex

            if isinstance(objs[0].index, MultiIndex):
                index = MultiIndex._concat([o.index for o in objs])
            else:
                index = Index._concat([o.index for o in objs])

        names = {obj.name for obj in objs}
        if len(names) == 1:
            [name] = names
        else:
            name = None
        col = ColumnBase._concat([o._column for o in objs])
        return cls(data=col, index=index, name=name)

    @property
    def valid_count(self):
        """Number of non-null values"""
        return self._column.valid_count

    @property
    def null_count(self):
        """Number of null values"""
        return self._column.null_count

    @property
    def nullable(self):
        """A boolean indicating whether a null-mask is needed"""
        return self._column.nullable

    @property
    def has_nulls(self):
        """
        Indicator whether Series contains null values.

        Returns
        -------
        out : bool
            If Series has atleast one null value, return True, if not
            return False.
        """
        return self._column.has_nulls

    def dropna(self):
        """
        Return a Series with null values removed.
        """
        return super().dropna(subset=[self.name])

    def drop_duplicates(self, keep="first", inplace=False, ignore_index=False):
        """
        Return Series with duplicate values removed
        """
        result = super().drop_duplicates(
            subset=[self.name], keep=keep, ignore_index=ignore_index
        )

        return self._mimic_inplace(result, inplace=inplace)

    def _mimic_inplace(self, result, inplace=False):
        if inplace:
            self._column._mimic_inplace(result._column, inplace=True)
            self.index._mimic_inplace(result.index, inplace=True)
            self._size = len(self._index)
            self.name = result.name
        else:
            return result

    def fill(self, fill_value, begin=0, end=-1, inplace=False):
        return self._fill([fill_value], begin, end, inplace)

    def fillna(self, value, method=None, axis=None, inplace=False, limit=None):
        """Fill null values with ``value`` without changing the series' type.

        Parameters
        ----------
        value : scalar or Series-like
            Value to use to fill nulls. If `value`'s dtype differs from the
            series, the fill value will be cast to the column's dtype before
            applying the fill. If Series-like, null values are filled with the
            values in corresponding indices of the given Series.

        Returns
        -------
        result : Series
            Copy with nulls filled.
        """
        if method is not None:
            raise NotImplementedError("The method keyword is not supported")
        if limit is not None:
            raise NotImplementedError("The limit keyword is not supported")
        if axis:
            raise NotImplementedError("The axis keyword is not supported")

        data = self._column.fillna(value)

        if inplace:
            self._column._mimic_inplace(data, inplace=True)
        else:
            return self._copy_construct(data=data)

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

        If ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        return self._column.to_array(fillna=fillna)

    def nans_to_nulls(self):
        """
        Convert nans (if any) to nulls
        """
        if self.dtype.kind == "f":
            sr = self.fillna(np.nan)
            newmask = libcudf.transform.nans_to_nulls(sr._column)
            return self.set_mask(newmask)
        else:
            return self

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """
        Return whether all elements are True in Series.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values. If the entire row/column is NA and
            skipna is True, then the result will be True, as for an
            empty row/column.
            If skipna is False, then NA are treated as True, because
            these are not equal to zero.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `bool_only`, `level`.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.all()
        True
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if bool_only not in (None, True):
            raise NotImplementedError(
                "bool_only parameter is not implemented yet"
            )

        if skipna:
            result_series = self.nans_to_nulls()
            if len(result_series) == result_series.null_count:
                return True
        else:
            result_series = self
        return result_series._column.all()

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """
        Return whether any elements is True in Series.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values. If the entire row/column is NA and
            skipna is True, then the result will be False, as for an
            empty row/column.
            If skipna is False, then NA are treated as True, because
            these are not equal to zero.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `bool_only`, `level`.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.any()
        True
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if bool_only not in (None, True):
            raise NotImplementedError(
                "bool_only parameter is not implemented yet"
            )

        if self.empty:
            return False

        if skipna:
            result_series = self.nans_to_nulls()
            if len(result_series) == result_series.null_count:
                return False

        else:
            result_series = self

        return result_series._column.any()

    def to_gpu_array(self, fillna=None):
        """Get a dense numba device array for the data.

        Parameters
        ----------
        fillna : str or None
            See *fillna* in ``.to_array``.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        return self._column.to_gpu_array(fillna=fillna)

    def to_pandas(self, index=True):
        """
        Convert to a Pandas Series.

        Parameters
        ----------
        index : Boolean, Default True
            If ``index`` is ``True``, converts the index of cudf.Series
            and sets it to the pandas.Series. If ``index`` is ``False``,
            no index conversion is performed and pandas.Series will assign
            a default index.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-3, 2, 0])
        >>> pds = ser.to_pandas()
        >>> pds
        0   -3
        1    2
        2    0
        dtype: int64
        >>> type(pds)
        <class 'pandas.core.series.Series'>
        """
        if index is True:
            index = self.index.to_pandas()
        s = self._column.to_pandas(index=index)
        s.name = self.name
        return s

    def to_arrow(self):
        """
        Convert Series to a PyArrow Array.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-3, 10, 15, 20])
        >>> ser.to_arrow()
        <pyarrow.lib.Int64Array object at 0x7f5e769499f0>
        [
        -3,
        10,
        15,
        20
        ]
        """
        return self._column.to_arrow()

    @property
    def data(self):
        """The gpu buffer for the data
        """
        return self._column.data

    @property
    def index(self):
        """The index object
        """
        return self._index

    @index.setter
    def index(self, _index):
        self._index = as_index(_index)

    @property
    def loc(self):
        """
        Select values by label.

        See also
        --------
        cudf.core.dataframe.Dataframe.loc
        """
        return _SeriesLocIndexer(self)

    @property
    def iloc(self):
        """
        Select values by position.

        See also
        --------
        cudf.core.dataframe.Dataframe.iloc
        """
        return _SeriesIlocIndexer(self)

    @property
    def nullmask(self):
        """The gpu buffer for the null-mask
        """
        return cudf.Series(self._column.nullmask)

    def as_mask(self):
        """Convert booleans to bitmask

        Returns
        -------
        device array
        """
        return self._column.as_mask()

    def astype(self, dtype, copy=False, errors="raise", **kwargs):
        """
        Cast the Series to the given dtype

        Parameters
        ----------

        dtype : data type, or dict of column name -> data type
            Use a numpy.dtype or Python type to cast Series object to
            the same type. Alternatively, use {col: dtype, ...}, where col is a
            series name and dtype is a numpy.dtype or Python type to cast to.
        copy : bool, default False
            Return a deep-copy when ``copy=True``. Note by default
            ``copy=False`` setting is used and hence changes to
            values then may propagate to other cudf objects.
        errors : {'raise', 'ignore', 'warn'}, default 'raise'
            Control raising of exceptions on invalid data for provided dtype.

            -   ``raise`` : allow exceptions to be raised
            -   ``ignore`` : suppress exceptions. On error return original
                object.
            -   ``warn`` : prints last exceptions as warnings and
                return original object.
        **kwargs : extra arguments to pass on to the constructor

        Returns
        -------
        out : Series
            Returns ``self.copy(deep=copy)`` if ``dtype`` is the same
            as ``self.dtype``.
        """
        if errors not in ("ignore", "raise", "warn"):
            raise ValueError("invalid error value specified")

        if is_dict_like(dtype):
            if len(dtype) > 1 or self.name not in dtype:
                raise KeyError(
                    "Only the Series name can be used for "
                    "the key in Series dtype mappings."
                )
            dtype = dtype[self.name]

        if pd.api.types.is_dtype_equal(dtype, self.dtype):
            return self.copy(deep=copy)
        try:
            data = self._column.astype(dtype, **kwargs)

            return self._copy_construct(
                data=data.copy(deep=True) if copy else data, index=self.index
            )

        except Exception as e:
            if errors == "raise":
                raise e
            elif errors == "warn":
                import traceback

                tb = traceback.format_exc()
                warnings.warn(tb)
            elif errors == "ignore":
                pass
            return self

    def argsort(self, ascending=True, na_position="last"):
        """Returns a Series of int64 index that will sort the series.

        Uses Thrust sort.

        Returns
        -------
        result: Series
        """
        return self._sort(ascending=ascending, na_position=na_position)[1]

    def sort_index(self, ascending=True):
        """Sort by the index.
        """
        inds = self.index.argsort(ascending=ascending)
        return self.take(inds)

    def sort_values(self, ascending=True, na_position="last"):
        """
        Sort by the values.

        Sort a Series in ascending or descending order by some criterion.

        Parameters
        ----------
        ascending : bool, default True
            If True, sort values in ascending order, otherwise descending.
        na_position : {‘first’, ‘last’}, default ‘last’
            'first' puts nulls at the beginning, 'last' puts nulls at the end.

        Returns
        -------
        sorted_obj : cuDF Series

        Notes
        -----
        Difference from pandas:
          * Not supporting: `inplace`, `kind`

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([1, 5, 2, 4, 3])
        >>> s.sort_values()
        0    1
        2    2
        4    3
        3    4
        1    5
        """
        if len(self) == 0:
            return self
        vals, inds = self._sort(ascending=ascending, na_position=na_position)
        index = self.index.take(inds)
        return vals.set_index(index)

    def _n_largest_or_smallest(self, largest, n, keep):
        direction = largest
        if keep == "first":
            if n < 0:
                n = 0
            return self.sort_values(ascending=not direction).head(n)
        elif keep == "last":
            data = self.sort_values(ascending=direction)
            if n <= 0:
                data = data[-n:-n]
            else:
                data = data.tail(n)
            return data.reverse()
        else:
            raise ValueError('keep must be either "first", "last"')

    def nlargest(self, n=5, keep="first"):
        """Returns a new Series of the *n* largest element.
        """
        return self._n_largest_or_smallest(n=n, keep=keep, largest=True)

    def nsmallest(self, n=5, keep="first"):
        """Returns a new Series of the *n* smallest element.
        """
        return self._n_largest_or_smallest(n=n, keep=keep, largest=False)

    def _sort(self, ascending=True, na_position="last"):
        """
        Sort by values

        Returns
        -------
        2-tuple of key and index
        """
        col_keys, col_inds = self._column.sort_by_values(
            ascending=ascending, na_position=na_position
        )
        sr_keys = self._copy_construct(data=col_keys)
        sr_inds = self._copy_construct(data=col_inds)
        return sr_keys, sr_inds

    def replace(
        self,
        to_replace=None,
        value=None,
        inplace=False,
        limit=None,
        regex=False,
        method=None,
    ):
        """
        Replace values given in ``to_replace`` with ``value``.

        Parameters
        ----------
        to_replace : numeric, str or list-like
            Value(s) to replace.

            * numeric or str:
                - values equal to ``to_replace`` will be replaced
                  with ``value``
            * list of numeric or str:
                - If ``value`` is also list-like, ``to_replace`` and
                  ``value`` must be of same length.
        value : numeric, str, list-like, or dict
            Value(s) to replace ``to_replace`` with.
        inplace : bool, default False
            If True, in place.

        See also
        --------
        Series.fillna

        Returns
        -------
        result : Series
            Series after replacement. The mask and index are preserved.

        Notes
        -----
        Parameters that are currently not supported are: `limit`, `regex`,
        `method`
        """
        if limit is not None:
            raise NotImplementedError("limit parameter is not implemented yet")

        if regex:
            raise NotImplementedError("regex parameter is not implemented yet")

        if method not in ("pad", None):
            raise NotImplementedError(
                "method parameter is not implemented yet"
            )

        result = super().replace(to_replace=to_replace, replacement=value)

        return self._mimic_inplace(result, inplace=inplace)

    def reverse(self):
        """Reverse the Series
        """
        rinds = cupy.arange((self._column.size - 1), -1, -1, dtype=np.int32)
        col = self._column[rinds]
        index = self.index._values[rinds]
        return self._copy_construct(data=col, index=index)

    def one_hot_encoding(self, cats, dtype="float64"):
        """Perform one-hot-encoding

        Parameters
        ----------
        cats : sequence of values
                values representing each category.
        dtype : numpy.dtype
                specifies the output dtype.

        Returns
        -------
        Sequence
            A sequence of new series for each category. Its length is
            determined by the length of ``cats``.
        """
        if hasattr(cats, "to_pandas"):
            cats = cats.to_pandas()
        else:
            cats = pd.Series(cats)
        dtype = np.dtype(dtype)
        return ((self == cat).fillna(False).astype(dtype) for cat in cats)

    def label_encoding(self, cats, dtype=None, na_sentinel=-1):
        """Perform label encoding

        Parameters
        ----------
        values : sequence of input values
        dtype: numpy.dtype; optional
               Specifies the output dtype.  If `None` is given, the
               smallest possible integer dtype (starting with np.int8)
               is used.
        na_sentinel : number
            Value to indicate missing category.

        Returns
        -------
        A sequence of encoded labels with value between 0 and n-1 classes(cats)
        """
        from cudf import DataFrame

        if dtype is None:
            dtype = min_scalar_type(len(cats), 8)

        cats = column.as_column(cats)
        try:
            # Where there is a type-cast from string to numeric types,
            # there is a possibility for ValueError when strings
            # are having non-numeric values, in such cases we have
            # to catch the exception and return encoded labels
            # with na_sentinel values as there would be no corresponding
            # encoded values of cats in self.
            cats = cats.astype(self.dtype)
        except ValueError:
            return Series(
                utils.scalar_broadcast_to(
                    na_sentinel, size=len(self), dtype=dtype
                ),
                index=self.index,
                name=None,
            )

        order = column.as_column(cupy.arange(len(self)))
        codes = column.as_column(cupy.arange(len(cats), dtype=dtype))

        value = DataFrame({"value": cats, "code": codes})
        codes = DataFrame(
            {"value": self._data.columns[0].copy(deep=False), "order": order}
        )

        codes = codes.merge(value, on="value", how="left")
        codes = codes.sort_values("order")["code"].fillna(na_sentinel)

        return codes._copy_construct(name=None, index=self.index)

    def factorize(self, na_sentinel=-1):
        """Encode the input values as integer labels

        Parameters
        ----------
        na_sentinel : number
            Value to indicate missing category.

        Returns
        --------
        (labels, cats) : (Series, Series)
            - *labels* contains the encoded values
            - *cats* contains the categories in order that the N-th
              item corresponds to the (N-1) code.
        """
        cats = self.unique().astype(self.dtype)

        name = self.name  # label_encoding mutates self.name
        labels = self.label_encoding(cats=cats)
        self.name = name

        return labels, cats

    # UDF related

    def applymap(self, udf, out_dtype=None):
        """Apply an elementwise function to transform the values in the Column.

        The user function is expected to take one argument and return the
        result, which will be stored to the output Series.  The function
        cannot reference globals except for other simple scalar objects.

        Parameters
        ----------
        udf : function
            Either a callable python function or a python function already
            decorated by ``numba.cuda.jit`` for call on the GPU as a device

        out_dtype  : numpy.dtype; optional
            The dtype for use in the output.
            Only used for ``numba.cuda.jit`` decorated udf.
            By default, the result will have the same dtype as the source.

        Returns
        -------
        result : Series
            The mask and index are preserved.

        Notes
        --------
        The supported Python features are listed in

          https://numba.pydata.org/numba-doc/dev/cuda/cudapysupported.html

        with these exceptions:

        * Math functions in `cmath` are not supported since `libcudf` does not
          have complex number support and output of `cmath` functions are most
          likely complex numbers.

        * These five functions in `math` are not supported since numba
          generates multiple PTX functions from them

          * math.sin()
          * math.cos()
          * math.tan()
          * math.gamma()
          * math.lgamma()

        * Series with string dtypes are not supported in `applymap` method.

        * Global variables need to be re-defined explicitly inside
          the udf, as numba considers them to be compile-time constants
          and there is no known way to obtain value of the global variable.

        Examples
        --------
        Returning a Series of booleans using only a literal pattern.

        >>> import cudf
        >>> s = cudf.Series([1, 10, -10, 200, 100])
        >>> s.applymap(lambda x: x)
        0      1
        1     10
        2    -10
        3    200
        4    100
        dtype: int64
        >>> s.applymap(lambda x: x in [1, 100, 59])
        0     True
        1    False
        2    False
        3    False
        4     True
        dtype: bool
        >>> s.applymap(lambda x: x ** 2)
        0        1
        1      100
        2      100
        3    40000
        4    10000
        dtype: int64
        >>> s.applymap(lambda x: (x ** 2) + (x / 2))
        0        1.5
        1      105.0
        2       95.0
        3    40100.0
        4    10050.0
        dtype: float64
        >>> def cube_function(a):
        ...     return a ** 3
        ...
        >>> s.applymap(cube_function)
        0          1
        1       1000
        2      -1000
        3    8000000
        4    1000000
        dtype: int64
        >>> def custom_udf(x):
        ...     if x > 0:
        ...         return x + 5
        ...     else:
        ...         return x - 5
        ...
        >>> s.applymap(custom_udf)
        0      6
        1     15
        2    -15
        3    205
        4    105
        dtype: int64
        """
        if is_string_dtype(self._column.dtype) or is_categorical_dtype(
            self._column.dtype
        ):
            raise TypeError(
                "User defined functions are currently not "
                "supported on Series with dtypes `str` and `category`."
            )

        if callable(udf):
            res_col = self._unaryop(udf)
        else:
            res_col = self._column.applymap(udf, out_dtype=out_dtype)
        return self._copy_construct(data=res_col)

    #
    # Stats
    #
    def count(self, level=None, **kwargs):
        """
        Return number of non-NA/null observations in the Series

        Returns
        -------
        int
            Number of non-null values in the Series.

        Notes
        -----
        Parameters currently not supported is `level`.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.count()
        5
        """

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        return self.valid_count

    def min(
        self,
        axis=None,
        skipna=None,
        dtype=None,
        level=None,
        numeric_only=None,
        **kwargs,
    ):
        """
        Return the minimum of the values in the Series.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        dtype : data type
            Data type to cast the result to.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.min()
        1
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if numeric_only not in (None, True):
            raise NotImplementedError(
                "numeric_only parameter is not implemented yet"
            )

        skipna = True if skipna is None else skipna

        if skipna:
            result_series = self.nans_to_nulls()
            if result_series.has_nulls:
                result_series = result_series.dropna()
        else:
            if self.has_nulls:
                return np.nan

            result_series = self

        return result_series._column.min(dtype=dtype)

    def max(
        self,
        axis=None,
        skipna=None,
        dtype=None,
        level=None,
        numeric_only=None,
        **kwargs,
    ):
        """
        Return the maximum of the values in the Series.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        dtype : data type
            Data type to cast the result to.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.max()
        5
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if numeric_only not in (None, True):
            raise NotImplementedError(
                "numeric_only parameter is not implemented yet"
            )

        skipna = True if skipna is None else skipna

        if skipna:
            result_series = self.nans_to_nulls()
            if result_series.has_nulls:
                result_series = result_series.dropna()
        else:
            if self.has_nulls:
                return np.nan

            result_series = self

        return result_series._column.max(dtype=dtype)

    def sum(
        self,
        axis=None,
        skipna=None,
        dtype=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):
        """
        Return sum of the values in the Series.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        dtype : data type
            Data type to cast the result to.

        min_count : int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.sum()
        15
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if numeric_only not in (None, True):
            raise NotImplementedError(
                "numeric_only parameter is not implemented yet"
            )

        skipna = True if skipna is None else skipna

        if skipna:
            result_series = self.nans_to_nulls()
            if result_series.has_nulls:
                result_series = result_series.dropna()
        else:
            if self.has_nulls:
                return np.nan

            result_series = self

        if min_count > 0:
            valid_count = len(result_series) - result_series.null_count
            if valid_count < min_count:
                return np.nan
        elif min_count < 0:
            msg = "min_count value cannot be negative({0}), will default to 0."
            warnings.warn(msg.format(min_count))

        return result_series._column.sum(dtype=dtype)

    def product(
        self,
        axis=None,
        skipna=None,
        dtype=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):
        """
        Return product of the values in the Series.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        dtype : data type
            Data type to cast the result to.

        min_count : int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.product()
        120
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if numeric_only not in (None, True):
            raise NotImplementedError(
                "numeric_only parameter is not implemented yet"
            )

        skipna = True if skipna is None else skipna

        if skipna:
            result_series = self.nans_to_nulls()
            if result_series.has_nulls:
                result_series = result_series.dropna()
        else:
            if self.has_nulls:
                return np.nan

            result_series = self

        if min_count > 0:
            valid_count = len(result_series) - result_series.null_count
            if valid_count < min_count:
                return np.nan
        elif min_count < 0:
            msg = "min_count value cannot be negative({0}), will default to 0."
            warnings.warn(msg.format(min_count))

        return result_series._column.product(dtype=dtype)

    def prod(
        self,
        axis=None,
        skipna=None,
        dtype=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):
        """
        Return product of the values in the series

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        dtype : data type
            Data type to cast the result to.

        min_count : int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.prod()
        120
        """
        return self.product(
            axis=axis,
            skipna=skipna,
            dtype=dtype,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def cummin(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return cumulative minimum of the Series.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported is `axis`

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.cummin()
        0    1
        1    1
        2    1
        3    1
        4    1
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        skipna = True if skipna is None else skipna

        if skipna:
            result_col = self.nans_to_nulls()._column
        else:
            result_col = self._column.copy()
            if result_col.has_nulls:
                # Workaround as find_first_value doesn't seem to work
                # incase of bools.
                first_index = int(
                    result_col.isnull().astype("int8").find_first_value(1)
                )
                result_col[first_index:] = None

        return Series(
            result_col._apply_scan_op("min"), name=self.name, index=self.index,
        )

    def cummax(self, axis=0, skipna=True, *args, **kwargs):
        """
        Return cumulative maximum of the Series.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported is `axis`

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.cummax()
        0    1
        1    5
        2    5
        3    5
        4    5
        """
        assert axis in (None, 0)

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        skipna = True if skipna is None else skipna

        if skipna:
            result_col = self.nans_to_nulls()._column
        else:
            result_col = self._column.copy()
            if result_col.has_nulls:
                first_index = int(
                    result_col.isnull().astype("int8").find_first_value(1)
                )
                result_col[first_index:] = None

        return Series(
            result_col._apply_scan_op("max"), name=self.name, index=self.index,
        )

    def cumsum(self, axis=0, skipna=True, *args, **kwargs):
        """
        Return cumulative sum of the Series.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.


        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported is `axis`

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.cumsum()
        0    1
        1    6
        2    8
        3    12
        4    15
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        skipna = True if skipna is None else skipna

        if skipna:
            result_col = self.nans_to_nulls()._column
        else:
            result_col = self._column.copy()
            if result_col.has_nulls:
                first_index = int(
                    result_col.isnull().astype("int8").find_first_value(1)
                )
                result_col[first_index:] = None

        # pandas always returns int64 dtype if original dtype is int or `bool`
        if np.issubdtype(result_col.dtype, np.integer) or np.issubdtype(
            result_col.dtype, np.bool_
        ):
            return Series(
                result_col.astype(np.int64)._apply_scan_op("sum"),
                name=self.name,
                index=self.index,
            )
        else:
            return Series(
                result_col._apply_scan_op("sum"),
                name=self.name,
                index=self.index,
            )

    def cumprod(self, axis=0, skipna=True, *args, **kwargs):
        """
        Return cumulative product of the Series.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported is `axis`

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.cumprod()
        0    1
        1    5
        2    10
        3    40
        4    120
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        skipna = True if skipna is None else skipna

        if skipna:
            result_col = self.nans_to_nulls()._column
        else:
            result_col = self._column.copy()
            if result_col.has_nulls:
                first_index = int(
                    result_col.isnull().astype("int8").find_first_value(1)
                )
                result_col[first_index:] = None

        # pandas always returns int64 dtype if original dtype is int or `bool`
        if np.issubdtype(result_col.dtype, np.integer) or np.issubdtype(
            result_col.dtype, np.bool_
        ):
            return Series(
                result_col.astype(np.int64)._apply_scan_op("product"),
                name=self.name,
                index=self.index,
            )
        else:
            return Series(
                result_col._apply_scan_op("product"),
                name=self.name,
                index=self.index,
            )

    def mean(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        """
        Return the mean of the values in the series.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `level` and
        `numeric_only`

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([10, 25, 3, 25, 24, 6])
        >>> ser.mean()
        15.5
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if numeric_only not in (None, True):
            raise NotImplementedError(
                "numeric_only parameter is not implemented yet"
            )

        skipna = True if skipna is None else skipna

        if skipna:
            result_series = self.nans_to_nulls()
            if result_series.has_nulls:
                result_series = result_series.dropna()
        else:
            if self.has_nulls:
                return np.nan

            result_series = self

        return result_series._column.mean()

    def std(
        self,
        axis=None,
        skipna=None,
        level=None,
        ddof=1,
        numeric_only=None,
        **kwargs,
    ):
        """
        Return sample standard deviation of the Series.

        Normalized by N-1 by default. This can be changed using
        the ddof argument

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.

        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations
            is N - ddof, where N represents the number of elements.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `level` and
        `numeric_only`
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if numeric_only not in (None, True):
            raise NotImplementedError(
                "numeric_only parameter is not implemented yet"
            )

        skipna = True if skipna is None else skipna

        if skipna:
            result_series = self.nans_to_nulls()
            if result_series.has_nulls:
                result_series = result_series.dropna()
        else:
            if self.has_nulls:
                return np.nan

            result_series = self

        return result_series._column.std(ddof=ddof)

    def var(
        self,
        axis=None,
        skipna=None,
        level=None,
        ddof=1,
        numeric_only=None,
        **kwargs,
    ):
        """
        Return unbiased variance of the Series.

        Normalized by N-1 by default. This can be changed using the
        ddof argument

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.

        ddof : int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is
            N - ddof, where N represents the number of elements.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `level` and
        `numeric_only`
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if numeric_only not in (None, True):
            raise NotImplementedError(
                "numeric_only parameter is not implemented yet"
            )

        skipna = True if skipna is None else skipna

        if skipna:
            result_series = self.nans_to_nulls()
            if result_series.has_nulls:
                result_series = result_series.dropna()
        else:
            if self.has_nulls:
                return np.nan

            result_series = self

        return result_series._column.var(ddof=ddof)

    def sum_of_squares(self, dtype=None):
        return self._column.sum_of_squares(dtype=dtype)

    def median(self, skipna=True):
        """Compute the median of the series
        """
        if not skipna and self.has_nulls:
            return np.nan
        # enforce linear in case the default ever changes
        return self.quantile(0.5, interpolation="linear", exact=True)

    def round(self, decimals=0):
        """Round a Series to a configurable number of decimal places.
        """
        return Series(
            self._column.round(decimals=decimals),
            name=self.name,
            index=self.index,
            dtype=self.dtype,
        )

    def kurtosis(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        """
        Return Fisher's unbiased kurtosis of a sample.

        Kurtosis obtained using Fisher’s definition of
        kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `level` and
        `numeric_only`
        """
        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if numeric_only not in (None, True):
            raise NotImplementedError(
                "numeric_only parameter is not implemented yet"
            )

        skipna = True if skipna is None else skipna

        if self.empty or (not skipna and self.has_nulls):
            return np.nan

        self = self.nans_to_nulls().dropna()

        if len(self) < 4:
            return np.nan

        n = len(self)
        miu = self.mean()
        m4_numerator = ((self - miu) ** 4).sum()
        V = self.var()

        if V == 0:
            return 0

        term_one_section_one = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        term_one_section_two = m4_numerator / (V ** 2)
        term_two = ((n - 1) ** 2) / ((n - 2) * (n - 3))
        kurt = term_one_section_one * term_one_section_two - 3 * term_two
        return kurt

    # Alias for kurtosis.
    kurt = kurtosis

    def skew(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        """
        Return unbiased Fisher-Pearson skew of a sample.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values when computing the result.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `axis`, `level` and
        `numeric_only`
        """

        if axis not in (None, 0):
            raise NotImplementedError("axis parameter is not implemented yet")

        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if numeric_only not in (None, True):
            raise NotImplementedError(
                "numeric_only parameter is not implemented yet"
            )

        skipna = True if skipna is None else skipna

        if self.empty or (not skipna and self.has_nulls):
            return np.nan

        self = self.nans_to_nulls().dropna()

        if len(self) < 3:
            return np.nan

        n = len(self)
        miu = self.mean()
        m3 = ((self - miu) ** 3).sum() / n
        m2 = self.var(ddof=0)

        if m2 == 0:
            return 0

        unbiased_coef = ((n * (n - 1)) ** 0.5) / (n - 2)
        skew = unbiased_coef * m3 / (m2 ** (3 / 2))
        return skew

    def cov(self, other, min_periods=None):
        """
        Compute covariance with Series, excluding missing values.

        Parameters
        ----------
        other : Series
            Series with which to compute the covariance.

        Returns
        -------
        float
            Covariance between Series and other normalized by N-1
            (unbiased estimator).

        Notes
        -----
        `min_periods` parameter is not yet supported.

        Examples
        --------
        >>> import cudf
        >>> ser1 = cudf.Series([0.9, 0.13, 0.62])
        >>> ser2 = cudf.Series([0.12, 0.26, 0.51])
        >>> ser1.cov(ser2)
        -0.015750000000000004
        """

        if min_periods is not None:
            raise NotImplementedError(
                "min_periods parameter is not implemented yet"
            )

        if self.empty or other.empty:
            return np.nan

        lhs = self.nans_to_nulls().dropna()
        rhs = other.nans_to_nulls().dropna()

        lhs, rhs = _align_indices([lhs, rhs], how="inner")

        if lhs.empty or rhs.empty or (len(lhs) == 1 and len(rhs) == 1):
            return np.nan

        result = (lhs - lhs.mean()) * (rhs - rhs.mean())
        cov_sample = result.sum() / (len(lhs) - 1)
        return cov_sample

    def corr(self, other, method="pearson", min_periods=None):
        """Calculates the sample correlation between two Series,
        excluding missing values.

        Examples
        --------
        >>> import cudf
        >>> ser1 = cudf.Series([0.9, 0.13, 0.62])
        >>> ser2 = cudf.Series([0.12, 0.26, 0.51])
        >>> ser1.corr(ser2)
        -0.20454263717316112
        """

        assert method in ("pearson",) and min_periods in (None,)

        if self.empty or other.empty:
            return np.nan

        lhs = self.nans_to_nulls().dropna()
        rhs = other.nans_to_nulls().dropna()
        lhs, rhs = _align_indices([lhs, rhs], how="inner")

        if lhs.empty or rhs.empty:
            return np.nan

        cov = lhs.cov(rhs)
        lhs_std, rhs_std = lhs.std(), rhs.std()

        if not cov or lhs_std == 0 or rhs_std == 0:
            return np.nan
        return cov / lhs_std / rhs_std

    def isin(self, values):
        """Check whether values are contained in Series.

        Parameters
        ----------
        values : set or list-like
            The sequence of values to test. Passing in a single string will
            raise a TypeError. Instead, turn a single string into a list
            of one element.

        Returns
        -------
        result : Series
            Series of booleans indicating if each element is in values.

        Raises
        -------
        TypeError
            If values is a string
        """

        if is_scalar(values):
            raise TypeError(
                "only list-like objects are allowed to be passed "
                f"to isin(), you passed a [{type(values).__name__}]"
            )

        return Series(
            self._column.isin(values), index=self.index, name=self.name
        )

    def unique(self):
        """
        Returns unique values of this Series.
        """
        res = self._column.unique()
        return Series(res, name=self.name)

    def nunique(self, method="sort", dropna=True):
        """Returns the number of unique values of the Series: approximate version,
        and exact version to be moved to libgdf
        """
        if method != "sort":
            msg = "non sort based distinct_count() not implemented yet"
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            return 0
        return self._column.distinct_count(method, dropna)

    def value_counts(
        self,
        normalize=False,
        sort=True,
        ascending=False,
        bins=None,
        dropna=True,
    ):
        """Return a Series containing counts of unique values.

        The resulting object will be in descending order so that
        the first element is the most frequently-occurring element.
        Excludes NA values by default.

        Parameters
        ----------
        normalize : bool, default False
            If True then the object returned will contain the
            relative frequencies of the unique values. normalize == True
            is not supported.

        sort : bool, default True
            Sort by frequencies.

        ascending : bool, default False
            Sort in ascending order.

        bins : int, optional
            Rather than count values, group them into half-open bins,
            works with numeric data. Not yet supported.

        dropna : bool, default True
            Don’t include counts of NaN and None.
            dropna == False is not supported

        Returns
        -------
        result : Series contanining counts of unique values.

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, None])
        >>> sr.value_counts(ascending=True)
        1.0    1
        2.0    2
        3.0    3
        dtype: int32
        """

        if normalize is not False:
            raise NotImplementedError(
                "Only normalize == False is currently supported"
            )
        if dropna is not True:
            raise NotImplementedError(
                "Only dropna == True is currently supported"
            )
        if bins is not None:
            raise NotImplementedError("bins is not yet supported")

        if self.null_count == len(self):
            return Series(np.array([], dtype=np.int32), name=self.name)

        res = self.groupby(self).count()
        res.index.name = None

        if sort:
            return res.sort_values(ascending=ascending)
        return res

    def scale(self):
        """Scale values to [0, 1] in float64
        """
        vmin = self.min()
        vmax = self.max()
        scaled = (self - vmin) / (vmax - vmin)
        return self._copy_construct(data=scaled)

    # Absolute
    def abs(self):
        """Absolute value of each element of the series.

        Returns a new Series.
        """
        return self._unaryop("abs")

    def __abs__(self):
        return self.abs()

    # Rounding
    def ceil(self):
        """Rounds each value upward to the smallest integral value not less
        than the original.

        Returns a new Series.
        """
        return self._unaryop("ceil")

    def floor(self):
        """Rounds each value downward to the largest integral value not greater
        than the original.

        Returns a new Series.
        """
        return self._unaryop("floor")

    def hash_values(self):
        """Compute the hash of values in this column.
        """
        return Series(self._hash()).values

    def hash_encode(self, stop, use_name=False):
        """Encode column values as ints in [0, stop) using hash function.

        Parameters
        ----------
        stop : int
            The upper bound on the encoding range.
        use_name : bool
            If ``True`` then combine hashed column values
            with hashed column name. This is useful for when the same
            values in different columns should be encoded
            with different hashed values.

        Returns
        -------
        result : Series
            The encoded Series.
        """
        assert stop > 0

        initial_hash = [hash(self.name) & 0xFFFFFFFF] if use_name else None
        hashed_values = Series(self._hash(initial_hash))

        if hashed_values.has_nulls:
            raise ValueError("Column must have no nulls.")

        mod_vals = hashed_values % stop
        return Series(mod_vals._column, index=self.index, name=self.name)

    def quantile(
        self, q=0.5, interpolation="linear", exact=True, quant_index=True
    ):
        """
        Return values at the given quantile.

        Parameters
        ----------

        q : float or array-like, default 0.5 (50% quantile)
            0 <= q <= 1, the quantile(s) to compute
        interpolation : {’linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points i and j:
        columns : list of str
            List of column names to include.
        exact : boolean
            Whether to use approximate or exact quantile algorithm.
        quant_index : boolean
            Whether to use the list of quantiles as index.

        Returns
        -------

        DataFrame
        """

        if isinstance(q, Number) or is_list_like(q):
            np_array_q = np.asarray(q)
            if np.logical_or(np_array_q < 0, np_array_q > 1).any():
                raise ValueError(
                    "percentiles should all \
                             be in the interval [0, 1]"
                )

        # Beyond this point, q either being scalar or list-like
        # will only have values in range [0, 1]

        if isinstance(q, Number):
            res = self._column.quantile(q, interpolation, exact)
            res = res[0]
            return np.nan if res is None else res

        if not quant_index:
            return Series(
                self._column.quantile(q, interpolation, exact), name=self.name
            )
        else:
            from cudf.core.column import column_empty_like

            np_array_q = np.asarray(q)
            if len(self) == 0:
                result = column_empty_like(
                    np_array_q,
                    dtype=self.dtype,
                    masked=True,
                    newsize=len(np_array_q),
                )
            else:
                result = self._column.quantile(q, interpolation, exact)
            return Series(result, index=as_index(np_array_q), name=self.name)

    def describe(self, percentiles=None, include=None, exclude=None):
        """Compute summary statistics of a Series. For numeric
        data, the output includes the minimum, maximum, mean, median,
        standard deviation, and various quantiles. For object data, the output
        includes the count, number of unique values, the most common value, and
        the number of occurrences of the most common value.

        Parameters
        ----------
        percentiles : list-like, optional
            The percentiles used to generate the output summary statistics.
            If None, the default percentiles used are the 25th, 50th and 75th.
            Values should be within the interval [0, 1].

        Returns
        -------
        A DataFrame containing summary statistics of relevant columns from
        the input DataFrame.

        Examples
        --------
        Describing a ``Series`` containing numeric values.

        >>> import cudf
        >>> s = cudf.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> print(s.describe())
           stats   values
        0  count     10.0
        1   mean      5.5
        2    std  3.02765
        3    min      1.0
        4    25%      2.5
        5    50%      5.5
        6    75%      7.5
        7    max     10.0
        """

        def _prepare_percentiles(percentiles):
            percentiles = list(percentiles)

            if not all(0 <= x <= 1 for x in percentiles):
                raise ValueError(
                    "All percentiles must be between 0 and 1, " "inclusive."
                )

            # describe always includes 50th percentile
            if 0.5 not in percentiles:
                percentiles.append(0.5)

            percentiles = np.sort(percentiles)
            return percentiles

        def _format_percentile_names(percentiles):
            return ["{0}%".format(int(x * 100)) for x in percentiles]

        def _format_stats_values(stats_data):
            return list(map(lambda x: round(x, 6), stats_data))

        def describe_numeric(self):
            # mimicking pandas
            names = (
                ["count", "mean", "std", "min"]
                + _format_percentile_names(percentiles)
                + ["max"]
            )
            data = (
                [self.count(), self.mean(), self.std(), self.min()]
                + self.quantile(percentiles).to_array(fillna="pandas").tolist()
                + [self.max()]
            )
            data = _format_stats_values(data)

            return Series(
                data=data, index=names, nan_as_null=False, name=self.name
            )

        def describe_categorical(self):
            # blocked by StringColumn/DatetimeColumn support for
            # value_counts/unique
            pass

        if percentiles is not None:
            percentiles = _prepare_percentiles(percentiles)
        else:
            # pandas defaults
            percentiles = np.array([0.25, 0.5, 0.75])

        if np.issubdtype(self.dtype, np.number):
            return describe_numeric(self)
        else:
            raise NotImplementedError(
                "Describing non-numeric columns is not " "yet supported"
            )

    def digitize(self, bins, right=False):
        """Return the indices of the bins to which each value in series belongs.

        Notes
        -----
        Monotonicity of bins is assumed and not checked.

        Parameters
        ----------
        bins : np.array
            1-D monotonically, increasing array with same type as this series.
        right : bool
            Indicates whether interval contains the right or left bin edge.

        Returns
        -------
        A new Series containing the indices.
        """
        from cudf.core.column import numerical

        return Series(numerical.digitize(self._column, bins, right))

    def diff(self, periods=1):
        """Calculate the difference between values at positions i and i - N in
        an array and store the output in a new array.

        Notes
        -----
        Diff currently only supports float and integer dtype columns with
        no null values.
        """
        if self.has_nulls:
            raise AssertionError(
                "Diff currently requires columns with no null values"
            )

        if not np.issubdtype(self.dtype, np.number):
            raise NotImplementedError(
                "Diff currently only supports numeric dtypes"
            )

        # TODO: move this libcudf
        input_col = self._column
        output_col = column_empty_like(input_col)
        output_mask = column_empty_like(input_col, dtype="bool")
        if output_col.size > 0:
            cudautils.gpu_diff.forall(output_col.size)(
                input_col, output_col, output_mask, periods
            )

        output_col = column.build_column(
            data=output_col.data,
            dtype=output_col.dtype,
            mask=bools_to_mask(output_mask),
        )

        return Series(output_col, name=self.name, index=self.index)

    @copy_docstring(SeriesGroupBy.__init__)
    def groupby(
        self,
        by=None,
        group_series=None,
        level=None,
        sort=True,
        group_keys=True,
        as_index=None,
        dropna=True,
        method=None,
    ):
        if group_keys is not True:
            raise NotImplementedError(
                "The group_keys keyword is not yet implemented"
            )
        else:
            if method is not None:
                warnings.warn(
                    "The 'method' argument is deprecated and will be unused",
                    DeprecationWarning,
                )
            return SeriesGroupBy(
                self, by=by, level=level, dropna=dropna, sort=sort
            )

    @copy_docstring(Rolling)
    def rolling(
        self, window, min_periods=None, center=False, axis=0, win_type=None
    ):
        return Rolling(
            self,
            window,
            min_periods=min_periods,
            center=center,
            axis=axis,
            win_type=win_type,
        )

    @ioutils.doc_to_json()
    def to_json(self, path_or_buf=None, *args, **kwargs):
        """{docstring}"""
        import cudf.io.json as json

        return json.to_json(self, path_or_buf=path_or_buf, *args, **kwargs)

    @ioutils.doc_to_hdf()
    def to_hdf(self, path_or_buf, key, *args, **kwargs):
        """{docstring}"""
        import cudf.io.hdf as hdf

        hdf.to_hdf(path_or_buf, key, self, *args, **kwargs)

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""
        import cudf.io.dlpack as dlpack

        return dlpack.to_dlpack(self)

    def rename(self, index=None, copy=True):
        """
        Alter Series name

        Change Series.name with a scalar value

        Parameters
        ----------
        index : Scalar, optional
            Scalar to alter the Series.name attribute
        copy : boolean, default True
            Also copy underlying data

        Returns
        -------
        Series

        Notes
        -----
        Difference from pandas:
          - Supports scalar values only for changing name attribute
          - Not supporting : inplace, level
        """
        out = self.copy(deep=False)
        out = out.set_index(self.index)
        if index:
            out.name = index

        return out.copy(deep=copy)

    @property
    def is_unique(self):
        """
        Return boolean if values in the object are unique.

        Returns
        -------
        out : bool
        """
        return self._column.is_unique

    @property
    def is_monotonic(self):
        """
        Return boolean if values in the object are monotonic_increasing.

        Returns
        -------
        out : bool
        """
        return self._column.is_monotonic_increasing

    @property
    def is_monotonic_increasing(self):
        """
        Return boolean if values in the object are monotonic_increasing.

        Returns
        -------
        out : bool
        """
        return self._column.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        """
        Return boolean if values in the object are monotonic_decreasing.

        Returns
        -------
        out : bool
        """
        return self._column.is_monotonic_decreasing

    @property
    def __cuda_array_interface__(self):
        return self._column.__cuda_array_interface__

    def _align_to_index(
        self, index, how="outer", sort=True, allow_non_unique=False
    ):
        """
        Align to the given Index. See _align_indices below.
        """
        from uuid import uuid4

        index = as_index(index)
        if self.index.equals(index):
            return self
        if not allow_non_unique:
            if len(self) != len(self.index.unique()) or len(index) != len(
                index.unique()
            ):
                raise ValueError("Cannot align indices with non-unique values")
        lhs = self.to_frame(0)
        rhs = cudf.DataFrame(index=as_index(index))
        if how == "left":
            tmp_col_id = str(uuid4())
            lhs[tmp_col_id] = cupy.arange(len(lhs))
        elif how == "right":
            tmp_col_id = str(uuid4())
            rhs[tmp_col_id] = cupy.arange(len(rhs))
        result = lhs.join(rhs, how=how, sort=sort)
        if how == "left" or how == "right":
            result = result.sort_values(tmp_col_id)[0]
        else:
            result = result[0]

        result.name = self.name
        result.index.names = index.names
        return result

    def merge(
        self,
        other,
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        how="inner",
        sort=False,
        lsuffix=None,
        rsuffix=None,
        method="hash",
        suffixes=("_x", "_y"),
    ):

        if left_on not in (self.name, None):
            raise ValueError(
                "Series to other merge uses series name as key implicitly"
            )

        lhs = self.copy(deep=False)
        rhs = other.copy(deep=False)

        result = super(Series, lhs)._merge(
            rhs,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            sort=sort,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            method=method,
            indicator=False,
            suffixes=suffixes,
        )

        return result


truediv_int_dtype_corrections = {
    "int8": "float32",
    "int16": "float32",
    "int32": "float32",
    "int64": "float64",
    "uint8": "float32",
    "uint16": "float32",
    "uint32": "float64",
    "uint64": "float64",
    "bool": "float32",
    "int": "float",
}


class DatetimeProperties(object):
    def __init__(self, series):
        self.series = series

    @property
    def year(self):
        return self._get_dt_field("year")

    @property
    def month(self):
        return self._get_dt_field("month")

    @property
    def day(self):
        return self._get_dt_field("day")

    @property
    def hour(self):
        return self._get_dt_field("hour")

    @property
    def minute(self):
        return self._get_dt_field("minute")

    @property
    def second(self):
        return self._get_dt_field("second")

    @property
    def weekday(self):
        return self._get_dt_field("weekday")

    def _get_dt_field(self, field):
        out_column = self.series._column.get_dt_field(field)
        return Series(
            data=out_column, index=self.series._index, name=self.series.name
        )


def _align_indices(series_list, how="outer", allow_non_unique=False):
    """
    Internal util to align the indices of a list of Series objects

    series_list : list of Series objects
    how : {"outer", "inner"}
        If "outer", the values of the resulting index are the
        unique values of the index obtained by concatenating
        the indices of all the series.
        If "inner", the values of the resulting index are
        the values common to the indices of all series.
    allow_non_unique : bool
        Whether or not to allow non-unique valued indices in the input
        series.
    """
    if len(series_list) <= 1:
        return series_list

    # check if all indices are the same
    head = series_list[0].index

    all_index_equal = True
    for sr in series_list[1:]:
        if not sr.index.equals(head):
            all_index_equal = False
            break

    # check if all names are the same
    all_names_equal = True
    for sr in series_list[1:]:
        if not sr.index.names == head.names:
            all_names_equal = False
    new_index_names = [None]
    if all_names_equal:
        new_index_names = head.names

    if all_index_equal:
        return series_list

    if how == "outer":
        combined_index = cudf.core.reshape.concat(
            [sr.index for sr in series_list]
        ).unique()
        combined_index.names = new_index_names
    else:
        combined_index = series_list[0].index
        for sr in series_list[1:]:
            combined_index = (
                cudf.DataFrame(index=sr.index).join(
                    cudf.DataFrame(index=combined_index),
                    sort=True,
                    how="inner",
                )
            ).index
        combined_index.names = new_index_names

    # align all Series to the combined index
    result = [
        sr._align_to_index(
            combined_index, how=how, allow_non_unique=allow_non_unique
        )
        for sr in series_list
    ]

    return result


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Returns a boolean array where two arrays are equal within a tolerance.

    Two values in ``a`` and ``b`` are  considiered equal when the following
    equation is satisfied.

    .. math::
       |a - b| \\le \\mathrm{atol} + \\mathrm{rtol} |b|

    Parameters
    ----------
        a : list-like, array-like or cudf.Series
            Input sequence to compare.
        b : list-like, array-like or cudf.Series
            Input sequence to compare.
        rtol : float
            The relative tolerance.
        atol : float
            The absolute tolerance.
        equal_nan : bool
            If ``True``, null's in ``a`` will be considered equal
            to null's in ``b``.

    Returns
    -------
    Series

    See Also
    --------
        np.isclose : Returns a boolean array where two arrays are element-wise
            equal within a tolerance.

    Examples
    --------
    >>> import cudf
    >>> s1 = cudf.Series([1.9876543,   2.9876654,   3.9876543, None, 9.9, 1.0])
    >>> s2 = cudf.Series([1.987654321, 2.987654321, 3.987654321, None, 19.9,
    ... None])
    >>> s1
    0    1.9876543
    1    2.9876654
    2    3.9876543
    3         null
    4          9.9
    5          1.0
    dtype: float64
    >>> s2
    0    1.987654321
    1    2.987654321
    2    3.987654321
    3           null
    4           19.9
    5           null
    dtype: float64
    >>> cudf.isclose(s1, s2)
    0     True
    1     True
    2     True
    3    False
    4    False
    5    False
    dtype: bool
    >>> cudf.isclose(s1, s2, equal_nan=True)
    0     True
    1     True
    2     True
    3     True
    4    False
    5    False
    dtype: bool
    >>> cudf.isclose(s1, s2, equal_nan=False)
    0     True
    1     True
    2     True
    3    False
    4    False
    5    False
    dtype: bool
    """

    index = None

    if not can_convert_to_column(a):
        raise TypeError(
            f"Parameter `a` is expected to be a "
            f"list-like or Series object, found:{type(a)}"
        )
    if not can_convert_to_column(b):
        raise TypeError(
            f"Parameter `b` is expected to be a "
            f"list-like or Series object, found:{type(a)}"
        )

    if isinstance(a, pd.Series):
        a = Series.from_pandas(a)
    if isinstance(b, pd.Series):
        b = Series.from_pandas(b)

    if isinstance(a, cudf.Series) and isinstance(b, cudf.Series):
        b = b.reindex(a.index)
        index = as_index(a.index)

    a_col = column.as_column(a)
    a_array = cupy.asarray(a_col.data_array_view)

    b_col = column.as_column(b)
    b_array = cupy.asarray(b_col.data_array_view)

    result = cupy.isclose(
        a=a_array, b=b_array, rtol=rtol, atol=atol, equal_nan=equal_nan
    )
    result_col = column.as_column(result)

    if a_col.null_count and b_col.null_count:
        a_nulls = a_col.isna()
        b_nulls = b_col.isna()
        null_values = a_nulls.binary_operator("or", b_nulls)

        if equal_nan is True:
            equal_nulls = a_nulls.binary_operator("and", b_nulls)

        del a_nulls, b_nulls
    elif a_col.null_count:
        null_values = a_col.isna()
    elif b_col.null_count:
        null_values = b_col.isna()
    else:
        return Series(result_col, index=index)

    result_col[null_values] = False
    if equal_nan is True and a_col.null_count and b_col.null_count:
        result_col[equal_nulls] = True

    return Series(result_col, index=index)
