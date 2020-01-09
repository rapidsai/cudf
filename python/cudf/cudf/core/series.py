# Copyright (c) 2018, NVIDIA CORPORATION.

import operator
import pickle
import warnings
from numbers import Number

import cupy
import numpy as np
import pandas as pd
from pandas.api.types import is_dict_like

import rmm

import cudf
import cudf._lib as libcudf
from cudf.core.column import ColumnBase, DatetimeColumn, column
from cudf.core.index import Index, RangeIndex, as_index
from cudf.core.indexing import _SeriesIlocIndexer, _SeriesLocIndexer
from cudf.core.table import Table
from cudf.core.window import Rolling
from cudf.utils import cudautils, ioutils, utils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    is_categorical_dtype,
    is_datetime_dtype,
    is_list_like,
    is_scalar,
    min_scalar_type,
    to_cudf_compatible_scalar,
)


class Series(Table):
    """
    Data and null-masks.

    ``Series`` objects are used as columns of ``DataFrame``.
    """

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
        mask : 1D array-like of numpy.uint8
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
        self, data=None, index=None, name=None, nan_as_null=True, dtype=None
    ):
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
            data = data.as_column()
            if dtype is not None:
                data = data.astype(dtype)

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

        super().__init__([data], [name])
        self._index = RangeIndex(len(data)) if index is None else index

    @property
    def _column(self):
        return self._data[self.name]

    @_column.setter
    def _column(self, value):
        self._data[self.name] = value

    def __contains__(self, item):
        return item in self._index

    @classmethod
    def from_pandas(cls, s, nan_as_null=True):
        return cls(s, nan_as_null=nan_as_null)

    @property
    def values(self):

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
        if self.dtype == np.dtype("object"):
            return np.array(
                self._column.data_array_view.to_host(), dtype="object"
            )
        elif is_categorical_dtype(self.dtype):
            return self._column.to_pandas().values
        else:
            return self._column.data_array_view.copy_to_host()

    @classmethod
    def from_arrow(cls, s):
        return cls(s)

    def serialize(self):
        header = {}
        frames = []
        header["index"], index_frames = self._index.serialize()
        frames.extend(index_frames)
        header["index_frame_count"] = len(index_frames)
        header["column"], column_frames = self._column.serialize()
        header["type"] = pickle.dumps(type(self))
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
        if isinstance(self._column, DatetimeColumn):
            return DatetimeProperties(self)
        else:
            raise AttributeError(
                "Can only use .dt accessor with datetimelike " "values"
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
        return next(iter(self._data))

    @name.setter
    def name(self, value):
        col = self._data.pop(self.name)
        self._data[value] = col

    @classmethod
    def deserialize(cls, header, frames):

        index_nframes = header["index_frame_count"]
        idx_typ = pickle.loads(header["index"]["type"])
        index = idx_typ.deserialize(header["index"], frames[:index_nframes])

        frames = frames[index_nframes:]

        column_nframes = header["column_frame_count"]
        col_typ = pickle.loads(header["column"]["type"])
        column = col_typ.deserialize(header["column"], frames[:column_nframes])
        return Series(column, index=index)

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
        result = self._copy_construct()
        if deep:
            result._column = self._column.copy(deep)
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

        return DataFrame({col: self}, index=self.index)

    def set_mask(self, mask, null_count=None):
        """Create new Series by setting a mask array.

        This will override the existing mask.  The returned Series will
        reference the same data buffer as this Series.

        Parameters
        ----------
        mask : 1D array-like of numpy.uint8
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
            return func(self)
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
        return not len(self)

    def __getitem__(self, arg):
        data = self._column[arg]
        index = self.index.take(arg)
        if is_scalar(data) or data is None:
            return data
        return self._copy_construct(data=data, index=index)

    def __setitem__(self, key, value):
        # coerce value into a scalar or column
        if is_scalar(value):
            value = to_cudf_compatible_scalar(value)
        else:
            value = column.as_column(value)

        if hasattr(value, "dtype") and pd.api.types.is_numeric_dtype(
            value.dtype
        ):
            # normalize types if necessary:
            if not pd.api.types.is_integer(key):
                to_dtype = np.result_type(value.dtype, self._column.dtype)
                value = value.astype(to_dtype)
                self._column._mimic_inplace(
                    self._column.astype(to_dtype), inplace=True
                )

        self._column[key] = value

    def take(self, indices, ignore_index=False):
        """Return Series by taking values from the corresponding *indices*.
        """
        result = self[indices]
        if ignore_index:
            index = RangeIndex(len(result))
            return result._copy_construct(index=index)
        else:
            return result

    def _get_mask_as_series(self):
        mask = Series(cudautils.ones(len(self), dtype=np.bool))
        if self._column.nullable:
            mask = mask.set_mask(self._column.mask).fillna(False)
        return mask

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

    def _binaryop(self, other, fn, reflect=False):
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

        libcudf.nvtx.nvtx_range_push("CUDF_BINARY_OP", "orange")
        result_name = utils.get_result_name(self, other)
        lhs, rhs = _align_indices(self, other)
        rhs = self._normalize_binop_value(rhs)
        outcol = lhs._column.binary_operator(fn, rhs, reflect=reflect)
        result = lhs._copy_construct(data=outcol, name=result_name)
        libcudf.nvtx.nvtx_range_pop()
        return result

    def _rbinaryop(self, other, fn):
        """
        Internal util to call a binary operator *fn* on operands *self*
        and *other* for reflected operations.  Return the output Series.
        The output dtype is determined by the input operands.
        """
        return self._binaryop(other, fn, reflect=True)

    def _filled_binaryop(self, other, fn, fill_value=None, reflect=False):
        def func(lhs, rhs):
            return fn(rhs, lhs) if reflect else fn(lhs, rhs)

        lhs, rhs = _align_indices(self, other)

        if fill_value is not None:
            if isinstance(rhs, Series):
                if lhs.nullable and rhs.nullable:
                    lmask = Series(data=lhs.nullmask)
                    rmask = Series(data=rhs.nullmask)
                    mask = (lmask | rmask).data
                    lhs = lhs.fillna(fill_value)
                    rhs = rhs.fillna(fill_value)
                    result = func(lhs, rhs)
                    data = column.build_column(
                        data=result.data, dtype=result.dtype, mask=mask,
                    )
                    return lhs._copy_construct(data=data)
                elif lhs.nullable:
                    return func(lhs.fillna(fill_value), rhs)
                elif rhs.nullable:
                    return func(lhs, rhs.fillna(fill_value))
            elif is_scalar(rhs):
                return func(lhs.fillna(fill_value), rhs)
        return func(lhs, rhs)

    def add(self, other, fill_value=None):
        """Addition of series and other, element-wise
        (binary operator add).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.add, fill_value)

    def __add__(self, other):
        return self._binaryop(other, "add")

    def radd(self, other, fill_value=None):
        """Addition of series and other, element-wise
        (binary operator radd).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.add, fill_value, True)

    def __radd__(self, other):
        return self._rbinaryop(other, "add")

    def sub(self, other, fill_value=None):
        """Subtraction of series and other, element-wise
        (binary operator sub).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.sub, fill_value)

    def __sub__(self, other):
        return self._binaryop(other, "sub")

    def rsub(self, other, fill_value=None):
        """Subtraction of series and other, element-wise
        (binary operator rsub).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.sub, fill_value, True)

    def __rsub__(self, other):
        return self._rbinaryop(other, "sub")

    def mul(self, other, fill_value=None):
        """Multiplication of series and other, element-wise
        (binary operator mul).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.mul, fill_value)

    def __mul__(self, other):
        return self._binaryop(other, "mul")

    def rmul(self, other, fill_value=None):
        """Multiplication of series and other, element-wise
        (binary operator rmul).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.mul, fill_value, True)

    def __rmul__(self, other):
        return self._rbinaryop(other, "mul")

    def mod(self, other, fill_value=None):
        """Modulo of series and other, element-wise
        (binary operator mod).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.mod, fill_value)

    def __mod__(self, other):
        return self._binaryop(other, "mod")

    def rmod(self, other, fill_value=None):
        """Modulo of series and other, element-wise
        (binary operator rmod).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.mod, fill_value, True)

    def __rmod__(self, other):
        return self._rbinaryop(other, "mod")

    def pow(self, other, fill_value=None):
        """Exponential power of series and other, element-wise
        (binary operator pow).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.pow, fill_value)

    def __pow__(self, other):
        return self._binaryop(other, "pow")

    def rpow(self, other, fill_value=None):
        """Exponential power of series and other, element-wise
        (binary operator rpow).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.pow, fill_value, True)

    def __rpow__(self, other):
        return self._rbinaryop(other, "pow")

    def floordiv(self, other, fill_value=None):
        """Integer division of series and other, element-wise
        (binary operator floordiv).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.floordiv, fill_value)

    def __floordiv__(self, other):
        return self._binaryop(other, "floordiv")

    def rfloordiv(self, other, fill_value=None):
        """Integer division of series and other, element-wise
        (binary operator rfloordiv).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(
            other, operator.floordiv, fill_value, True
        )

    def __rfloordiv__(self, other):
        return self._rbinaryop(other, "floordiv")

    def truediv(self, other, fill_value=None):
        """Floating division of series and other, element-wise
        (binary operator truediv).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.truediv, fill_value)

    def __truediv__(self, other):
        if self.dtype in list(truediv_int_dtype_corrections.keys()):
            truediv_type = truediv_int_dtype_corrections[str(self.dtype)]
            return self.astype(truediv_type)._binaryop(other, "truediv")
        else:
            return self._binaryop(other, "truediv")

    def rtruediv(self, other, fill_value=None):
        """Floating division of series and other, element-wise
        (binary operator rtruediv).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.truediv, fill_value, True)

    def __rtruediv__(self, other):
        if self.dtype in list(truediv_int_dtype_corrections.keys()):
            truediv_type = truediv_int_dtype_corrections[str(self.dtype)]
            return self.astype(truediv_type)._rbinaryop(other, "truediv")
        else:
            return self._rbinaryop(other, "truediv")

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

    def _unordered_compare(self, other, cmpops):
        libcudf.nvtx.nvtx_range_push("CUDF_UNORDERED_COMP", "orange")
        result_name = utils.get_result_name(self, other)
        other = self._normalize_binop_value(other)
        outcol = self._column.unordered_compare(cmpops, other)
        result = self._copy_construct(data=outcol, name=result_name)
        libcudf.nvtx.nvtx_range_pop()
        return result

    def _ordered_compare(self, other, cmpops):
        libcudf.nvtx.nvtx_range_push("CUDF_ORDERED_COMP", "orange")
        result_name = utils.get_result_name(self, other)
        other = self._normalize_binop_value(other)
        outcol = self._column.ordered_compare(cmpops, other)
        result = self._copy_construct(data=outcol, name=result_name)
        libcudf.nvtx.nvtx_range_pop()
        return result

    def eq(self, other, fill_value=None):
        """Equal to of series and other, element-wise
        (binary operator eq).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.eq, fill_value)

    def __eq__(self, other):
        return self._unordered_compare(other, "eq")

    def equals(self, other):
        if self is other:
            return True
        if other is None or len(self) != len(other):
            return False
        return self._unordered_compare(other, "eq").min()

    def ne(self, other, fill_value=None):
        """Not equal to of series and other, element-wise
        (binary operator ne).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.ne, fill_value)

    def __ne__(self, other):
        return self._unordered_compare(other, "ne")

    def lt(self, other, fill_value=None):
        """Less than of series and other, element-wise
        (binary operator lt).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.lt, fill_value)

    def __lt__(self, other):
        return self._ordered_compare(other, "lt")

    def le(self, other, fill_value=None):
        """Less than or equal to of series and other, element-wise
        (binary operator le).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.le, fill_value)

    def __le__(self, other):
        return self._ordered_compare(other, "le")

    def gt(self, other, fill_value=None):
        """Greater than of series and other, element-wise
        (binary operator gt).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.gt, fill_value)

    def __gt__(self, other):
        return self._ordered_compare(other, "gt")

    def ge(self, other, fill_value=None):
        """Greater than or equal to of series and other, element-wise
        (binary operator ge).

        Parameters
        ----------
        other: Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null
        """
        return self._filled_binaryop(other, operator.ge, fill_value)

    def __ge__(self, other):
        return self._ordered_compare(other, "ge")

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
        """Negatated value (-) for each element

        Returns a new Series.
        """
        return self.__mul__(-1)

    @property
    def cat(self):
        return self._column.cat()

    @property
    def str(self):
        return self._column.str(self.index, self.name)

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
        return self._column.has_nulls

    def drop_duplicates(self, keep="first", inplace=False):
        """
        Return Series with duplicate values removed
        """
        in_cols = [self._column]
        in_index = self.index
        from cudf.core.multiindex import MultiIndex

        if isinstance(in_index, MultiIndex):
            in_index = RangeIndex(len(in_index))
        out_cols, new_index = libcudf.stream_compaction.drop_duplicates(
            [in_index.as_column()], in_cols, None, keep
        )
        new_index = as_index(new_index)
        if self.index.equals(new_index):
            new_index = self.index
        if isinstance(self.index, MultiIndex):
            new_index = self.index.take(new_index)

        result = Series(out_cols[0], index=new_index, name=self.name)
        return self._mimic_inplace(result, inplace=inplace)

    def _mimic_inplace(self, result, inplace=False):
        if inplace:
            self._data = result._data
            self._index = result._index
            self._size = len(self._index)
            self.name = result.name
        else:
            return result

    def dropna(self):
        """
        Return a Series with null values removed.
        """
        if not self.has_nulls:
            return self
        name = self.name
        result = self.to_frame(name="_").dropna()
        result = result["_"]
        result.name = name
        self.name = name
        return result

    def fillna(self, value, method=None, axis=None, inplace=False, limit=None):
        """Fill null values with ``value``.

        Parameters
        ----------
        value : scalar or Series-like
            Value to use to fill nulls. If Series-like, null values
            are filled with the values in corresponding indices of the
            given Series.

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

        data = self._column.fillna(value, inplace=inplace)

        if not inplace:
            return self._copy_construct(data=data)

    def where(self, cond, other=None, axis=None):
        """
        Replace values with other where the condition is False.

        Parameters
        ----------
        cond : boolean
            Where cond is True, keep the original value. Where False,
            replace with corresponding value from other.
        other: scalar, default None
            Entries where cond is False are replaced with
            corresponding value from other.

        Returns
        -------
        result : Series

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> print(ser.where(ser > 2, 10))
        0     4
        1     3
        2    10
        3    10
        4    10
        >>> print(ser.where(ser > 2))
        0    4
        1    3
        2
        3
        4
        """
        if not pd.api.types.is_numeric_dtype(self.dtype):
            raise TypeError(
                "`Series.where` is only supported for numeric column types."
            )
        to_replace = self._column.apply_boolean_mask(~cond & self.notna())
        if is_scalar(other):
            all_nan = other is None
            if all_nan:
                new_value = [other] * len(to_replace)
            else:
                new_value = [other]
        else:
            raise NotImplementedError(
                "Replacement arg of {} is not supported.".format(type(other))
            )

        result = self._column.find_and_replace(
            to_replace, new_value, all_nan=all_nan
        )

        # To replace nulls:: If there are nulls in `cond` series, then we will
        # fill them with `False`, which means, by default, elements containing
        # nulls, are failing the given condition.
        # But, if condition is deliberately setting the `True` for nulls (i.e.
        # `s.isnulls()`), then there are no nulls in `cond`
        if not all_nan and (~cond.fillna(False) & self.isnull()).any():
            result = result.fillna(other)
        return self._copy_construct(data=result)

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
        return self._column.to_array(fillna=fillna)

    def isnull(self):
        """Identify missing values in a Series.
        """
        result = Series(
            self._column.isnull(), name=self.name, index=self.index
        )
        return result

    def isna(self):
        """Identify missing values in a Series. Alias for isnull.
        """
        return self.isnull()

    def notna(self):
        """Identify non-missing values in a Series.
        """
        result = Series(self._column.notna(), name=self.name, index=self.index)
        return result

    def notnull(self):
        """Identify non-missing values in a Series. Alias for notna.
        """
        return self.notna()

    def nans_to_nulls(self):
        """
        Convert nans (if any) to nulls
        """
        if self.dtype.kind == "f":
            sr = self.fillna(np.nan)
            newmask = libcudf.unaryops.nans_to_nulls(sr._column)
            self._column.mask = newmask
        return self

    def all(self, axis=0, skipna=True, level=None):
        """
        """
        assert axis in (None, 0) and skipna is True and level in (None,)
        if self.dtype.kind not in "biuf":
            raise NotImplementedError(
                "All does not currently support columns of {} dtype.".format(
                    self.dtype
                )
            )
        return self._column.all()

    def any(self, axis=0, skipna=True, level=None):
        """
        """
        assert axis in (None, 0) and skipna is True and level in (None,)
        if self.dtype.kind not in "biuf":
            raise NotImplementedError(
                "Any does not currently support columns of {} dtype.".format(
                    self.dtype
                )
            )
        return self._column.any()

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
        if index is True:
            index = self.index.to_pandas()
        s = self._column.to_pandas(index=index)
        s.name = self.name
        return s

    def to_arrow(self):
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

        See DataFrame.loc
        """
        return _SeriesLocIndexer(self)

    @property
    def iloc(self):
        """
        Select values by position.

        See DataFrame.iloc
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
        if self.has_nulls:
            raise ValueError("Column must have no nulls.")
        return cudautils.compact_mask_bytes(self._column.data_array_view)

    def astype(self, dtype, errors="raise", **kwargs):
        """
        Cast the Series to the given dtype

        Parameters
        ----------

        dtype : data type
        **kwargs : extra arguments to pass on to the constructor

        Returns
        -------
        out : Series
            Copy of ``self`` cast to the given dtype. Returns
            ``self`` if ``dtype`` is the same as ``self.dtype``.
        """
        if errors not in ("ignore", "raise", "warn"):
            raise ValueError("invalid error value specified")

        if pd.api.types.is_dtype_equal(dtype, self.dtype):
            return self
        try:
            return self._copy_construct(
                data=self._column.astype(dtype, **kwargs)
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

        Difference from pandas:
          * Not supporting: inplace, kind

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
        if not (0 <= n <= len(self)):
            raise ValueError("n out-of-bound")
        direction = largest
        if keep == "first":
            return self.sort_values(ascending=not direction)[:n]
        elif keep == "last":
            return self.sort_values(ascending=direction)[-n:].reverse()
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

    def replace(self, to_replace, replacement):
        """
        Replace values given in *to_replace* with *replacement*.

        Parameters
        ----------
        to_replace : numeric, str or list-like
            Value(s) to replace.
            * numeric or str:
                - values equal to *to_replace* will be replaced with *value*
            * list of numeric or str:
                - If *replacement* is also list-like, *to_replace* and
                  *replacement* must be of same length.
        replacement : numeric, str, list-like, or dict
            Value(s) to replace `to_replace` with.

        See also
        --------
        Series.fillna

        Returns
        -------
        result : Series
            Series after replacement. The mask and index are preserved.
        """
        # if all the elements of replacement column are None then propagate the
        # same dtype as self.dtype in column.as_column() for replacement
        all_nan = False
        if not is_scalar(to_replace):
            if is_scalar(replacement):
                all_nan = replacement is None
                if all_nan:
                    replacement = [replacement] * len(to_replace)
                # Do not broadcast numeric dtypes
                elif pd.api.types.is_numeric_dtype(self.dtype):
                    replacement = [replacement]
                else:
                    replacement = utils.scalar_broadcast_to(
                        replacement,
                        (len(to_replace),),
                        np.dtype(type(replacement)),
                    )
            else:
                # If both are non-scalar
                if len(to_replace) != len(replacement):
                    raise ValueError(
                        "Replacement lists must be "
                        "of same length."
                        "Expected {}, got {}.".format(
                            len(to_replace), len(replacement)
                        )
                    )
        else:
            if not is_scalar(replacement):
                raise TypeError(
                    "Incompatible types '{}' and '{}' "
                    "for *to_replace* and *replacement*.".format(
                        type(to_replace).__name__, type(replacement).__name__
                    )
                )
            to_replace = [to_replace]
            replacement = [replacement]

        if is_dict_like(to_replace) or is_dict_like(replacement):
            raise TypeError("Dict-like args not supported in Series.replace()")

        if isinstance(replacement, list):
            all_nan = replacement.count(None) == len(replacement)
        result = self._column.find_and_replace(
            to_replace, replacement, all_nan
        )

        return self._copy_construct(data=result)

    def reverse(self):
        """Reverse the Series
        """
        rinds = cudautils.arange_reversed(self._column.size, dtype=np.int32)
        col = self._column[rinds]
        index = self.index.as_column()[rinds]
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
        A sequence of new series for each category.  Its length is determined
        by the length of ``cats``.
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
               smallest possible integer dtype (starting with np.int32)
               is used.
        na_sentinel : number
            Value to indicate missing category.
        Returns
        -------
        A sequence of encoded labels with value between 0 and n-1 classes(cats)
        """
        from cudf import DataFrame

        if dtype is None:
            dtype = min_scalar_type(len(cats), 32)

        cats = Series(cats).astype(self.dtype)
        order = Series(cudautils.arange(len(self)))
        codes = Series(cudautils.arange(len(cats), dtype=dtype))

        value = DataFrame({"value": cats, "code": codes})
        codes = DataFrame({"value": self.copy(), "order": order})
        codes = codes.merge(value, on="value", how="left")
        codes = codes.sort_values("order")["code"].fillna(na_sentinel)

        cats.name = None  # because it was mutated above

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
        """Apply a elemenwise function to transform the values in the Column.

        The user function is expected to take one argument and return the
        result, which will be stored to the output Series.  The function
        cannot reference globals except for other simple scalar objects.

        Parameters
        ----------
        udf : Either a callable python function or a python function already
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

        """
        if callable(udf):
            res_col = self._unaryop(udf)
        else:
            res_col = self._column.applymap(udf, out_dtype=out_dtype)
        return self._copy_construct(data=res_col)

    # Find / Search

    def find_first_value(self, value):
        """
        Returns offset of first value that matches
        """
        return self._column.find_first_value(value)

    def find_last_value(self, value):
        """
        Returns offset of last value that matches
        """
        return self._column.find_last_value(value)

    #
    # Stats
    #
    def count(self, axis=None, skipna=True):
        """The number of non-null values"""
        assert axis in (None, 0) and skipna is True
        return self.valid_count

    def min(self, axis=None, skipna=True, dtype=None):
        """Compute the min of the series
        """
        assert axis in (None, 0) and skipna is True
        return self._column.min(dtype=dtype)

    def max(self, axis=None, skipna=True, dtype=None):
        """Compute the max of the series
        """
        assert axis in (None, 0) and skipna is True
        return self._column.max(dtype=dtype)

    def sum(self, axis=None, skipna=True, dtype=None):
        """Compute the sum of the series"""
        assert axis in (None, 0) and skipna is True
        return self._column.sum(dtype=dtype)

    def product(self, axis=None, skipna=True, dtype=None):
        """Compute the product of the series"""
        assert axis in (None, 0) and skipna is True
        return self._column.product(dtype=dtype)

    def prod(self, axis=None, skipna=True, dtype=None):
        """Alias for product"""
        return self.product(axis=axis, skipna=skipna, dtype=dtype)

    def cummin(self, axis=0, skipna=True):
        """Compute the cumulative minimum of the series"""
        assert axis in (None, 0) and skipna is True
        return Series(
            self._column._apply_scan_op("min"),
            name=self.name,
            index=self.index,
        )

    def cummax(self, axis=0, skipna=True):
        """Compute the cumulative maximum of the series"""
        assert axis in (None, 0) and skipna is True
        return Series(
            self._column._apply_scan_op("max"),
            name=self.name,
            index=self.index,
        )

    def cumsum(self, axis=0, skipna=True):
        """Compute the cumulative sum of the series"""
        assert axis in (None, 0) and skipna is True

        # pandas always returns int64 dtype if original dtype is int or `bool`
        if np.issubdtype(self.dtype, np.integer) or np.issubdtype(
            self.dtype, np.bool_
        ):
            return Series(
                self.astype(np.int64)._column._apply_scan_op("sum"),
                name=self.name,
                index=self.index,
            )
        else:
            return Series(
                self._column._apply_scan_op("sum"),
                name=self.name,
                index=self.index,
            )

    def cumprod(self, axis=0, skipna=True):
        """Compute the cumulative product of the series"""
        assert axis in (None, 0) and skipna is True

        # pandas always returns int64 dtype if original dtype is int or `bool`
        if np.issubdtype(self.dtype, np.integer) or np.issubdtype(
            self.dtype, np.bool_
        ):
            return Series(
                self.astype(np.int64)._column._apply_scan_op("product"),
                name=self.name,
                index=self.index,
            )
        else:
            return Series(
                self._column._apply_scan_op("product"),
                name=self.name,
                index=self.index,
            )

    def mean(self, axis=None, skipna=True):
        """Compute the mean of the series
        """
        assert axis in (None, 0) and skipna is True
        return self._column.mean()

    def std(self, ddof=1, axis=None, skipna=True):
        """Compute the standard deviation of the series
        """
        assert axis in (None, 0) and skipna is True
        return self._column.std(ddof=ddof)

    def var(self, ddof=1, axis=None, skipna=True):
        """Compute the variance of the series
        """
        assert axis in (None, 0) and skipna is True
        return self._column.var(ddof=ddof)

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

    def kurtosis(self, axis=None, skipna=None, level=None, numeric_only=None):
        """Calculates Fisher's unbiased kurtosis of a sample.
        """
        assert (
            axis in (None, 0)
            and skipna in (None, True)
            and level in (None,)
            and numeric_only in (None, True)
        )

        if self.empty:
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

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None):
        """Calculates the unbiased Fisher-Pearson skew of a sample.
        """
        assert (
            axis in (None, 0)
            and skipna in (None, True)
            and level in (None,)
            and numeric_only in (None, True)
        )

        if self.empty:
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
        """Calculates the sample covariance between two Series,
        excluding missing values.
        """
        assert min_periods in (None,)

        if self.empty or other.empty:
            return np.nan

        lhs = self.nans_to_nulls().dropna()
        rhs = other.nans_to_nulls().dropna()
        lhs, rhs = _align_indices(lhs, rhs, join="inner")

        if lhs.empty or rhs.empty or (len(lhs) == 1 and len(rhs) == 1):
            return np.nan

        result = (lhs - lhs.mean()) * (rhs - rhs.mean())
        cov_sample = result.sum() / (len(lhs) - 1)
        return cov_sample

    def corr(self, other, method="pearson", min_periods=None):
        """Calculates the sample correlation between two Series,
        excluding missing values.
        """
        assert method in ("pearson",) and min_periods in (None,)

        if self.empty or other.empty:
            return np.nan

        lhs = self.nans_to_nulls().dropna()
        rhs = other.nans_to_nulls().dropna()
        lhs, rhs = _align_indices(lhs, rhs, join="inner")

        if lhs.empty or rhs.empty:
            return np.nan

        cov = lhs.cov(rhs)
        lhs_std, rhs_std = lhs.std(), rhs.std()

        if not cov or lhs_std == 0 or rhs_std == 0:
            return np.nan
        return cov / lhs_std / rhs_std

    def isin(self, test):
        from cudf import DataFrame

        lhs = self
        rhs = None

        try:
            rhs = column.as_column(test, dtype=self.dtype)
            # if necessary, convert values via typecast
            rhs = Series(rhs.astype(self.dtype))
        except Exception:
            # pandas functionally returns all False when cleansing via
            # typecasting fails
            return Series(cudautils.zeros(len(self), dtype="bool"))

        # If categorical, combine categories first
        if is_categorical_dtype(lhs):
            lhs_cats = lhs.cat.categories.as_column()
            rhs_cats = rhs.cat.categories.as_column()
            if np.issubdtype(rhs_cats.dtype, lhs_cats.dtype):
                # if the categories are the same dtype, we can combine them
                cats = Series(lhs_cats.append(rhs_cats)).drop_duplicates()
                lhs = lhs.cat.set_categories(cats, is_unique=True)
                rhs = rhs.cat.set_categories(cats, is_unique=True)
            else:
                # If they're not the same dtype, short-circuit if the test
                # list doesn't have any nulls. If it does have nulls, make
                # the test list a Categorical with a single null
                if not rhs.has_nulls:
                    return Series(cudautils.zeros(len(self), dtype="bool"))
                rhs = Series(pd.Categorical.from_codes([-1], categories=[]))
                rhs = rhs.cat.set_categories(lhs_cats).astype(self.dtype)

        # fillna so we can find nulls
        if rhs.has_nulls:
            lhs = lhs.fillna(lhs._column.default_na_value())
            rhs = rhs.fillna(lhs._column.default_na_value())

        lhs = DataFrame({"x": lhs, "orig_order": cudautils.arange(len(lhs))})
        rhs = DataFrame({"x": rhs, "bool": cudautils.ones(len(rhs), "bool")})
        res = lhs.merge(rhs, on="x", how="left").sort_values(by="orig_order")
        res = res.drop_duplicates(subset="orig_order").reset_index(drop=True)
        res = res["bool"].fillna(False)
        res.name = self.name

        return res

    def unique_k(self, k):
        warnings.warn("Use .unique() instead", DeprecationWarning)
        return self.unique()

    def unique(self, method="sort", sort=True):
        """Returns unique values of this Series.
        default='sort' will be changed to 'hash' when implemented.
        """
        if method != "sort":
            msg = "non sort based unique() not implemented yet"
            raise NotImplementedError(msg)
        if not sort:
            msg = "not sorted unique not implemented yet."
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            res = column.column_empty_like(self._column, newsize=0)
            return self._copy_construct(data=res)
        res = self._column.unique(method=method)
        return Series(res, name=self.name)

    def nunique(self, method="sort", dropna=True):
        """Returns the number of unique values of the Series: approximate version,
        and exact version to be moved to libgdf
        """
        if method != "sort":
            msg = "non sort based unique_count() not implemented yet"
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            return 0
        return self._column.unique_count(method=method, dropna=dropna)
        # return len(self._column.unique())

    def value_counts(self, sort=True):
        """Returns unique values of this Series.
        """

        if self.null_count == len(self):
            return Series(np.array([], dtype=np.int32), name=self.name)

        res = self.groupby(self).count()
        res.index.name = None

        if sort:
            return res.sort_values(ascending=False)
        return res

    def scale(self):
        """Scale values to [0, 1] in float64
        """
        if self.has_nulls:
            msg = "masked series not supported by this operation"
            raise NotImplementedError(msg)

        vmin = self.min()
        vmax = self.max()
        gpuarr = self._column.data_array_view
        scaled = cudautils.compute_scale(gpuarr, vmin, vmax)
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
        from cudf.core.column import numerical

        return Series(numerical.column_hash_values(self._column))

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
        result: Series
            The encoded Series.
        """
        assert stop > 0

        from cudf.core.column import numerical

        initial_hash = np.asarray(hash(self.name)) if use_name else None
        hashed_values = numerical.column_hash_values(
            self._column, initial_hash_values=initial_hash
        )

        if hashed_values.has_nulls:
            raise ValueError("Column must have no nulls.")

        # TODO: Binary op when https://github.com/rapidsai/cudf/pull/892 merged
        mod_vals = cudautils.modulo(hashed_values.to_gpu_array(), stop)
        return Series(mod_vals, index=self.index)

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
            if len(res) == 0:
                return np.nan
            else:
                # if q is an int/float, we shouldn't be constructing series
                return res.pop()

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

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """Shift values of an input array by periods positions and store the
        output in a new array.

        Notes
        -----
        Shift currently only supports float and integer dtype columns with
        no null values.
        """
        assert axis in (None, 0) and freq is None and fill_value is None

        if self.has_nulls:
            raise AssertionError(
                "Shift currently requires columns with no " "null values"
            )

        if not np.issubdtype(self.dtype, np.number):
            raise NotImplementedError(
                "Shift currently only supports " "numeric dtypes"
            )
        if periods == 0:
            return self

        input_dary = self.to_gpu_array()
        output_dary = rmm.device_array_like(input_dary)
        if output_dary.size > 0:
            cudautils.gpu_shift.forall(output_dary.size)(
                input_dary, output_dary, periods
            )
        return Series(output_dary, name=self.name, index=self.index)

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
                "Diff currently requires columns with no " "null values"
            )

        if not np.issubdtype(self.dtype, np.number):
            raise NotImplementedError(
                "Diff currently only supports " "numeric dtypes"
            )

        input_dary = self.to_gpu_array()
        output_dary = rmm.device_array_like(input_dary)
        if output_dary.size > 0:
            cudautils.gpu_diff.forall(output_dary.size)(
                input_dary, output_dary, periods
            )
        return Series(output_dary, name=self.name, index=self.index)

    def groupby(
        self,
        by=None,
        group_series=None,
        level=None,
        sort=True,
        group_keys=True,
        as_index=None,
        dropna=True,
    ):
        if group_keys is not True:
            raise NotImplementedError(
                "The group_keys keyword is not yet implemented"
            )

        from cudf.core.groupby.groupby import SeriesGroupBy

        return SeriesGroupBy(
            self,
            by=by,
            level=level,
            sort=sort,
            as_index=as_index,
            dropna=dropna,
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

        json.to_json(self, path_or_buf=path_or_buf, *args, **kwargs)

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
        Alter Series name.

        Change Series.name with a scalar value.

        Parameters
        ----------
        index : Scalar, optional
            Scalar to alter the Series.name attribute
        copy : boolean, default True
            Also copy underlying data

        Returns
        -------
        Series

        Difference from pandas:
          * Supports scalar values only for changing name attribute
          * Not supporting: inplace, level
        """
        out = self.copy(deep=False)
        out = out.set_index(self.index)
        if index:
            out.name = index

        return out.copy(deep=copy)

    def searchsorted(self, value, side="left"):
        """Find indices where elements should be inserted to maintain order

        Parameters
        ----------
        value : array_like
            Column of values to search for
        side : str {‘left’, ‘right’} optional
            If ‘left’, the index of the first suitable location found is given.
            If ‘right’, return the last such index

        Returns
        -------
        A Column of insertion points with the same shape as value
        """
        outcol = self._column.searchsorted(value, side)
        return Series(outcol)

    @property
    def is_unique(self):
        return self._column.is_unique

    @property
    def is_monotonic(self):
        return self._column.is_monotonic_increasing

    @property
    def is_monotonic_increasing(self):
        return self._column.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        return self._column.is_monotonic_decreasing

    @property
    def __cuda_array_interface__(self):
        return self._column.__cuda_array_interface__

    def repeat(self, repeats, axis=None):
        assert axis in (None, 0)
        data = self._column.repeat(repeats)
        new_index = self.index.repeat(repeats)
        return Series(data, index=new_index, name=self.name)


truediv_int_dtype_corrections = {
    "int16": "float32",
    "int32": "float32",
    "int64": "float64",
    "bool": "float32",
    "int": "float",
}


class DatetimeProperties(object):
    def __init__(self, series):
        self.series = series

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

    def get_dt_field(self, field):
        out_column = self.series._column.get_dt_field(field)
        return Series(
            data=out_column, index=self.series._index, name=self.series.name
        )


def _align_indices(lhs, rhs, join="outer"):
    """
    Internal util to align the indices of two Series. Returns a tuple of the
    aligned series, or the original arguments if the indices are the same, or
    if rhs isn't a Series.
    """
    if isinstance(rhs, Series) and not lhs.index.equals(rhs.index):
        lhs, rhs = lhs.to_frame(0), rhs.to_frame(1)
        lhs_rhs = lhs.join(rhs, how=join, sort=True)
        lhs = lhs_rhs.iloc[:, 0]
        rhs = lhs_rhs.iloc[:, 1]
    return lhs, rhs
