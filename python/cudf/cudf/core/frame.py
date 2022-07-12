# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from __future__ import annotations

import copy
import operator
import pickle
import warnings
from collections import abc
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
from cudf import _lib as libcudf
from cudf._typing import Dtype
from cudf.api.types import is_dtype_equal, is_scalar
from cudf.core.column import (
    ColumnBase,
    as_column,
    build_categorical_column,
    deserialize_columns,
    serialize_columns,
)
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.mixins import BinaryOperand, Scannable
from cudf.core.window import Rolling
from cudf.utils import ioutils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import find_common_type
from cudf.utils.utils import _array_ufunc, _cudf_nvtx_annotate

T = TypeVar("T", bound="Frame")


# TODO: It looks like Frame is missing a declaration of `copy`, need to add
class Frame(BinaryOperand, Scannable):
    """A collection of Column objects with an optional index.

    Parameters
    ----------
    data : dict
        An dict mapping column names to Columns
    index : Table
        A Frame representing the (optional) index columns.
    """

    _data: "ColumnAccessor"

    _VALID_BINARY_OPERATIONS = BinaryOperand._SUPPORTED_BINARY_OPERATIONS

    def __init__(self, data=None):
        if data is None:
            data = {}
        self._data = cudf.core.column_accessor.ColumnAccessor(data)

    @property
    def _num_columns(self) -> int:
        return len(self._data)

    @property
    def _num_rows(self) -> int:
        return 0 if self._num_columns == 0 else len(self._data.columns[0])

    @property
    def _column_names(self) -> Tuple[Any, ...]:  # TODO: Tuple[str]?
        return tuple(self._data.names)

    @property
    def _columns(self) -> Tuple[Any, ...]:  # TODO: Tuple[Column]?
        return tuple(self._data.columns)

    def serialize(self):
        header = {
            "type-serialized": pickle.dumps(type(self)),
            "column_names": pickle.dumps(tuple(self._data.names)),
        }
        header["columns"], frames = serialize_columns(self._columns)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        cls_deserialize = pickle.loads(header["type-serialized"])
        column_names = pickle.loads(header["column_names"])
        columns = deserialize_columns(header["columns"], frames)
        return cls_deserialize._from_data(dict(zip(column_names, columns)))

    @classmethod
    @_cudf_nvtx_annotate
    def _from_data(cls, data: MutableMapping):
        obj = cls.__new__(cls)
        Frame.__init__(obj, data)
        return obj

    @_cudf_nvtx_annotate
    def _from_data_like_self(self, data: MutableMapping):
        return self._from_data(data)

    @classmethod
    @_cudf_nvtx_annotate
    def _from_columns(
        cls,
        columns: List[ColumnBase],
        column_names: abc.Iterable[str],
    ):
        """Construct a `Frame` object from a list of columns."""
        data = {name: columns[i] for i, name in enumerate(column_names)}
        return cls._from_data(data)

    @_cudf_nvtx_annotate
    def _from_columns_like_self(
        self,
        columns: List[ColumnBase],
        column_names: Optional[abc.Iterable[str]] = None,
    ):
        """Construct a Frame from a list of columns with metadata from self.

        If `column_names` is None, use column names from self.
        """
        if column_names is None:
            column_names = self._column_names
        frame = self.__class__._from_columns(columns, column_names)
        return frame._copy_type_metadata(self)

    def _mimic_inplace(
        self: T, result: T, inplace: bool = False
    ) -> Optional[Frame]:
        if inplace:
            for col in self._data:
                if col in result._data:
                    self._data[col]._mimic_inplace(
                        result._data[col], inplace=True
                    )
            self._data = result._data
            return None
        else:
            return result

    @property
    def size(self):
        """
        Return the number of elements in the underlying data.

        Returns
        -------
        size : Size of the DataFrame / Index / Series / MultiIndex

        Examples
        --------
        Size of an empty dataframe is 0.

        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df
        Empty DataFrame
        Columns: []
        Index: []
        >>> df.size
        0
        >>> df = cudf.DataFrame(index=[1, 2, 3])
        >>> df
        Empty DataFrame
        Columns: []
        Index: [1, 2, 3]
        >>> df.size
        0

        DataFrame with values

        >>> df = cudf.DataFrame({'a': [10, 11, 12],
        ...         'b': ['hello', 'rapids', 'ai']})
        >>> df
            a       b
        0  10   hello
        1  11  rapids
        2  12      ai
        >>> df.size
        6
        >>> df.index
        RangeIndex(start=0, stop=3)
        >>> df.index.size
        3

        Size of an Index

        >>> index = cudf.Index([])
        >>> index
        Float64Index([], dtype='float64')
        >>> index.size
        0
        >>> index = cudf.Index([1, 2, 3, 10])
        >>> index
        Int64Index([1, 2, 3, 10], dtype='int64')
        >>> index.size
        4

        Size of a MultiIndex

        >>> midx = cudf.MultiIndex(
        ...                 levels=[["a", "b", "c", None], ["1", None, "5"]],
        ...                 codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...                 names=["x", "y"],
        ...             )
        >>> midx
        MultiIndex([( 'a',  '1'),
                    ( 'a',  '5'),
                    ( 'b', <NA>),
                    ( 'c', <NA>),
                    (<NA>,  '1')],
                   names=['x', 'y'])
        >>> midx.size
        5
        """
        return self._num_columns * self._num_rows

    @property
    def shape(self):
        """Returns a tuple representing the dimensionality of the DataFrame."""
        return self._num_rows, self._num_columns

    @property
    def empty(self):
        """
        Indicator whether DataFrame or Series is empty.

        True if DataFrame/Series is entirely empty (no items),
        meaning any of the axes are of length 0.

        Returns
        -------
        out : bool
            If DataFrame/Series is empty, return True, if not return False.

        Notes
        -----
        If DataFrame/Series contains only `null` values, it is still not
        considered empty. See the example below.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'A' : []})
        >>> df
        Empty DataFrame
        Columns: [A]
        Index: []
        >>> df.empty
        True

        If we only have `null` values in our DataFrame, it is
        not considered empty! We will need to drop
        the `null`'s to make the DataFrame empty:

        >>> df = cudf.DataFrame({'A' : [None, None]})
        >>> df
              A
        0  <NA>
        1  <NA>
        >>> df.empty
        False
        >>> df.dropna().empty
        True

        Non-empty and empty Series example:

        >>> s = cudf.Series([1, 2, None])
        >>> s
        0       1
        1       2
        2    <NA>
        dtype: int64
        >>> s.empty
        False
        >>> s = cudf.Series([])
        >>> s
        Series([], dtype: float64)
        >>> s.empty
        True
        """
        return self.size == 0

    def memory_usage(self, deep=False):
        """Return the memory usage of an object.

        Parameters
        ----------
        deep : bool
            The deep parameter is ignored and is only included for pandas
            compatibility.

        Returns
        -------
        The total bytes used.
        """
        raise NotImplementedError

    def __len__(self):
        return self._num_rows

    @_cudf_nvtx_annotate
    def astype(self, dtype, copy=False, **kwargs):
        result = {}
        for col_name, col in self._data.items():
            dt = dtype.get(col_name, col.dtype)
            if not is_dtype_equal(dt, col.dtype):
                result[col_name] = col.astype(dt, copy=copy, **kwargs)
            else:
                result[col_name] = col.copy() if copy else col

        return result

    @_cudf_nvtx_annotate
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

        Comparing Series with `equals`:

        >>> s = cudf.Series([1, 2, 3])
        >>> other = cudf.Series([1, 2, 3])
        >>> s.equals(other)
        True
        >>> different = cudf.Series([1.5, 2, 3])
        >>> s.equals(different)
        False

        Comparing DataFrames with `equals`:

        >>> df = cudf.DataFrame({1: [10], 2: [20]})
        >>> df
            1   2
        0  10  20
        >>> exactly_equal = cudf.DataFrame({1: [10], 2: [20]})
        >>> exactly_equal
            1   2
        0  10  20
        >>> df.equals(exactly_equal)
        True

        For two DataFrames to compare equal, the types of column
        values must be equal, but the types of column labels
        need not:

        >>> different_column_type = cudf.DataFrame({1.0: [10], 2.0: [20]})
        >>> different_column_type
           1.0  2.0
        0   10   20
        >>> df.equals(different_column_type)
        True
        """
        if self is other:
            return True
        if (
            other is None
            or not isinstance(other, type(self))
            or len(self) != len(other)
        ):
            return False

        return all(
            self_col.equals(other_col, check_dtypes=True)
            for self_col, other_col in zip(
                self._data.values(), other._data.values()
            )
        )

    @_cudf_nvtx_annotate
    def _get_columns_by_label(self, labels, downcast=False):
        """
        Returns columns of the Frame specified by `labels`

        """
        return self._data.select_by_label(labels)

    @property
    def values(self):
        """
        Return a CuPy representation of the DataFrame.

        Only the values in the DataFrame will be returned, the axes labels will
        be removed.

        Returns
        -------
        cupy.ndarray
            The values of the DataFrame.
        """
        return self.to_cupy()

    @property
    def values_host(self):
        """
        Return a NumPy representation of the data.

        Only the values in the DataFrame will be returned, the axes labels will
        be removed.

        Returns
        -------
        numpy.ndarray
            A host representation of the underlying data.
        """
        return self.to_numpy()

    def __array__(self, dtype=None):
        raise TypeError(
            "Implicit conversion to a host NumPy array via __array__ is not "
            "allowed, To explicitly construct a GPU matrix, consider using "
            ".to_cupy()\nTo explicitly construct a host matrix, consider "
            "using .to_numpy()."
        )

    def __arrow_array__(self, type=None):
        raise TypeError(
            "Implicit conversion to a host PyArrow object via __arrow_array__ "
            "is not allowed. Consider using .to_arrow()"
        )

    def _to_array(
        self,
        get_column_values: Callable,
        make_empty_matrix: Callable,
        dtype: Union[Dtype, None] = None,
        na_value=None,
    ) -> Union[cupy.ndarray, np.ndarray]:
        # Internal function to implement to_cupy and to_numpy, which are nearly
        # identical except for the attribute they access to generate values.

        def get_column_values_na(col):
            if na_value is not None:
                col = col.fillna(na_value)
            return get_column_values(col)

        # Early exit for an empty Frame.
        ncol = self._num_columns
        if ncol == 0:
            return make_empty_matrix(
                shape=(0, 0), dtype=np.dtype("float64"), order="F"
            )

        if dtype is None:
            dtype = find_common_type(
                [col.dtype for col in self._data.values()]
            )

        matrix = make_empty_matrix(
            shape=(len(self), ncol), dtype=dtype, order="F"
        )
        for i, col in enumerate(self._data.values()):
            # TODO: col.values may fail if there is nullable data or an
            # unsupported dtype. We may want to catch and provide a more
            # suitable error.
            matrix[:, i] = get_column_values_na(col)
        return matrix

    # TODO: As of now, calling cupy.asarray is _much_ faster than calling
    # to_cupy. We should investigate the reasons why and whether we can provide
    # a more efficient method here by exploiting __cuda_array_interface__. In
    # particular, we need to benchmark how much of the overhead is coming from
    # (potentially unavoidable) local copies in to_cupy and how much comes from
    # inefficiencies in the implementation.
    @_cudf_nvtx_annotate
    def to_cupy(
        self,
        dtype: Union[Dtype, None] = None,
        copy: bool = False,
        na_value=None,
    ) -> cupy.ndarray:
        """Convert the Frame to a CuPy array.

        Parameters
        ----------
        dtype : str or :class:`numpy.dtype`, optional
            The dtype to pass to :func:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_cupy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, default None
            The value to use for missing values. The default value depends on
            dtype and the dtypes of the DataFrame columns.

        Returns
        -------
        cupy.ndarray
        """
        return self._to_array(
            (lambda col: col.values.copy())
            if copy
            else (lambda col: col.values),
            cupy.empty,
            dtype,
            na_value,
        )

    @_cudf_nvtx_annotate
    def to_numpy(
        self,
        dtype: Union[Dtype, None] = None,
        copy: bool = True,
        na_value=None,
    ) -> np.ndarray:
        """Convert the Frame to a NumPy array.

        Parameters
        ----------
        dtype : str or :class:`numpy.dtype`, optional
            The dtype to pass to :func:`numpy.asarray`.
        copy : bool, default True
            Whether to ensure that the returned value is not a view on
            another array. This parameter must be ``True`` since cuDF must copy
            device memory to host to provide a numpy array.
        na_value : Any, default None
            The value to use for missing values. The default value depends on
            dtype and the dtypes of the DataFrame columns.

        Returns
        -------
        numpy.ndarray
        """
        if not copy:
            raise ValueError(
                "copy=False is not supported because conversion to a numpy "
                "array always copies the data."
            )

        return self._to_array(
            (lambda col: col.values_host), np.empty, dtype, na_value
        )

    @_cudf_nvtx_annotate
    def where(self, cond, other=None, inplace=False):
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : bool Series/DataFrame, array-like
            Where cond is True, keep the original value.
            Where False, replace with corresponding value from other.
            Callables are not supported.
        other: scalar, list of scalars, Series/DataFrame
            Entries where cond is False are replaced with
            corresponding value from other. Callables are not
            supported. Default is None.

            DataFrame expects only Scalar or array like with scalars or
            dataframe with same dimension as self.

            Series expects only scalar or series like with same length
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        Same type as caller

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A":[1, 4, 5], "B":[3, 5, 8]})
        >>> df.where(df % 2 == 0, [-1, -1])
           A  B
        0 -1 -1
        1  4 -1
        2 -1  8

        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> ser.where(ser > 2, 10)
        0     4
        1     3
        2    10
        3    10
        4    10
        dtype: int64
        >>> ser.where(ser > 2)
        0       4
        1       3
        2    <NA>
        3    <NA>
        4    <NA>
        dtype: int64
        """
        raise NotImplementedError

    @_cudf_nvtx_annotate
    def mask(self, cond, other=None, inplace=False):
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : bool Series/DataFrame, array-like
            Where cond is False, keep the original value.
            Where True, replace with corresponding value from other.
            Callables are not supported.
        other: scalar, list of scalars, Series/DataFrame
            Entries where cond is True are replaced with
            corresponding value from other. Callables are not
            supported. Default is None.

            DataFrame expects only Scalar or array like with scalars or
            dataframe with same dimension as self.

            Series expects only scalar or series like with same length
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        Same type as caller

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A":[1, 4, 5], "B":[3, 5, 8]})
        >>> df.mask(df % 2 == 0, [-1, -1])
           A  B
        0  1  3
        1 -1  5
        2  5 -1

        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> ser.mask(ser > 2, 10)
        0    10
        1    10
        2     2
        3     1
        4     0
        dtype: int64
        >>> ser.mask(ser > 2)
        0    <NA>
        1    <NA>
        2       2
        3       1
        4       0
        dtype: int64
        """

        if not hasattr(cond, "__invert__"):
            # We Invert `cond` below and call `where`, so
            # making sure the object supports
            # `~`(inversion) operator or `__invert__` method
            cond = cupy.asarray(cond)

        return self.where(cond=~cond, other=other, inplace=inplace)

    @_cudf_nvtx_annotate
    def pipe(self, func, *args, **kwargs):
        """
        Apply ``func(self, *args, **kwargs)``.

        Parameters
        ----------
        func : function
            Function to apply to the Series/DataFrame/Index.
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the Series/DataFrame/Index.
        args : iterable, optional
            Positional arguments passed into ``func``.
        kwargs : mapping, optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        Examples
        --------

        Use ``.pipe`` when chaining together functions that expect
        Series, DataFrames or GroupBy objects. Instead of writing

        >>> func(g(h(df), arg1=a), arg2=b, arg3=c)

        You can write

        >>> (df.pipe(h)
        ...    .pipe(g, arg1=a)
        ...    .pipe(func, arg2=b, arg3=c)
        ... )

        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``f`` takes its data as ``arg2``:

        >>> (df.pipe(h)
        ...    .pipe(g, arg1=a)
        ...    .pipe((func, 'arg2'), arg1=a, arg3=c)
        ...  )
        """
        return cudf.core.common.pipe(self, func, *args, **kwargs)

    @_cudf_nvtx_annotate
    def fillna(
        self, value=None, method=None, axis=None, inplace=False, limit=None
    ):
        """Fill null values with ``value`` or specified ``method``.

        Parameters
        ----------
        value : scalar, Series-like or dict
            Value to use to fill nulls. If Series-like, null values
            are filled with values in corresponding indices.
            A dict can be used to provide different values to fill nulls
            in different columns. Cannot be used with ``method``.

        method : {'ffill', 'bfill'}, default None
            Method to use for filling null values in the dataframe or series.
            `ffill` propagates the last non-null values forward to the next
            non-null value. `bfill` propagates backward with the next non-null
            value. Cannot be used with ``value``.

        Returns
        -------
        result : DataFrame, Series, or Index
            Copy with nulls filled.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, None], 'b': [3, None, 5]})
        >>> df
              a     b
        0     1     3
        1     2  <NA>
        2  <NA>     5
        >>> df.fillna(4)
           a  b
        0  1  3
        1  2  4
        2  4  5
        >>> df.fillna({'a': 3, 'b': 4})
           a  b
        0  1  3
        1  2  4
        2  3  5

        ``fillna`` on a Series object:

        >>> ser = cudf.Series(['a', 'b', None, 'c'])
        >>> ser
        0       a
        1       b
        2    <NA>
        3       c
        dtype: object
        >>> ser.fillna('z')
        0    a
        1    b
        2    z
        3    c
        dtype: object

        ``fillna`` can also supports inplace operation:

        >>> ser.fillna('z', inplace=True)
        >>> ser
        0    a
        1    b
        2    z
        3    c
        dtype: object
        >>> df.fillna({'a': 3, 'b': 4}, inplace=True)
        >>> df
           a  b
        0  1  3
        1  2  4
        2  3  5

        ``fillna`` specified with fill ``method``

        >>> ser = cudf.Series([1, None, None, 2, 3, None, None])
        >>> ser.fillna(method='ffill')
        0    1
        1    1
        2    1
        3    2
        4    3
        5    3
        6    3
        dtype: int64
        >>> ser.fillna(method='bfill')
        0       1
        1       2
        2       2
        3       2
        4       3
        5    <NA>
        6    <NA>
        dtype: int64
        """
        if limit is not None:
            raise NotImplementedError("The limit keyword is not supported")
        if axis:
            raise NotImplementedError("The axis keyword is not supported")

        if value is not None and method is not None:
            raise ValueError("Cannot specify both 'value' and 'method'.")

        if method:
            if method not in {"ffill", "bfill", "pad", "backfill"}:
                raise NotImplementedError(
                    f"Fill method {method} is not supported"
                )
            if method == "pad":
                method = "ffill"
            elif method == "backfill":
                method = "bfill"

        # TODO: This logic should be handled in different subclasses since
        # different Frames support different types of values.
        if isinstance(value, cudf.Series):
            value = value.reindex(self._data.names)
        elif isinstance(value, cudf.DataFrame):
            if not self.index.equals(value.index):
                value = value.reindex(self.index)
            else:
                value = value
        elif not isinstance(value, abc.Mapping):
            value = {name: copy.deepcopy(value) for name in self._data.names}
        else:
            value = {
                key: value.reindex(self.index)
                if isinstance(value, cudf.Series)
                else value
                for key, value in value.items()
            }

        filled_data = {}
        for col_name, col in self._data.items():
            if col_name in value and method is None:
                replace_val = value[col_name]
            else:
                replace_val = None
            should_fill = (
                col_name in value
                and col.contains_na_entries
                and not libcudf.scalar._is_null_host_scalar(replace_val)
            ) or method is not None
            if should_fill:
                filled_data[col_name] = col.fillna(replace_val, method)
            else:
                filled_data[col_name] = col.copy(deep=True)

        return self._mimic_inplace(
            self._from_data(
                data=ColumnAccessor._create_unsafe(
                    data=filled_data,
                    multiindex=self._data.multiindex,
                    level_names=self._data.level_names,
                )
            ),
            inplace=inplace,
        )

    @_cudf_nvtx_annotate
    def _drop_column(self, name):
        """Drop a column by *name*"""
        if name not in self._data:
            raise KeyError(f"column '{name}' does not exist")
        del self._data[name]

    @_cudf_nvtx_annotate
    def _drop_na_columns(self, how="any", subset=None, thresh=None):
        """
        Drop columns containing nulls
        """
        out_cols = []

        if subset is None:
            df = self
        else:
            df = self.take(subset)

        if thresh is None:
            if how == "all":
                thresh = 1
            else:
                thresh = len(df)

        for name, col in df._data.items():
            try:
                check_col = col.nans_to_nulls()
            except AttributeError:
                check_col = col
            no_threshold_valid_count = (
                len(col) - check_col.null_count
            ) < thresh
            if no_threshold_valid_count:
                continue
            out_cols.append(name)

        return self[out_cols]

    @_cudf_nvtx_annotate
    def _quantiles(
        self,
        q,
        interpolation="LINEAR",
        is_sorted=False,
        column_order=(),
        null_precedence=(),
    ):
        interpolation = libcudf.types.Interpolation[interpolation]

        is_sorted = libcudf.types.Sorted["YES" if is_sorted else "NO"]

        column_order = [libcudf.types.Order[key] for key in column_order]

        null_precedence = [
            libcudf.types.NullOrder[key] for key in null_precedence
        ]

        return self._from_columns_like_self(
            libcudf.quantiles.quantiles(
                [*self._columns],
                q,
                interpolation,
                is_sorted,
                column_order,
                null_precedence,
            ),
            column_names=self._column_names,
        )

    @classmethod
    @_cudf_nvtx_annotate
    def from_arrow(cls, data):
        """Convert from PyArrow Table to Frame

        Parameters
        ----------
        data : PyArrow Table

        Raises
        ------
        TypeError for invalid input type.

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> data = pa.table({"a":[1, 2, 3], "b":[4, 5, 6]})
        >>> cudf.core.frame.Frame.from_arrow(data)
           a  b
        0  1  4
        1  2  5
        2  3  6
        """

        if not isinstance(data, (pa.Table)):
            raise TypeError(
                "To create a multicolumn cudf data, "
                "the data should be an arrow Table"
            )

        column_names = data.column_names
        pandas_dtypes = None
        np_dtypes = None
        if isinstance(data.schema.pandas_metadata, dict):
            metadata = data.schema.pandas_metadata
            pandas_dtypes = {
                col["field_name"]: col["pandas_type"]
                for col in metadata["columns"]
                if "field_name" in col
            }
            np_dtypes = {
                col["field_name"]: col["numpy_type"]
                for col in metadata["columns"]
                if "field_name" in col
            }

        # Currently we don't have support for
        # pyarrow.DictionaryArray -> cudf Categorical column,
        # so handling indices and dictionary as two different columns.
        # This needs be removed once we have hooked libcudf dictionary32
        # with categorical.
        dict_indices = {}
        dict_dictionaries = {}
        dict_ordered = {}
        for field in data.schema:
            if isinstance(field.type, pa.DictionaryType):
                dict_ordered[field.name] = field.type.ordered
                dict_indices[field.name] = pa.chunked_array(
                    [chunk.indices for chunk in data[field.name].chunks],
                    type=field.type.index_type,
                )
                dict_dictionaries[field.name] = pa.chunked_array(
                    [chunk.dictionary for chunk in data[field.name].chunks],
                    type=field.type.value_type,
                )

        # Handle dict arrays
        cudf_category_frame = {}
        if len(dict_indices):

            dict_indices_table = pa.table(dict_indices)
            data = data.drop(dict_indices_table.column_names)
            indices_columns = libcudf.interop.from_arrow(dict_indices_table)
            # as dictionary size can vary, it can't be a single table
            cudf_dictionaries_columns = {
                name: ColumnBase.from_arrow(dict_dictionaries[name])
                for name in dict_dictionaries.keys()
            }

            cudf_category_frame = {
                name: build_categorical_column(
                    cudf_dictionaries_columns[name],
                    codes,
                    mask=codes.base_mask,
                    size=codes.size,
                    ordered=dict_ordered[name],
                )
                for name, codes in zip(
                    dict_indices_table.column_names, indices_columns
                )
            }

        # Handle non-dict arrays
        cudf_non_category_frame = {
            name: col
            for name, col in zip(
                data.column_names, libcudf.interop.from_arrow(data)
            )
        }

        result = {**cudf_non_category_frame, **cudf_category_frame}

        # There are some special cases that need to be handled
        # based on metadata.
        if pandas_dtypes:
            for name in result:
                dtype = None
                if (
                    len(result[name]) == 0
                    and pandas_dtypes[name] == "categorical"
                ):
                    # When pandas_dtype is a categorical column and the size
                    # of column is 0(i.e., empty) then we will have an
                    # int8 column in result._data[name] returned by libcudf,
                    # which needs to be type-casted to 'category' dtype.
                    dtype = "category"
                elif (
                    pandas_dtypes[name] == "empty"
                    and np_dtypes[name] == "object"
                ):
                    # When a string column has all null values, pandas_dtype is
                    # is specified as 'empty' and np_dtypes as 'object',
                    # hence handling this special case to type-cast the empty
                    # float column to str column.
                    dtype = np_dtypes[name]
                elif pandas_dtypes[
                    name
                ] == "object" and cudf.api.types.is_struct_dtype(
                    np_dtypes[name]
                ):
                    # Incase of struct column, libcudf is not aware of names of
                    # struct fields, hence renaming the struct fields is
                    # necessary by extracting the field names from arrow
                    # struct types.
                    result[name] = result[name]._rename_fields(
                        [field.name for field in data[name].type]
                    )

                if dtype is not None:
                    result[name] = result[name].astype(dtype)

        return cls._from_data({name: result[name] for name in column_names})

    @_cudf_nvtx_annotate
    def to_arrow(self):
        """
        Convert to arrow Table

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame(
        ...     {"a":[1, 2, 3], "b":[4, 5, 6]}, index=[1, 2, 3])
        >>> df.to_arrow()
        pyarrow.Table
        a: int64
        b: int64
        index: int64
        ----
        a: [[1,2,3]]
        b: [[4,5,6]]
        index: [[1,2,3]]
        """
        return pa.Table.from_pydict(
            {name: col.to_arrow() for name, col in self._data.items()}
        )

    def _positions_from_column_names(self, column_names):
        """Map each column name into their positions in the frame.

        The order of indices returned corresponds to the column order in this
        Frame.
        """
        return [
            i
            for i, name in enumerate(self._column_names)
            if name in set(column_names)
        ]

    def _copy_type_metadata(self: T, other: T) -> T:
        """
        Copy type metadata from each column of `other` to the corresponding
        column of `self`.
        See `ColumnBase._with_type_metadata` for more information.
        """
        for name, col, other_col in zip(
            self._data.keys(), self._data.values(), other._data.values()
        ):
            self._data.set_by_label(
                name, col._with_type_metadata(other_col.dtype), validate=False
            )

        return self

    @_cudf_nvtx_annotate
    def isnull(self):
        """
        Identify missing values.

        Return a boolean same-sized object indicating if
        the values are ``<NA>``. ``<NA>`` values gets mapped to
        ``True`` values. Everything else gets mapped to
        ``False`` values. ``<NA>`` values include:

        * Values where null mask is set.
        * ``NaN`` in float dtype.
        * ``NaT`` in datetime64 and timedelta64 types.

        Characters such as empty strings ``''`` or
        ``inf`` incase of float are not
        considered ``<NA>`` values.

        Returns
        -------
        DataFrame/Series/Index
            Mask of bool values for each element in
            the object that indicates whether an element is an NA value.

        Examples
        --------

        Show which entries in a DataFrame are NA.

        >>> import cudf
        >>> import numpy as np
        >>> import pandas as pd
        >>> df = cudf.DataFrame({'age': [5, 6, np.NaN],
        ...                    'born': [pd.NaT, pd.Timestamp('1939-05-27'),
        ...                             pd.Timestamp('1940-04-25')],
        ...                    'name': ['Alfred', 'Batman', ''],
        ...                    'toy': [None, 'Batmobile', 'Joker']})
        >>> df
            age                        born    name        toy
        0     5                        <NA>  Alfred       <NA>
        1     6  1939-05-27 00:00:00.000000  Batman  Batmobile
        2  <NA>  1940-04-25 00:00:00.000000              Joker
        >>> df.isnull()
             age   born   name    toy
        0  False   True  False   True
        1  False  False  False  False
        2   True  False  False  False

        Show which entries in a Series are NA.

        >>> ser = cudf.Series([5, 6, np.NaN, np.inf, -np.inf])
        >>> ser
        0     5.0
        1     6.0
        2    <NA>
        3     Inf
        4    -Inf
        dtype: float64
        >>> ser.isnull()
        0    False
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        Show which entries in an Index are NA.

        >>> idx = cudf.Index([1, 2, None, np.NaN, 0.32, np.inf])
        >>> idx
        Float64Index([1.0, 2.0, <NA>, <NA>, 0.32, Inf], dtype='float64')
        >>> idx.isnull()
        GenericIndex([False, False, True, True, False, False], dtype='bool')
        """
        data_columns = (col.isnull() for col in self._columns)
        return self._from_data_like_self(zip(self._column_names, data_columns))

    # Alias for isnull
    isna = isnull

    @_cudf_nvtx_annotate
    def notnull(self):
        """
        Identify non-missing values.

        Return a boolean same-sized object indicating if
        the values are not ``<NA>``. Non-missing values get
        mapped to ``True``. ``<NA>`` values get mapped to
        ``False`` values. ``<NA>`` values include:

        * Values where null mask is set.
        * ``NaN`` in float dtype.
        * ``NaT`` in datetime64 and timedelta64 types.

        Characters such as empty strings ``''`` or
        ``inf`` incase of float are not
        considered ``<NA>`` values.

        Returns
        -------
        DataFrame/Series/Index
            Mask of bool values for each element in
            the object that indicates whether an element is not an NA value.

        Examples
        --------

        Show which entries in a DataFrame are NA.

        >>> import cudf
        >>> import numpy as np
        >>> import pandas as pd
        >>> df = cudf.DataFrame({'age': [5, 6, np.NaN],
        ...                    'born': [pd.NaT, pd.Timestamp('1939-05-27'),
        ...                             pd.Timestamp('1940-04-25')],
        ...                    'name': ['Alfred', 'Batman', ''],
        ...                    'toy': [None, 'Batmobile', 'Joker']})
        >>> df
            age                        born    name        toy
        0     5                        <NA>  Alfred       <NA>
        1     6  1939-05-27 00:00:00.000000  Batman  Batmobile
        2  <NA>  1940-04-25 00:00:00.000000              Joker
        >>> df.notnull()
             age   born  name    toy
        0   True  False  True  False
        1   True   True  True   True
        2  False   True  True   True

        Show which entries in a Series are NA.

        >>> ser = cudf.Series([5, 6, np.NaN, np.inf, -np.inf])
        >>> ser
        0     5.0
        1     6.0
        2    <NA>
        3     Inf
        4    -Inf
        dtype: float64
        >>> ser.notnull()
        0     True
        1     True
        2    False
        3     True
        4     True
        dtype: bool

        Show which entries in an Index are NA.

        >>> idx = cudf.Index([1, 2, None, np.NaN, 0.32, np.inf])
        >>> idx
        Float64Index([1.0, 2.0, <NA>, <NA>, 0.32, Inf], dtype='float64')
        >>> idx.notnull()
        GenericIndex([True, True, False, False, True, True], dtype='bool')
        """
        data_columns = (col.notnull() for col in self._columns)
        return self._from_data_like_self(zip(self._column_names, data_columns))

    # Alias for notnull
    notna = notnull

    @_cudf_nvtx_annotate
    def searchsorted(
        self, values, side="left", ascending=True, na_position="last"
    ):
        """Find indices where elements should be inserted to maintain order

        Parameters
        ----------
        value : Frame (Shape must be consistent with self)
            Values to be hypothetically inserted into Self
        side : str {‘left’, ‘right’} optional, default ‘left‘
            If ‘left’, the index of the first suitable location found is given
            If ‘right’, return the last such index
        ascending : bool optional, default True
            Sorted Frame is in ascending order (otherwise descending)
        na_position : str {‘last’, ‘first’} optional, default ‘last‘
            Position of null values in sorted order

        Returns
        -------
        1-D cupy array of insertion points

        Examples
        --------
        >>> s = cudf.Series([1, 2, 3])
        >>> s.searchsorted(4)
        3
        >>> s.searchsorted([0, 4])
        array([0, 3], dtype=int32)
        >>> s.searchsorted([1, 3], side='left')
        array([0, 2], dtype=int32)
        >>> s.searchsorted([1, 3], side='right')
        array([1, 3], dtype=int32)

        If the values are not monotonically sorted, wrong
        locations may be returned:

        >>> s = cudf.Series([2, 1, 3])
        >>> s.searchsorted(1)
        0   # wrong result, correct would be 1

        >>> df = cudf.DataFrame({'a': [1, 3, 5, 7], 'b': [10, 12, 14, 16]})
        >>> df
           a   b
        0  1  10
        1  3  12
        2  5  14
        3  7  16
        >>> values_df = cudf.DataFrame({'a': [0, 2, 5, 6],
        ... 'b': [10, 11, 13, 15]})
        >>> values_df
           a   b
        0  0  10
        1  2  17
        2  5  13
        3  6  15
        >>> df.searchsorted(values_df, ascending=False)
        array([4, 4, 4, 0], dtype=int32)
        """
        # Call libcudf++ search_sorted primitive

        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")

        scalar_flag = None
        if is_scalar(values):
            scalar_flag = True

        if not isinstance(values, Frame):
            values = [as_column(values)]
        else:
            values = [*values._columns]
        if len(values) != len(self._data):
            raise ValueError("Mismatch number of columns to search for.")

        sources = [
            col
            if is_dtype_equal(col.dtype, val.dtype)
            else col.astype(val.dtype)
            for col, val in zip(self._columns, values)
        ]
        outcol = libcudf.search.search_sorted(
            sources,
            values,
            side,
            ascending=ascending,
            na_position=na_position,
        )

        # Return result as cupy array if the values is non-scalar
        # If values is scalar, result is expected to be scalar.
        result = cupy.asarray(outcol.data_array_view)
        if scalar_flag:
            return result[0].item()
        else:
            return result

    @_cudf_nvtx_annotate
    def argsort(
        self,
        by=None,
        axis=0,
        kind="quicksort",
        order=None,
        ascending=True,
        na_position="last",
    ):
        """Return the integer indices that would sort the Series values.

        Parameters
        ----------
        by : str or list of str, default None
            Name or list of names to sort by. If None, sort by all columns.
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
        na_position : {‘first’ or ‘last’}, default ‘last’
            Argument ‘first’ puts NaNs at the beginning, ‘last’ puts NaNs
            at the end.

        Returns
        -------
        cupy.ndarray: The indices sorted based on input.

        Examples
        --------
        **Series**

        >>> import cudf
        >>> s = cudf.Series([3, 1, 2])
        >>> s
        0    3
        1    1
        2    2
        dtype: int64
        >>> s.argsort()
        0    1
        1    2
        2    0
        dtype: int32
        >>> s[s.argsort()]
        1    1
        2    2
        0    3
        dtype: int64

        **DataFrame**
        >>> import cudf
        >>> df = cudf.DataFrame({'foo': [3, 1, 2]})
        >>> df.argsort()
        array([1, 2, 0], dtype=int32)

        **Index**
        >>> import cudf
        >>> idx = cudf.Index([3, 1, 2])
        >>> idx.argsort()
        array([1, 2, 0], dtype=int32)
        """  # noqa: E501
        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")
        if kind != "quicksort":
            if kind not in {"mergesort", "heapsort", "stable"}:
                raise AttributeError(
                    f"{kind} is not a valid sorting algorithm for "
                    f"'DataFrame' object"
                )
            warnings.warn(
                f"GPU-accelerated {kind} is currently not supported, "
                "defaulting to quicksort."
            )

        if isinstance(by, str):
            by = [by]
        return self._get_sorted_inds(
            by=by, ascending=ascending, na_position=na_position
        ).values

    def _get_sorted_inds(self, by=None, ascending=True, na_position="last"):
        # Get an int64 column consisting of the indices required to sort self
        # according to the columns specified in by.

        to_sort = [
            *(
                self
                if by is None
                else self._get_columns_by_label(list(by), downcast=False)
            )._columns
        ]

        # If given a scalar need to construct a sequence of length # of columns
        if np.isscalar(ascending):
            ascending = [ascending] * len(to_sort)

        return libcudf.sort.order_by(to_sort, ascending, na_position)

    @_cudf_nvtx_annotate
    def abs(self):
        """
        Return a Series/DataFrame with absolute numeric value of each element.

        This function only applies to elements that are all numeric.

        Returns
        -------
        DataFrame/Series
            Absolute value of each element.

        Examples
        --------
        Absolute numeric values in a Series

        >>> s = cudf.Series([-1.10, 2, -3.33, 4])
        >>> s.abs()
        0    1.10
        1    2.00
        2    3.33
        3    4.00
        dtype: float64
        """
        return self._unaryop("abs")

    @_cudf_nvtx_annotate
    def _is_sorted(self, ascending=None, null_position=None):
        """
        Returns a boolean indicating whether the data of the Frame are sorted
        based on the parameters given. Does not account for the index.

        Parameters
        ----------
        self : Frame
            Frame whose columns are to be checked for sort order
        ascending : None or list-like of booleans
            None or list-like of boolean values indicating expected sort order
            of each column. If list-like, size of list-like must be
            len(columns). If None, all columns expected sort order is set to
            ascending. False (0) - ascending, True (1) - descending.
        null_position : None or list-like of booleans
            None or list-like of boolean values indicating desired order of
            nulls compared to other elements. If list-like, size of list-like
            must be len(columns). If None, null order is set to before. False
            (0) - before, True (1) - after.

        Returns
        -------
        returns : boolean
            Returns True, if sorted as expected by ``ascending`` and
            ``null_position``, False otherwise.
        """
        if ascending is not None and not cudf.api.types.is_list_like(
            ascending
        ):
            raise TypeError(
                f"Expected a list-like or None for `ascending`, got "
                f"{type(ascending)}"
            )
        if null_position is not None and not cudf.api.types.is_list_like(
            null_position
        ):
            raise TypeError(
                f"Expected a list-like or None for `null_position`, got "
                f"{type(null_position)}"
            )
        return libcudf.sort.is_sorted(
            [*self._columns], ascending=ascending, null_position=null_position
        )

    @_cudf_nvtx_annotate
    def _split(self, splits):
        """Split a frame with split points in ``splits``. Returns a list of
        Frames of length `len(splits) + 1`.
        """
        return [
            self._from_columns_like_self(
                libcudf.copying.columns_split([*self._data.columns], splits)[
                    split_idx
                ],
                self._column_names,
            )
            for split_idx in range(len(splits) + 1)
        ]

    @_cudf_nvtx_annotate
    def _encode(self):
        columns, indices = libcudf.transform.table_encode([*self._columns])
        keys = self._from_columns_like_self(columns)
        return keys, indices

    @_cudf_nvtx_annotate
    def _unaryop(self, op):
        data_columns = (col.unary_operator(op) for col in self._columns)
        return self._from_data_like_self(zip(self._column_names, data_columns))

    @classmethod
    @_cudf_nvtx_annotate
    def _colwise_binop(
        cls,
        operands: Dict[Optional[str], Tuple[ColumnBase, Any, bool, Any]],
        fn: str,
    ):
        """Implement binary ops between two frame-like objects.

        Binary operations for Frames can be reduced to a sequence of binary
        operations between column-like objects. Different types of frames need
        to preprocess different inputs, so subclasses should implement binary
        operations as a preprocessing step that calls this method.

        Parameters
        ----------
        operands : Dict[Optional[str], Tuple[ColumnBase, Any, bool, Any]]
            A mapping from column names to a tuple containing left and right
            operands as well as a boolean indicating whether or not to reflect
            an operation and fill value for nulls.
        fn : str
            The operation to perform.

        Returns
        -------
        Dict[ColumnBase]
            A dict of columns constructed from the result of performing the
            requested operation on the operands.
        """
        # Now actually perform the binop on the columns in left and right.
        output = {}
        for (
            col,
            (left_column, right_column, reflect, fill_value),
        ) in operands.items():
            output_mask = None
            if fill_value is not None:
                left_is_column = isinstance(left_column, ColumnBase)
                right_is_column = isinstance(right_column, ColumnBase)

                if left_is_column and right_is_column:
                    # If both columns are nullable, pandas semantics dictate
                    # that nulls that are present in both left_column and
                    # right_column are not filled.
                    if left_column.nullable and right_column.nullable:
                        lmask = as_column(left_column.nullmask)
                        rmask = as_column(right_column.nullmask)
                        output_mask = (lmask | rmask).data
                        left_column = left_column.fillna(fill_value)
                        right_column = right_column.fillna(fill_value)
                    elif left_column.nullable:
                        left_column = left_column.fillna(fill_value)
                    elif right_column.nullable:
                        right_column = right_column.fillna(fill_value)
                elif left_is_column:
                    if left_column.nullable:
                        left_column = left_column.fillna(fill_value)
                elif right_is_column:
                    if right_column.nullable:
                        right_column = right_column.fillna(fill_value)
                else:
                    assert False, "At least one operand must be a column."

            # TODO: Disable logical and binary operators between columns that
            # are not numerical using the new binops mixin.

            outcol = (
                getattr(operator, fn)(right_column, left_column)
                if reflect
                else getattr(operator, fn)(left_column, right_column)
            )

            if output_mask is not None:
                outcol = outcol.set_mask(output_mask)

            output[col] = outcol

        return output

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _array_ufunc(self, ufunc, method, inputs, kwargs)

    def _apply_cupy_ufunc_to_operands(
        self, ufunc, cupy_func, operands, **kwargs
    ):
        # Note: There are some operations that may be supported by libcudf but
        # are not supported by pandas APIs. In particular, libcudf binary
        # operations support logical and/or operations as well as
        # trigonometric, but those operations are not defined on
        # pd.Series/DataFrame. For now those operations will dispatch to cupy,
        # but if ufuncs are ever a bottleneck we could add special handling to
        # dispatch those (or any other) functions that we could implement
        # without cupy.

        mask = None
        data = [{} for _ in range(ufunc.nout)]
        for name, (left, right, _, _) in operands.items():
            cupy_inputs = []
            for inp in (left, right) if ufunc.nin == 2 else (left,):
                if isinstance(inp, ColumnBase) and inp.has_nulls():
                    new_mask = as_column(inp.nullmask)

                    # TODO: This is a hackish way to perform a bitwise and
                    # of bitmasks. Once we expose
                    # cudf::detail::bitwise_and, then we can use that
                    # instead.
                    mask = new_mask if mask is None else (mask & new_mask)

                    # Arbitrarily fill with zeros. For ufuncs, we assume
                    # that the end result propagates nulls via a bitwise
                    # and, so these elements are irrelevant.
                    inp = inp.fillna(0)
                cupy_inputs.append(cupy.asarray(inp))

            cp_output = cupy_func(*cupy_inputs, **kwargs)
            if ufunc.nout == 1:
                cp_output = (cp_output,)
            for i, out in enumerate(cp_output):
                data[i][name] = as_column(out).set_mask(mask)
        return data

    @_cudf_nvtx_annotate
    def dot(self, other, reflect=False):
        """
        Get dot product of frame and other, (binary operator `dot`).

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`,
        `dot`) to arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`,
        `@`.

        Parameters
        ----------
        other : Sequence, Series, or DataFrame
            Any multiple element data structure, or list-like object.
        reflect : bool, default False
            If ``True``, swap the order of the operands. See
            https://docs.python.org/3/reference/datamodel.html#object.__ror__
            for more information on when this is necessary.

        Returns
        -------
        scalar, Series, or DataFrame
            The result of the operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame([[1, 2, 3, 4],
        ...                      [5, 6, 7, 8]])
        >>> df @ df.T
            0    1
        0  30   70
        1  70  174
        >>> s = cudf.Series([1, 1, 1, 1])
        >>> df @ s
        0    10
        1    26
        dtype: int64
        >>> [1, 2, 3, 4] @ s
        10
        """
        # TODO: This function does not currently support nulls.
        # TODO: This function does not properly support misaligned indexes.
        lhs = self.values
        if isinstance(other, Frame):
            rhs = other.values
        elif isinstance(other, cupy.ndarray):
            rhs = other
        elif isinstance(
            other, (abc.Sequence, np.ndarray, pd.DataFrame, pd.Series)
        ):
            rhs = cupy.asarray(other)
        else:
            # TODO: This should raise an exception, not return NotImplemented,
            # but __matmul__ relies on the current behavior. We should either
            # move this implementation to __matmul__ and call it from here
            # (checking for NotImplemented and raising NotImplementedError if
            # that's what's returned), or __matmul__ should catch a
            # NotImplementedError from here and return NotImplemented. The
            # latter feels cleaner (putting the implementation in this method
            # rather than in the operator) but will be slower in the (highly
            # unlikely) case that we're multiplying a cudf object with another
            # type of object that somehow supports this behavior.
            return NotImplemented
        if reflect:
            lhs, rhs = rhs, lhs

        result = lhs.dot(rhs)
        if len(result.shape) == 1:
            return cudf.Series(result)
        if len(result.shape) == 2:
            return cudf.DataFrame(result)
        return result.item()

    def __matmul__(self, other):
        return self.dot(other)

    def __rmatmul__(self, other):
        return self.dot(other, reflect=True)

    # Unary logical operators
    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self.copy(deep=True)

    def __abs__(self):
        return self._unaryop("abs")

    # Reductions
    @classmethod
    def _get_axis_from_axis_arg(cls, axis):
        try:
            return cls._SUPPORT_AXIS_LOOKUP[axis]
        except KeyError:
            raise ValueError(f"No axis named {axis} for object type {cls}")

    def _reduce(self, *args, **kwargs):
        raise NotImplementedError(
            f"Reductions are not supported for objects of type {type(self)}."
        )

    @_cudf_nvtx_annotate
    def min(
        self,
        axis=None,
        skipna=True,
        level=None,
        numeric_only=None,
        **kwargs,
    ):
        """
        Return the minimum of the values in the DataFrame.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        level: int or level name, default None
            If the axis is a MultiIndex (hierarchical), count along a
            particular level, collapsing into a Series.
        numeric_only: bool, default None
            Include only float, int, boolean columns. If None, will attempt to
            use everything, then use only numeric data.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.min()
        a    1
        b    7
        dtype: int64
        """
        return self._reduce(
            "min",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def max(
        self,
        axis=None,
        skipna=True,
        level=None,
        numeric_only=None,
        **kwargs,
    ):
        """
        Return the maximum of the values in the DataFrame.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        level: int or level name, default None
            If the axis is a MultiIndex (hierarchical), count along a
            particular level, collapsing into a Series.
        numeric_only: bool, default None
            Include only float, int, boolean columns. If None, will attempt to
            use everything, then use only numeric data.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.max()
        a     4
        b    10
        dtype: int64
        """
        return self._reduce(
            "max",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def sum(
        self,
        axis=None,
        skipna=True,
        dtype=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):
        """
        Return sum of the values in the DataFrame.

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        dtype: data type
            Data type to cast the result to.
        min_count: int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.sum()
        a    10
        b    34
        dtype: int64
        """
        return self._reduce(
            "sum",
            axis=axis,
            skipna=skipna,
            dtype=dtype,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def product(
        self,
        axis=None,
        skipna=True,
        dtype=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):
        """
        Return product of the values in the DataFrame.

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        dtype: data type
            Data type to cast the result to.
        min_count: int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.product()
        a      24
        b    5040
        dtype: int64
        """
        axis = self._get_axis_from_axis_arg(axis)
        return self._reduce(
            # cuDF columns use "product" as the op name, but cupy uses "prod"
            # and we need cupy if axis == 1.
            "product" if axis == 0 else "prod",
            axis=axis,
            skipna=skipna,
            dtype=dtype,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    # Alias for pandas compatibility.
    prod = product

    @_cudf_nvtx_annotate
    def mean(
        self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs
    ):
        """
        Return the mean of the values for the requested axis.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}
            Axis for the function to be applied on.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        level : int or level name, default None
            If the axis is a MultiIndex (hierarchical), count along a
            particular level, collapsing into a Series.
        numeric_only : bool, default None
            Include only float, int, boolean columns. If None, will attempt to
            use everything, then use only numeric data. Not implemented for
            Series.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        mean : Series or DataFrame (if level specified)

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.mean()
        a    2.5
        b    8.5
        dtype: float64
        """
        return self._reduce(
            "mean",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def std(
        self,
        axis=None,
        skipna=True,
        level=None,
        ddof=1,
        numeric_only=None,
        **kwargs,
    ):
        """
        Return sample standard deviation of the DataFrame.

        Normalized by N-1 by default. This can be changed using
        the `ddof` argument

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof: int, default 1
            Delta Degrees of Freedom. The divisor used in calculations
            is N - ddof, where N represents the number of elements.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `level` and
        `numeric_only`

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.std()
        a    1.290994
        b    1.290994
        dtype: float64
        """

        return self._reduce(
            "std",
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def var(
        self,
        axis=None,
        skipna=True,
        level=None,
        ddof=1,
        numeric_only=None,
        **kwargs,
    ):
        """
        Return unbiased variance of the DataFrame.

        Normalized by N-1 by default. This can be changed using the
        ddof argument

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof: int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is
            N - ddof, where N represents the number of elements.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `level` and
        `numeric_only`

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.var()
        a    1.666667
        b    1.666667
        dtype: float64
        """
        return self._reduce(
            "var",
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def kurtosis(
        self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs
    ):
        """
        Return Fisher's unbiased kurtosis of a sample.

        Kurtosis obtained using Fisher's definition of
        kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.

        Returns
        -------
        Series or scalar

        Notes
        -----
        Parameters currently not supported are `level` and `numeric_only`

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series([1, 2, 3, 4])
        >>> series.kurtosis()
        -1.1999999999999904

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.kurt()
        a   -1.2
        b   -1.2
        dtype: float64
        """
        if axis not in (0, "index", None):
            raise NotImplementedError("Only axis=0 is currently supported.")

        return self._reduce(
            "kurtosis",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    # Alias for kurtosis.
    @copy_docstring(kurtosis)
    def kurt(
        self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs
    ):
        return self.kurtosis(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def skew(
        self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs
    ):
        """
        Return unbiased Fisher-Pearson skew of a sample.

        Parameters
        ----------
        skipna: bool, default True
            Exclude NA/null values when computing the result.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `axis`, `level` and
        `numeric_only`

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series([1, 2, 3, 4, 5, 6, 6])
        >>> series
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        6    6
        dtype: int64

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 8, 10, 10]})
        >>> df.skew()
        a    0.00000
        b   -0.37037
        dtype: float64
        """
        if axis not in (0, "index", None):
            raise NotImplementedError("Only axis=0 is currently supported.")

        return self._reduce(
            "skew",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def all(self, axis=0, skipna=True, level=None, **kwargs):
        """
        Return whether all elements are True in DataFrame.

        Parameters
        ----------

        skipna: bool, default True
            Exclude NA/null values. If the entire row/column is NA and
            skipna is True, then the result will be True, as for an
            empty row/column.
            If skipna is False, then NA are treated as True, because
            these are not equal to zero.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `axis`, `bool_only`, `level`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 0, 10, 10]})
        >>> df.all()
        a     True
        b    False
        dtype: bool
        """
        return self._reduce(
            "all",
            axis=axis,
            skipna=skipna,
            level=level,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def any(self, axis=0, skipna=True, level=None, **kwargs):
        """
        Return whether any elements is True in DataFrame.

        Parameters
        ----------

        skipna: bool, default True
            Exclude NA/null values. If the entire row/column is NA and
            skipna is True, then the result will be False, as for an
            empty row/column.
            If skipna is False, then NA are treated as True, because
            these are not equal to zero.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `axis`, `bool_only`, `level`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 0, 10, 10]})
        >>> df.any()
        a    True
        b    True
        dtype: bool
        """
        return self._reduce(
            "any",
            axis=axis,
            skipna=skipna,
            level=level,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def sum_of_squares(self, dtype=None):
        """Return the sum of squares of values.

        Parameters
        ----------
        dtype: data type
            Data type to cast the result to.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 0, 10, 10]})
        >>> df.sum_of_squares()
        a     38
        b    249
        dtype: int64
        """
        return self._reduce("sum_of_squares", dtype=dtype)

    @_cudf_nvtx_annotate
    def median(
        self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs
    ):
        """
        Return the median of the values for the requested axis.

        Parameters
        ----------

        skipna : bool, default True
            Exclude NA/null values when computing the result.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `level` and `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([10, 25, 3, 25, 24, 6])
        >>> ser
        0    10
        1    25
        2     3
        3    25
        4    24
        5     6
        dtype: int64
        >>> ser.median()
        17.0
        """
        return self._reduce(
            "median",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    @ioutils.doc_to_json()
    def to_json(self, path_or_buf=None, *args, **kwargs):
        """{docstring}"""

        return cudf.io.json.to_json(
            self, path_or_buf=path_or_buf, *args, **kwargs
        )

    @_cudf_nvtx_annotate
    @ioutils.doc_to_hdf()
    def to_hdf(self, path_or_buf, key, *args, **kwargs):
        """{docstring}"""

        cudf.io.hdf.to_hdf(path_or_buf, key, self, *args, **kwargs)

    @_cudf_nvtx_annotate
    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""

        return cudf.io.dlpack.to_dlpack(self)

    @_cudf_nvtx_annotate
    def to_string(self):
        r"""
        Convert to string

        cuDF uses Pandas internals for efficient string formatting.
        Set formatting options using pandas string formatting options and
        cuDF objects will print identically to Pandas objects.

        cuDF supports `null/None` as a value in any column type, which
        is transparently supported during this output process.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2]
        >>> df['val'] = [float(i + 10) for i in range(3)]
        >>> df.to_string()
        '   key   val\n0    0  10.0\n1    1  11.0\n2    2  12.0'
        """
        return repr(self)

    def __str__(self):
        return self.to_string()

    @_cudf_nvtx_annotate
    def __deepcopy__(self, memo):
        return self.copy(deep=True)

    @_cudf_nvtx_annotate
    def __copy__(self):
        return self.copy(deep=False)

    @_cudf_nvtx_annotate
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
        DataFrame or Series
            The first `n` rows of the caller object.

        See Also
        --------
        Frame.tail: Returns the last `n` rows.

        Examples
        --------

        **Series**

        >>> ser = cudf.Series(['alligator', 'bee', 'falcon',
        ... 'lion', 'monkey', 'parrot', 'shark', 'whale', 'zebra'])
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

        **DataFrame**

        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2, 3, 4]
        >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
        >>> df.head(2)
           key   val
        0    0  10.0
        1    1  11.0
        """
        return self.iloc[:n]

    @_cudf_nvtx_annotate
    def tail(self, n=5):
        """
        Returns the last n rows as a new DataFrame or Series

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2, 3, 4]
        >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
        >>> df.tail(2)
           key   val
        3    3  13.0
        4    4  14.0

        **Series**

        >>> import cudf
        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> ser.tail(2)
        3    1
        4    0
        """
        if n == 0:
            return self.iloc[0:0]

        return self.iloc[-n:]

    @_cudf_nvtx_annotate
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

    @_cudf_nvtx_annotate
    def nans_to_nulls(self):
        """
        Convert nans (if any) to nulls

        Returns
        -------
        DataFrame or Series

        Examples
        --------

        **Series**

        >>> import cudf, numpy as np
        >>> series = cudf.Series([1, 2, np.nan, None, 10], nan_as_null=False)
        >>> series
        0     1.0
        1     2.0
        2     NaN
        3    <NA>
        4    10.0
        dtype: float64
        >>> series.nans_to_nulls()
        0     1.0
        1     2.0
        2    <NA>
        3    <NA>
        4    10.0
        dtype: float64

        **DataFrame**

        >>> df = cudf.DataFrame()
        >>> df['a'] = cudf.Series([1, None, np.nan], nan_as_null=False)
        >>> df['b'] = cudf.Series([None, 3.14, np.nan], nan_as_null=False)
        >>> df
              a     b
        0   1.0  <NA>
        1  <NA>  3.14
        2   NaN   NaN
        >>> df.nans_to_nulls()
              a     b
        0   1.0  <NA>
        1  <NA>  3.14
        2  <NA>  <NA>
        """
        result_data = {}
        for name, col in self._data.items():
            try:
                result_data[name] = col.nans_to_nulls()
            except AttributeError:
                result_data[name] = col.copy()
        return self._from_data_like_self(result_data)

    @_cudf_nvtx_annotate
    def __invert__(self):
        """Bitwise invert (~) for integral dtypes, logical NOT for bools."""
        return self._from_data_like_self(
            {
                name: _apply_inverse_column(col)
                for name, col in self._data.items()
            }
        )

    def nunique(self, dropna: bool = True):
        """
        Returns a per column mapping with counts of unique values for
        each column.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        dict
            Name and unique value counts of each column in frame.
        """
        return {
            name: col.distinct_count(dropna=dropna)
            for name, col in self._data.items()
        }

    @staticmethod
    def _repeat(
        columns: List[ColumnBase], repeats, axis=None
    ) -> List[ColumnBase]:
        if axis is not None:
            raise NotImplementedError(
                "Only axis=`None` supported at this time."
            )

        if not is_scalar(repeats):
            repeats = as_column(repeats)

        return libcudf.filling.repeat(columns, repeats)


def _apply_inverse_column(col: ColumnBase) -> ColumnBase:
    """Bitwise invert (~) for integral dtypes, logical NOT for bools."""
    if np.issubdtype(col.dtype, np.integer):
        return col.unary_operator("invert")
    elif np.issubdtype(col.dtype, np.bool_):
        return col.unary_operator("not")
    else:
        raise TypeError(
            f"Operation `~` not supported on {col.dtype.type.__name__}"
        )
