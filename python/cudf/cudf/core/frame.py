# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from __future__ import annotations

import copy
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
    cast,
)

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
from nvtx import annotate

import cudf
from cudf import _lib as libcudf
from cudf._typing import ColumnLike, DataFrameOrSeries, Dtype
from cudf.api.types import (
    _is_non_decimal_numeric_dtype,
    is_bool_dtype,
    is_decimal_dtype,
    is_dict_like,
    is_integer_dtype,
    is_scalar,
    issubdtype,
)
from cudf.core.column import (
    ColumnBase,
    as_column,
    build_categorical_column,
    column_empty,
    deserialize_columns,
    serialize_columns,
)
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.join import Merge, MergeSemi
from cudf.core.udf.pipeline import compile_or_get, supported_cols_from_frame
from cudf.core.window import Rolling
from cudf.utils import ioutils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import find_common_type, is_column_like
from cudf.utils.utils import _gather_map_is_valid

T = TypeVar("T", bound="Frame")


class Frame:
    """A collection of Column objects with an optional index.

    Parameters
    ----------
    data : dict
        An dict mapping column names to Columns
    index : Table
        A Frame representing the (optional) index columns.
    """

    _data: "ColumnAccessor"
    # TODO: Once all dependence on Frame having an index is removed, this
    # attribute should be moved to IndexedFrame.
    _index: Optional[cudf.core.index.BaseIndex]

    def __init__(self, data=None, index=None):
        if data is None:
            data = {}
        self._data = cudf.core.column_accessor.ColumnAccessor(data)
        self._index = index

    @property
    def _num_columns(self) -> int:
        return len(self._data)

    @property
    def _num_indices(self) -> int:
        if self._index is None:
            return 0
        else:
            return len(self._index_names)

    @property
    def _num_rows(self) -> int:
        if self._index is not None:
            return len(self._index)
        if len(self._data) == 0:
            return 0
        return len(self._data.columns[0])

    @property
    def _column_names(self) -> List[Any]:  # TODO: List[str]?
        return self._data.names

    @property
    def _index_names(self) -> List[Any]:  # TODO: List[str]?
        # TODO: Temporarily suppressing mypy warnings to avoid introducing bugs
        # by returning an empty list where one is not expected.
        return (
            None  # type: ignore
            if self._index is None
            else self._index._data.names
        )

    @property
    def _columns(self) -> List[Any]:  # TODO: List[Column]?
        return self._data.columns

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
    def _from_data(
        cls,
        data: MutableMapping,
        index: Optional[cudf.core.index.BaseIndex] = None,
    ):
        obj = cls.__new__(cls)
        Frame.__init__(obj, data, index)
        return obj

    @classmethod
    def _from_columns(
        cls,
        columns: List[ColumnBase],
        column_names: List[str],
        index_names: Optional[List[str]] = None,
    ):
        """Construct a `Frame` object from a list of columns.

        If `index_names` is set, the first `len(index_names)` columns are
        used to construct the index of the frame.
        """
        index = None
        n_index_columns = 0
        if index_names is not None:
            n_index_columns = len(index_names)
            index = cudf.core.index._index_from_data(
                dict(zip(range(n_index_columns), columns))
            )
            if isinstance(index, cudf.MultiIndex):
                index.names = index_names
            else:
                index.name = index_names[0]

        data = {
            name: columns[i + n_index_columns]
            for i, name in enumerate(column_names)
        }

        return cls._from_data(data, index)

    def _mimic_inplace(
        self: T, result: Frame, inplace: bool = False
    ) -> Optional[Frame]:
        if inplace:
            for col in self._data:
                if col in result._data:
                    self._data[col]._mimic_inplace(
                        result._data[col], inplace=True
                    )
            self._data = result._data
            self._index = result._index
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
    def _is_homogeneous(self):
        # make sure that the dataframe has columns
        if not self._data.columns:
            return True

        first_type = self._data.columns[0].dtype.name
        return all(x.dtype.name == first_type for x in self._data.columns)

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

    def __len__(self):
        return self._num_rows

    def copy(self: T, deep: bool = True) -> T:
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
        new_frame = self.__class__.__new__(self.__class__)
        new_frame._data = self._data.copy(deep=deep)

        if self._index is not None:
            new_frame._index = self._index.copy(deep=deep)
        else:
            new_frame._index = None

        return new_frame

    def equals(self, other, **kwargs):
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

        check_types = kwargs.get("check_types", True)

        if check_types:
            if type(self) is not type(other):
                return False

        if other is None or len(self) != len(other):
            return False

        # check data:
        for self_col, other_col in zip(
            self._data.values(), other._data.values()
        ):
            if not self_col.equals(other_col, check_dtypes=check_types):
                return False

        # check index:
        if self._index is None:
            return other._index is None
        else:
            return self._index.equals(other._index)

    def _explode(self, explode_column: Any, ignore_index: bool):
        """Helper function for `explode` in `Series` and `Dataframe`, explodes
        a specified nested column. Other columns' corresponding rows are
        duplicated. If ignore_index is set, the original index is not exploded
        and will be replaced with a `RangeIndex`.
        """
        explode_column_num = self._column_names.index(explode_column)
        if not ignore_index and self._index is not None:
            explode_column_num += self._index.nlevels

        res = self.__class__._from_data(  # type: ignore
            *libcudf.lists.explode_outer(
                self, explode_column_num, ignore_index
            )
        )

        res._data.multiindex = self._data.multiindex
        res._data._level_names = self._data._level_names

        if not ignore_index and self._index is not None:
            res.index.names = self._index.names
        return res

    def _get_columns_by_label(self, labels, downcast=False):
        """
        Returns columns of the Frame specified by `labels`

        """
        return self._data.select_by_label(labels)

    def _get_columns_by_index(self, indices):
        """
        Returns columns of the Frame specified by `labels`

        """
        data = self._data.select_by_index(indices)
        return self.__class__(
            data, columns=data.to_pandas_index(), index=self.index
        )

    def _gather(
        self, gather_map, keep_index=True, nullify=False, check_bounds=True
    ):
        """Gather rows of frame specified by indices in `gather_map`.

        Skip bounds checking if check_bounds is False.
        Set rows to null for all out of bound indices if nullify is `True`.
        """
        # TODO: `keep_index` argument is to be removed.
        gather_map = cudf.core.column.as_column(gather_map)

        # TODO: For performance, the check and conversion of gather map should
        # be done by the caller. This check will be removed in future release.
        if not is_integer_dtype(gather_map.dtype):
            gather_map = gather_map.astype("int32")

        if not _gather_map_is_valid(
            gather_map, len(self), check_bounds, nullify
        ):
            raise IndexError("Gather map index is out of bounds.")

        result = self.__class__._from_columns(
            libcudf.copying.gather(
                list(self._columns), gather_map, nullify=nullify,
            ),
            self._column_names,
        )

        result._copy_type_metadata(self)
        return result

    def _hash(self, method, initial_hash=None):
        return libcudf.hash.hash(self, method, initial_hash)

    def _hash_partition(
        self, columns_to_hash, num_partitions, keep_index=True
    ):
        output_data, output_index, offsets = libcudf.hash.hash_partition(
            self, columns_to_hash, num_partitions, keep_index
        )
        output = self.__class__._from_data(output_data, output_index)
        output._copy_type_metadata(self, include_index=keep_index)
        return output, offsets

    def _as_column(self):
        """
        _as_column : Converts a single columned Frame to Column
        """
        assert (
            self._num_columns == 1
            and self._index is None
            and self._column_names[0] is None
        ), """There should be only one data column,
            no index and None as the name to use this method"""

        return self._data[None].copy(deep=False)

    def _empty_like(self, keep_index=True):
        result = self.__class__._from_data(
            *libcudf.copying.table_empty_like(self, keep_index)
        )

        result._copy_type_metadata(self, include_index=keep_index)
        return result

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

    def to_cupy(
        self,
        dtype: Union[Dtype, None] = None,
        copy: bool = False,
        na_value=None,
    ) -> cupy.ndarray:
        """Convert the Frame to a CuPy array.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
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

    def to_numpy(
        self,
        dtype: Union[Dtype, None] = None,
        copy: bool = True,
        na_value=None,
    ) -> np.ndarray:
        """Convert the Frame to a NumPy array.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
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

    def clip(self, lower=None, upper=None, inplace=False, axis=1):
        """
        Trim values at input threshold(s).

        Assigns values outside boundary to boundary values.
        Thresholds can be singular values or array like,
        and in the latter case the clipping is performed
        element-wise in the specified axis. Currently only
        `axis=1` is supported.

        Parameters
        ----------
        lower : scalar or array_like, default None
            Minimum threshold value. All values below this
            threshold will be set to it. If it is None,
            there will be no clipping based on lower.
            In case of Series/Index, lower is expected to be
            a scalar or an array of size 1.
        upper : scalar or array_like, default None
            Maximum threshold value. All values below this
            threshold will be set to it. If it is None,
            there will be no clipping based on upper.
            In case of Series, upper is expected to be
            a scalar or an array of size 1.
        inplace : bool, default False

        Returns
        -------
        Clipped DataFrame/Series/Index/MultiIndex

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"a":[1, 2, 3, 4], "b":['a', 'b', 'c', 'd']})
        >>> df.clip(lower=[2, 'b'], upper=[3, 'c'])
           a  b
        0  2  b
        1  2  b
        2  3  c
        3  3  c

        >>> df.clip(lower=None, upper=[3, 'c'])
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  3  c

        >>> df.clip(lower=[2, 'b'], upper=None)
           a  b
        0  2  b
        1  2  b
        2  3  c
        3  4  d

        >>> df.clip(lower=2, upper=3, inplace=True)
        >>> df
           a  b
        0  2  2
        1  2  3
        2  3  3
        3  3  3

        >>> import cudf
        >>> sr = cudf.Series([1, 2, 3, 4])
        >>> sr.clip(lower=2, upper=3)
        0    2
        1    2
        2    3
        3    3
        dtype: int64

        >>> sr.clip(lower=None, upper=3)
        0    1
        1    2
        2    3
        3    3
        dtype: int64

        >>> sr.clip(lower=2, upper=None, inplace=True)
        >>> sr
        0    2
        1    2
        2    3
        3    4
        dtype: int64
        """

        if axis != 1:
            raise NotImplementedError("`axis is not yet supported in clip`")

        if lower is None and upper is None:
            return None if inplace is True else self.copy(deep=True)

        if is_scalar(lower):
            lower = np.full(self._num_columns, lower)
        if is_scalar(upper):
            upper = np.full(self._num_columns, upper)

        if len(lower) != len(upper):
            raise ValueError("Length of lower and upper should be equal")

        if len(lower) != self._num_columns:
            raise ValueError(
                """Length of lower/upper should be
                equal to number of columns in
                DataFrame/Series/Index/MultiIndex"""
            )

        output = self.copy(deep=False)
        if output.ndim == 1:
            # In case of series and Index,
            # swap lower and upper if lower > upper
            if (
                lower[0] is not None
                and upper[0] is not None
                and (lower[0] > upper[0])
            ):
                lower[0], upper[0] = upper[0], lower[0]

        for i, name in enumerate(self._data):
            output._data[name] = self._data[name].clip(lower[i], upper[i])

        output._copy_type_metadata(self, include_index=False)

        return self._mimic_inplace(output, inplace=inplace)

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
        import cudf.core._internals.where

        return cudf.core._internals.where.where(
            frame=self, cond=cond, other=other, inplace=inplace
        )

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

    def _partition(self, scatter_map, npartitions, keep_index=True):

        data, index, output_offsets = libcudf.partitioning.partition(
            self, scatter_map, npartitions, keep_index
        )
        partitioned = self.__class__._from_data(data, index)

        # due to the split limitation mentioned
        # here: https://github.com/rapidsai/cudf/issues/4607
        # we need to remove first & last elements in offsets.
        # TODO: Remove this after the above issue is fixed.
        output_offsets = output_offsets[1:-1]

        result = partitioned._split(output_offsets, keep_index=keep_index)

        for frame in result:
            frame._copy_type_metadata(self, include_index=keep_index)

        if npartitions:
            for _ in range(npartitions - len(result)):
                result.append(self._empty_like(keep_index))

        return result

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

    @annotate("SCATTER_BY_MAP", color="green", domain="cudf_python")
    def scatter_by_map(
        self, map_index, map_size=None, keep_index=True, **kwargs
    ):
        """Scatter to a list of dataframes.

        Uses map_index to determine the destination
        of each row of the original DataFrame.

        Parameters
        ----------
        map_index : Series, str or list-like
            Scatter assignment for each row
        map_size : int
            Length of output list. Must be >= uniques in map_index
        keep_index : bool
            Conserve original index values for each row

        Returns
        -------
        A list of cudf.DataFrame objects.
        """

        # map_index might be a column name or array,
        # make it a Column
        if isinstance(map_index, str):
            map_index = self._data[map_index]
        elif isinstance(map_index, cudf.Series):
            map_index = map_index._column
        else:
            map_index = as_column(map_index)

        # Convert float to integer
        if map_index.dtype.kind == "f":
            map_index = map_index.astype(np.int32)

        # Convert string or categorical to integer
        if isinstance(map_index, cudf.core.column.StringColumn):
            map_index = map_index.as_categorical_column(
                "category"
            ).as_numerical
            warnings.warn(
                "Using StringColumn for map_index in scatter_by_map. "
                "Use an integer array/column for better performance."
            )
        elif isinstance(map_index, cudf.core.column.CategoricalColumn):
            map_index = map_index.as_numerical
            warnings.warn(
                "Using CategoricalColumn for map_index in scatter_by_map. "
                "Use an integer array/column for better performance."
            )

        if kwargs.get("debug", False) == 1 and map_size is not None:
            count = map_index.distinct_count()
            if map_size < count:
                raise ValueError(
                    f"ERROR: map_size must be >= {count} (got {map_size})."
                )

        tables = self._partition(map_index, map_size, keep_index)

        return tables

    def dropna(
        self, axis=0, how="any", thresh=None, subset=None, inplace=False
    ):
        """
        Drops rows (or columns) containing nulls from a Column.

        Parameters
        ----------
        axis : {0, 1}, optional
            Whether to drop rows (axis=0, default) or columns (axis=1)
            containing nulls.
        how : {"any", "all"}, optional
            Specifies how to decide whether to drop a row (or column).
            any (default) drops rows (or columns) containing at least
            one null value. all drops only rows (or columns) containing
            *all* null values.
        thresh: int, optional
            If specified, then drops every row (or column) containing
            less than `thresh` non-null values
        subset : list, optional
            List of columns to consider when dropping rows (all columns
            are considered by default). Alternatively, when dropping
            columns, subset is a list of rows to consider.
        inplace : bool, default False
            If True, do operation inplace and return None.

        Returns
        -------
        Copy of the DataFrame with rows/columns containing nulls dropped.

        See also
        --------
        cudf.DataFrame.isna
            Indicate null values.

        cudf.DataFrame.notna
            Indicate non-null values.

        cudf.DataFrame.fillna
            Replace null values.

        cudf.Series.dropna
            Drop null values.

        cudf.Index.dropna
            Drop null indices.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
        ...                    "toy": ['Batmobile', None, 'Bullwhip'],
        ...                    "born": [np.datetime64("1940-04-25"),
        ...                             np.datetime64("NaT"),
        ...                             np.datetime64("NaT")]})
        >>> df
               name        toy                 born
        0    Alfred  Batmobile  1940-04-25 00:00:00
        1    Batman       <NA>                 <NA>
        2  Catwoman   Bullwhip                 <NA>

        Drop the rows where at least one element is null.

        >>> df.dropna()
             name        toy       born
        0  Alfred  Batmobile 1940-04-25

        Drop the columns where at least one element is null.

        >>> df.dropna(axis='columns')
               name
        0    Alfred
        1    Batman
        2  Catwoman

        Drop the rows where all elements are null.

        >>> df.dropna(how='all')
               name        toy                 born
        0    Alfred  Batmobile  1940-04-25 00:00:00
        1    Batman       <NA>                 <NA>
        2  Catwoman   Bullwhip                 <NA>

        Keep only the rows with at least 2 non-null values.

        >>> df.dropna(thresh=2)
               name        toy                 born
        0    Alfred  Batmobile  1940-04-25 00:00:00
        2  Catwoman   Bullwhip                 <NA>

        Define in which columns to look for null values.

        >>> df.dropna(subset=['name', 'born'])
             name        toy       born
        0  Alfred  Batmobile 1940-04-25

        Keep the DataFrame with valid entries in the same variable.

        >>> df.dropna(inplace=True)
        >>> df
             name        toy       born
        0  Alfred  Batmobile 1940-04-25
        """
        if axis == 0:
            result = self._drop_na_rows(
                how=how, subset=subset, thresh=thresh, drop_nan=True
            )
        else:
            result = self._drop_na_columns(
                how=how, subset=subset, thresh=thresh
            )

        return self._mimic_inplace(result, inplace=inplace)

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
        result : DataFrame
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

        if isinstance(value, cudf.Series):
            value = value.reindex(self._data.names)
        elif isinstance(value, cudf.DataFrame):
            if not self.index.equals(value.index):
                value = value.reindex(self.index)
            else:
                value = value
        elif not isinstance(value, abc.Mapping):
            value = {name: copy.deepcopy(value) for name in self._data.names}
        elif isinstance(value, abc.Mapping):
            value = {
                key: value.reindex(self.index)
                if isinstance(value, cudf.Series)
                else value
                for key, value in value.items()
            }

        copy_data = self._data.copy(deep=True)

        for name in copy_data.keys():
            should_fill = (
                name in value
                and not libcudf.scalar._is_null_host_scalar(value[name])
            ) or method is not None
            if should_fill:
                copy_data[name] = copy_data[name].fillna(value[name], method)
        result = self._from_data(copy_data, self._index)

        return self._mimic_inplace(result, inplace=inplace)

    def ffill(self):
        return self.fillna(method="ffill")

    def bfill(self):
        return self.fillna(method="bfill")

    def _drop_na_rows(
        self, how="any", subset=None, thresh=None, drop_nan=False
    ):
        """
        Drops null rows from `self`.

        how : {"any", "all"}, optional
            Specifies how to decide whether to drop a row.
            any (default) drops rows containing at least
            one null value. all drops only rows containing
            *all* null values.
        subset : list, optional
            List of columns to consider when dropping rows.
        thresh: int, optional
            If specified, then drops every row containing
            less than `thresh` non-null values.
        """
        if subset is None:
            subset = self._column_names
        elif (
            not np.iterable(subset)
            or isinstance(subset, str)
            or isinstance(subset, tuple)
            and subset in self._data.names
        ):
            subset = (subset,)
        diff = set(subset) - set(self._data)
        if len(diff) != 0:
            raise KeyError(f"columns {diff} do not exist")

        if len(subset) == 0:
            return self.copy(deep=True)

        frame = self.copy(deep=False)
        if drop_nan:
            for name, col in frame._data.items():
                if name in subset and isinstance(
                    col, cudf.core.column.NumericalColumn
                ):
                    frame._data[name] = col.nans_to_nulls()
                else:
                    frame._data[name] = col

        result = self.__class__._from_columns(
            libcudf.stream_compaction.drop_nulls(
                list(self._index._data.columns + frame._columns),
                how=how,
                keys=self._positions_from_column_names(
                    subset, offset_by_index_columns=True
                ),
                thresh=thresh,
            ),
            self._column_names,
            self._index.names,
        )
        result._copy_type_metadata(frame)
        return result

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

        for col in self._data.names:
            no_threshold_valid_count = (
                len(df[col]) - df[col].nans_to_nulls().null_count
            ) < thresh
            if no_threshold_valid_count:
                continue
            out_cols.append(col)

        return self[out_cols]

    def _apply_boolean_mask(self, boolean_mask):
        """
        Applies boolean mask to each row of `self`,
        rows corresponding to `False` is dropped
        """
        boolean_mask = as_column(boolean_mask)

        result = self.__class__._from_data(
            *libcudf.stream_compaction.apply_boolean_mask(
                self, as_column(boolean_mask)
            )
        )
        result._copy_type_metadata(self)
        return result

    def interpolate(
        self,
        method="linear",
        axis=0,
        limit=None,
        inplace=False,
        limit_direction=None,
        limit_area=None,
        downcast=None,
        **kwargs,
    ):
        """
        Interpolate data values between some points.

        Parameters
        ----------
        method : str, default 'linear'
            Interpolation technique to use. Currently,
            only 'linear` is supported.
            * 'linear': Ignore the index and treat the values as
            equally spaced. This is the only method supported on MultiIndexes.
            * 'index', 'values': linearly interpolate using the index as
            an x-axis. Unsorted indices can lead to erroneous results.
        axis : int, default 0
            Axis to interpolate along. Currently,
            only 'axis=0' is supported.
        inplace : bool, default False
            Update the data in place if possible.

        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller, interpolated at
            some or all ``NaN`` values

        """

        if method in {"pad", "ffill"} and limit_direction != "forward":
            raise ValueError(
                f"`limit_direction` must be 'forward' for method `{method}`"
            )
        if method in {"backfill", "bfill"} and limit_direction != "backward":
            raise ValueError(
                f"`limit_direction` must be 'backward' for method `{method}`"
            )

        data = self

        if not isinstance(data._index, cudf.RangeIndex):
            perm_sort = data._index.argsort()
            data = data._gather(perm_sort)

        interpolator = cudf.core.algorithms.get_column_interpolator(method)
        columns = {}
        for colname, col in data._data.items():
            if col.nullable:
                col = col.astype("float64").fillna(np.nan)

            # Interpolation methods may or may not need the index
            columns[colname] = interpolator(col, index=data._index)

        result = self._from_data(columns, index=data._index)

        return (
            result
            if isinstance(data._index, cudf.RangeIndex)
            else result._gather(perm_sort.argsort())
        )

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

        result = self.__class__._from_data(
            *libcudf.quantiles.quantiles(
                self,
                q,
                interpolation,
                is_sorted,
                column_order,
                null_precedence,
            )
        )

        result._copy_type_metadata(self)
        return result

    @annotate("APPLY", color="purple", domain="cudf_python")
    def _apply(self, func, *args):
        """
        Apply `func` across the rows of the frame.
        """
        kernel, retty = compile_or_get(self, func, args)

        # Mask and data column preallocated
        ans_col = cupy.empty(len(self), dtype=retty)
        ans_mask = cudf.core.column.column_empty(len(self), dtype="bool")
        launch_args = [(ans_col, ans_mask), len(self)]
        offsets = []

        # if compile_or_get succeeds, it is safe to create a kernel that only
        # consumes the columns that are of supported dtype
        for col in supported_cols_from_frame(self).values():
            data = col.data
            mask = col.mask
            if mask is None:
                launch_args.append(data)
            else:
                launch_args.append((data, mask))
            offsets.append(col.offset)
        launch_args += offsets
        launch_args += list(args)
        kernel.forall(len(self))(*launch_args)

        col = as_column(ans_col)
        col.set_base_mask(libcudf.transform.bools_to_mask(ans_mask))
        result = cudf.Series._from_data({None: col}, self._index)

        return result

    def rank(
        self,
        axis=0,
        method="average",
        numeric_only=None,
        na_option="keep",
        ascending=True,
        pct=False,
    ):
        """
        Compute numerical data ranks (1 through n) along axis.
        By default, equal values are assigned a rank that is the average of the
        ranks of those values.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Index to direct ranking.
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            How to rank the group of records that have the same value
            (i.e. ties):
            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups.
        numeric_only : bool, optional
            For DataFrame objects, rank only numeric columns if set to True.
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            How to rank NaN values:
            * keep: assign NaN rank to NaN values
            * top: assign smallest rank to NaN values if ascending
            * bottom: assign highest rank to NaN values if ascending.
        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order.
        pct : bool, default False
            Whether or not to display the returned rankings in percentile
            form.

        Returns
        -------
        same type as caller
            Return a Series or DataFrame with data ranks as values.
        """
        if method not in {"average", "min", "max", "first", "dense"}:
            raise KeyError(method)

        method_enum = libcudf.sort.RankMethod[method.upper()]
        if na_option not in {"keep", "top", "bottom"}:
            raise ValueError(
                "na_option must be one of 'keep', 'top', or 'bottom'"
            )

        if axis not in (0, "index"):
            raise NotImplementedError(
                f"axis must be `0`/`index`, "
                f"axis={axis} is not yet supported in rank"
            )

        source = self
        if numeric_only:
            numeric_cols = (
                name
                for name in self._data.names
                if _is_non_decimal_numeric_dtype(self._data[name])
            )
            source = self._get_columns_by_label(numeric_cols)
            if source.empty:
                return source.astype("float64")

        data, index = libcudf.sort.rank_columns(
            source, method_enum, na_option, ascending, pct
        )

        return self._from_data(data, index).astype(np.float64)

    def repeat(self, repeats, axis=None):
        """Repeats elements consecutively.

        Returns a new object of caller type(DataFrame/Series/Index) where each
        element of the current object is repeated consecutively a given
        number of times.

        Parameters
        ----------
        repeats : int, or array of ints
            The number of repetitions for each element. This should
            be a non-negative integer. Repeating 0 times will return
            an empty object.

        Returns
        -------
        Series/DataFrame/Index
            A newly created object of same type as caller
            with repeated elements.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        >>> df
           a   b
        0  1  10
        1  2  20
        2  3  30
        >>> df.repeat(3)
           a   b
        0  1  10
        0  1  10
        0  1  10
        1  2  20
        1  2  20
        1  2  20
        2  3  30
        2  3  30
        2  3  30

        Repeat on Series

        >>> s = cudf.Series([0, 2])
        >>> s
        0    0
        1    2
        dtype: int64
        >>> s.repeat([3, 4])
        0    0
        0    0
        0    0
        1    2
        1    2
        1    2
        1    2
        dtype: int64
        >>> s.repeat(2)
        0    0
        0    0
        1    2
        1    2
        dtype: int64

        Repeat on Index

        >>> index = cudf.Index([10, 22, 33, 55])
        >>> index
        Int64Index([10, 22, 33, 55], dtype='int64')
        >>> index.repeat(5)
        Int64Index([10, 10, 10, 10, 10, 22, 22, 22, 22, 22, 33,
                    33, 33, 33, 33, 55, 55, 55, 55, 55],
                dtype='int64')
        """
        if axis is not None:
            raise NotImplementedError(
                "Only axis=`None` supported at this time."
            )

        return self._repeat(repeats)

    def _repeat(self, count):
        if not is_scalar(count):
            count = as_column(count)

        result = self.__class__._from_data(
            *libcudf.filling.repeat(self, count)
        )

        result._copy_type_metadata(self)
        return result

    def _fill(self, fill_values, begin, end, inplace):
        col_and_fill = zip(self._columns, fill_values)

        if not inplace:
            data_columns = (c._fill(v, begin, end) for (c, v) in col_and_fill)
            return self.__class__._from_data(
                zip(self._column_names, data_columns), self._index
            )

        for (c, v) in col_and_fill:
            c.fill(v, begin, end, inplace=True)

        return self

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """Shift values by `periods` positions."""
        assert axis in (None, 0) and freq is None
        return self._shift(periods)

    def _shift(self, offset, fill_value=None):
        data_columns = (col.shift(offset, fill_value) for col in self._columns)
        return self.__class__._from_data(
            zip(self._column_names, data_columns), self._index
        )

    def round(self, decimals=0, how="half_even"):
        """
        Round to a variable number of decimal places.

        Parameters
        ----------
        decimals : int, dict, Series
            Number of decimal places to round each column to. This parameter
            must be an int for a Series.  For a DataFrame, a dict or a Series
            are also valid inputs. If an int is given, round each column to the
            same number of places.  Otherwise dict and Series round to variable
            numbers of places.  Column names should be in the keys if
            `decimals` is a dict-like, or in the index if `decimals` is a
            Series. Any columns not included in `decimals` will be left as is.
            Elements of `decimals` which are not columns of the input will be
            ignored.
        how : str, optional
            Type of rounding. Can be either "half_even" (default)
            of "half_up" rounding.

        Returns
        -------
        Series or DataFrame
            A Series or DataFrame with the affected columns rounded to the
            specified number of decimal places.

        Examples
        --------
        **Series**

        >>> s = cudf.Series([0.1, 1.4, 2.9])
        >>> s.round()
        0    0.0
        1    1.0
        2    3.0
        dtype: float64

        **DataFrame**

        >>> df = cudf.DataFrame(
                [(.21, .32), (.01, .67), (.66, .03), (.21, .18)],
        ...     columns=['dogs', 'cats']
        ... )
        >>> df
           dogs  cats
        0  0.21  0.32
        1  0.01  0.67
        2  0.66  0.03
        3  0.21  0.18

        By providing an integer each column is rounded to the same number
        of decimal places

        >>> df.round(1)
           dogs  cats
        0   0.2   0.3
        1   0.0   0.7
        2   0.7   0.0
        3   0.2   0.2

        With a dict, the number of places for specific columns can be
        specified with the column names as key and the number of decimal
        places as value

        >>> df.round({'dogs': 1, 'cats': 0})
           dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0

        Using a Series, the number of places for specific columns can be
        specified with the column names as index and the number of
        decimal places as value

        >>> decimals = cudf.Series([0, 1], index=['cats', 'dogs'])
        >>> df.round(decimals)
           dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0
        """

        if isinstance(decimals, cudf.Series):
            decimals = decimals.to_pandas()

        if isinstance(decimals, pd.Series):
            if not decimals.index.is_unique:
                raise ValueError("Index of decimals must be unique")
            decimals = decimals.to_dict()
        elif isinstance(decimals, int):
            decimals = {name: decimals for name in self._column_names}
        elif not isinstance(decimals, abc.Mapping):
            raise TypeError(
                "decimals must be an integer, a dict-like or a Series"
            )

        cols = {
            name: col.round(decimals[name], how=how)
            if (name in decimals and _is_non_decimal_numeric_dtype(col.dtype))
            else col.copy(deep=True)
            for name, col in self._data.items()
        }

        return self.__class__._from_data(
            data=cudf.core.column_accessor.ColumnAccessor(
                cols,
                multiindex=self._data.multiindex,
                level_names=self._data.level_names,
            ),
            index=self._index,
        )

    @annotate("SAMPLE", color="orange", domain="cudf_python")
    def sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None,
        keep_index=True,
    ):
        """Return a random sample of items from an axis of object.

        You can use random_state for reproducibility.

        Parameters
        ----------
        n : int, optional
            Number of items from axis to return. Cannot be used with frac.
            Default = 1 if frac = None.
        frac : float, optional
            Fraction of axis items to return. Cannot be used with n.
        replace : bool, default False
            Allow or disallow sampling of the same row more than once.
            replace == True is not yet supported for axis = 1/"columns"
        weights : str or ndarray-like, optional
            Only supported for axis=1/"columns"
        random_state : int, numpy RandomState or None, default None
            Seed for the random number generator (if int), or None.
            If None, a random seed will be chosen.
            if RandomState, seed will be extracted from current state.
        axis : {0 or ‘index’, 1 or ‘columns’, None}, default None
            Axis to sample. Accepts axis number or name.
            Default is stat axis for given data type
            (0 for Series and DataFrames). Series and Index doesn't
            support axis=1.

        Returns
        -------
        Series or DataFrame or Index
            A new object of same type as caller containing n items
            randomly sampled from the caller object.

        Examples
        --------
        >>> import cudf as cudf
        >>> df = cudf.DataFrame({"a":{1, 2, 3, 4, 5}})
        >>> df.sample(3)
           a
        1  2
        3  4
        0  1

        >>> sr = cudf.Series([1, 2, 3, 4, 5])
        >>> sr.sample(10, replace=True)
        1    4
        3    1
        2    4
        0    5
        0    1
        4    5
        4    1
        0    2
        0    3
        3    2
        dtype: int64

        >>> df = cudf.DataFrame(
        ... {"a":[1, 2], "b":[2, 3], "c":[3, 4], "d":[4, 5]})
        >>> df.sample(2, axis=1)
           a  c
        0  1  3
        1  2  4
        """

        if frac is not None and frac > 1 and not replace:
            raise ValueError(
                "Replace has to be set to `True` "
                "when upsampling the population `frac` > 1."
            )
        elif frac is not None and n is not None:
            raise ValueError(
                "Please enter a value for `frac` OR `n`, not both"
            )

        if frac is None and n is None:
            n = 1
        elif frac is not None:
            if axis is None or axis == 0 or axis == "index":
                n = int(round(self.shape[0] * frac))
            else:
                n = int(round(self.shape[1] * frac))

        if axis is None or axis == 0 or axis == "index":
            if n > 0 and self.shape[0] == 0:
                raise ValueError(
                    "Cannot take a sample larger than 0 when axis is empty"
                )

            if not replace and n > self.shape[0]:
                raise ValueError(
                    "Cannot take a larger sample than population "
                    "when 'replace=False'"
                )

            if weights is not None:
                raise NotImplementedError(
                    "weights is not yet supported for axis=0/index"
                )

            if random_state is None:
                seed = np.random.randint(
                    np.iinfo(np.int64).max, dtype=np.int64
                )
            elif isinstance(random_state, np.random.mtrand.RandomState):
                _, keys, pos, _, _ = random_state.get_state()
                seed = 0 if pos >= len(keys) else pos
            else:
                seed = np.int64(random_state)

            result = self.__class__._from_data(
                *libcudf.copying.sample(
                    self,
                    n=n,
                    replace=replace,
                    seed=seed,
                    keep_index=keep_index,
                )
            )
            result._copy_type_metadata(self)

            return result
        else:
            if len(self.shape) != 2:
                raise ValueError(
                    f"No axis named {axis} for "
                    f"object type {self.__class__}"
                )

            if replace:
                raise NotImplementedError(
                    "Sample is not supported for "
                    f"axis {axis} when 'replace=True'"
                )

            if n > 0 and self.shape[1] == 0:
                raise ValueError(
                    "Cannot take a sample larger than 0 when axis is empty"
                )

            columns = np.asarray(self._data.names)
            if not replace and n > columns.size:
                raise ValueError(
                    "Cannot take a larger sample "
                    "than population when 'replace=False'"
                )

            if weights is not None:
                if is_column_like(weights):
                    weights = np.asarray(weights)
                else:
                    raise ValueError(
                        "Strings can only be passed to weights "
                        "when sampling from rows on a DataFrame"
                    )

                if columns.size != len(weights):
                    raise ValueError(
                        "Weights and axis to be sampled must be of same length"
                    )

                total_weight = weights.sum()
                if total_weight != 1:
                    if not isinstance(weights.dtype, float):
                        weights = weights.astype("float64")
                    weights = weights / total_weight

            np.random.seed(random_state)
            gather_map = np.random.choice(
                columns, size=n, replace=replace, p=weights
            )

            if isinstance(self, cudf.MultiIndex):
                # TODO: Need to update this once MultiIndex is refactored,
                # should be able to treat it similar to other Frame object
                result = cudf.Index(self.to_frame(index=False)[gather_map])
            else:
                result = self[gather_map]
                if not keep_index:
                    result.index = None

            return result

    @classmethod
    @annotate("FROM_ARROW", color="orange", domain="cudf_python")
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
            cudf_indices_frame, _ = libcudf.interop.from_arrow(
                dict_indices_table, dict_indices_table.column_names
            )
            # as dictionary size can vary, it can't be a single table
            cudf_dictionaries_columns = {
                name: ColumnBase.from_arrow(dict_dictionaries[name])
                for name in dict_dictionaries.keys()
            }

            for name, codes in cudf_indices_frame.items():
                cudf_category_frame[name] = build_categorical_column(
                    cudf_dictionaries_columns[name],
                    codes,
                    mask=codes.base_mask,
                    size=codes.size,
                    ordered=dict_ordered[name],
                )

        # Handle non-dict arrays
        cudf_non_category_frame = (
            {}
            if data.num_columns == 0
            else libcudf.interop.from_arrow(data, data.column_names)[0]
        )

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

    @annotate("TO_ARROW", color="orange", domain="cudf_python")
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
        """
        return pa.Table.from_pydict(
            {name: col.to_arrow() for name, col in self._data.items()}
        )

    def drop_duplicates(
        self, keep="first", nulls_are_equal=True,
    ):
        """
        Drop duplicate rows in frame.

        keep : ["first", "last", False], default "first"
            "first" will keep the first duplicate entry, "last" will keep the
            last duplicate entry, and False will drop all duplicates.
        nulls_are_equal: bool, default True
            Null elements are considered equal to other null elements.
        """

        result = self.__class__._from_columns(
            libcudf.stream_compaction.drop_duplicates(
                list(self._columns),
                keys=range(len(self._columns)),
                keep=keep,
                nulls_are_equal=nulls_are_equal,
            ),
            self._column_names,
        )
        # TODO: _copy_type_metadata is a common pattern to apply after the
        # roundtrip from libcudf. We should build this into a factory function
        # to increase reusability.
        result._copy_type_metadata(self)
        return result

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

    def replace(
        self,
        to_replace=None,
        value=None,
        inplace=False,
        limit=None,
        regex=False,
        method=None,
    ):
        """Replace values given in ``to_replace`` with ``value``.

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
            * dict:
                - Dicts can be used to specify different replacement values
                  for different existing values. For example, {'a': 'b',
                  'y': 'z'} replaces the value ‘a’ with ‘b’ and
                  ‘y’ with ‘z’.
                  To use a dict in this way the ``value`` parameter should
                  be ``None``.
        value : scalar, dict, list-like, str, default None
            Value to replace any values matching ``to_replace`` with.
        inplace : bool, default False
            If True, in place.

        See also
        --------
        Series.fillna

        Raises
        ------
        TypeError
            - If ``to_replace`` is not a scalar, array-like, dict, or None
            - If ``to_replace`` is a dict and value is not a list, dict,
              or Series
        ValueError
            - If a list is passed to ``to_replace`` and ``value`` but they
              are not the same length.

        Returns
        -------
        result : Series
            Series after replacement. The mask and index are preserved.

        Notes
        -----
        Parameters that are currently not supported are: `limit`, `regex`,
        `method`

        Examples
        --------
        **Series**

        Scalar ``to_replace`` and ``value``

        >>> import cudf
        >>> s = cudf.Series([0, 1, 2, 3, 4])
        >>> s
        0    0
        1    1
        2    2
        3    3
        4    4
        dtype: int64
        >>> s.replace(0, 5)
        0    5
        1    1
        2    2
        3    3
        4    4
        dtype: int64

        List-like ``to_replace``

        >>> s.replace([1, 2], 10)
        0     0
        1    10
        2    10
        3     3
        4     4
        dtype: int64

        dict-like ``to_replace``

        >>> s.replace({1:5, 3:50})
        0     0
        1     5
        2     2
        3    50
        4     4
        dtype: int64
        >>> s = cudf.Series(['b', 'a', 'a', 'b', 'a'])
        >>> s
        0     b
        1     a
        2     a
        3     b
        4     a
        dtype: object
        >>> s.replace({'a': None})
        0       b
        1    <NA>
        2    <NA>
        3       b
        4    <NA>
        dtype: object

        If there is a mimatch in types of the values in
        ``to_replace`` & ``value`` with the actual series, then
        cudf exhibits different behaviour with respect to pandas
        and the pairs are ignored silently:

        >>> s = cudf.Series(['b', 'a', 'a', 'b', 'a'])
        >>> s
        0    b
        1    a
        2    a
        3    b
        4    a
        dtype: object
        >>> s.replace('a', 1)
        0    b
        1    a
        2    a
        3    b
        4    a
        dtype: object
        >>> s.replace(['a', 'c'], [1, 2])
        0    b
        1    a
        2    a
        3    b
        4    a
        dtype: object

        **DataFrame**

        Scalar ``to_replace`` and ``value``

        >>> import cudf
        >>> df = cudf.DataFrame({'A': [0, 1, 2, 3, 4],
        ...                    'B': [5, 6, 7, 8, 9],
        ...                    'C': ['a', 'b', 'c', 'd', 'e']})
        >>> df
           A  B  C
        0  0  5  a
        1  1  6  b
        2  2  7  c
        3  3  8  d
        4  4  9  e
        >>> df.replace(0, 5)
           A  B  C
        0  5  5  a
        1  1  6  b
        2  2  7  c
        3  3  8  d
        4  4  9  e

        List-like ``to_replace``

        >>> df.replace([0, 1, 2, 3], 4)
           A  B  C
        0  4  5  a
        1  4  6  b
        2  4  7  c
        3  4  8  d
        4  4  9  e
        >>> df.replace([0, 1, 2, 3], [4, 3, 2, 1])
           A  B  C
        0  4  5  a
        1  3  6  b
        2  2  7  c
        3  1  8  d
        4  4  9  e

        dict-like ``to_replace``

        >>> df.replace({0: 10, 1: 100})
             A  B  C
        0   10  5  a
        1  100  6  b
        2    2  7  c
        3    3  8  d
        4    4  9  e
        >>> df.replace({'A': 0, 'B': 5}, 100)
             A    B  C
        0  100  100  a
        1    1    6  b
        2    2    7  c
        3    3    8  d
        4    4    9  e
        """
        if limit is not None:
            raise NotImplementedError("limit parameter is not implemented yet")

        if regex:
            raise NotImplementedError("regex parameter is not implemented yet")

        if method not in ("pad", None):
            raise NotImplementedError(
                "method parameter is not implemented yet"
            )

        if not (to_replace is None and value is None):
            copy_data = self._data.copy(deep=False)
            (
                all_na_per_column,
                to_replace_per_column,
                replacements_per_column,
            ) = _get_replacement_values_for_columns(
                to_replace=to_replace,
                value=value,
                columns_dtype_map={
                    col: copy_data._data[col].dtype for col in copy_data._data
                },
            )

            for name, col in copy_data.items():
                try:
                    copy_data[name] = col.find_and_replace(
                        to_replace_per_column[name],
                        replacements_per_column[name],
                        all_na_per_column[name],
                    )
                except (KeyError, OverflowError):
                    # We need to create a deep copy if:
                    # i. `find_and_replace` was not successful or any of
                    #    `to_replace_per_column`, `replacements_per_column`,
                    #    `all_na_per_column` don't contain the `name`
                    #    that exists in `copy_data`.
                    # ii. There is an OverflowError while trying to cast
                    #     `to_replace_per_column` to `replacements_per_column`.
                    copy_data[name] = col.copy(deep=True)
        else:
            copy_data = self._data.copy(deep=True)

        result = self._from_data(copy_data, self._index)

        return self._mimic_inplace(result, inplace=inplace)

    def _copy_type_metadata(
        self, other: Frame, include_index: bool = True
    ) -> Frame:
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

        if include_index:
            if self._index is not None and other._index is not None:
                self._index._copy_type_metadata(other._index)  # type: ignore
                # When other._index is a CategoricalIndex, the current index
                # will be a NumericalIndex with an underlying CategoricalColumn
                # (the above _copy_type_metadata call will have converted the
                # column). Calling cudf.Index on that column generates the
                # appropriate index.
                if isinstance(
                    other._index, cudf.core.index.CategoricalIndex
                ) and not isinstance(
                    self._index, cudf.core.index.CategoricalIndex
                ):
                    self._index = cudf.Index(
                        cast(
                            cudf.core.index.NumericIndex, self._index
                        )._column,
                        name=self._index.name,
                    )

        return self

    def _copy_interval_data(self, other, include_index=True):
        for name, col, other_col in zip(
            self._data.keys(), self._data.values(), other._data.values()
        ):
            if isinstance(other_col, cudf.core.column.IntervalColumn):
                self._data[name] = cudf.core.column.IntervalColumn(col)

    def _postprocess_columns(self, other, include_index=True):
        self._copy_categories(other, include_index=include_index)
        self._copy_struct_names(other, include_index=include_index)
        self._copy_interval_data(other, include_index=include_index)

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
        return self.__class__._from_data(
            zip(self._column_names, data_columns), self._index
        )

    # Alias for isnull
    isna = isnull

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
        return self.__class__._from_data(
            zip(self._column_names, data_columns), self._index
        )

    # Alias for notnull
    notna = notnull

    def interleave_columns(self):
        """
        Interleave Series columns of a table into a single column.

        Converts the column major table `cols` into a row major column.

        Parameters
        ----------
        cols : input Table containing columns to interleave.

        Examples
        --------
        >>> df = DataFrame([['A1', 'A2', 'A3'], ['B1', 'B2', 'B3']])
        >>> df
        0    [A1, A2, A3]
        1    [B1, B2, B3]
        >>> df.interleave_columns()
        0    A1
        1    B1
        2    A2
        3    B2
        4    A3
        5    B3

        Returns
        -------
        The interleaved columns as a single column
        """
        if ("category" == self.dtypes).any():
            raise ValueError(
                "interleave_columns does not support 'category' dtype."
            )

        result = self._constructor_sliced(
            libcudf.reshape.interleave_columns(self)
        )

        return result

    def tile(self, count):
        """
        Repeats the rows from `self` DataFrame `count` times to form a
        new DataFrame.

        Parameters
        ----------
        self : input Table containing columns to interleave.
        count : Number of times to tile "rows". Must be non-negative.

        Examples
        --------
        >>> df  = Dataframe([[8, 4, 7], [5, 2, 3]])
        >>> count = 2
        >>> df.tile(df, count)
           0  1  2
        0  8  4  7
        1  5  2  3
        0  8  4  7
        1  5  2  3

        Returns
        -------
        The table containing the tiled "rows".
        """
        result = self.__class__._from_data(*libcudf.reshape.tile(self, count))
        result._copy_type_metadata(self)
        return result

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
            values = as_column(values)
            if values.dtype != self.dtype:
                self = self.astype(values.dtype)
            values = values.as_frame()
        outcol = libcudf.search.search_sorted(
            self, values, side, ascending=ascending, na_position=na_position
        )

        # Retrun result as cupy array if the values is non-scalar
        # If values is scalar, result is expected to be scalar.
        result = cupy.asarray(outcol.data_array_view)
        if scalar_flag:
            return result[0].item()
        else:
            return result

    @annotate("ARGSORT", color="yellow", domain="cudf_python")
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

        to_sort = (
            self
            if by is None
            else self._get_columns_by_label(list(by), downcast=False)
        )

        # If given a scalar need to construct a sequence of length # of columns
        if np.isscalar(ascending):
            ascending = [ascending] * to_sort._num_columns

        return libcudf.sort.order_by(to_sort, ascending, na_position)

    def take(self, indices, keep_index=None):
        """Return a new object containing the rows specified by *positions*

        Parameters
        ----------
        indices : array-like
            Array of ints indicating which positions to take.
        keep_index : bool, default True
            Whether to retain the index in result or not.

        Returns
        -------
        out : Series or DataFrame or Index
            New object with desired subset of rows.

        Examples
        --------
        **Series**
        >>> s = cudf.Series(['a', 'b', 'c', 'd', 'e'])
        >>> s.take([2, 0, 4, 3])
        2    c
        0    a
        4    e
        3    d
        dtype: object

        **DataFrame**

        >>> a = cudf.DataFrame({'a': [1.0, 2.0, 3.0],
        ...                    'b': cudf.Series(['a', 'b', 'c'])})
        >>> a.take([0, 2, 2])
             a  b
        0  1.0  a
        2  3.0  c
        2  3.0  c
        >>> a.take([True, False, True])
             a  b
        0  1.0  a
        2  3.0  c

        **Index**

        >>> idx = cudf.Index(['a', 'b', 'c', 'd', 'e'])
        >>> idx.take([2, 0, 4, 3])
        StringIndex(['c' 'a' 'e' 'd'], dtype='object')
        """
        # TODO: When we remove keep_index we should introduce the axis
        # parameter. We could also introduce is_copy, but that's already
        # deprecated in pandas so it's probably unnecessary. We also need to
        # introduce Index.take's allow_fill and fill_value parameters.
        if keep_index is not None:
            warnings.warn(
                "keep_index is deprecated and will be removed in the future.",
                FutureWarning,
            )
        else:
            keep_index = True

        indices = as_column(indices)
        if is_bool_dtype(indices):
            warnings.warn(
                "Calling take with a boolean array is deprecated and will be "
                "removed in the future.",
                FutureWarning,
            )
            return self._apply_boolean_mask(indices)
        return self._gather(indices, keep_index=keep_index)

    def sin(self):
        """
        Get Trigonometric sine, element-wise.

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([0.0, 0.32434, 0.5, 45, 90, 180, 360])
        >>> ser
        0      0.00000
        1      0.32434
        2      0.50000
        3     45.00000
        4     90.00000
        5    180.00000
        6    360.00000
        dtype: float64
        >>> ser.sin()
        0    0.000000
        1    0.318683
        2    0.479426
        3    0.850904
        4    0.893997
        5   -0.801153
        6    0.958916
        dtype: float64

        `sin` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [0.0, 5, 10, 15],
        ...                      'second': [100.0, 360, 720, 300]})
        >>> df
           first  second
        0    0.0   100.0
        1    5.0   360.0
        2   10.0   720.0
        3   15.0   300.0
        >>> df.sin()
              first    second
        0  0.000000 -0.506366
        1 -0.958924  0.958916
        2 -0.544021 -0.544072
        3  0.650288 -0.999756

        `sin` operation on Index:

        >>> index = cudf.Index([-0.4, 100, -180, 90])
        >>> index
        Float64Index([-0.4, 100.0, -180.0, 90.0], dtype='float64')
        >>> index.sin()
        Float64Index([-0.3894183423086505, -0.5063656411097588,
                    0.8011526357338306, 0.8939966636005579],
                    dtype='float64')
        """
        return self._unaryop("sin")

    def cos(self):
        """
        Get Trigonometric cosine, element-wise.

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([0.0, 0.32434, 0.5, 45, 90, 180, 360])
        >>> ser
        0      0.00000
        1      0.32434
        2      0.50000
        3     45.00000
        4     90.00000
        5    180.00000
        6    360.00000
        dtype: float64
        >>> ser.cos()
        0    1.000000
        1    0.947861
        2    0.877583
        3    0.525322
        4   -0.448074
        5   -0.598460
        6   -0.283691
        dtype: float64

        `cos` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [0.0, 5, 10, 15],
        ...                      'second': [100.0, 360, 720, 300]})
        >>> df
           first  second
        0    0.0   100.0
        1    5.0   360.0
        2   10.0   720.0
        3   15.0   300.0
        >>> df.cos()
              first    second
        0  1.000000  0.862319
        1  0.283662 -0.283691
        2 -0.839072 -0.839039
        3 -0.759688 -0.022097

        `cos` operation on Index:

        >>> index = cudf.Index([-0.4, 100, -180, 90])
        >>> index
        Float64Index([-0.4, 100.0, -180.0, 90.0], dtype='float64')
        >>> index.cos()
        Float64Index([ 0.9210609940028851,  0.8623188722876839,
                    -0.5984600690578581, -0.4480736161291701],
                    dtype='float64')
        """
        return self._unaryop("cos")

    def tan(self):
        """
        Get Trigonometric tangent, element-wise.

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([0.0, 0.32434, 0.5, 45, 90, 180, 360])
        >>> ser
        0      0.00000
        1      0.32434
        2      0.50000
        3     45.00000
        4     90.00000
        5    180.00000
        6    360.00000
        dtype: float64
        >>> ser.tan()
        0    0.000000
        1    0.336213
        2    0.546302
        3    1.619775
        4   -1.995200
        5    1.338690
        6   -3.380140
        dtype: float64

        `tan` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [0.0, 5, 10, 15],
        ...                      'second': [100.0, 360, 720, 300]})
        >>> df
           first  second
        0    0.0   100.0
        1    5.0   360.0
        2   10.0   720.0
        3   15.0   300.0
        >>> df.tan()
              first     second
        0  0.000000  -0.587214
        1 -3.380515  -3.380140
        2  0.648361   0.648446
        3 -0.855993  45.244742

        `tan` operation on Index:

        >>> index = cudf.Index([-0.4, 100, -180, 90])
        >>> index
        Float64Index([-0.4, 100.0, -180.0, 90.0], dtype='float64')
        >>> index.tan()
        Float64Index([-0.4227932187381618,  -0.587213915156929,
                    -1.3386902103511544, -1.995200412208242],
                    dtype='float64')
        """
        return self._unaryop("tan")

    def asin(self):
        """
        Get Trigonometric inverse sine, element-wise.

        The inverse of sine so that, if y = x.sin(), then x = y.asin()

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-1, 0, 1, 0.32434, 0.5])
        >>> ser.asin()
        0   -1.570796
        1    0.000000
        2    1.570796
        3    0.330314
        4    0.523599
        dtype: float64

        `asin` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-1, 0, 0.5],
        ...                      'second': [0.234, 0.3, 0.1]})
        >>> df
           first  second
        0   -1.0   0.234
        1    0.0   0.300
        2    0.5   0.100
        >>> df.asin()
              first    second
        0 -1.570796  0.236190
        1  0.000000  0.304693
        2  0.523599  0.100167

        `asin` operation on Index:

        >>> index = cudf.Index([-1, 0.4, 1, 0.3])
        >>> index
        Float64Index([-1.0, 0.4, 1.0, 0.3], dtype='float64')
        >>> index.asin()
        Float64Index([-1.5707963267948966, 0.41151684606748806,
                    1.5707963267948966, 0.3046926540153975],
                    dtype='float64')
        """
        return self._unaryop("asin")

    def acos(self):
        """
        Get Trigonometric inverse cosine, element-wise.

        The inverse of cos so that, if y = x.cos(), then x = y.acos()

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-1, 0, 1, 0.32434, 0.5])
        >>> ser.acos()
        0    3.141593
        1    1.570796
        2    0.000000
        3    1.240482
        4    1.047198
        dtype: float64

        `acos` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-1, 0, 0.5],
        ...                      'second': [0.234, 0.3, 0.1]})
        >>> df
           first  second
        0   -1.0   0.234
        1    0.0   0.300
        2    0.5   0.100
        >>> df.acos()
              first    second
        0  3.141593  1.334606
        1  1.570796  1.266104
        2  1.047198  1.470629

        `acos` operation on Index:

        >>> index = cudf.Index([-1, 0.4, 1, 0, 0.3])
        >>> index
        Float64Index([-1.0, 0.4, 1.0, 0.0, 0.3], dtype='float64')
        >>> index.acos()
        Float64Index([ 3.141592653589793, 1.1592794807274085, 0.0,
                    1.5707963267948966,  1.266103672779499],
                    dtype='float64')
        """
        result = self.copy(deep=False)
        for col in result._data:
            min_float_dtype = cudf.utils.dtypes.get_min_float_dtype(
                result._data[col]
            )
            result._data[col] = result._data[col].astype(min_float_dtype)
        result = result._unaryop("acos")
        result = result.mask((result < 0) | (result > np.pi + 1))
        return result

    def atan(self):
        """
        Get Trigonometric inverse tangent, element-wise.

        The inverse of tan so that, if y = x.tan(), then x = y.atan()

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-1, 0, 1, 0.32434, 0.5, -10])
        >>> ser
        0    -1.00000
        1     0.00000
        2     1.00000
        3     0.32434
        4     0.50000
        5   -10.00000
        dtype: float64
        >>> ser.atan()
        0   -0.785398
        1    0.000000
        2    0.785398
        3    0.313635
        4    0.463648
        5   -1.471128
        dtype: float64

        `atan` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-1, -10, 0.5],
        ...                      'second': [0.234, 0.3, 10]})
        >>> df
           first  second
        0   -1.0   0.234
        1  -10.0   0.300
        2    0.5  10.000
        >>> df.atan()
              first    second
        0 -0.785398  0.229864
        1 -1.471128  0.291457
        2  0.463648  1.471128

        `atan` operation on Index:

        >>> index = cudf.Index([-1, 0.4, 1, 0, 0.3])
        >>> index
        Float64Index([-1.0, 0.4, 1.0, 0.0, 0.3], dtype='float64')
        >>> index.atan()
        Float64Index([-0.7853981633974483,  0.3805063771123649,
                                    0.7853981633974483, 0.0,
                                    0.2914567944778671],
                    dtype='float64')
        """
        return self._unaryop("atan")

    def exp(self):
        """
        Get the exponential of all elements, element-wise.

        Exponential is the inverse of the log function,
        so that x.exp().log() = x

        Returns
        -------
        DataFrame/Series/Index
            Result of the element-wise exponential.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-1, 0, 1, 0.32434, 0.5, -10, 100])
        >>> ser
        0     -1.00000
        1      0.00000
        2      1.00000
        3      0.32434
        4      0.50000
        5    -10.00000
        6    100.00000
        dtype: float64
        >>> ser.exp()
        0    3.678794e-01
        1    1.000000e+00
        2    2.718282e+00
        3    1.383117e+00
        4    1.648721e+00
        5    4.539993e-05
        6    2.688117e+43
        dtype: float64

        `exp` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-1, -10, 0.5],
        ...                      'second': [0.234, 0.3, 10]})
        >>> df
           first  second
        0   -1.0   0.234
        1  -10.0   0.300
        2    0.5  10.000
        >>> df.exp()
              first        second
        0  0.367879      1.263644
        1  0.000045      1.349859
        2  1.648721  22026.465795

        `exp` operation on Index:

        >>> index = cudf.Index([-1, 0.4, 1, 0, 0.3])
        >>> index
        Float64Index([-1.0, 0.4, 1.0, 0.0, 0.3], dtype='float64')
        >>> index.exp()
        Float64Index([0.36787944117144233,  1.4918246976412703,
                      2.718281828459045, 1.0,  1.3498588075760032],
                    dtype='float64')
        """
        return self._unaryop("exp")

    def log(self):
        """
        Get the natural logarithm of all elements, element-wise.

        Natural logarithm is the inverse of the exp function,
        so that x.log().exp() = x

        Returns
        -------
        DataFrame/Series/Index
            Result of the element-wise natural logarithm.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-1, 0, 1, 0.32434, 0.5, -10, 100])
        >>> ser
        0     -1.00000
        1      0.00000
        2      1.00000
        3      0.32434
        4      0.50000
        5    -10.00000
        6    100.00000
        dtype: float64
        >>> ser.log()
        0         NaN
        1        -inf
        2    0.000000
        3   -1.125963
        4   -0.693147
        5         NaN
        6    4.605170
        dtype: float64

        `log` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-1, -10, 0.5],
        ...                      'second': [0.234, 0.3, 10]})
        >>> df
           first  second
        0   -1.0   0.234
        1  -10.0   0.300
        2    0.5  10.000
        >>> df.log()
              first    second
        0       NaN -1.452434
        1       NaN -1.203973
        2 -0.693147  2.302585

        `log` operation on Index:

        >>> index = cudf.Index([10, 11, 500.0])
        >>> index
        Float64Index([10.0, 11.0, 500.0], dtype='float64')
        >>> index.log()
        Float64Index([2.302585092994046, 2.3978952727983707,
                    6.214608098422191], dtype='float64')
        """
        return self._unaryop("log")

    def sqrt(self):
        """
        Get the non-negative square-root of all elements, element-wise.

        Returns
        -------
        DataFrame/Series/Index
            Result of the non-negative
            square-root of each element.

        Examples
        --------
        >>> import cudf
        >>> import cudf
        >>> ser = cudf.Series([10, 25, 81, 1.0, 100])
        >>> ser
        0     10.0
        1     25.0
        2     81.0
        3      1.0
        4    100.0
        dtype: float64
        >>> ser.sqrt()
        0     3.162278
        1     5.000000
        2     9.000000
        3     1.000000
        4    10.000000
        dtype: float64

        `sqrt` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-10.0, 100, 625],
        ...                      'second': [1, 2, 0.4]})
        >>> df
           first  second
        0  -10.0     1.0
        1  100.0     2.0
        2  625.0     0.4
        >>> df.sqrt()
           first    second
        0    NaN  1.000000
        1   10.0  1.414214
        2   25.0  0.632456

        `sqrt` operation on Index:

        >>> index = cudf.Index([-10.0, 100, 625])
        >>> index
        Float64Index([-10.0, 100.0, 625.0], dtype='float64')
        >>> index.sqrt()
        Float64Index([nan, 10.0, 25.0], dtype='float64')
        """
        return self._unaryop("sqrt")

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

    # Rounding
    def ceil(self):
        """
        Rounds each value upward to the smallest integral value not less
        than the original.

        Returns
        -------
        DataFrame or Series
            Ceiling value of each element.

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([1.1, 2.8, 3.5, 4.5])
        >>> series
        0    1.1
        1    2.8
        2    3.5
        3    4.5
        dtype: float64
        >>> series.ceil()
        0    2.0
        1    3.0
        2    4.0
        3    5.0
        dtype: float64
        """

        warnings.warn(
            "Series.ceil and DataFrame.ceil are deprecated and will be \
                removed in the future",
            DeprecationWarning,
        )

        return self._unaryop("ceil")

    def floor(self):
        """Rounds each value downward to the largest integral value not greater
        than the original.

        Returns
        -------
        DataFrame or Series
            Flooring value of each element.

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([-1.9, 2, 0.2, 1.5, 0.0, 3.0])
        >>> series
        0   -1.9
        1    2.0
        2    0.2
        3    1.5
        4    0.0
        5    3.0
        dtype: float64
        >>> series.floor()
        0   -2.0
        1    2.0
        2    0.0
        3    1.0
        4    0.0
        5    3.0
        dtype: float64
        """

        warnings.warn(
            "Series.ceil and DataFrame.ceil are deprecated and will be \
                removed in the future",
            DeprecationWarning,
        )

        return self._unaryop("floor")

    def scale(self):
        """
        Scale values to [0, 1] in float64

        Returns
        -------
        DataFrame or Series
            Values scaled to [0, 1].

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([10, 11, 12, 0.5, 1])
        >>> series
        0    10.0
        1    11.0
        2    12.0
        3     0.5
        4     1.0
        dtype: float64
        >>> series.scale()
        0    0.826087
        1    0.913043
        2    1.000000
        3    0.000000
        4    0.043478
        dtype: float64
        """
        vmin = self.min()
        vmax = self.max()
        scaled = (self - vmin) / (vmax - vmin)
        scaled._index = self._index.copy(deep=False)
        return scaled

    def _merge(
        self,
        right,
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        how="inner",
        sort=False,
        indicator=False,
        suffixes=("_x", "_y"),
    ):
        lhs, rhs = self, right
        merge_cls = Merge
        if how == "right":
            # Merge doesn't support right, so just swap
            how = "left"
            lhs, rhs = right, self
            left_on, right_on = right_on, left_on
            left_index, right_index = right_index, left_index
            suffixes = (suffixes[1], suffixes[0])
        elif how in {"leftsemi", "leftanti"}:
            merge_cls = MergeSemi

        return merge_cls(
            lhs,
            rhs,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            sort=sort,
            indicator=indicator,
            suffixes=suffixes,
        ).perform_merge()

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
        return libcudf.sort.is_sorted(
            self, ascending=ascending, null_position=null_position
        )

    def _split(self, splits, keep_index=True):
        results = libcudf.copying.table_split(
            self, splits, keep_index=keep_index
        )
        return [self.__class__._from_data(*result) for result in results]

    def _encode(self):
        data, index, indices = libcudf.transform.table_encode(self)
        for name, col in data.items():
            data[name] = col._with_type_metadata(self._data[name].dtype)
        keys = self.__class__._from_data(data, index)
        return keys, indices

    def _unaryop(self, op):
        data_columns = (col.unary_operator(op) for col in self._columns)
        return self.__class__._from_data(
            zip(self._column_names, data_columns), self._index
        )

    def _binaryop(
        self,
        other: T,
        fn: str,
        fill_value: Any = None,
        reflect: bool = False,
        *args,
        **kwargs,
    ) -> Frame:
        """Perform a binary operation between two frames.

        Parameters
        ----------
        other : Frame
            The second operand.
        fn : str
            The operation to perform.
        fill_value : Any, default None
            The value to replace null values with. If ``None``, nulls are not
            filled before the operation.
        reflect : bool, default False
            If ``True``, swap the order of the operands. See
            https://docs.python.org/3/reference/datamodel.html#object.__ror__
            for more information on when this is necessary.

        Returns
        -------
        Frame
            A new instance containing the result of the operation.
        """
        raise NotImplementedError

    @classmethod
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

            # Handle object columns that are empty or
            # all nulls when performing binary operations
            if (
                left_column.dtype == "object"
                and left_column.null_count == len(left_column)
                and fill_value is None
            ):
                if fn in (
                    "add",
                    "sub",
                    "mul",
                    "mod",
                    "pow",
                    "truediv",
                    "floordiv",
                ):
                    output[col] = left_column
                elif fn in ("eq", "lt", "le", "gt", "ge"):
                    output[col] = left_column.notnull()
                elif fn == "ne":
                    output[col] = left_column.isnull()
                continue

            if right_column is cudf.NA:
                right_column = cudf.Scalar(
                    right_column, dtype=left_column.dtype
                )
            elif not isinstance(right_column, ColumnBase):
                right_column = left_column.normalize_binop_value(right_column)

            fn_apply = fn
            if fn == "truediv":
                # Decimals in libcudf don't support truediv, see
                # https://github.com/rapidsai/cudf/pull/7435 for explanation.
                if is_decimal_dtype(left_column.dtype):
                    fn_apply = "div"

                # Division with integer types results in a suitable float.
                truediv_type = {
                    np.int8: np.float32,
                    np.int16: np.float32,
                    np.int32: np.float32,
                    np.int64: np.float64,
                    np.uint8: np.float32,
                    np.uint16: np.float32,
                    np.uint32: np.float64,
                    np.uint64: np.float64,
                    np.bool_: np.float32,
                }.get(left_column.dtype.type)
                if truediv_type is not None:
                    left_column = left_column.astype(truediv_type)

            output_mask = None
            if fill_value is not None:
                if is_scalar(right_column):
                    if left_column.nullable:
                        left_column = left_column.fillna(fill_value)
                else:
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

            # For bitwise operations we must verify whether the input column
            # types are valid, and if so, whether we need to coerce the output
            # columns to booleans.
            coerce_to_bool = False
            if fn_apply in {"and", "or", "xor"}:
                err_msg = (
                    f"Operation 'bitwise {fn_apply}' not supported between "
                    f"{left_column.dtype.type.__name__} and {{}}"
                )
                if right_column is None:
                    raise TypeError(err_msg.format(type(None)))

                try:
                    left_is_bool = issubdtype(left_column.dtype, np.bool_)
                    right_is_bool = issubdtype(right_column.dtype, np.bool_)
                except TypeError:
                    raise TypeError(err_msg.format(type(right_column)))

                coerce_to_bool = left_is_bool or right_is_bool

                if not (
                    (left_is_bool or issubdtype(left_column.dtype, np.integer))
                    and (
                        right_is_bool
                        or issubdtype(right_column.dtype, np.integer)
                    )
                ):
                    raise TypeError(
                        err_msg.format(right_column.dtype.type.__name__)
                    )

            outcol = (
                left_column.binary_operator(
                    fn_apply, right_column, reflect=reflect
                )
                if right_column is not None
                else column_empty(
                    left_column.size, left_column.dtype, masked=True
                )
            )

            if output_mask is not None:
                outcol = outcol.set_mask(output_mask)

            if coerce_to_bool:
                outcol = outcol.astype(np.bool_)

            output[col] = outcol

        return output

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
            return NotImplemented
        if reflect:
            lhs, rhs = rhs, lhs

        result = lhs.dot(rhs)
        if len(result.shape) == 1:
            return cudf.Series(result)
        if len(result.shape) == 2:
            return cudf.DataFrame(result)
        return result.item()

    # Binary arithmetic operations.
    def __add__(self, other):
        return self._binaryop(other, "add")

    def __radd__(self, other):
        return self._binaryop(other, "add", reflect=True)

    def __sub__(self, other):
        return self._binaryop(other, "sub")

    def __rsub__(self, other):
        return self._binaryop(other, "sub", reflect=True)

    def __matmul__(self, other):
        return self.dot(other)

    def __rmatmul__(self, other):
        return self.dot(other, reflect=True)

    def __mul__(self, other):
        return self._binaryop(other, "mul")

    def __rmul__(self, other):
        return self._binaryop(other, "mul", reflect=True)

    def __mod__(self, other):
        return self._binaryop(other, "mod")

    def __rmod__(self, other):
        return self._binaryop(other, "mod", reflect=True)

    def __pow__(self, other):
        return self._binaryop(other, "pow")

    def __rpow__(self, other):
        return self._binaryop(other, "pow", reflect=True)

    def __floordiv__(self, other):
        return self._binaryop(other, "floordiv")

    def __rfloordiv__(self, other):
        return self._binaryop(other, "floordiv", reflect=True)

    def __truediv__(self, other):
        return self._binaryop(other, "truediv")

    def __rtruediv__(self, other):
        return self._binaryop(other, "truediv", reflect=True)

    def __and__(self, other):
        return self._binaryop(other, "and")

    def __or__(self, other):
        return self._binaryop(other, "or")

    def __xor__(self, other):
        return self._binaryop(other, "xor")

    # Binary rich comparison operations.
    def __eq__(self, other):
        return self._binaryop(other, "eq")

    def __ne__(self, other):
        return self._binaryop(other, "ne")

    def __lt__(self, other):
        return self._binaryop(other, "lt")

    def __le__(self, other):
        return self._binaryop(other, "le")

    def __gt__(self, other):
        return self._binaryop(other, "gt")

    def __ge__(self, other):
        return self._binaryop(other, "ge")

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

    def min(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs,
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

    def max(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs,
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

    def mean(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
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

    def kurtosis(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        """
        Return Fisher's unbiased kurtosis of a sample.

        Kurtosis obtained using Fisher’s definition of
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
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        return self.kurtosis(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    def skew(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
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
            "all", axis=axis, skipna=skipna, level=level, **kwargs,
        )

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
            "any", axis=axis, skipna=skipna, level=level, **kwargs,
        )

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

    def median(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
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

    # Scans
    def _scan(self, op, axis=None, skipna=True, cast_to_int=False):
        skipna = True if skipna is None else skipna

        results = {}
        for name, col in self._data.items():
            if skipna:
                result_col = self._data[name].nans_to_nulls()
            else:
                result_col = self._data[name].copy()
                if result_col.has_nulls:
                    # Workaround as find_first_value doesn't seem to work
                    # incase of bools.
                    first_index = int(
                        result_col.isnull().astype("int8").find_first_value(1)
                    )
                    result_col[first_index:] = None

            if (
                cast_to_int
                and not is_decimal_dtype(result_col.dtype)
                and (
                    np.issubdtype(result_col.dtype, np.integer)
                    or np.issubdtype(result_col.dtype, np.bool_)
                )
            ):
                # For reductions that accumulate a value (e.g. sum, not max)
                # pandas returns an int64 dtype for all int or bool dtypes.
                result_col = result_col.astype(np.int64)
            results[name] = result_col._apply_scan_op(op)
        # TODO: This will work for Index because it's passing self._index
        # (which is None), but eventually we may want to remove that parameter
        # for Index._from_data and simplify.
        return self._from_data(results, index=self._index)

    def cummin(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return cumulative minimum of the Series or DataFrame.

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.

        Returns
        -------
        Series or DataFrame

        Examples
        --------
        **Series**

        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.cummin()
        0    1
        1    1
        2    1
        3    1
        4    1

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.cummin()
           a  b
        0  1  7
        1  1  7
        2  1  7
        3  1  7
        """
        return self._scan("min", axis=axis, skipna=skipna, *args, **kwargs)

    def cummax(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return cumulative maximum of the Series or DataFrame.

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.

        Returns
        -------
        Series or DataFrame

        Examples
        --------
        **Series**

        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.cummax()
        0    1
        1    5
        2    5
        3    5
        4    5

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.cummax()
           a   b
        0  1   7
        1  2   8
        2  3   9
        3  4  10
        """
        return self._scan("max", axis=axis, skipna=skipna, *args, **kwargs)

    def cumsum(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return cumulative sum of the Series or DataFrame.

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.


        Returns
        -------
        Series or DataFrame

        Examples
        --------
        **Series**

        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.cumsum()
        0    1
        1    6
        2    8
        3    12
        4    15

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> s.cumsum()
            a   b
        0   1   7
        1   3  15
        2   6  24
        3  10  34
        """
        return self._scan(
            "sum", axis=axis, skipna=skipna, cast_to_int=True, *args, **kwargs
        )

    def cumprod(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return cumulative product of the Series or DataFrame.

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.

        Returns
        -------
        Series or DataFrame

        Examples
        --------
        **Series**

        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.cumprod()
        0    1
        1    5
        2    10
        3    40
        4    120

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> s.cumprod()
            a     b
        0   1     7
        1   2    56
        2   6   504
        3  24  5040
        """
        return self._scan(
            "prod", axis=axis, skipna=skipna, cast_to_int=True, *args, **kwargs
        )

    @ioutils.doc_to_json()
    def to_json(self, path_or_buf=None, *args, **kwargs):
        """{docstring}"""

        return cudf.io.json.to_json(
            self, path_or_buf=path_or_buf, *args, **kwargs
        )

    @ioutils.doc_to_hdf()
    def to_hdf(self, path_or_buf, key, *args, **kwargs):
        """{docstring}"""

        cudf.io.hdf.to_hdf(path_or_buf, key, self, *args, **kwargs)

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""

        return cudf.io.dlpack.to_dlpack(self)

    def to_string(self):
        """
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
        '   key   val\\n0    0  10.0\\n1    1  11.0\\n2    2  12.0'
        """
        return self.__repr__()

    def __str__(self):
        return self.to_string()

    def __deepcopy__(self, memo):
        return self.copy(deep=True)

    def __copy__(self):
        return self.copy(deep=False)

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
        return self._from_data(
            {
                name: col.copy().nans_to_nulls()
                for name, col in self._data.items()
            },
            self._index,
        )

    def __invert__(self):
        """Bitwise invert (~) for integral dtypes, logical NOT for bools."""
        return self._from_data(
            {
                name: _apply_inverse_column(col)
                for name, col in self._data.items()
            },
            self._index,
        )

    def add(self, other, axis, level=None, fill_value=None):
        """
        Get Addition of dataframe or series and other, element-wise (binary
        operator `add`).

        Equivalent to ``frame + other``, but with support to substitute a
        ``fill_value`` for missing data in one of the inputs.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361
        >>> df.add(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        **Series**

        >>> a = cudf.Series([1, 1, 1, None], index=['a', 'b', 'c', 'd'])
        >>> b = cudf.Series([1, None, 1, None], index=['a', 'b', 'd', 'e'])
        >>> a.add(b)
        a       2
        b    <NA>
        c    <NA>
        d    <NA>
        e    <NA>
        dtype: int64
        >>> a.add(b, fill_value=0)
        a       2
        b       1
        c       1
        d       1
        e    <NA>
        dtype: int64
        """
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "add", fill_value)

    def radd(self, other, axis, level=None, fill_value=None):
        """
        Get Addition of dataframe or series and other, element-wise (binary
        operator `radd`).

        Equivalent to ``other + frame``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `add`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df + 1
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361
        >>> df.radd(1)
                   angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361

        **Series**

        >>> a = cudf.Series([1, 2, 3, None], index=['a', 'b', 'c', 'd'])
        >>> a
        a       1
        b       2
        c       3
        d    <NA>
        dtype: int64
        >>> b = cudf.Series([1, None, 1, None], index=['a', 'b', 'd', 'e'])
        >>> b
        a       1
        b    <NA>
        d       1
        e    <NA>
        dtype: int64
        >>> a.add(b, fill_value=0)
        a       2
        b       2
        c       3
        d       1
        e    <NA>
        dtype: int64

        """

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "add", fill_value, reflect=True)

    def subtract(self, other, axis, level=None, fill_value=None):
        """
        Get Subtraction of dataframe or series and other, element-wise (binary
        operator `sub`).

        Equivalent to ``frame - other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rsub`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df.sub(1)
                   angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359
        >>> df.sub([1, 2])
                   angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358

        **Series**

        >>> a = cudf.Series([10, 20, None, 30, None], index=['a', 'b', 'c', 'd', 'e'])
        >>> a
        a      10
        b      20
        c    <NA>
        d      30
        e    <NA>
        dtype: int64
        >>> b = cudf.Series([1, None, 2, 30], index=['a', 'c', 'b', 'd'])
        >>> b
        a       1
        c    <NA>
        b       2
        d      30
        dtype: int64
        >>> a.subtract(b, fill_value=2)
        a       9
        b      18
        c    <NA>
        d       0
        e    <NA>
        dtype: int64

        """  # noqa: E501

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "sub", fill_value)

    sub = subtract

    def rsub(self, other, axis, level=None, fill_value=None):
        """
        Get Subtraction of dataframe or series and other, element-wise (binary
        operator `rsub`).

        Equivalent to ``other - frame``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `sub`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360
        >>> df.rsub(1)
                   angles  degrees
        circle          1     -359
        triangle       -2     -179
        rectangle      -3     -359
        >>> df.rsub([1, 2])
                   angles  degrees
        circle          1     -358
        triangle       -2     -178
        rectangle      -3     -358

        **Series**

        >>> import cudf
        >>> a = cudf.Series([1, 2, 3, None], index=['a', 'b', 'c', 'd'])
        >>> a
        a       1
        b       2
        c       3
        d    <NA>
        dtype: int64
        >>> b = cudf.Series([1, None, 2, None], index=['a', 'b', 'd', 'e'])
        >>> b
        a       1
        b    <NA>
        d       2
        e    <NA>
        dtype: int64
        >>> a.rsub(b, fill_value=10)
        a       0
        b       8
        c       7
        d      -8
        e    <NA>
        dtype: int64
        """
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "sub", fill_value, reflect=True)

    def multiply(self, other, axis, level=None, fill_value=None):
        """
        Get Multiplication of dataframe or series and other, element-wise
        (binary operator `mul`).

        Equivalent to ``frame * other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rmul`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> other = cudf.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> df * other
                   angles degrees
        circle          0    <NA>
        triangle        9    <NA>
        rectangle      16    <NA>
        >>> df.mul(other, fill_value=0)
                   angles  degrees
        circle          0        0
        triangle        9        0
        rectangle      16        0

        **Series**

        >>> import cudf
        >>> a = cudf.Series([1, 2, 3, None], index=['a', 'b', 'c', 'd'])
        >>> a
        a       1
        b       2
        c       3
        d    <NA>
        dtype: int64
        >>> b = cudf.Series([1, None, 2, None], index=['a', 'b', 'd', 'e'])
        >>> b
        a       1
        b    <NA>
        d       2
        e    <NA>
        dtype: int64
        >>> a.multiply(b, fill_value=0)
        a       1
        b       0
        c       0
        d       0
        e    <NA>
        dtype: int64

        """

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "mul", fill_value)

    mul = multiply

    def rmul(self, other, axis, level=None, fill_value=None):
        """
        Get Multiplication of dataframe or series and other, element-wise
        (binary operator `rmul`).

        Equivalent to ``other * frame``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `mul`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> other = cudf.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other * df
                   angles degrees
        circle          0    <NA>
        triangle        9    <NA>
        rectangle      16    <NA>
        >>> df.rmul(other, fill_value=0)
                   angles  degrees
        circle          0        0
        triangle        9        0
        rectangle      16        0

        **Series**

        >>> import cudf
        >>> a = cudf.Series([10, 20, None, 30, 40], index=['a', 'b', 'c', 'd', 'e'])
        >>> a
        a      10
        b      20
        c    <NA>
        d      30
        e      40
        dtype: int64
        >>> b = cudf.Series([None, 1, 20, 5, 4], index=['a', 'b', 'd', 'e', 'f'])
        >>> b
        a    <NA>
        b       1
        d      20
        e       5
        f       4
        dtype: int64
        >>> a.rmul(b, fill_value=2)
        a      20
        b      20
        c    <NA>
        d     600
        e     200
        f       8
        dtype: int64
        """  # noqa E501

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "mul", fill_value, reflect=True)

    def mod(self, other, axis, level=None, fill_value=None):
        """
        Get Modulo division of dataframe or series and other, element-wise
        (binary operator `mod`).

        Equivalent to ``frame % other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rmod`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df % 100
                   angles  degrees
        circle          0       60
        triangle        3       80
        rectangle       4       60
        >>> df.mod(100)
                   angles  degrees
        circle          0       60
        triangle        3       80
        rectangle       4       60

        **Series**

        >>> import cudf
        >>> series = cudf.Series([10, 20, 30])
        >>> series
        0    10
        1    20
        2    30
        dtype: int64
        >>> series.mod(4)
        0    2
        1    0
        2    2
        dtype: int64


        """
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "mod", fill_value)

    def rmod(self, other, axis, level=None, fill_value=None):
        """
        Get Modulo division of dataframe or series and other, element-wise
        (binary operator `rmod`).

        Equivalent to ``other % frame``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `mod`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [1, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> 100 % df
                   angles  degrees
        circle          0      100
        triangle        1      100
        rectangle       0      100
        >>> df.rmod(100)
                   angles  degrees
        circle          0      100
        triangle        1      100
        rectangle       0      100

        **Series**

        >>> import cudf
        >>> a = cudf.Series([10, 20, None, 30, 40], index=['a', 'b', 'c', 'd', 'e'])
        >>> a
        a      10
        b      20
        c    <NA>
        d      30
        e      40
        dtype: int64
        >>> b = cudf.Series([None, 1, 20, 5, 4], index=['a', 'b', 'd', 'e', 'f'])
        >>> b
        a    <NA>
        b       1
        d      20
        e       5
        f       4
        dtype: int64
        >>> a.rmod(b, fill_value=10)
        a       0
        b       1
        c    <NA>
        d      20
        e       5
        f       4
        dtype: int64
        """  # noqa E501

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "mod", fill_value, reflect=True)

    def pow(self, other, axis, level=None, fill_value=None):
        """
        Get Exponential power of dataframe series and other, element-wise
        (binary operator `pow`).

        Equivalent to ``frame ** other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rpow`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [1, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df ** 2
                   angles  degrees
        circle          0   129600
        triangle        9    32400
        rectangle      16   129600
        >>> df.pow(2)
                   angles  degrees
        circle          0   129600
        triangle        9    32400
        rectangle      16   129600

        **Series**

        >>> import cudf
        >>> a = cudf.Series([1, 2, 3, None], index=['a', 'b', 'c', 'd'])
        >>> a
        a       1
        b       2
        c       3
        d    <NA>
        dtype: int64
        >>> b = cudf.Series([10, None, 12, None], index=['a', 'b', 'd', 'e'])
        >>> b
        a      10
        b    <NA>
        d      12
        e    <NA>
        dtype: int64
        >>> a.pow(b, fill_value=0)
        a       1
        b       1
        c       1
        d       0
        e    <NA>
        dtype: int64
        """

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "pow", fill_value)

    def rpow(self, other, axis, level=None, fill_value=None):
        """
        Get Exponential power of dataframe or series and other, element-wise
        (binary operator `pow`).

        Equivalent to ``other ** frame``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `pow`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [1, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> 1 ** df
                   angles  degrees
        circle          1        1
        triangle        1        1
        rectangle       1        1
        >>> df.rpow(1)
                   angles  degrees
        circle          1        1
        triangle        1        1
        rectangle       1        1

        **Series**

        >>> import cudf
        >>> a = cudf.Series([1, 2, 3, None], index=['a', 'b', 'c', 'd'])
        >>> a
        a       1
        b       2
        c       3
        d    <NA>
        dtype: int64
        >>> b = cudf.Series([10, None, 12, None], index=['a', 'b', 'd', 'e'])
        >>> b
        a      10
        b    <NA>
        d      12
        e    <NA>
        dtype: int64
        >>> a.rpow(b, fill_value=0)
        a      10
        b       0
        c       0
        d       1
        e    <NA>
        dtype: int64
        """

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "pow", fill_value, reflect=True)

    def floordiv(self, other, axis, level=None, fill_value=None):
        """
        Get Integer division of dataframe or series and other, element-wise
        (binary operator `floordiv`).

        Equivalent to ``frame // other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rfloordiv`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [1, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df.floordiv(2)
                   angles  degrees
        circle          0      180
        triangle        1       90
        rectangle       2      180
        >>> df // 2
                   angles  degrees
        circle          0      180
        triangle        1       90
        rectangle       2      180

        **Series**

        >>> import cudf
        >>> a = cudf.Series([1, 1, 1, None], index=['a', 'b', 'c', 'd'])
        >>> a
        a       1
        b       1
        c       1
        d    <NA>
        dtype: int64
        >>> b = cudf.Series([1, None, 1, None], index=['a', 'b', 'd', 'e'])
        >>> b
        a       1
        b    <NA>
        d       1
        e    <NA>
        dtype: int64
        >>> a.floordiv(b)
        a       1
        b    <NA>
        c    <NA>
        d    <NA>
        e    <NA>
        dtype: int64
        """

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "floordiv", fill_value)

    def rfloordiv(self, other, axis, level=None, fill_value=None):
        """
        Get Integer division of dataframe or series and other, element-wise
        (binary operator `rfloordiv`).

        Equivalent to ``other // dataframe``, but with support to substitute
        a fill_value for missing data in one of the inputs. With reverse
        version, `floordiv`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'col1': [10, 11, 23],
        ... 'col2': [101, 122, 321]})
        >>> df
           col1  col2
        0    10   101
        1    11   122
        2    23   321
        >>> df.rfloordiv(df)
           col1  col2
        0     1     1
        1     1     1
        2     1     1
        >>> df.rfloordiv(200)
           col1  col2
        0    20     1
        1    18     1
        2     8     0
        >>> df.rfloordiv(100)
           col1  col2
        0    10     0
        1     9     0
        2     4     0

        **Series**

        >>> import cudf
        >>> s = cudf.Series([1, 2, 10, 17])
        >>> s
        0     1
        1     2
        2    10
        3    17
        dtype: int64
        >>> s.rfloordiv(100)
        0    100
        1     50
        2     10
        3      5
        dtype: int64
        >>> s = cudf.Series([10, 20, None])
        >>> s
        0      10
        1      20
        2    <NA>
        dtype: int64
        >>> s.rfloordiv(200)
        0      20
        1      10
        2    <NA>
        dtype: int64
        >>> s.rfloordiv(200, fill_value=2)
        0     20
        1     10
        2    100
        dtype: int64
        """

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "floordiv", fill_value, reflect=True)

    def truediv(self, other, axis, level=None, fill_value=None):
        """
        Get Floating division of dataframe or series and other, element-wise
        (binary operator `truediv`).

        Equivalent to ``frame / other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rtruediv`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df.truediv(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0
        >>> df.div(10)
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0
        >>> df / 10
                   angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0

        **Series**

        >>> import cudf
        >>> a = cudf.Series([1, 10, 20, None], index=['a', 'b', 'c', 'd'])
        >>> a
        a       1
        b      10
        c      20
        d    <NA>
        dtype: int64
        >>> b = cudf.Series([1, None, 2, None], index=['a', 'b', 'd', 'e'])
        >>> b
        a       1
        b    <NA>
        d       2
        e    <NA>
        dtype: int64
        >>> a.truediv(b, fill_value=0)
        a     1.0
        b     Inf
        c     Inf
        d     0.0
        e    <NA>
        dtype: float64
        """

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "truediv", fill_value)

    # Alias for truediv
    div = truediv
    divide = truediv

    def rtruediv(self, other, axis, level=None, fill_value=None):
        """
        Get Floating division of dataframe or series and other, element-wise
        (binary operator `rtruediv`).

        Equivalent to ``other / frame``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `truediv`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        axis : int or string
            Only ``0`` is supported for series, ``1`` or ``columns`` supported
            for dataframe
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame or Series
            Result of the arithmetic operation.

        Examples
        --------

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360
        >>> df.rtruediv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778
        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778
        >>> 10 / df
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778

        **Series**

        >>> import cudf
        >>> a = cudf.Series([10, 20, None, 30], index=['a', 'b', 'c', 'd'])
        >>> a
        a      10
        b      20
        c    <NA>
        d      30
        dtype: int64
        >>> b = cudf.Series([1, None, 2, 3], index=['a', 'b', 'd', 'e'])
        >>> b
        a       1
        b    <NA>
        d       2
        e       3
        dtype: int64
        >>> a.rtruediv(b, fill_value=0)
        a            0.1
        b            0.0
        c           <NA>
        d    0.066666667
        e            Inf
        dtype: float64
        """

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "truediv", fill_value, reflect=True)

    # Alias for rtruediv
    rdiv = rtruediv

    def eq(self, other, axis="columns", level=None, fill_value=None):
        """Equal to, element-wise (binary operator eq).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null

        Returns
        -------
        Frame
            The result of the operation.

        Examples
        --------
        **DataFrame**

        >>> left = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'c': [7, 8, 9]}
        ... )
        >>> right = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'd': [10, 12, 12]}
        ... )
        >>> left.eq(right)
        a     b     c     d
        0  True  True  <NA>  <NA>
        1  True  True  <NA>  <NA>
        2  True  True  <NA>  <NA>
        >>> left.eq(right, fill_value=7)
        a     b      c      d
        0  True  True   True  False
        1  True  True  False  False
        2  True  True  False  False

        **Series**

        >>> a = cudf.Series([1, 2, 3, None, 10, 20],
        ...                 index=['a', 'c', 'd', 'e', 'f', 'g'])
        >>> a
        a       1
        c       2
        d       3
        e    <NA>
        f      10
        g      20
        dtype: int64
        >>> b = cudf.Series([-10, 23, -1, None, None],
        ...                 index=['a', 'b', 'c', 'd', 'e'])
        >>> b
        a     -10
        b      23
        c      -1
        d    <NA>
        e    <NA>
        dtype: int64
        >>> a.eq(b, fill_value=2)
        a    False
        b    False
        c    False
        d    False
        e     <NA>
        f    False
        g    False
        dtype: bool
        """
        return self._binaryop(
            other=other, fn="eq", fill_value=fill_value, can_reindex=True
        )

    def ne(self, other, axis="columns", level=None, fill_value=None):
        """Not equal to, element-wise (binary operator ne).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null

        Returns
        -------
        Frame
            The result of the operation.

        Examples
        --------
        **DataFrame**

        >>> left = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'c': [7, 8, 9]}
        ... )
        >>> right = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'd': [10, 12, 12]}
        ... )
        >>> left.ne(right)
        a      b     c     d
        0  False  False  <NA>  <NA>
        1  False  False  <NA>  <NA>
        2  False  False  <NA>  <NA>
        >>> left.ne(right, fill_value=7)
        a      b      c     d
        0  False  False  False  True
        1  False  False   True  True
        2  False  False   True  True

        **Series**

        >>> a = cudf.Series([1, 2, 3, None, 10, 20],
        ...                 index=['a', 'c', 'd', 'e', 'f', 'g'])
        >>> a
        a       1
        c       2
        d       3
        e    <NA>
        f      10
        g      20
        dtype: int64
        >>> b = cudf.Series([-10, 23, -1, None, None],
        ...                 index=['a', 'b', 'c', 'd', 'e'])
        >>> b
        a     -10
        b      23
        c      -1
        d    <NA>
        e    <NA>
        dtype: int64
        >>> a.ne(b, fill_value=2)
        a    True
        b    True
        c    True
        d    True
        e    <NA>
        f    True
        g    True
        dtype: bool
        """  # noqa: E501
        return self._binaryop(
            other=other, fn="ne", fill_value=fill_value, can_reindex=True
        )

    def lt(self, other, axis="columns", level=None, fill_value=None):
        """Less than, element-wise (binary operator lt).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null

        Returns
        -------
        Frame
            The result of the operation.

        Examples
        --------
        **DataFrame**

        >>> left = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'c': [7, 8, 9]}
        ... )
        >>> right = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'd': [10, 12, 12]}
        ... )
        >>> left.lt(right)
        a      b     c     d
        0  False  False  <NA>  <NA>
        1  False  False  <NA>  <NA>
        2  False  False  <NA>  <NA>
        >>> left.lt(right, fill_value=7)
        a      b      c     d
        0  False  False  False  True
        1  False  False  False  True
        2  False  False  False  True

        **Series**

        >>> a = cudf.Series([1, 2, 3, None, 10, 20],
        ...                 index=['a', 'c', 'd', 'e', 'f', 'g'])
        >>> a
        a       1
        c       2
        d       3
        e    <NA>
        f      10
        g      20
        dtype: int64
        >>> b = cudf.Series([-10, 23, -1, None, None],
        ...                 index=['a', 'b', 'c', 'd', 'e'])
        >>> b
        a     -10
        b      23
        c      -1
        d    <NA>
        e    <NA>
        dtype: int64
        >>> a.lt(b, fill_value=-10)
        a    False
        b     True
        c    False
        d    False
        e     <NA>
        f    False
        g    False
        dtype: bool
        """  # noqa: E501
        return self._binaryop(
            other=other, fn="lt", fill_value=fill_value, can_reindex=True
        )

    def le(self, other, axis="columns", level=None, fill_value=None):
        """Less than or equal, element-wise (binary operator le).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null

        Returns
        -------
        Frame
            The result of the operation.

        Examples
        --------
        **DataFrame**

        >>> left = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'c': [7, 8, 9]}
        ... )
        >>> right = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'd': [10, 12, 12]}
        ... )
        >>> left.le(right)
        a     b     c     d
        0  True  True  <NA>  <NA>
        1  True  True  <NA>  <NA>
        2  True  True  <NA>  <NA>
        >>> left.le(right, fill_value=7)
        a     b      c     d
        0  True  True   True  True
        1  True  True  False  True
        2  True  True  False  True

        **Series**

        >>> a = cudf.Series([1, 2, 3, None, 10, 20],
        ...                 index=['a', 'c', 'd', 'e', 'f', 'g'])
        >>> a
        a       1
        c       2
        d       3
        e    <NA>
        f      10
        g      20
        dtype: int64
        >>> b = cudf.Series([-10, 23, -1, None, None],
        ...                 index=['a', 'b', 'c', 'd', 'e'])
        >>> b
        a     -10
        b      23
        c      -1
        d    <NA>
        e    <NA>
        dtype: int64
        >>> a.le(b, fill_value=-10)
        a    False
        b     True
        c    False
        d    False
        e     <NA>
        f    False
        g    False
        dtype: bool
        """  # noqa: E501
        return self._binaryop(
            other=other, fn="le", fill_value=fill_value, can_reindex=True
        )

    def gt(self, other, axis="columns", level=None, fill_value=None):
        """Greater than, element-wise (binary operator gt).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null

        Returns
        -------
        Frame
            The result of the operation.

        Examples
        --------
        **DataFrame**

        >>> left = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'c': [7, 8, 9]}
        ... )
        >>> right = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'd': [10, 12, 12]}
        ... )
        >>> left.gt(right)
        a      b     c     d
        0  False  False  <NA>  <NA>
        1  False  False  <NA>  <NA>
        2  False  False  <NA>  <NA>
        >>> left.gt(right, fill_value=7)
        a      b      c      d
        0  False  False  False  False
        1  False  False   True  False
        2  False  False   True  False

        **Series**

        >>> a = cudf.Series([1, 2, 3, None, 10, 20],
        ...                 index=['a', 'c', 'd', 'e', 'f', 'g'])
        >>> a
        a       1
        c       2
        d       3
        e    <NA>
        f      10
        g      20
        dtype: int64
        >>> b = cudf.Series([-10, 23, -1, None, None],
        ...                 index=['a', 'b', 'c', 'd', 'e'])
        >>> b
        a     -10
        b      23
        c      -1
        d    <NA>
        e    <NA>
        dtype: int64
        >>> a.gt(b)
        a     True
        b    False
        c     True
        d    False
        e    False
        f    False
        g    False
        dtype: bool
        """  # noqa: E501
        return self._binaryop(
            other=other, fn="gt", fill_value=fill_value, can_reindex=True
        )

    def ge(self, other, axis="columns", level=None, fill_value=None):
        """Greater than or equal, element-wise (binary operator ge).

        Parameters
        ----------
        other : Series or scalar value
        fill_value : None or value
            Value to fill nulls with before computation. If data in both
            corresponding Series locations is null the result will be null

        Returns
        -------
        Frame
            The result of the operation.

        Examples
        --------
        **DataFrame**

        >>> left = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'c': [7, 8, 9]}
        ... )
        >>> right = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'd': [10, 12, 12]}
        ... )
        >>> left.ge(right)
        a     b     c     d
        0  True  True  <NA>  <NA>
        1  True  True  <NA>  <NA>
        2  True  True  <NA>  <NA>
        >>> left.ge(right, fill_value=7)
        a     b     c      d
        0  True  True  True  False
        1  True  True  True  False
        2  True  True  True  False

        **Series**

        >>> a = cudf.Series([1, 2, 3, None, 10, 20],
        ...                 index=['a', 'c', 'd', 'e', 'f', 'g'])
        >>> a
        a       1
        c       2
        d       3
        e    <NA>
        f      10
        g      20
        dtype: int64
        >>> b = cudf.Series([-10, 23, -1, None, None],
        ...                 index=['a', 'b', 'c', 'd', 'e'])
        >>> b
        a     -10
        b      23
        c      -1
        d    <NA>
        e    <NA>
        dtype: int64
        >>> a.ge(b)
        a     True
        b    False
        c     True
        d    False
        e    False
        f    False
        g    False
        dtype: bool
        """  # noqa: E501
        return self._binaryop(
            other=other, fn="ge", fill_value=fill_value, can_reindex=True
        )


def _get_replacement_values_for_columns(
    to_replace: Any, value: Any, columns_dtype_map: Dict[Any, Any]
) -> Tuple[Dict[Any, bool], Dict[Any, Any], Dict[Any, Any]]:
    """
    Returns a per column mapping for the values to be replaced, new
    values to be replaced with and if all the values are empty.

    Parameters
    ----------
    to_replace : numeric, str, list-like or dict
        Contains the values to be replaced.
    value : numeric, str, list-like, or dict
        Contains the values to replace `to_replace` with.
    columns_dtype_map : dict
        A column to dtype mapping representing dtype of columns.

    Returns
    -------
    all_na_columns : dict
        A dict mapping of all columns if they contain all na values
    to_replace_columns : dict
        A dict mapping of all columns and the existing values that
        have to be replaced.
    values_columns : dict
        A dict mapping of all columns and the corresponding values
        to be replaced with.
    """
    to_replace_columns: Dict[Any, Any] = {}
    values_columns: Dict[Any, Any] = {}
    all_na_columns: Dict[Any, Any] = {}

    if is_scalar(to_replace) and is_scalar(value):
        to_replace_columns = {col: [to_replace] for col in columns_dtype_map}
        values_columns = {col: [value] for col in columns_dtype_map}
    elif cudf.api.types.is_list_like(to_replace) or isinstance(
        to_replace, ColumnBase
    ):
        if is_scalar(value):
            to_replace_columns = {col: to_replace for col in columns_dtype_map}
            values_columns = {
                col: [value]
                if _is_non_decimal_numeric_dtype(columns_dtype_map[col])
                else cudf.utils.utils.scalar_broadcast_to(
                    value, (len(to_replace),), cudf.dtype(type(value)),
                )
                for col in columns_dtype_map
            }
        elif cudf.api.types.is_list_like(value):
            if len(to_replace) != len(value):
                raise ValueError(
                    f"Replacement lists must be "
                    f"of same length."
                    f" Expected {len(to_replace)}, got {len(value)}."
                )
            else:
                to_replace_columns = {
                    col: to_replace for col in columns_dtype_map
                }
                values_columns = {col: value for col in columns_dtype_map}
        elif cudf.utils.dtypes.is_column_like(value):
            to_replace_columns = {col: to_replace for col in columns_dtype_map}
            values_columns = {col: value for col in columns_dtype_map}
        else:
            raise TypeError(
                "value argument must be scalar, list-like or Series"
            )
    elif _is_series(to_replace):
        if value is None:
            to_replace_columns = {
                col: as_column(to_replace.index) for col in columns_dtype_map
            }
            values_columns = {col: to_replace for col in columns_dtype_map}
        elif is_dict_like(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: value[col] for col in to_replace_columns if col in value
            }
        elif is_scalar(value) or _is_series(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: [value] if is_scalar(value) else value[col]
                for col in to_replace_columns
                if col in value
            }
        else:
            raise ValueError(
                "Series.replace cannot use dict-like to_replace and non-None "
                "value"
            )
    elif is_dict_like(to_replace):
        if value is None:
            to_replace_columns = {
                col: list(to_replace.keys()) for col in columns_dtype_map
            }
            values_columns = {
                col: list(to_replace.values()) for col in columns_dtype_map
            }
        elif is_dict_like(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: value[col] for col in columns_dtype_map if col in value
            }
        elif is_scalar(value) or _is_series(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: [value] if is_scalar(value) else value
                for col in columns_dtype_map
                if col in to_replace
            }
        else:
            raise TypeError("value argument must be scalar, dict, or Series")
    else:
        raise TypeError(
            "Expecting 'to_replace' to be either a scalar, array-like, "
            "dict or None, got invalid type "
            f"'{type(to_replace).__name__}'"
        )

    to_replace_columns = {
        key: [value] if is_scalar(value) else value
        for key, value in to_replace_columns.items()
    }
    values_columns = {
        key: [value] if is_scalar(value) else value
        for key, value in values_columns.items()
    }

    for i in to_replace_columns:
        if i in values_columns:
            if isinstance(values_columns[i], list):
                all_na = values_columns[i].count(None) == len(
                    values_columns[i]
                )
            else:
                all_na = False
            all_na_columns[i] = all_na

    return all_na_columns, to_replace_columns, values_columns


def _is_series(obj):
    """
    Checks if the `obj` is of type `cudf.Series`
    instead of checking for isinstance(obj, cudf.Series)
    """
    return isinstance(obj, Frame) and obj.ndim == 1 and obj._index is not None


def _drop_rows_by_labels(
    obj: DataFrameOrSeries,
    labels: Union[ColumnLike, abc.Iterable, str],
    level: Union[int, str],
    errors: str,
) -> DataFrameOrSeries:
    """Remove rows specified by `labels`. If `errors=True`, an error is raised
    if some items in `labels` do not exist in `obj._index`.

    Will raise if level(int) is greater or equal to index nlevels
    """
    if isinstance(level, int) and level >= obj.index.nlevels:
        raise ValueError("Param level out of bounds.")

    if not isinstance(labels, cudf.core.single_column_frame.SingleColumnFrame):
        labels = as_column(labels)

    if isinstance(obj._index, cudf.MultiIndex):
        if level is None:
            level = 0

        levels_index = obj.index.get_level_values(level)
        if errors == "raise" and not labels.isin(levels_index).all():
            raise KeyError("One or more values not found in axis")

        if isinstance(level, int):
            ilevel = level
        else:
            ilevel = obj._index.names.index(level)

        # 1. Merge Index df and data df along column axis:
        # | id | ._index df | data column(s) |
        idx_nlv = obj._index.nlevels
        working_df = obj._index.to_frame(index=False)
        working_df.columns = [i for i in range(idx_nlv)]
        for i, col in enumerate(obj._data):
            working_df[idx_nlv + i] = obj._data[col]
        # 2. Set `level` as common index:
        # | level | ._index df w/o level | data column(s) |
        working_df = working_df.set_index(level)

        # 3. Use "leftanti" join to drop
        # TODO: use internal API with "leftanti" and specify left and right
        # join keys to bypass logic check
        to_join = cudf.DataFrame(index=cudf.Index(labels, name=level))
        join_res = working_df.join(to_join, how="leftanti")

        # 4. Reconstruct original layout, and rename
        join_res.insert(
            ilevel, name=join_res._index.name, value=join_res._index
        )
        join_res = join_res.reset_index(drop=True)

        midx = cudf.MultiIndex.from_frame(
            join_res.iloc[:, 0:idx_nlv], names=obj._index.names
        )

        if isinstance(obj, cudf.Series):
            return obj.__class__._from_data(
                join_res.iloc[:, idx_nlv:]._data, index=midx, name=obj.name
            )
        else:
            return obj.__class__._from_data(
                join_res.iloc[:, idx_nlv:]._data,
                index=midx,
                columns=obj.columns,
            )

    else:
        if errors == "raise" and not labels.isin(obj.index).all():
            raise KeyError("One or more values not found in axis")

        key_df = cudf.DataFrame(index=labels)
        if isinstance(obj, cudf.Series):
            res = obj.to_frame(name="tmp").join(key_df, how="leftanti")["tmp"]
            res.name = obj.name
            return res
        else:
            return obj.join(key_df, how="leftanti")


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
