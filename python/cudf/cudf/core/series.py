# Copyright (c) 2018-2021, NVIDIA CORPORATION.

from __future__ import annotations

import functools
import inspect
import pickle
import warnings
from collections import abc as abc
from hashlib import sha256
from numbers import Number
from shutil import get_terminal_size
from typing import Any, MutableMapping, Optional, Set, Union

import cupy
import numpy as np
import pandas as pd
from numba import cuda
from pandas._config import get_option

import cudf
from cudf import _lib as libcudf
from cudf._lib.scalar import _is_null_host_scalar
from cudf._lib.transform import bools_to_mask
from cudf._typing import ColumnLike, DataFrameOrSeries, ScalarLike
from cudf.api.types import (
    _is_non_decimal_numeric_dtype,
    _is_scalar_or_zero_d_array,
    is_bool_dtype,
    is_categorical_dtype,
    is_decimal_dtype,
    is_dict_like,
    is_dtype_equal,
    is_integer,
    is_integer_dtype,
    is_interval_dtype,
    is_list_dtype,
    is_list_like,
    is_scalar,
    is_struct_dtype,
)
from cudf.core.abc import Serializable
from cudf.core.column import (
    DatetimeColumn,
    TimeDeltaColumn,
    arange,
    as_column,
    column,
    column_empty_like,
    full,
)
from cudf.core.column.categorical import (
    CategoricalAccessor as CategoricalAccessor,
)
from cudf.core.column.column import concat_columns
from cudf.core.column.lists import ListMethods
from cudf.core.column.string import StringMethods
from cudf.core.column.struct import StructMethods
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.frame import Frame, _drop_rows_by_labels
from cudf.core.groupby.groupby import SeriesGroupBy
from cudf.core.index import BaseIndex, RangeIndex, as_index
from cudf.core.indexed_frame import (
    IndexedFrame,
    _FrameIndexer,
    _get_label_range_or_mask,
    _indices_from_labels,
)
from cudf.core.single_column_frame import SingleColumnFrame
from cudf.utils import cudautils, docutils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    can_convert_to_column,
    find_common_type,
    is_mixed_with_object_dtype,
    min_scalar_type,
)
from cudf.utils.utils import (
    get_appropriate_dispatched_func,
    get_relevant_submodule,
    to_cudf_compatible_scalar,
)


def _append_new_row_inplace(col: ColumnLike, value: ScalarLike):
    """Append a scalar `value` to the end of `col` inplace.
    Cast to common type if possible
    """
    to_type = find_common_type([type(value), col.dtype])
    val_col = as_column(value, dtype=to_type)
    old_col = col.astype(to_type)

    col._mimic_inplace(concat_columns([old_col, val_col]), inplace=True)


class _SeriesIlocIndexer(_FrameIndexer):
    """
    For integer-location based selection.
    """

    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            arg = list(arg)
        data = self._frame._column[arg]

        if (
            isinstance(data, (dict, list))
            or _is_scalar_or_zero_d_array(data)
            or _is_null_host_scalar(data)
        ):
            return data
        return self._frame._from_data(
            {self._frame.name: data}, index=cudf.Index(self._frame.index[arg]),
        )

    def __setitem__(self, key, value):
        from cudf.core.column import column

        if isinstance(key, tuple):
            key = list(key)

        # coerce value into a scalar or column
        if is_scalar(value):
            value = to_cudf_compatible_scalar(value)
        elif not (
            isinstance(value, (list, dict))
            and isinstance(
                self._frame._column.dtype, (cudf.ListDtype, cudf.StructDtype)
            )
        ):
            value = column.as_column(value)

        if (
            not isinstance(
                self._frame._column.dtype,
                (
                    cudf.Decimal64Dtype,
                    cudf.Decimal32Dtype,
                    cudf.CategoricalDtype,
                ),
            )
            and hasattr(value, "dtype")
            and _is_non_decimal_numeric_dtype(value.dtype)
        ):
            # normalize types if necessary:
            if not is_integer(key):
                to_dtype = np.result_type(
                    value.dtype, self._frame._column.dtype
                )
                value = value.astype(to_dtype)
                self._frame._column._mimic_inplace(
                    self._frame._column.astype(to_dtype), inplace=True
                )

        self._frame._column[key] = value


class _SeriesLocIndexer(_FrameIndexer):
    """
    Label-based selection
    """

    def __getitem__(self, arg: Any) -> Union[ScalarLike, DataFrameOrSeries]:
        if isinstance(arg, pd.MultiIndex):
            arg = cudf.from_pandas(arg)

        if isinstance(self._frame.index, cudf.MultiIndex) and not isinstance(
            arg, cudf.MultiIndex
        ):
            result = self._frame.index._get_row_major(self._frame, arg)
            if (
                isinstance(arg, tuple)
                and len(arg) == self._frame._index.nlevels
                and not any((isinstance(x, slice) for x in arg))
            ):
                result = result.iloc[0]
            return result
        try:
            arg = self._loc_to_iloc(arg)
        except (TypeError, KeyError, IndexError, ValueError):
            raise KeyError(arg)

        return self._frame.iloc[arg]

    def __setitem__(self, key, value):
        try:
            key = self._loc_to_iloc(key)
        except KeyError as e:
            if (
                is_scalar(key)
                and not isinstance(self._frame.index, cudf.MultiIndex)
                and is_scalar(value)
            ):
                _append_new_row_inplace(self._frame.index._values, key)
                _append_new_row_inplace(self._frame._column, value)
                return
            else:
                raise e
        if isinstance(value, (pd.Series, cudf.Series)):
            value = cudf.Series(value)
            value = value._align_to_index(self._frame.index, how="right")
        self._frame.iloc[key] = value

    def _loc_to_iloc(self, arg):
        if _is_scalar_or_zero_d_array(arg):
            if not _is_non_decimal_numeric_dtype(self._frame.index.dtype):
                # TODO: switch to cudf.utils.dtypes.is_integer(arg)
                if isinstance(arg, cudf.Scalar) and is_integer_dtype(
                    arg.dtype
                ):
                    found_index = arg.value
                    return found_index
                elif is_integer(arg):
                    found_index = arg
                    return found_index
            try:
                found_index = self._frame.index._values.find_first_value(
                    arg, closest=False
                )
                return found_index
            except (TypeError, KeyError, IndexError, ValueError):
                raise KeyError("label scalar is out of bound")

        elif isinstance(arg, slice):
            return _get_label_range_or_mask(
                self._frame.index, arg.start, arg.stop, arg.step
            )
        elif isinstance(arg, (cudf.MultiIndex, pd.MultiIndex)):
            if isinstance(arg, pd.MultiIndex):
                arg = cudf.MultiIndex.from_pandas(arg)

            return _indices_from_labels(self._frame, arg)

        else:
            arg = cudf.core.series.Series(cudf.core.column.as_column(arg))
            if arg.dtype in (bool, np.bool_):
                return arg
            else:
                indices = _indices_from_labels(self._frame, arg)
                if indices.null_count > 0:
                    raise KeyError("label scalar is out of bound")
                return indices


class Series(SingleColumnFrame, IndexedFrame, Serializable):
    """
    One-dimensional GPU array (including time series).

    Labels need not be unique but must be a hashable type. The object
    supports both integer- and label-based indexing and provides a
    host of methods for performing operations involving the index.
    Statistical methods from ndarray have been overridden to
    automatically exclude missing data (currently represented
    as null/NaN).

    Operations between Series (`+`, `-`, `/`, `*`, `**`) align
    values based on their associated index values-– they need
    not be the same length. The result index will be the
    sorted union of the two indexes.

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

    _accessors: Set[Any] = set()
    _loc_indexer_type = _SeriesLocIndexer
    _iloc_indexer_type = _SeriesIlocIndexer

    # The `constructor*` properties are used by `dask` (and `dask_cudf`)
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
        return cudf.DataFrame

    @classmethod
    def from_categorical(cls, categorical, codes=None):
        """Creates from a pandas.Categorical

        Parameters
        ----------
        categorical : pandas.Categorical
            Contains data stored in a pandas Categorical.

        codes : array-like, optional.
            The category codes of this categorical. If ``codes`` are
            defined, they are used instead of ``categorical.codes``

        Returns
        -------
        Series
            A cudf categorical series.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> pd_categorical = pd.Categorical(pd.Series(['a', 'b', 'c', 'a'], dtype='category'))
        >>> pd_categorical
        ['a', 'b', 'c', 'a']
        Categories (3, object): ['a', 'b', 'c']
        >>> series = cudf.Series.from_categorical(pd_categorical)
        >>> series
        0    a
        1    b
        2    c
        3    a
        dtype: category
        Categories (3, object): ['a', 'b', 'c']
        """  # noqa: E501
        col = cudf.core.column.categorical.pandas_categorical_as_column(
            categorical, codes=codes
        )
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

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> a = cudf.Series([1, 2, 3, None, 4, None])
        >>> a
        0       1
        1       2
        2       3
        3    <NA>
        4       4
        5    <NA>
        dtype: int64
        >>> b = cudf.Series([10, 11, 12, 13, 14])
        >>> cudf.Series.from_masked_array(data=b, mask=a._column.mask)
        0      10
        1      11
        2      12
        3    <NA>
        4      14
        dtype: int64
        """
        col = column.as_column(data).set_mask(mask)
        return cls(data=col)

    def __init__(
        self, data=None, index=None, dtype=None, name=None, nan_as_null=True,
    ):
        if isinstance(data, pd.Series):
            if name is None:
                name = data.name
            if isinstance(data.index, pd.MultiIndex):
                index = cudf.from_pandas(data.index)
            else:
                index = as_index(data.index)
        elif isinstance(data, pd.Index):
            name = data.name
            data = data.values
        elif isinstance(data, BaseIndex):
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

        if isinstance(data, dict):
            index = data.keys()
            data = column.as_column(
                list(data.values()), nan_as_null=nan_as_null, dtype=dtype
            )

        if data is None:
            if index is not None:
                data = column.column_empty(
                    row_count=len(index), dtype=None, masked=True
                )
            else:
                data = {}

        if not isinstance(data, column.ColumnBase):
            data = column.as_column(data, nan_as_null=nan_as_null, dtype=dtype)
        else:
            if dtype is not None:
                data = data.astype(dtype)

        if index is not None and not isinstance(index, BaseIndex):
            index = as_index(index)

        assert isinstance(data, column.ColumnBase)

        super().__init__({name: data})
        self._index = RangeIndex(len(data)) if index is None else index

    @classmethod
    def _from_data(
        cls,
        data: MutableMapping,
        index: Optional[BaseIndex] = None,
        name: Any = None,
    ) -> Series:
        """
        Construct the Series from a ColumnAccessor
        """
        out: Series = super()._from_data(data, index, name)
        if index is None:
            out._index = RangeIndex(out._data.nrows)
        return out

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
        3    <NA>
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
        elif isinstance(self._column, TimeDeltaColumn):
            return TimedeltaProperties(self)
        else:
            raise AttributeError(
                "Can only use .dt accessor with datetimelike values"
            )

    def serialize(self):
        header, frames = super().serialize()

        header["index"], index_frames = self._index.serialize()
        header["index_frame_count"] = len(index_frames)
        # For backwards compatibility with older versions of cuDF, index
        # columns are placed before data columns.
        frames = index_frames + frames

        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        if "column" in header:
            warnings.warn(
                "Series objects serialized in cudf version "
                "21.10 or older will no longer be deserializable "
                "after version 21.12. Please load and resave any "
                "pickles before upgrading to version 22.02.",
                FutureWarning,
            )
            header["columns"] = [header.pop("column")]
            header["column_names"] = pickle.dumps(
                [pickle.loads(header["name"])]
            )

        index_nframes = header["index_frame_count"]
        obj = super().deserialize(
            header, frames[header["index_frame_count"] :]
        )

        idx_typ = pickle.loads(header["index"]["type-serialized"])
        index = idx_typ.deserialize(header["index"], frames[:index_nframes])
        obj._index = index

        return obj

    def _get_columns_by_label(self, labels, downcast=False):
        """Return the column specified by `labels`

        For cudf.Series, either the column, or an empty series is returned.
        Parameter `downcast` does not have effects.
        """
        new_data = super()._get_columns_by_label(labels, downcast)

        return (
            self.__class__(data=new_data, index=self.index)
            if len(new_data) > 0
            else self.__class__(dtype=self.dtype, name=self.name)
        )

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        """
        Return Series with specified index labels removed.

        Remove elements of a Series based on specifying the index labels.
        When using a multi-index, labels on different levels can be removed by
        specifying the level.

        Parameters
        ----------
        labels : single label or list-like
            Index labels to drop.
        axis : 0, default 0
            Redundant for application on Series.
        index : single label or list-like
            Redundant for application on Series. But ``index`` can be used
            instead of ``labels``
        columns : single label or list-like
            This parameter is ignored. Use ``index`` or ``labels`` to specify.
        level : int or level name, optional
            For MultiIndex, level from which the labels will be removed.
        inplace : bool, default False
            If False, return a copy. Otherwise, do operation
            inplace and return None.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are
            dropped.

        Returns
        -------
        Series or None
            Series with specified index labels removed or None if
            ``inplace=True``

        Raises
        ------
        KeyError
            If any of the labels is not found in the selected axis and
            ``error='raise'``

        See Also
        --------
        Series.reindex
            Return only specified index labels of Series
        Series.dropna
            Return series without null values
        Series.drop_duplicates
            Return series with duplicate values removed
        cudf.DataFrame.drop
            Drop specified labels from rows or columns in dataframe

        Examples
        --------
        >>> s = cudf.Series([1,2,3], index=['x', 'y', 'z'])
        >>> s
        x    1
        y    2
        z    3
        dtype: int64

        Drop labels x and z

        >>> s.drop(labels=['x', 'z'])
        y    2
        dtype: int64

        Drop a label from the second level in MultiIndex Series.

        >>> midx = cudf.MultiIndex.from_product([[0, 1, 2], ['x', 'y']])
        >>> s = cudf.Series(range(6), index=midx)
        >>> s
        0  x    0
           y    1
        1  x    2
           y    3
        2  x    4
           y    5
        >>> s.drop(labels='y', level=1)
        0  x    0
        1  x    2
        2  x    4
        """
        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError(
                    "Cannot specify both 'labels' and 'index'/'columns'"
                )
            if axis == 1:
                raise ValueError("No axis named 1 for object type Series")
            target = labels
        elif index is not None:
            target = index
        elif columns is not None:
            target = []  # Ignore parameter columns
        else:
            raise ValueError(
                "Need to specify at least one of 'labels', "
                "'index' or 'columns'"
            )

        if inplace:
            out = self
        else:
            out = self.copy()

        dropped = _drop_rows_by_labels(out, target, level, errors)

        out._data = dropped._data
        out._index = dropped._index

        if not inplace:
            return out

    def append(self, to_append, ignore_index=False, verify_integrity=False):
        """Append values from another ``Series`` or array-like object.
        If ``ignore_index=True``, the index is reset.

        Parameters
        ----------
        to_append : Series or list/tuple of Series
            Series to append with self.
        ignore_index : boolean, default False.
            If True, do not use the index.
        verify_integrity : bool, default False
            This Parameter is currently not supported.

        Returns
        -------
        Series
            A new concatenated series

        See Also
        --------
        cudf.concat : General function to concatenate DataFrame or
            Series objects.

        Examples
        --------
        >>> import cudf
        >>> s1 = cudf.Series([1, 2, 3])
        >>> s2 = cudf.Series([4, 5, 6])
        >>> s1
        0    1
        1    2
        2    3
        dtype: int64
        >>> s2
        0    4
        1    5
        2    6
        dtype: int64
        >>> s1.append(s2)
        0    1
        1    2
        2    3
        0    4
        1    5
        2    6
        dtype: int64

        >>> s3 = cudf.Series([4, 5, 6], index=[3, 4, 5])
        >>> s3
        3    4
        4    5
        5    6
        dtype: int64
        >>> s1.append(s3)
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        dtype: int64

        With `ignore_index` set to True:

        >>> s1.append(s2, ignore_index=True)
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        dtype: int64
        """
        if verify_integrity not in (None, False):
            raise NotImplementedError(
                "verify_integrity parameter is not supported yet."
            )

        if is_list_like(to_append):
            to_concat = [self]
            to_concat.extend(to_append)
        else:
            to_concat = [self, to_append]

        return cudf.concat(to_concat, ignore_index=ignore_index)

    def reindex(self, index=None, copy=True):
        """Return a Series that conforms to a new index

        Parameters
        ----------
        index : Index, Series-convertible, default None
        copy : boolean, default True

        Returns
        -------
        A new Series that conforms to the supplied index

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
        >>> series
        a    10
        b    20
        c    30
        d    40
        dtype: int64
        >>> series.reindex(['a', 'b', 'y', 'z'])
        a      10
        b      20
        y    <NA>
        z    <NA>
        dtype: int64
        """
        name = self.name or 0
        idx = self._index if index is None else index
        series = self.to_frame(name).reindex(idx, copy=copy)[name]
        series.name = self.name
        return series

    def reset_index(self, drop=False, inplace=False):
        """
        Reset index to RangeIndex

        Parameters
        ----------
        drop : bool, default False
            Just reset the index, without inserting it as a column in
            the new DataFrame.
        inplace : bool, default False
            Modify the Series in place (do not create a new object).

        Returns
        -------
        Series or DataFrame or None
            When `drop` is False (the default), a DataFrame is returned.
            The newly created columns will come first in the DataFrame,
            followed by the original Series values.
            When `drop` is True, a `Series` is returned.
            In either case, if ``inplace=True``, no value is returned.

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series(['a', 'b', 'c', 'd'], index=[10, 11, 12, 13])
        >>> series
        10    a
        11    b
        12    c
        13    d
        dtype: object
        >>> series.reset_index()
           index  0
        0     10  a
        1     11  b
        2     12  c
        3     13  d
        >>> series.reset_index(drop=True)
        0    a
        1    b
        2    c
        3    d
        dtype: object
        """
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
                return self._from_data(self._data, index=RangeIndex(len(self)))

    def set_index(self, index):
        """Returns a new Series with a different index.

        Parameters
        ----------
        index : Index, Series-convertible
            the new index or values for the new index

        Returns
        -------
        Series
            A new Series with assigned index.

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([10, 11, 12, 13, 14])
        >>> series
        0    10
        1    11
        2    12
        3    13
        4    14
        dtype: int64
        >>> series.set_index(['a', 'b', 'c', 'd', 'e'])
        a    10
        b    11
        c    12
        d    13
        e    14
        dtype: int64
        """
        warnings.warn(
            "Series.set_index is deprecated and will be removed in the future",
            FutureWarning,
        )
        index = index if isinstance(index, BaseIndex) else as_index(index)
        return self._from_data(self._data, index, self.name)

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

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series(['a', 'b', 'c', None, 'd'], name='sample', index=[10, 11, 12, 13, 15])
        >>> series
        10       a
        11       b
        12       c
        13    <NA>
        15       d
        Name: sample, dtype: object
        >>> series.to_frame()
           sample
        10      a
        11      b
        12      c
        13   <NA>
        15      d
        """  # noqa: E501

        if name is not None:
            col = name
        elif self.name is None:
            col = 0
        else:
            col = self.name

        return cudf.DataFrame({col: self._column}, index=self.index)

    def set_mask(self, mask, null_count=None):
        warnings.warn(
            "Series.set_mask is deprecated and will be removed in the future.",
            FutureWarning,
        )
        return self._from_data(
            {self.name: self._column.set_mask(mask)}, self._index
        )

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
        cudf.DataFrame.memory_usage : Bytes consumed by
            a DataFrame.

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
        if deep:
            warnings.warn(
                "The deep parameter is ignored and is only included "
                "for pandas compatibility."
            )
        n = self._column.memory_usage()
        if index:
            n += self._index.memory_usage()
        return n

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            return get_appropriate_dispatched_func(
                cudf, cudf.Series, cupy, ufunc, inputs, kwargs
            )
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        handled_types = [cudf.Series]
        for t in types:
            if t not in handled_types:
                return NotImplemented

        cudf_submodule = get_relevant_submodule(func, cudf)
        cudf_ser_submodule = get_relevant_submodule(func, cudf.Series)
        cupy_submodule = get_relevant_submodule(func, cupy)

        return get_appropriate_dispatched_func(
            cudf_submodule,
            cudf_ser_submodule,
            cupy_submodule,
            func,
            args,
            kwargs,
        )

    def map(self, arg, na_action=None) -> "Series":
        """
        Map values of Series according to input correspondence.

        Used for substituting each value in a Series with another value,
        that may be derived from a function, a ``dict`` or
        a :class:`Series`.

        Parameters
        ----------
        arg : function, collections.abc.Mapping subclass or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to the
            mapping correspondence.

        Returns
        -------
        Series
            Same index as caller.

        Examples
        --------
        >>> s = cudf.Series(['cat', 'dog', np.nan, 'rabbit'])
        >>> s
        0      cat
        1      dog
        2     <NA>
        3   rabbit
        dtype: object

        ``map`` accepts a ``dict`` or a ``Series``. Values that are not found
        in the ``dict`` are converted to ``NaN``, default values in dicts are
        currently not supported.:

        >>> s.map({'cat': 'kitten', 'dog': 'puppy'})
        0   kitten
        1    puppy
        2     <NA>
        3     <NA>
        dtype: object

        It also accepts numeric functions:

        >>> s = cudf.Series([1, 2, 3, 4, np.nan])
        >>> s.map(lambda x: x ** 2)
        0       1
        1       4
        2       9
        3       16
        4     <NA>
        dtype: int64

        Notes
        -----
        Please note map currently only supports fixed-width numeric
        type functions.
        """
        if isinstance(arg, dict):
            if hasattr(arg, "__missing__"):
                raise NotImplementedError(
                    "default values in dicts are currently not supported."
                )
            lhs = cudf.DataFrame({"x": self, "orig_order": arange(len(self))})
            rhs = cudf.DataFrame(
                {
                    "x": arg.keys(),
                    "s": arg.values(),
                    "bool": full(len(arg), True, dtype=self.dtype),
                }
            )
            res = lhs.merge(rhs, on="x", how="left").sort_values(
                by="orig_order"
            )
            result = res["s"]
            result.name = self.name
            result.index = self.index
        elif isinstance(arg, cudf.Series):
            if not arg.index.is_unique:
                raise ValueError(
                    "Reindexing only valid with"
                    " uniquely valued Index objects"
                )
            lhs = cudf.DataFrame({"x": self, "orig_order": arange(len(self))})
            rhs = cudf.DataFrame(
                {
                    "x": arg.keys(),
                    "s": arg,
                    "bool": full(len(arg), True, dtype=self.dtype),
                }
            )
            res = lhs.merge(rhs, on="x", how="left").sort_values(
                by="orig_order"
            )
            result = res["s"]
            result.name = self.name
            result.index = self.index
        else:
            result = self.applymap(arg)
        return result

    def __getitem__(self, arg):
        if isinstance(arg, slice):
            return self.iloc[arg]
        else:
            return self.loc[arg]

    iteritems = SingleColumnFrame.__iter__

    items = SingleColumnFrame.__iter__

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self.iloc[key] = value
        else:
            self.loc[key] = value

    def take(self, indices, axis=0, keep_index=True):
        # Validate but don't use the axis.
        _ = self._get_axis_from_axis_arg(axis)
        return super().take(indices, keep_index)

    def __repr__(self):
        _, height = get_terminal_size()
        max_rows = (
            height
            if get_option("display.max_rows") == 0
            else get_option("display.max_rows")
        )
        if max_rows not in (0, None) and len(self) > max_rows:
            top = self.head(int(max_rows / 2 + 1))
            bottom = self.tail(int(max_rows / 2 + 1))
            preprocess = cudf.concat([top, bottom])
        else:
            preprocess = self.copy()
        preprocess.index = preprocess.index._clean_nulls_from_index()
        if (
            preprocess.nullable
            and not isinstance(
                preprocess._column, cudf.core.column.CategoricalColumn
            )
            and not is_list_dtype(preprocess.dtype)
            and not is_struct_dtype(preprocess.dtype)
            and not is_decimal_dtype(preprocess.dtype)
            and not is_struct_dtype(preprocess.dtype)
        ) or isinstance(
            preprocess._column, cudf.core.column.timedelta.TimeDeltaColumn
        ):
            output = (
                preprocess.astype("O")
                .fillna(cudf._NA_REP)
                .to_pandas()
                .__repr__()
            )
        elif isinstance(
            preprocess._column, cudf.core.column.CategoricalColumn
        ):
            min_rows = (
                height
                if get_option("display.min_rows") == 0
                else get_option("display.min_rows")
            )
            show_dimensions = get_option("display.show_dimensions")
            if preprocess._column.categories.dtype.kind == "f":
                pd_series = (
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
            else:
                if is_categorical_dtype(self):
                    if is_interval_dtype(
                        self.dtype.categories
                    ) and is_struct_dtype(preprocess._column.categories):
                        # with a series input the IntervalIndex's are turn
                        # into a struct dtype this line will change the
                        # categories back to an intervalIndex.
                        preprocess.dtype._categories = self.dtype.categories
                pd_series = preprocess.to_pandas()
            output = pd_series.to_string(
                name=self.name,
                dtype=self.dtype,
                min_rows=min_rows,
                max_rows=max_rows,
                length=show_dimensions,
                na_rep=cudf._NA_REP,
            )
        else:
            output = preprocess.to_pandas().__repr__()

        lines = output.split("\n")

        if isinstance(preprocess._column, cudf.core.column.CategoricalColumn):
            category_memory = lines[-1]
            if preprocess._column.categories.dtype.kind == "f":
                category_memory = category_memory.replace("'", "").split(": ")
                category_memory = (
                    category_memory[0].replace(
                        "object", preprocess._column.categories.dtype.name
                    )
                    + ": "
                    + category_memory[1]
                )
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
            lines[-1] = " dtype: %s)" % self.dtype
            return ",".join(lines)
        if isinstance(preprocess._column, cudf.core.column.CategoricalColumn):
            lines.append(category_memory)
        return "\n".join(lines)

    def _binaryop(
        self,
        other: Frame,
        fn: str,
        fill_value: Any = None,
        reflect: bool = False,
        can_reindex: bool = False,
        *args,
        **kwargs,
    ):
        # Specialize binops to align indices.
        if isinstance(other, SingleColumnFrame):
            if (
                # TODO: The can_reindex logic also needs to be applied for
                # DataFrame (the methods that need it just don't exist yet).
                not can_reindex
                and fn in cudf.utils.utils._EQUALITY_OPS
                and (
                    isinstance(other, Series)
                    # TODO: mypy doesn't like this line because the index
                    # property is not defined on SingleColumnFrame (or Index,
                    # for that matter). Ignoring is the easy solution for now,
                    # a cleaner fix requires reworking the type hierarchy.
                    and not self.index.equals(other.index)  # type: ignore
                )
            ):
                raise ValueError(
                    "Can only compare identically-labeled Series objects"
                )
            lhs, other = _align_indices([self, other], allow_non_unique=True)
        else:
            lhs = self

        operands = lhs._make_operands_for_binop(other, fill_value, reflect)
        return (
            lhs._from_data(
                data=lhs._colwise_binop(operands, fn), index=lhs._index,
            )
            if operands is not NotImplemented
            else NotImplemented
        )

    def logical_and(self, other):
        return self._binaryop(other, "l_and").astype(np.bool_)

    def remainder(self, other):
        return self._binaryop(other, "mod")

    def logical_or(self, other):
        return self._binaryop(other, "l_or").astype(np.bool_)

    def logical_not(self):
        return self._unaryop("not")

    @copy_docstring(CategoricalAccessor)  # type: ignore
    @property
    def cat(self):
        return CategoricalAccessor(parent=self)

    @copy_docstring(StringMethods)  # type: ignore
    @property
    def str(self):
        return StringMethods(parent=self)

    @copy_docstring(ListMethods)  # type: ignore
    @property
    def list(self):
        return ListMethods(parent=self)

    @copy_docstring(StructMethods)  # type: ignore
    @property
    def struct(self):
        return StructMethods(parent=self)

    @property
    def dtype(self):
        """dtype of the Series"""
        return self._column.dtype

    @classmethod
    def _concat(cls, objs, axis=0, index=True):
        # Concatenate index if not provided
        if index is True:
            if isinstance(objs[0].index, cudf.MultiIndex):
                index = cudf.MultiIndex._concat([o.index for o in objs])
            else:
                index = cudf.core.index.GenericIndex._concat(
                    [o.index for o in objs]
                )

        names = {obj.name for obj in objs}
        if len(names) == 1:
            [name] = names
        else:
            name = None

        if len(objs) > 1:
            dtype_mismatch = False
            for obj in objs[1:]:
                if (
                    obj.null_count == len(obj)
                    or len(obj) == 0
                    or isinstance(
                        obj._column, cudf.core.column.CategoricalColumn
                    )
                    or isinstance(
                        objs[0]._column, cudf.core.column.CategoricalColumn
                    )
                ):
                    continue

                if (
                    not dtype_mismatch
                    and (
                        not isinstance(
                            objs[0]._column, cudf.core.column.CategoricalColumn
                        )
                        and not isinstance(
                            obj._column, cudf.core.column.CategoricalColumn
                        )
                    )
                    and objs[0].dtype != obj.dtype
                ):
                    dtype_mismatch = True

                if is_mixed_with_object_dtype(objs[0], obj):
                    raise TypeError(
                        "cudf does not support mixed types, please type-cast "
                        "both series to same dtypes."
                    )

            if dtype_mismatch:
                common_dtype = find_common_type([obj.dtype for obj in objs])
                objs = [obj.astype(common_dtype) for obj in objs]

        col = concat_columns([o._column for o in objs])

        if isinstance(col, cudf.core.column.Decimal64Column):
            col = col._with_type_metadata(objs[0]._column.dtype)

        if isinstance(col, cudf.core.column.StructColumn):
            col = col._with_type_metadata(objs[0].dtype)

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

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([1, 2, None, 3, 4])
        >>> series
        0       1
        1       2
        2    <NA>
        3       3
        4       4
        dtype: int64
        >>> series.has_nulls
        True
        >>> series.dropna().has_nulls
        False
        """
        return self._column.has_nulls

    def dropna(self, axis=0, inplace=False, how=None):
        """
        Return a Series with null values removed.

        Parameters
        ----------
        axis : {0 or ‘index’}, default 0
            There is only one axis to drop values from.
        inplace : bool, default False
            If True, do operation inplace and return None.
        how : str, optional
            Not in use. Kept for compatibility.

        Returns
        -------
        Series
            Series with null entries dropped from it.

        See Also
        --------
        Series.isna : Indicate null values.

        Series.notna : Indicate non-null values.

        Series.fillna : Replace null values.

        cudf.DataFrame.dropna : Drop rows or columns which
            contain null values.

        cudf.Index.dropna : Drop null indices.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 2, None])
        >>> ser
        0       1
        1       2
        2    null
        dtype: int64

        Drop null values from a Series.

        >>> ser.dropna()
        0    1
        1    2
        dtype: int64

        Keep the Series with valid entries in the same variable.

        >>> ser.dropna(inplace=True)
        >>> ser
        0    1
        1    2
        dtype: int64

        Empty strings are not considered null values.
        `None` is considered a null value.

        >>> ser = cudf.Series(['', None, 'abc'])
        >>> ser
        0
        1    <NA>
        2     abc
        dtype: object
        >>> ser.dropna()
        0
        2    abc
        dtype: object
        """
        if axis not in (0, "index"):
            raise ValueError(
                "Series.dropna supports only one axis to drop values from"
            )

        result = super().dropna(axis=axis)

        return self._mimic_inplace(result, inplace=inplace)

    def drop_duplicates(self, keep="first", inplace=False, ignore_index=False):
        """
        Return Series with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            Method to handle dropping duplicates:

            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        inplace : bool, default ``False``
            If ``True``, performs operation inplace and returns None.

        Returns
        -------
        Series or None
            Series with duplicates dropped or None if ``inplace=True``.

        Examples
        --------
        >>> s = cudf.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'],
        ...               name='animal')
        >>> s
        0      lama
        1       cow
        2      lama
        3    beetle
        4      lama
        5     hippo
        Name: animal, dtype: object

        With the `keep` parameter, the selection behaviour of duplicated
        values can be changed. The value ‘first’ keeps the first
        occurrence for each set of duplicated entries.
        The default value of keep is ‘first’. Note that order of
        the rows being returned is not guaranteed
        to be sorted.

        >>> s.drop_duplicates()
        3    beetle
        1       cow
        5     hippo
        0      lama
        Name: animal, dtype: object

        The value ‘last’ for parameter `keep` keeps the last occurrence
        for each set of duplicated entries.

        >>> s.drop_duplicates(keep='last')
        3    beetle
        1       cow
        5     hippo
        4      lama
        Name: animal, dtype: object

        The value `False` for parameter `keep` discards all sets
        of duplicated entries. Setting the value of ‘inplace’ to
        `True` performs the operation inplace and returns `None`.

        >>> s.drop_duplicates(keep=False, inplace=True)
        >>> s
        3    beetle
        1       cow
        5     hippo
        Name: animal, dtype: object
        """
        result = super().drop_duplicates(keep=keep, ignore_index=ignore_index)

        return self._mimic_inplace(result, inplace=inplace)

    def fill(self, fill_value, begin=0, end=-1, inplace=False):
        return self._fill([fill_value], begin, end, inplace)

    def fillna(
        self, value=None, method=None, axis=None, inplace=False, limit=None
    ):
        if isinstance(value, pd.Series):
            value = Series.from_pandas(value)

        if not (is_scalar(value) or isinstance(value, (abc.Mapping, Series))):
            raise TypeError(
                f'"value" parameter must be a scalar, dict '
                f"or Series, but you passed a "
                f'"{type(value).__name__}"'
            )

        if isinstance(value, (abc.Mapping, Series)):
            value = Series(value)
            if not self.index.equals(value.index):
                value = value.reindex(self.index)
            value = value._column

        return super().fillna(
            value=value, method=method, axis=axis, inplace=inplace, limit=limit
        )

    # TODO: When this method is removed we can also remove ColumnBase.to_array.
    def to_array(self, fillna=None):
        warnings.warn(
            "The to_array method will be removed in a future cuDF "
            "release. Consider using `to_numpy` instead.",
            FutureWarning,
        )
        return self._column.to_array(fillna=fillna)

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        if bool_only not in (None, True):
            raise NotImplementedError(
                "The bool_only parameter is not supported for Series."
            )
        return super().all(axis, skipna, level, **kwargs)

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        if bool_only not in (None, True):
            raise NotImplementedError(
                "The bool_only parameter is not supported for Series."
            )
        return super().any(axis, skipna, level, **kwargs)

    def to_pandas(self, index=True, nullable=False, **kwargs):
        """
        Convert to a Pandas Series.

        Parameters
        ----------
        index : Boolean, Default True
            If ``index`` is ``True``, converts the index of cudf.Series
            and sets it to the pandas.Series. If ``index`` is ``False``,
            no index conversion is performed and pandas.Series will assign
            a default index.
        nullable : Boolean, Default False
            If ``nullable`` is ``True``, the resulting series will be
            having a corresponding nullable Pandas dtype. If ``nullable``
            is ``False``, the resulting series will either convert null
            values to ``np.nan`` or ``None`` depending on the dtype.

        Returns
        -------
        out : Pandas Series

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

        ``nullable`` parameter can be used to control
        whether dtype can be Pandas Nullable or not:

        >>> ser = cudf.Series([10, 20, None, 30])
        >>> ser
        0      10
        1      20
        2    <NA>
        3      30
        dtype: int64
        >>> ser.to_pandas(nullable=True)
        0      10
        1      20
        2    <NA>
        3      30
        dtype: Int64
        >>> ser.to_pandas(nullable=False)
        0    10.0
        1    20.0
        2     NaN
        3    30.0
        dtype: float64
        """
        if index is True:
            index = self.index.to_pandas()
        s = self._column.to_pandas(index=index, nullable=nullable)
        s.name = self.name
        return s

    @property
    def data(self):
        """The gpu buffer for the data

        Returns
        -------
        out : The GPU buffer of the Series.

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([1, 2, 3, 4])
        >>> series
        0    1
        1    2
        2    3
        3    4
        dtype: int64
        >>> series.data
        <cudf.core.buffer.Buffer object at 0x7f23c192d110>
        >>> series.data.to_host_array()
        array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
               0, 0, 4, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)
        """  # noqa: E501
        return self._column.data

    @property
    def nullmask(self):
        """The gpu buffer for the null-mask"""
        return cudf.Series(self._column.nullmask)

    def as_mask(self):
        """Convert booleans to bitmask

        Returns
        -------
        device array

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([True, False, True])
        >>> s.as_mask()
        <cudf.core.buffer.Buffer object at 0x7f23c3eed0d0>
        >>> s.as_mask().to_host_array()
        array([  5,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,
                 0,   0,   0,   2,   0,   0,   0,   0,   0,   0,   0, 181, 164,
               188,   1,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255,
               127, 253, 214,  62, 241,   1,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
             dtype=uint8)
        """
        if not is_bool_dtype(self.dtype):
            raise TypeError(
                f"Series must of boolean dtype, found: {self.dtype}"
            )

        return self._column.as_mask()

    def astype(self, dtype, copy=False, errors="raise"):
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

            - ``raise`` : allow exceptions to be raised
            - ``ignore`` : suppress exceptions. On error return original
              object.
            - ``warn`` : prints last exceptions as warnings and
              return original object.

        Returns
        -------
        out : Series
            Returns ``self.copy(deep=copy)`` if ``dtype`` is the same
            as ``self.dtype``.

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([1, 2], dtype='int32')
        >>> series
        0    1
        1    2
        dtype: int32
        >>> series.astype('int64')
        0    1
        1    2
        dtype: int64

        Convert to categorical type:

        >>> series.astype('category')
        0    1
        1    2
        dtype: category
        Categories (2, int64): [1, 2]

        Convert to ordered categorical type with custom ordering:

        >>> cat_dtype = cudf.CategoricalDtype(categories=[2, 1], ordered=True)
        >>> series.astype(cat_dtype)
        0    1
        1    2
        dtype: category
        Categories (2, int64): [2 < 1]

        Note that using ``copy=False`` (enabled by default)
        and changing data on a new Series will
        propagate changes:

        >>> s1 = cudf.Series([1, 2])
        >>> s1
        0    1
        1    2
        dtype: int64
        >>> s2 = s1.astype('int64', copy=False)
        >>> s2[0] = 10
        >>> s1
        0    10
        1     2
        dtype: int64
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

        if is_dtype_equal(dtype, self.dtype):
            return self.copy(deep=copy)
        try:
            data = self._column.astype(dtype)

            return self._from_data(
                {self.name: (data.copy(deep=True) if copy else data)},
                index=self._index,
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

    def sort_index(self, axis=0, *args, **kwargs):
        if axis not in (0, "index"):
            raise ValueError("Only axis=0 is valid for Series.")
        return super().sort_index(axis=axis, *args, **kwargs)

    def sort_values(
        self,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
    ):
        """Sort by the values along either axis.

        Parameters
        ----------
        ascending : bool or list of bool, default True
            Sort ascending vs. descending. Specify list for multiple sort
            orders. If this is a list of bools, must match the length of the
            by.
        na_position : {‘first’, ‘last’}, default ‘last’
            'first' puts nulls at the beginning, 'last' puts nulls at the end
        ignore_index : bool, default False
            If True, index will not be sorted.

        Returns
        -------
        Series : Series with sorted values.

        Notes
        -----
        Difference from pandas:
          * Support axis='index' only.
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
        dtype: int64
        """
        return super().sort_values(
            by=self.name,
            axis=axis,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
            ignore_index=ignore_index,
        )

    def nlargest(self, n=5, keep="first"):
        """Returns a new Series of the *n* largest element.

        Parameters
        ----------
        n : int, default 5
            Return this many descending sorted values.
        keep : {'first', 'last'}, default 'first'
            When there are duplicate values that cannot all fit in a
            Series of `n` elements:

            - ``first`` : return the first `n` occurrences in order
              of appearance.
            - ``last`` : return the last `n` occurrences in reverse
              order of appearance.

        Returns
        -------
        Series
            The `n` largest values in the Series, sorted in decreasing order.

        Examples
        --------
        >>> import cudf
        >>> countries_population = {"Italy": 59000000, "France": 65000000,
        ...                         "Malta": 434000, "Maldives": 434000,
        ...                         "Brunei": 434000, "Iceland": 337000,
        ...                         "Nauru": 11300, "Tuvalu": 11300,
        ...                         "Anguilla": 11300, "Montserrat": 5200}
        >>> series = cudf.Series(countries_population)
        >>> series
        Italy         59000000
        France        65000000
        Malta           434000
        Maldives        434000
        Brunei          434000
        Iceland         337000
        Nauru            11300
        Tuvalu           11300
        Anguilla         11300
        Montserrat        5200
        dtype: int64
        >>> series.nlargest()
        France      65000000
        Italy       59000000
        Malta         434000
        Maldives      434000
        Brunei        434000
        dtype: int64
        >>> series.nlargest(3)
        France    65000000
        Italy     59000000
        Malta       434000
        dtype: int64
        >>> series.nlargest(3, keep='last')
        France    65000000
        Italy     59000000
        Brunei      434000
        dtype: int64
        """
        return self._n_largest_or_smallest(True, n, [self.name], keep)

    def nsmallest(self, n=5, keep="first"):
        """
        Returns a new Series of the *n* smallest element.

        Parameters
        ----------
        n : int, default 5
            Return this many ascending sorted values.
        keep : {'first', 'last'}, default 'first'
            When there are duplicate values that cannot all fit in a
            Series of `n` elements:

            - ``first`` : return the first `n` occurrences in order
              of appearance.
            - ``last`` : return the last `n` occurrences in reverse
              order of appearance.

        Returns
        -------
        Series
            The `n` smallest values in the Series, sorted in increasing order.

        Examples
        --------
        >>> import cudf
        >>> countries_population = {"Italy": 59000000, "France": 65000000,
        ...                         "Brunei": 434000, "Malta": 434000,
        ...                         "Maldives": 434000, "Iceland": 337000,
        ...                         "Nauru": 11300, "Tuvalu": 11300,
        ...                         "Anguilla": 11300, "Montserrat": 5200}
        >>> s = cudf.Series(countries_population)
        >>> s
        Italy       59000000
        France      65000000
        Brunei        434000
        Malta         434000
        Maldives      434000
        Iceland       337000
        Nauru          11300
        Tuvalu         11300
        Anguilla       11300
        Montserrat      5200
        dtype: int64

        The `n` smallest elements where ``n=5`` by default.

        >>> s.nsmallest()
        Montserrat    5200
        Nauru        11300
        Tuvalu       11300
        Anguilla     11300
        Iceland     337000
        dtype: int64

        The `n` smallest elements where ``n=3``. Default `keep` value is
        'first' so Nauru and Tuvalu will be kept.

        >>> s.nsmallest(3)
        Montserrat   5200
        Nauru       11300
        Tuvalu      11300
        dtype: int64

        The `n` smallest elements where ``n=3`` and keeping the last
        duplicates. Anguilla and Tuvalu will be kept since they are the last
        with value 11300 based on the index order.

        >>> s.nsmallest(3, keep='last')
        Montserrat   5200
        Anguilla    11300
        Tuvalu      11300
        dtype: int64
        """
        return self._n_largest_or_smallest(False, n, [self.name], keep)

    def argsort(
        self,
        axis=0,
        kind="quicksort",
        order=None,
        ascending=True,
        na_position="last",
    ):
        obj = self.__class__._from_data(
            {
                None: super().argsort(
                    axis=axis,
                    kind=kind,
                    order=order,
                    ascending=ascending,
                    na_position=na_position,
                )
            }
        )
        obj.name = self.name
        return obj

    def replace(self, to_replace=None, value=None, *args, **kwargs):
        if is_dict_like(to_replace) and value is not None:
            raise ValueError(
                "Series.replace cannot use dict-like to_replace and non-None "
                "value"
            )

        return super().replace(to_replace, value, *args, **kwargs)

    def update(self, other):
        """
        Modify Series in place using values from passed Series.
        Uses non-NA values from passed Series to make updates. Aligns
        on index.

        Parameters
        ----------
        other : Series, or object coercible into Series

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.update(cudf.Series([4, 5, 6]))
        >>> s
        0    4
        1    5
        2    6
        dtype: int64
        >>> s = cudf.Series(['a', 'b', 'c'])
        >>> s
        0    a
        1    b
        2    c
        dtype: object
        >>> s.update(cudf.Series(['d', 'e'], index=[0, 2]))
        >>> s
        0    d
        1    b
        2    e
        dtype: object
        >>> s = cudf.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.update(cudf.Series([4, 5, 6, 7, 8]))
        >>> s
        0    4
        1    5
        2    6
        dtype: int64

        If ``other`` contains NaNs the corresponding values are not updated
        in the original Series.

        >>> s = cudf.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.update(cudf.Series([4, np.nan, 6], nan_as_null=False))
        >>> s
        0    4
        1    2
        2    6
        dtype: int64

        ``other`` can also be a non-Series object type
        that is coercible into a Series

        >>> s = cudf.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.update([4, np.nan, 6])
        >>> s
        0    4
        1    2
        2    6
        dtype: int64
        >>> s = cudf.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.update({1: 9})
        >>> s
        0    1
        1    9
        2    3
        dtype: int64
        """

        if not isinstance(other, cudf.Series):
            other = cudf.Series(other)

        if not self.index.equals(other.index):
            other = other.reindex(index=self.index)
        mask = other.notna()

        self.mask(mask, other, inplace=True)

    def reverse(self):
        warnings.warn(
            "Series.reverse is deprecated and will be removed in the future.",
            FutureWarning,
        )
        rinds = column.arange((self._column.size - 1), -1, -1, dtype=np.int32)
        return self._from_data(
            {self.name: self._column[rinds]}, self.index._values[rinds]
        )

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

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['a', 'b', 'c', 'a'])
        >>> s
        0    a
        1    b
        2    c
        3    a
        dtype: object
        >>> s.one_hot_encoding(['a', 'c', 'b'])
        [0    1.0
        1    0.0
        2    0.0
        3    1.0
        dtype: float64, 0    0.0
        1    0.0
        2    1.0
        3    0.0
        dtype: float64, 0    0.0
        1    1.0
        2    0.0
        3    0.0
        dtype: float64]
        """

        warnings.warn(
            "Series.one_hot_encoding is deprecated and will be removed in "
            "future, use `get_dummies` instead.",
            FutureWarning,
        )

        if hasattr(cats, "to_arrow"):
            cats = cats.to_pandas()
        else:
            cats = pd.Series(cats, dtype="object")
        dtype = cudf.dtype(dtype)

        try:
            cats_col = as_column(cats, nan_as_null=False, dtype=self.dtype)
        except TypeError:
            raise ValueError("Cannot convert `cats` as cudf column.")

        if self._column.size * cats_col.size >= np.iinfo("int32").max:
            raise ValueError(
                "Size limitation exceeded: series.size * category.size < "
                "np.iinfo('int32').max. Consider reducing size of category"
            )

        res = libcudf.transform.one_hot_encode(self._column, cats_col)
        if dtype.type == np.bool_:
            return [
                Series._from_data({None: x}, index=self._index)
                for x in list(res.values())
            ]
        else:
            return [
                Series._from_data({None: x.astype(dtype)}, index=self._index)
                for x in list(res.values())
            ]

    def label_encoding(self, cats, dtype=None, na_sentinel=-1):
        """Perform label encoding.

        Parameters
        ----------
        values : sequence of input values
        dtype : numpy.dtype; optional
            Specifies the output dtype.  If `None` is given, the
            smallest possible integer dtype (starting with np.int8)
            is used.
        na_sentinel : number, default -1
            Value to indicate missing category.

        Returns
        -------
        A sequence of encoded labels with value between 0 and n-1 classes(cats)

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([1, 2, 3, 4, 10])
        >>> s.label_encoding([2, 3])
        0   -1
        1    0
        2    1
        3   -1
        4   -1
        dtype: int8

        `na_sentinel` parameter can be used to
        control the value when there is no encoding.

        >>> s.label_encoding([2, 3], na_sentinel=10)
        0    10
        1     0
        2     1
        3    10
        4    10
        dtype: int8

        When none of `cats` values exist in s, entire
        Series will be `na_sentinel`.

        >>> s.label_encoding(['a', 'b', 'c'])
        0   -1
        1   -1
        2   -1
        3   -1
        4   -1
        dtype: int8
        """

        warnings.warn(
            "Series.label_encoding is deprecated and will be removed in the "
            "future. Consider using cuML's LabelEncoder instead.",
            FutureWarning,
        )

        return self._label_encoding(cats, dtype, na_sentinel)

    def _label_encoding(self, cats, dtype=None, na_sentinel=-1):
        # Private implementation of deprecated public label_encoding method
        def _return_sentinel_series():
            return Series(
                cudf.core.column.full(
                    size=len(self), fill_value=na_sentinel, dtype=dtype
                ),
                index=self.index,
                name=None,
            )

        if dtype is None:
            dtype = min_scalar_type(max(len(cats), na_sentinel), 8)

        cats = column.as_column(cats)
        if is_mixed_with_object_dtype(self, cats):
            return _return_sentinel_series()

        try:
            # Where there is a type-cast failure, we have
            # to catch the exception and return encoded labels
            # with na_sentinel values as there would be no corresponding
            # encoded values of cats in self.
            cats = cats.astype(self.dtype)
        except ValueError:
            return _return_sentinel_series()

        order = column.arange(len(self))
        codes = column.arange(len(cats), dtype=dtype)

        value = cudf.DataFrame({"value": cats, "code": codes})
        codes = cudf.DataFrame(
            {"value": self._data.columns[0].copy(deep=False), "order": order}
        )

        codes = codes.merge(value, on="value", how="left")
        codes = codes.sort_values("order")["code"].fillna(na_sentinel)

        codes.name = None
        codes.index = self._index
        return codes

    # UDF related
    def apply(self, func, convert_dtype=True, args=(), **kwargs):
        """
        Apply a scalar function to the values of a Series.

        Similar to `pandas.Series.apply. Applies a user
        defined function elementwise over a series.

        Parameters
        ----------
        func : function
            Scalar Python function to apply.
        convert_dtype : bool, default True
            In cuDF, this parameter is always True. Because
            cuDF does not support arbitrary object dtypes,
            the result will always be the common type as determined
            by numba based on the function logic and argument types.
            See examples for details.
        args : tuple
            Not supported
        **kwargs
            Not supported

        Notes
        -----
        UDFs are cached in memory to avoid recompilation. The first
        call to the UDF will incur compilation overhead. `func` may
        call nested functions that are decorated with the decorator
        `numba.cuda.jit(device=True)`, otherwise numba will raise a
        typing error.

        Examples
        --------

        Apply a basic function to a series
        >>> sr = cudf.Series([1,2,3])
        >>> def f(x):
        ...     return x + 1
        >>> sr.apply(f)
        0    2
        1    3
        2    4
        dtype: int64

        Apply a basic function to a series with nulls

        >>> sr = cudf.Series([1,cudf.NA,3])
        >>> def f(x):
        ...     return x + 1
        >>> sr.apply(f)
        0       2
        1    <NA>
        2       4
        dtype: int64

        Use a function that does something conditionally,
        based on if the value is or is not null

        >>> sr = cudf.Series([1,cudf.NA,3])
        >>> def f(x):
        ...     if x is cudf.NA:
        ...         return 42
        ...     else:
        ...         return x - 1
        >>> sr.apply(f)
        0     0
        1    42
        2     2
        dtype: int64

        Results will be upcast to the common dtype required
        as derived from the UDFs logic. Note that this means
        the common type will be returned even if such data
        is passed that would not result in any values of that
        dtype.

        >>> sr = cudf.Series([1,cudf.NA,3])
        >>> def f(x):
        ...     return x + 1.5
        >>> sr.apply(f)
        0     2.5
        1    <NA>
        2     4.5
        dtype: float64
        """
        if args or kwargs:
            raise ValueError(
                "UDFs using *args or **kwargs are not yet supported."
            )

        # these functions are generally written as functions of scalar
        # values rather than rows. Rather than writing an entirely separate
        # numba kernel that is not built around a row object, its simpler
        # to just turn this into the equivalent single column dataframe case
        name = self.name or "__temp_srname"
        df = cudf.DataFrame({name: self})
        f_ = cuda.jit(device=True)(func)

        return df.apply(lambda row: f_(row[name]))

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
        -----
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
        if not callable(udf):
            raise ValueError("Input UDF must be a callable object.")
        return self._from_data({self.name: self._unaryop(udf)}, self._index)

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

    def mode(self, dropna=True):
        """
        Return the mode(s) of the dataset.

        Always returns Series even if only one value is returned.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NA/NaN/NaT.

        Returns
        -------
        Series
            Modes of the Series in sorted order.

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([7, 6, 5, 4, 3, 2, 1])
        >>> series
        0    7
        1    6
        2    5
        3    4
        4    3
        5    2
        6    1
        dtype: int64
        >>> series.mode()
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        6    7
        dtype: int64

        We can include ``<NA>`` values in mode by
        passing ``dropna=False``.

        >>> series = cudf.Series([7, 4, 3, 3, 7, None, None])
        >>> series
        0       7
        1       4
        2       3
        3       3
        4       7
        5    <NA>
        6    <NA>
        dtype: int64
        >>> series.mode()
        0    3
        1    7
        dtype: int64
        >>> series.mode(dropna=False)
        0       3
        1       7
        2    <NA>
        dtype: int64
        """
        val_counts = self.value_counts(ascending=False, dropna=dropna)
        if len(val_counts) > 0:
            val_counts = val_counts[val_counts == val_counts.iloc[0]]

        return Series(val_counts.index.sort_values(), name=self.name)

    def round(self, decimals=0, how="half_even"):
        if not is_integer(decimals):
            raise ValueError(
                f"decimals must be an int, got {type(decimals).__name__}"
            )
        decimals = int(decimals)
        return super().round(decimals, how)

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
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        lhs = self.nans_to_nulls().dropna()
        rhs = other.nans_to_nulls().dropna()

        lhs, rhs = _align_indices([lhs, rhs], how="inner")

        return lhs._column.cov(rhs._column)

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

        if method not in ("pearson",):
            raise ValueError(f"Unknown method {method}")

        if min_periods not in (None,):
            raise NotImplementedError("Unsupported argument 'min_periods'")

        if self.empty or other.empty:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        lhs = self.nans_to_nulls().dropna()
        rhs = other.nans_to_nulls().dropna()
        lhs, rhs = _align_indices([lhs, rhs], how="inner")

        return lhs._column.corr(rhs._column)

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

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['lama', 'cow', 'lama', 'beetle', 'lama',
        ...                'hippo'], name='animal')
        >>> s.isin(['cow', 'lama'])
        0     True
        1     True
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        Passing a single string as ``s.isin('lama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(['lama'])
        0     True
        1    False
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        Strings and integers are distinct and are therefore not comparable:

        >>> cudf.Series([1]).isin(['1'])
        0    False
        dtype: bool
        >>> cudf.Series([1.1]).isin(['1.1'])
        0    False
        dtype: bool
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

        Returns
        -------
        Series
            A series with only the unique values.

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series(['a', 'a', 'b', None, 'b', None, 'c'])
        >>> series
        0       a
        1       a
        2       b
        3    <NA>
        4       b
        5    <NA>
        6       c
        dtype: object
        >>> series.unique()
        0    <NA>
        1       a
        2       b
        3       c
        dtype: object
        """
        res = self._column.unique()
        return Series(res, name=self.name)

    def nunique(self, method="sort", dropna=True):
        """Returns the number of unique values of the Series: approximate version,
        and exact version to be moved to libcudf

        Excludes NA values by default.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NA values in the count.

        Returns
        -------
        int

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([1, 3, 5, 7, 7])
        >>> s
        0    1
        1    3
        2    5
        3    7
        4    7
        dtype: int64
        >>> s.nunique()
        4
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
            If True then the object returned will contain
            the relative frequencies of the unique values.

        sort : bool, default True
            Sort by frequencies.

        ascending : bool, default False
            Sort in ascending order.

        bins : int, optional
            Rather than count values, group them into half-open bins,
            works with numeric data. This Parameter is not yet supported.

        dropna : bool, default True
            Don’t include counts of NaN and None.

        Returns
        -------
        result : Series containing counts of unique values.

        See also
        --------
        Series.count
            Number of non-NA elements in a Series.

        cudf.DataFrame.count
            Number of non-NA elements in a DataFrame.

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, None])
        >>> sr
        0     1.0
        1     2.0
        2     2.0
        3     3.0
        4     3.0
        5     3.0
        6    <NA>
        dtype: float64
        >>> sr.value_counts()
        3.0    3
        2.0    2
        1.0    1
        dtype: int32

        The order of the counts can be changed by passing ``ascending=True``:

        >>> sr.value_counts(ascending=True)
        1.0    1
        2.0    2
        3.0    3
        dtype: int32

        With ``normalize`` set to True, returns the relative frequency
        by dividing all values by the sum of values.

        >>> sr.value_counts(normalize=True)
        3.0    0.500000
        2.0    0.333333
        1.0    0.166667
        dtype: float64

        To include ``NA`` value counts, pass ``dropna=False``:

        >>> sr = cudf.Series([1.0, 2.0, 2.0, 3.0, None, 3.0, 3.0, None])
        >>> sr
        0     1.0
        1     2.0
        2     2.0
        3     3.0
        4    <NA>
        5     3.0
        6     3.0
        7    <NA>
        dtype: float64
        >>> sr.value_counts(dropna=False)
        3.0     3
        2.0     2
        <NA>    2
        1.0     1
        dtype: int32
        """
        if bins is not None:
            raise NotImplementedError("bins is not yet supported")

        if dropna and self.null_count == len(self):
            return Series(
                [],
                dtype=np.int32,
                name=self.name,
                index=cudf.Index([], dtype=self.dtype),
            )

        res = self.groupby(self, dropna=dropna).count(dropna=dropna)
        res.index.name = None

        if sort:
            res = res.sort_values(ascending=ascending)

        if normalize:
            res = res / float(res._column.sum())
        return res

    def hash_values(self, method="murmur3"):
        """Compute the hash of values in this column.

        Parameters
        ----------
        method : {'murmur3', 'md5'}, default 'murmur3'
            Hash function to use:
            * murmur3: MurmurHash3 hash function.
            * md5: MD5 hash function.

        Returns
        -------
        Series
            A Series with hash values.

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([10, 120, 30])
        >>> series
        0     10
        1    120
        2     30
        dtype: int64
        >>> series.hash_values(method="murmur3")
        0   -1930516747
        1     422619251
        2    -941520876
        dtype: int32
        >>> series.hash_values(method="md5")
        0    7be4bbacbfdb05fb3044e36c22b41e8b
        1    947ca8d2c5f0f27437f156cfbfab0969
        2    d0580ef52d27c043c8e341fd5039b166
        dtype: object
        """
        return Series._from_data(
            {None: self._hash(method=method)}, index=self.index
        )

    def hash_encode(self, stop, use_name=False):
        """Encode column values as ints in [0, stop) using hash function.

        This method is deprecated. Replace ``series.hash_encode(stop,
        use_name=False)`` with ``series.hash_values(method="murmur3") % stop``.

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

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([10, 120, 30])
        >>> series.hash_encode(stop=200)
        0     53
        1     51
        2    124
        dtype: int32

        You can choose to include name while hash
        encoding by specifying `use_name=True`

        >>> series.hash_encode(stop=200, use_name=True)
        0    131
        1     29
        2     76
        dtype: int32
        """
        warnings.warn(
            "The `hash_encode` method will be removed in a future cuDF "
            "release. Replace `series.hash_encode(stop, use_name=False)` "
            'with `series.hash_values(method="murmur3") % stop`.',
            FutureWarning,
        )

        if not stop > 0:
            raise ValueError("stop must be a positive integer.")

        if use_name:
            name_hasher = sha256()
            name_hasher.update(str(self.name).encode())
            name_hash_bytes = name_hasher.digest()[:4]
            name_hash_int = (
                int.from_bytes(name_hash_bytes, "little", signed=False)
                & 0xFFFFFFFF
            )
            initial_hash = [name_hash_int]
        else:
            initial_hash = None

        hashed_values = Series._from_data(
            {
                self.name: self._hash(
                    method="murmur3", initial_hash=initial_hash
                )
            },
            self.index,
        )

        if hashed_values.has_nulls:
            raise ValueError("Column must have no nulls.")

        return hashed_values % stop

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
        float or Series
            If ``q`` is an array, a Series will be returned where the
            index is ``q`` and the values are the quantiles, otherwise
            a float will be returned.

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([1, 2, 3, 4])
        >>> series
        0    1
        1    2
        2    3
        3    4
        dtype: int64
        >>> series.quantile(0.5)
        2.5
        >>> series.quantile([0.25, 0.5, 0.75])
        0.25    1.75
        0.50    2.50
        0.75    3.25
        dtype: float64
        """

        result = self._column.quantile(q, interpolation, exact)

        if isinstance(q, Number):
            return result

        if quant_index:
            index = np.asarray(q)
            if len(self) == 0:
                result = column_empty_like(
                    index, dtype=self.dtype, masked=True, newsize=len(index),
                )
        else:
            index = None

        return Series(result, index=index, name=self.name)

    @docutils.doc_describe()
    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
        datetime_is_numeric=False,
    ):
        """{docstring}"""

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

        def _describe_numeric(self):
            # mimicking pandas
            index = (
                ["count", "mean", "std", "min"]
                + _format_percentile_names(percentiles)
                + ["max"]
            )
            data = (
                [self.count(), self.mean(), self.std(), self.min()]
                + self.quantile(percentiles).to_numpy(na_value=np.nan).tolist()
                + [self.max()]
            )
            data = _format_stats_values(data)

            return Series(
                data=data, index=index, nan_as_null=False, name=self.name,
            )

        def _describe_timedelta(self):
            # mimicking pandas
            index = (
                ["count", "mean", "std", "min"]
                + _format_percentile_names(percentiles)
                + ["max"]
            )

            data = (
                [
                    str(self.count()),
                    str(self.mean()),
                    str(self.std()),
                    str(pd.Timedelta(self.min())),
                ]
                + self.quantile(percentiles)
                .astype("str")
                .to_numpy(na_value=None)
                .tolist()
                + [str(pd.Timedelta(self.max()))]
            )

            return Series(
                data=data,
                index=index,
                dtype="str",
                nan_as_null=False,
                name=self.name,
            )

        def _describe_categorical(self):
            # blocked by StringColumn/DatetimeColumn support for
            # value_counts/unique
            index = ["count", "unique", "top", "freq"]
            val_counts = self.value_counts(ascending=False)
            data = [self.count(), self.unique().size]

            if data[1] > 0:
                top, freq = val_counts.index[0], val_counts.iloc[0]
                data += [str(top), freq]
            # If the DataFrame is empty, set 'top' and 'freq' to None
            # to maintain output shape consistency
            else:
                data += [None, None]

            return Series(
                data=data,
                dtype="str",
                index=index,
                nan_as_null=False,
                name=self.name,
            )

        def _describe_timestamp(self):

            index = (
                ["count", "mean", "min"]
                + _format_percentile_names(percentiles)
                + ["max"]
            )

            data = (
                [
                    str(self.count()),
                    str(self.mean().to_numpy().astype("datetime64[ns]")),
                    str(pd.Timestamp(self.min().astype("datetime64[ns]"))),
                ]
                + self.quantile(percentiles)
                .astype("str")
                .to_numpy(na_value=None)
                .tolist()
                + [str(pd.Timestamp((self.max()).astype("datetime64[ns]")))]
            )

            return Series(
                data=data,
                dtype="str",
                index=index,
                nan_as_null=False,
                name=self.name,
            )

        if percentiles is not None:
            percentiles = _prepare_percentiles(percentiles)
        else:
            # pandas defaults
            percentiles = np.array([0.25, 0.5, 0.75])

        if is_bool_dtype(self.dtype):
            return _describe_categorical(self)
        elif isinstance(self._column, cudf.core.column.NumericalColumn):
            return _describe_numeric(self)
        elif isinstance(self._column, cudf.core.column.TimeDeltaColumn):
            return _describe_timedelta(self)
        elif isinstance(self._column, cudf.core.column.DatetimeColumn):
            return _describe_timestamp(self)
        else:
            return _describe_categorical(self)

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

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([0.2, 6.4, 3.0, 1.6])
        >>> bins = cudf.Series([0.0, 1.0, 2.5, 4.0, 10.0])
        >>> inds = s.digitize(bins)
        >>> inds
        0    1
        1    4
        2    3
        3    2
        dtype: int32
        """
        return Series(
            cudf.core.column.numerical.digitize(self._column, bins, right)
        )

    def diff(self, periods=1):
        """Calculate the difference between values at positions i and i - N in
        an array and store the output in a new array.

        Returns
        -------
        Series
            First differences of the Series.

        Notes
        -----
        Diff currently only supports float and integer dtype columns with
        no null values.

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([1, 1, 2, 3, 5, 8])
        >>> series
        0    1
        1    1
        2    2
        3    3
        4    5
        5    8
        dtype: int64

        Difference with previous row

        >>> series.diff()
        0    <NA>
        1       0
        2       1
        3       1
        4       2
        5       3
        dtype: int64

        Difference with 3rd previous row

        >>> series.diff(periods=3)
        0    <NA>
        1    <NA>
        2    <NA>
        3       2
        4       4
        5       6
        dtype: int64

        Difference with following row

        >>> series.diff(periods=-1)
        0       0
        1      -1
        2      -1
        3      -2
        4      -3
        5    <NA>
        dtype: int64
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

    @copy_docstring(SeriesGroupBy)
    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=False,
        group_keys=True,
        squeeze=False,
        observed=False,
        dropna=True,
    ):
        import cudf.core.resample

        if axis not in (0, "index"):
            raise NotImplementedError("axis parameter is not yet implemented")

        if group_keys is not True:
            raise NotImplementedError(
                "The group_keys keyword is not yet implemented"
            )

        if squeeze is not False:
            raise NotImplementedError(
                "squeeze parameter is not yet implemented"
            )

        if observed is not False:
            raise NotImplementedError(
                "observed parameter is not yet implemented"
            )

        if by is None and level is None:
            raise TypeError(
                "groupby() requires either by or level to be specified."
            )

        return (
            cudf.core.resample.SeriesResampler(self, by=by)
            if isinstance(by, cudf.Grouper) and by.freq
            else SeriesGroupBy(
                self, by=by, level=level, dropna=dropna, sort=sort
            )
        )

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

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([10, 20, 30])
        >>> series
        0    10
        1    20
        2    30
        dtype: int64
        >>> series.name
        >>> renamed_series = series.rename('numeric_series')
        >>> renamed_series
        0    10
        1    20
        2    30
        Name: numeric_series, dtype: int64
        >>> renamed_series.name
        'numeric_series'
        """
        out = self.copy(deep=False)
        out = out.set_index(self.index)
        if index:
            out.name = index

        return out.copy(deep=copy)

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

        if lsuffix or rsuffix:
            raise ValueError(
                "The lsuffix and rsuffix keywords have been replaced with the "
                "``suffixes=`` keyword.  "
                "Please provide the following instead: \n\n"
                "    suffixes=('%s', '%s')"
                % (lsuffix or "_x", rsuffix or "_y")
            )
        else:
            lsuffix, rsuffix = suffixes

        result = super()._merge(
            other,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            sort=sort,
            indicator=False,
            suffixes=suffixes,
        )

        return result

    def keys(self):
        """
        Return alias for index.

        Returns
        -------
        Index
            Index of the Series.

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series([10, 11, 12, 13, 14, 15])
        >>> sr
        0    10
        1    11
        2    12
        3    13
        4    14
        5    15
        dtype: int64

        >>> sr.keys()
        RangeIndex(start=0, stop=6)
        >>> sr = cudf.Series(['a', 'b', 'c'])
        >>> sr
        0    a
        1    b
        2    c
        dtype: object
        >>> sr.keys()
        RangeIndex(start=0, stop=3)
        >>> sr = cudf.Series([1, 2, 3], index=['a', 'b', 'c'])
        >>> sr
        a    1
        b    2
        c    3
        dtype: int64
        >>> sr.keys()
        StringIndex(['a' 'b' 'c'], dtype='object')
        """
        return self.index

    def explode(self, ignore_index=False):
        """
        Transform each element of a list-like to a row, replicating index
        values.

        Parameters
        ----------
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([[1, 2, 3], [], None, [4, 5]])
        >>> s
        0    [1, 2, 3]
        1           []
        2         None
        3       [4, 5]
        dtype: list
        >>> s.explode()
        0       1
        0       2
        0       3
        1    <NA>
        2    <NA>
        3       4
        3       5
        dtype: int64
        """
        if not is_list_dtype(self._column.dtype):
            data = self._data.copy(deep=True)
            idx = None if ignore_index else self._index.copy(deep=True)
            return self.__class__._from_data(data, index=idx)

        return super()._explode(self._column_names[0], ignore_index)

    def pct_change(
        self, periods=1, fill_method="ffill", limit=None, freq=None
    ):
        """
        Calculates the percent change between sequential elements
        in the Series.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for forming percent change.
        fill_method : str, default 'ffill'
            How to handle NAs before computing percent changes.
        limit : int, optional
            The number of consecutive NAs to fill before stopping.
            Not yet implemented.
        freq : str, optional
            Increment to use from time series API.
            Not yet implemented.

        Returns
        -------
        Series
        """
        if limit is not None:
            raise NotImplementedError("limit parameter not supported yet.")
        if freq is not None:
            raise NotImplementedError("freq parameter not supported yet.")
        elif fill_method not in {"ffill", "pad", "bfill", "backfill"}:
            raise ValueError(
                "fill_method must be one of 'ffill', 'pad', "
                "'bfill', or 'backfill'."
            )

        data = self.fillna(method=fill_method, limit=limit)
        diff = data.diff(periods=periods)
        change = diff / data.shift(periods=periods, freq=freq)
        return change


def make_binop_func(op):
    # This function is used to wrap binary operations in Frame with an
    # appropriate API for Series as required for pandas compatibility. The
    # main effect is reordering and error-checking parameters in
    # Series-specific ways.
    wrapped_func = getattr(Frame, op)

    @functools.wraps(wrapped_func)
    def wrapper(self, other, level=None, fill_value=None, axis=0):
        if axis != 0:
            raise NotImplementedError("Only axis=0 supported at this time.")
        return wrapped_func(self, other, axis, level, fill_value)

    # functools.wraps copies module level attributes to `wrapper` and sets
    # __wrapped__ attributes to `wrapped_func`. Cpython looks up the signature
    # string of a function by recursively delving into __wrapped__ until
    # it hits the first function that has __signature__ attribute set. To make
    # the signature string of `wrapper` matches with its actual parameter list,
    # we directly set the __signature__ attribute of `wrapper` below.

    new_sig = inspect.signature(
        lambda self, other, level=None, fill_value=None, axis=0: None
    )
    wrapper.__signature__ = new_sig
    return wrapper


# Wrap all Frame binop functions with the expected API for Series.
for binop in (
    "add",
    "radd",
    "subtract",
    "sub",
    "rsub",
    "multiply",
    "mul",
    "rmul",
    "mod",
    "rmod",
    "pow",
    "rpow",
    "floordiv",
    "rfloordiv",
    "truediv",
    "div",
    "divide",
    "rtruediv",
    "rdiv",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
):
    setattr(Series, binop, make_binop_func(binop))


class DatetimeProperties(object):
    """
    Accessor object for datetimelike properties of the Series values.

    Returns
    -------
    Returns a Series indexed like the original Series.

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> seconds_series = cudf.Series(pd.date_range("2000-01-01", periods=3,
    ...     freq="s"))
    >>> seconds_series
    0   2000-01-01 00:00:00
    1   2000-01-01 00:00:01
    2   2000-01-01 00:00:02
    dtype: datetime64[ns]
    >>> seconds_series.dt.second
    0    0
    1    1
    2    2
    dtype: int16
    >>> hours_series = cudf.Series(pd.date_range("2000-01-01", periods=3,
    ...     freq="h"))
    >>> hours_series
    0   2000-01-01 00:00:00
    1   2000-01-01 01:00:00
    2   2000-01-01 02:00:00
    dtype: datetime64[ns]
    >>> hours_series.dt.hour
    0    0
    1    1
    2    2
    dtype: int16
    >>> weekday_series = cudf.Series(pd.date_range("2000-01-01", periods=3,
    ...     freq="q"))
    >>> weekday_series
    0   2000-03-31
    1   2000-06-30
    2   2000-09-30
    dtype: datetime64[ns]
    >>> weekday_series.dt.weekday
    0    4
    1    4
    2    5
    dtype: int16
    """

    def __init__(self, series):
        self.series = series

    @property
    def year(self):
        """
        The year of the datetime.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> datetime_series = cudf.Series(pd.date_range("2000-01-01",
        ...         periods=3, freq="Y"))
        >>> datetime_series
        0   2000-12-31
        1   2001-12-31
        2   2002-12-31
        dtype: datetime64[ns]
        >>> datetime_series.dt.year
        0    2000
        1    2001
        2    2002
        dtype: int16
        """
        return self._get_dt_field("year")

    @property
    def month(self):
        """
        The month as January=1, December=12.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_series = cudf.Series(pd.date_range("2000-01-01",
        ...         periods=3, freq="M"))
        >>> datetime_series
        0   2000-01-31
        1   2000-02-29
        2   2000-03-31
        dtype: datetime64[ns]
        >>> datetime_series.dt.month
        0    1
        1    2
        2    3
        dtype: int16
        """
        return self._get_dt_field("month")

    @property
    def day(self):
        """
        The day of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_series = cudf.Series(pd.date_range("2000-01-01",
        ...         periods=3, freq="D"))
        >>> datetime_series
        0   2000-01-01
        1   2000-01-02
        2   2000-01-03
        dtype: datetime64[ns]
        >>> datetime_series.dt.day
        0    1
        1    2
        2    3
        dtype: int16
        """
        return self._get_dt_field("day")

    @property
    def hour(self):
        """
        The hours of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_series = cudf.Series(pd.date_range("2000-01-01",
        ...         periods=3, freq="h"))
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 01:00:00
        2   2000-01-01 02:00:00
        dtype: datetime64[ns]
        >>> datetime_series.dt.hour
        0    0
        1    1
        2    2
        dtype: int16
        """
        return self._get_dt_field("hour")

    @property
    def minute(self):
        """
        The minutes of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_series = cudf.Series(pd.date_range("2000-01-01",
        ...         periods=3, freq="T"))
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 00:01:00
        2   2000-01-01 00:02:00
        dtype: datetime64[ns]
        >>> datetime_series.dt.minute
        0    0
        1    1
        2    2
        dtype: int16
        """
        return self._get_dt_field("minute")

    @property
    def second(self):
        """
        The seconds of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_series = cudf.Series(pd.date_range("2000-01-01",
        ...         periods=3, freq="s"))
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 00:00:01
        2   2000-01-01 00:00:02
        dtype: datetime64[ns]
        >>> datetime_series.dt.second
        0    0
        1    1
        2    2
        dtype: int16
        """
        return self._get_dt_field("second")

    @property
    def weekday(self):
        """
        The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_series = cudf.Series(pd.date_range('2016-12-31',
        ...     '2017-01-08', freq='D'))
        >>> datetime_series
        0   2016-12-31
        1   2017-01-01
        2   2017-01-02
        3   2017-01-03
        4   2017-01-04
        5   2017-01-05
        6   2017-01-06
        7   2017-01-07
        8   2017-01-08
        dtype: datetime64[ns]
        >>> datetime_series.dt.weekday
        0    5
        1    6
        2    0
        3    1
        4    2
        5    3
        6    4
        7    5
        8    6
        dtype: int16
        """
        return self._get_dt_field("weekday")

    @property
    def dayofweek(self):
        """
        The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_series = cudf.Series(pd.date_range('2016-12-31',
        ...     '2017-01-08', freq='D'))
        >>> datetime_series
        0   2016-12-31
        1   2017-01-01
        2   2017-01-02
        3   2017-01-03
        4   2017-01-04
        5   2017-01-05
        6   2017-01-06
        7   2017-01-07
        8   2017-01-08
        dtype: datetime64[ns]
        >>> datetime_series.dt.dayofweek
        0    5
        1    6
        2    0
        3    1
        4    2
        5    3
        6    4
        7    5
        8    6
        dtype: int16
        """
        return self._get_dt_field("weekday")

    @property
    def dayofyear(self):
        """
        The day of the year, from 1-365 in non-leap years and
        from 1-366 in leap years.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_series = cudf.Series(pd.date_range('2016-12-31',
        ...     '2017-01-08', freq='D'))
        >>> datetime_series
        0   2016-12-31
        1   2017-01-01
        2   2017-01-02
        3   2017-01-03
        4   2017-01-04
        5   2017-01-05
        6   2017-01-06
        7   2017-01-07
        8   2017-01-08
        dtype: datetime64[ns]
        >>> datetime_series.dt.dayofyear
        0    366
        1      1
        2      2
        3      3
        4      4
        5      5
        6      6
        7      7
        8      8
        dtype: int16
        """
        return self._get_dt_field("day_of_year")

    @property
    def day_of_year(self):
        """
        The day of the year, from 1-365 in non-leap years and
        from 1-366 in leap years.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_series = cudf.Series(pd.date_range('2016-12-31',
        ...     '2017-01-08', freq='D'))
        >>> datetime_series
        0   2016-12-31
        1   2017-01-01
        2   2017-01-02
        3   2017-01-03
        4   2017-01-04
        5   2017-01-05
        6   2017-01-06
        7   2017-01-07
        8   2017-01-08
        dtype: datetime64[ns]
        >>> datetime_series.dt.day_of_year
        0    366
        1      1
        2      2
        3      3
        4      4
        5      5
        6      6
        7      7
        8      8
        dtype: int16
        """
        return self._get_dt_field("day_of_year")

    @property
    def is_leap_year(self):
        """
        Boolean indicator if the date belongs to a leap year.

        A leap year is a year, which has 366 days (instead of 365) including
        29th of February as an intercalary day. Leap years are years which are
        multiples of four with the exception of years divisible by 100 but not
        by 400.

        Returns
        -------
        Series
        Booleans indicating if dates belong to a leap year.

        Example
        -------
        >>> import pandas as pd, cudf
        >>> s = cudf.Series(
        ...     pd.date_range(start='2000-02-01', end='2013-02-01', freq='1Y'))
        >>> s
        0    2000-12-31
        1    2001-12-31
        2    2002-12-31
        3    2003-12-31
        4    2004-12-31
        5    2005-12-31
        6    2006-12-31
        7    2007-12-31
        8    2008-12-31
        9    2009-12-31
        10   2010-12-31
        11   2011-12-31
        12   2012-12-31
        dtype: datetime64[ns]
        >>> s.dt.is_leap_year
        0      True
        1     False
        2     False
        3     False
        4      True
        5     False
        6     False
        7     False
        8      True
        9     False
        10    False
        11    False
        12     True
        dtype: bool
        """
        res = libcudf.datetime.is_leap_year(self.series._column).fillna(False)
        return Series._from_data(
            ColumnAccessor({None: res}),
            index=self.series._index,
            name=self.series.name,
        )

    @property
    def quarter(self):
        """
        Integer indicator for which quarter of the year the date belongs in.

        There are 4 quarters in a year. With the first quarter being from
        January - March, second quarter being April - June, third quarter
        being July - September and fourth quarter being October - December.

        Returns
        -------
        Series
        Integer indicating which quarter the date belongs to.

        Examples
        -------
        >>> import cudf
        >>> s = cudf.Series(["2020-05-31 08:00:00","1999-12-31 18:40:00"],
        ...     dtype="datetime64[ms]")
        >>> s.dt.quarter
        0    2
        1    4
        dtype: int8
        """
        res = libcudf.datetime.extract_quarter(self.series._column).astype(
            np.int8
        )
        return Series._from_data(
            {None: res}, index=self.series._index, name=self.series.name,
        )

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
        >>> ser = cudf.Series(pd.date_range(start="2021-07-25",
        ... end="2021-07-30"))
        >>> ser.dt.isocalendar()
           year  week  day
        0  2021    29    7
        1  2021    30    1
        2  2021    30    2
        3  2021    30    3
        4  2021    30    4
        5  2021    30    5
        >>> ser.dt.isocalendar().week
        0    29
        1    30
        2    30
        3    30
        4    30
        5    30
        Name: week, dtype: object

        >>> serIndex = cudf.to_datetime(pd.Series(["2010-01-01", pd.NaT]))
        >>> serIndex.dt.isocalendar()
            year  week  day
        0  2009    53     5
        1  <NA>  <NA>  <NA>
        >>> serIndex.dt.isocalendar().year
        0    2009
        1    <NA>
        Name: year, dtype: object
        """
        return cudf.core.tools.datetimes._to_iso_calendar(self)

    @property
    def is_month_start(self):
        """
        Booleans indicating if dates are the first day of the month.
        """
        return (self.day == 1).fillna(False)

    @property
    def days_in_month(self):
        """
        Get the total number of days in the month that the date falls on.

        Returns
        -------
        Series
        Integers representing the number of days in month

        Example
        -------
        >>> import pandas as pd, cudf
        >>> s = cudf.Series(
        ...     pd.date_range(start='2000-08-01', end='2001-08-01', freq='1M'))
        >>> s
        0    2000-08-31
        1    2000-09-30
        2    2000-10-31
        3    2000-11-30
        4    2000-12-31
        5    2001-01-31
        6    2001-02-28
        7    2001-03-31
        8    2001-04-30
        9    2001-05-31
        10   2001-06-30
        11   2001-07-31
        dtype: datetime64[ns]
        >>> s.dt.days_in_month
        0     31
        1     30
        2     31
        3     30
        4     31
        5     31
        6     28
        7     31
        8     30
        9     31
        10    30
        11    31
        dtype: int16
        """
        res = libcudf.datetime.days_in_month(self.series._column)
        return Series._from_data(
            ColumnAccessor({None: res}),
            index=self.series._index,
            name=self.series.name,
        )

    @property
    def is_month_end(self):
        """
        Boolean indicator if the date is the last day of the month.

        Returns
        -------
        Series
        Booleans indicating if dates are the last day of the month.

        Example
        -------
        >>> import pandas as pd, cudf
        >>> s = cudf.Series(
        ...     pd.date_range(start='2000-08-26', end='2000-09-03', freq='1D'))
        >>> s
        0   2000-08-26
        1   2000-08-27
        2   2000-08-28
        3   2000-08-29
        4   2000-08-30
        5   2000-08-31
        6   2000-09-01
        7   2000-09-02
        8   2000-09-03
        dtype: datetime64[ns]
        >>> s.dt.is_month_end
        0    False
        1    False
        2    False
        3    False
        4    False
        5     True
        6    False
        7    False
        8    False
        dtype: bool
        """  # noqa: E501
        last_day = libcudf.datetime.last_day_of_month(self.series._column)
        last_day = Series._from_data(
            ColumnAccessor({None: last_day}),
            index=self.series._index,
            name=self.series.name,
        )
        return (self.day == last_day.dt.day).fillna(False)

    @property
    def is_quarter_start(self):
        """
        Boolean indicator if the date is the first day of a quarter.

        Returns
        -------
        Series
        Booleans indicating if dates are the begining of a quarter

        Example
        -------
        >>> import pandas as pd, cudf
        >>> s = cudf.Series(
        ...     pd.date_range(start='2000-09-26', end='2000-10-03', freq='1D'))
        >>> s
        0   2000-09-26
        1   2000-09-27
        2   2000-09-28
        3   2000-09-29
        4   2000-09-30
        5   2000-10-01
        6   2000-10-02
        7   2000-10-03
        dtype: datetime64[ns]
        >>> s.dt.is_quarter_start
        0    False
        1    False
        2    False
        3    False
        4    False
        5     True
        6    False
        7    False
        dtype: bool
        """
        day = self.series._column.get_dt_field("day")
        first_month = self.series._column.get_dt_field("month").isin(
            [1, 4, 7, 10]
        )

        result = ((day == cudf.Scalar(1)) & first_month).fillna(False)
        return Series._from_data(
            {None: result}, index=self.series._index, name=self.series.name,
        )

    @property
    def is_quarter_end(self):
        """
        Boolean indicator if the date is the last day of a quarter.

        Returns
        -------
        Series
        Booleans indicating if dates are the end of a quarter

        Example
        -------
        >>> import pandas as pd, cudf
        >>> s = cudf.Series(
        ...     pd.date_range(start='2000-09-26', end='2000-10-03', freq='1D'))
        >>> s
        0   2000-09-26
        1   2000-09-27
        2   2000-09-28
        3   2000-09-29
        4   2000-09-30
        5   2000-10-01
        6   2000-10-02
        7   2000-10-03
        dtype: datetime64[ns]
        >>> s.dt.is_quarter_end
        0    False
        1    False
        2    False
        3    False
        4     True
        5    False
        6    False
        7    False
        dtype: bool
        """
        day = self.series._column.get_dt_field("day")
        last_day = libcudf.datetime.last_day_of_month(self.series._column)
        last_day = last_day.get_dt_field("day")
        last_month = self.series._column.get_dt_field("month").isin(
            [3, 6, 9, 12]
        )

        result = ((day == last_day) & last_month).fillna(False)
        return Series._from_data(
            {None: result}, index=self.series._index, name=self.series.name,
        )

    @property
    def is_year_start(self):
        """
        Boolean indicator if the date is the first day of the year.

        Returns
        -------
        Series
        Booleans indicating if dates are the first day of the year.

        Example
        -------
        >>> import pandas as pd, cudf
        >>> s = cudf.Series(pd.date_range("2017-12-30", periods=3))
        >>> dates
        0   2017-12-30
        1   2017-12-31
        2   2018-01-01
        dtype: datetime64[ns]
        >>> dates.dt.is_year_start
        0    False
        1    False
        2    True
        dtype: bool
        """
        outcol = self.series._column.get_dt_field(
            "day_of_year"
        ) == cudf.Scalar(1)
        return Series._from_data(
            {None: outcol.fillna(False)},
            index=self.series._index,
            name=self.series.name,
        )

    @property
    def is_year_end(self):
        """
        Boolean indicator if the date is the last day of the year.

        Returns
        -------
        Series
        Booleans indicating if dates are the last day of the year.

        Example
        -------
        >>> import pandas as pd, cudf
        >>> dates = cudf.Series(pd.date_range("2017-12-30", periods=3))
        >>> dates
        0   2017-12-30
        1   2017-12-31
        2   2018-01-01
        dtype: datetime64[ns]
        >>> dates.dt.is_year_end
        0    False
        1     True
        2    False
        dtype: bool
        """
        day_of_year = self.series._column.get_dt_field("day_of_year")
        leap_dates = libcudf.datetime.is_leap_year(self.series._column)

        leap = day_of_year == cudf.Scalar(366)
        non_leap = day_of_year == cudf.Scalar(365)
        result = cudf._lib.copying.copy_if_else(leap, non_leap, leap_dates)
        result = result.fillna(False)
        return Series._from_data(
            {None: result}, index=self.series._index, name=self.series.name,
        )

    def _get_dt_field(self, field):
        out_column = self.series._column.get_dt_field(field)
        return Series(
            data=out_column, index=self.series._index, name=self.series.name
        )

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
        Series
            Series with all timestamps rounded up to the specified frequency.
            The index is preserved.

        Examples
        --------
        >>> import cudf
        >>> t = cudf.Series(["2001-01-01 00:04:45", "2001-01-01 00:04:58",
        ... "2001-01-01 00:05:04"], dtype="datetime64[ns]")
        >>> t.dt.ceil("T")
        0   2001-01-01 00:05:00
        1   2001-01-01 00:05:00
        2   2001-01-01 00:06:00
        dtype: datetime64[ns]
        """
        out_column = self.series._column.ceil(freq)

        return Series._from_data(
            data={self.series.name: out_column}, index=self.series._index
        )

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
        Series
            Series with all timestamps rounded up to the specified frequency.
            The index is preserved.

        Examples
        --------
        >>> import cudf
        >>> t = cudf.Series(["2001-01-01 00:04:45", "2001-01-01 00:04:58",
        ... "2001-01-01 00:05:04"], dtype="datetime64[ns]")
        >>> t.dt.floor("T")
        0   2001-01-01 00:04:00
        1   2001-01-01 00:04:00
        2   2001-01-01 00:05:00
        dtype: datetime64[ns]
        """
        out_column = self.series._column.floor(freq)

        return Series._from_data(
            data={self.series.name: out_column}, index=self.series._index
        )

    def strftime(self, date_format, *args, **kwargs):
        """
        Convert to Series using specified ``date_format``.

        Return a Series of formatted strings specified by ``date_format``,
        which supports the same string format as the python standard library.
        Details of the string format can be found in `python string format doc
        <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior>`_.

        Parameters
        ----------
        date_format : str
            Date format string (e.g. “%Y-%m-%d”).

        Returns
        -------
        Series
            Series of formatted strings.

        Notes
        -----

        The following date format identifiers are not yet
        supported: ``%c``, ``%x``,``%X``

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> weekday_series = cudf.Series(pd.date_range("2000-01-01", periods=3,
        ...      freq="q"))
        >>> weekday_series.dt.strftime("%Y-%m-%d")
        >>> weekday_series
        0   2000-03-31
        1   2000-06-30
        2   2000-09-30
        dtype: datetime64[ns]
        0    2000-03-31
        1    2000-06-30
        2    2000-09-30
        dtype: object
        >>> weekday_series.dt.strftime("%Y %d %m")
        0    2000 31 03
        1    2000 30 06
        2    2000 30 09
        dtype: object
        >>> weekday_series.dt.strftime("%Y / %d / %m")
        0    2000 / 31 / 03
        1    2000 / 30 / 06
        2    2000 / 30 / 09
        dtype: object
        """

        if not isinstance(date_format, str):
            raise TypeError(
                f"'date_format' must be str, not {type(date_format)}"
            )

        # TODO: Remove following validations
        # once https://github.com/rapidsai/cudf/issues/5991
        # is implemented
        not_implemented_formats = {
            "%c",
            "%x",
            "%X",
        }
        for d_format in not_implemented_formats:
            if d_format in date_format:
                raise NotImplementedError(
                    f"{d_format} date-time format is not "
                    f"supported yet, Please follow this issue "
                    f"https://github.com/rapidsai/cudf/issues/5991 "
                    f"for tracking purposes."
                )
        str_col = self.series._column.as_string_column(
            dtype="str", format=date_format
        )
        return Series(
            data=str_col, index=self.series._index, name=self.series.name
        )


class TimedeltaProperties(object):
    """
    Accessor object for timedeltalike properties of the Series values.

    Returns
    -------
    Returns a Series indexed like the original Series.

    Examples
    --------
    >>> import cudf
    >>> seconds_series = cudf.Series([1, 2, 3], dtype='timedelta64[s]')
    >>> seconds_series
    0    00:00:01
    1    00:00:02
    2    00:00:03
    dtype: timedelta64[s]
    >>> seconds_series.dt.seconds
    0    1
    1    2
    2    3
    dtype: int64
    >>> series = cudf.Series([12231312123, 1231231231, 1123236768712, 2135656,
    ...     3244334234], dtype='timedelta64[ms]')
    >>> series
    0      141 days 13:35:12.123
    1       14 days 06:00:31.231
    2    13000 days 10:12:48.712
    3        0 days 00:35:35.656
    4       37 days 13:12:14.234
    dtype: timedelta64[ms]
    >>> series.dt.components
        days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
    0    141     13       35       12           123             0            0
    1     14      6        0       31           231             0            0
    2  13000     10       12       48           712             0            0
    3      0      0       35       35           656             0            0
    4     37     13       12       14           234             0            0
    >>> series.dt.days
    0      141
    1       14
    2    13000
    3        0
    4       37
    dtype: int64
    >>> series.dt.seconds
    0    48912
    1    21631
    2    36768
    3     2135
    4    47534
    dtype: int64
    >>> series.dt.microseconds
    0    123000
    1    231000
    2    712000
    3    656000
    4    234000
    dtype: int64
    >>> s.dt.nanoseconds
    0    0
    1    0
    2    0
    3    0
    4    0
    dtype: int64
    """

    def __init__(self, series):
        self.series = series

    @property
    def days(self):
        """
        Number of days.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([12231312123, 1231231231, 1123236768712, 2135656,
        ...     3244334234], dtype='timedelta64[ms]')
        >>> s
        0      141 days 13:35:12.123
        1       14 days 06:00:31.231
        2    13000 days 10:12:48.712
        3        0 days 00:35:35.656
        4       37 days 13:12:14.234
        dtype: timedelta64[ms]
        >>> s.dt.days
        0      141
        1       14
        2    13000
        3        0
        4       37
        dtype: int64
        """
        return self._get_td_field("days")

    @property
    def seconds(self):
        """
        Number of seconds (>= 0 and less than 1 day).

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([12231312123, 1231231231, 1123236768712, 2135656,
        ...     3244334234], dtype='timedelta64[ms]')
        >>> s
        0      141 days 13:35:12.123
        1       14 days 06:00:31.231
        2    13000 days 10:12:48.712
        3        0 days 00:35:35.656
        4       37 days 13:12:14.234
        dtype: timedelta64[ms]
        >>> s.dt.seconds
        0    48912
        1    21631
        2    36768
        3     2135
        4    47534
        dtype: int64
        >>> s.dt.microseconds
        0    123000
        1    231000
        2    712000
        3    656000
        4    234000
        dtype: int64
        """
        return self._get_td_field("seconds")

    @property
    def microseconds(self):
        """
        Number of microseconds (>= 0 and less than 1 second).

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([12231312123, 1231231231, 1123236768712, 2135656,
        ...     3244334234], dtype='timedelta64[ms]')
        >>> s
        0      141 days 13:35:12.123
        1       14 days 06:00:31.231
        2    13000 days 10:12:48.712
        3        0 days 00:35:35.656
        4       37 days 13:12:14.234
        dtype: timedelta64[ms]
        >>> s.dt.microseconds
        0    123000
        1    231000
        2    712000
        3    656000
        4    234000
        dtype: int64
        """
        return self._get_td_field("microseconds")

    @property
    def nanoseconds(self):
        """
        Return the number of nanoseconds (n), where 0 <= n < 1 microsecond.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([12231312123, 1231231231, 1123236768712, 2135656,
        ...     3244334234], dtype='timedelta64[ns]')
        >>> s
        0    00:00:12.231312123
        1    00:00:01.231231231
        2    00:18:43.236768712
        3    00:00:00.002135656
        4    00:00:03.244334234
        dtype: timedelta64[ns]
        >>> s.dt.nanoseconds
        0    123
        1    231
        2    712
        3    656
        4    234
        dtype: int64
        """
        return self._get_td_field("nanoseconds")

    @property
    def components(self):
        """
        Return a Dataframe of the components of the Timedeltas.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> s = cudf.Series([12231312123, 1231231231, 1123236768712, 2135656, 3244334234], dtype='timedelta64[ms]')
        >>> s
        0      141 days 13:35:12.123
        1       14 days 06:00:31.231
        2    13000 days 10:12:48.712
        3        0 days 00:35:35.656
        4       37 days 13:12:14.234
        dtype: timedelta64[ms]
        >>> s.dt.components
            days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0    141     13       35       12           123             0            0
        1     14      6        0       31           231             0            0
        2  13000     10       12       48           712             0            0
        3      0      0       35       35           656             0            0
        4     37     13       12       14           234             0            0
        """  # noqa: E501
        return self.series._column.components(index=self.series._index)

    def _get_td_field(self, field):
        out_column = getattr(self.series._column, field)
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
    new_index_names = [None] * head.nlevels
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

    Two values in ``a`` and ``b`` are  considered equal when the following
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
    3         <NA>
    4          9.9
    5          1.0
    dtype: float64
    >>> s2
    0    1.987654321
    1    2.987654321
    2    3.987654321
    3           <NA>
    4           19.9
    5           <NA>
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

    index = None

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
        a_nulls = a_col.isnull()
        b_nulls = b_col.isnull()
        null_values = a_nulls | b_nulls

        if equal_nan is True:
            equal_nulls = a_nulls & b_nulls

        del a_nulls, b_nulls
    elif a_col.null_count:
        null_values = a_col.isnull()
    elif b_col.null_count:
        null_values = b_col.isnull()
    else:
        return Series(result_col, index=index)

    result_col[null_values] = False
    if equal_nan is True and a_col.null_count and b_col.null_count:
        result_col[equal_nulls] = True

    return Series(result_col, index=index)
