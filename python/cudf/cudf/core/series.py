# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from __future__ import annotations

import functools
import inspect
import pickle
import textwrap
import warnings
from collections import abc
from shutil import get_terminal_size
from typing import TYPE_CHECKING, Any, Literal

import cupy
import numpy as np
import pandas as pd
from typing_extensions import Self, assert_never

import cudf
from cudf import _lib as libcudf
from cudf.api.extensions import no_default
from cudf.api.types import (
    _is_non_decimal_numeric_dtype,
    _is_scalar_or_zero_d_array,
    is_dict_like,
    is_integer,
    is_scalar,
)
from cudf.core import indexing_utils
from cudf.core._compat import PANDAS_LT_300
from cudf.core.abc import Serializable
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column import (
    ColumnBase,
    DatetimeColumn,
    IntervalColumn,
    TimeDeltaColumn,
    as_column,
)
from cudf.core.column.categorical import (
    _DEFAULT_CATEGORICAL_VALUE,
    CategoricalAccessor as CategoricalAccessor,
    CategoricalColumn,
)
from cudf.core.column.column import concat_columns
from cudf.core.column.lists import ListMethods
from cudf.core.column.string import StringMethods
from cudf.core.column.struct import StructMethods
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.groupby.groupby import SeriesGroupBy, groupby_doc_template
from cudf.core.index import BaseIndex, DatetimeIndex, RangeIndex, ensure_index
from cudf.core.indexed_frame import (
    IndexedFrame,
    _FrameIndexer,
    _get_label_range_or_mask,
    _indices_from_labels,
    doc_reset_index_template,
)
from cudf.core.resample import SeriesResampler
from cudf.core.single_column_frame import SingleColumnFrame
from cudf.core.udf.scalar_function import _get_scalar_kernel
from cudf.errors import MixedTypeError
from cudf.utils import docutils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    can_convert_to_column,
    find_common_type,
    is_mixed_with_object_dtype,
    to_cudf_compatible_scalar,
)
from cudf.utils.performance_tracking import _performance_tracking

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    import pyarrow as pa

    from cudf._typing import (
        ColumnLike,
        DataFrameOrSeries,
        NotImplementedType,
        ScalarLike,
    )


def _format_percentile_names(percentiles):
    return [f"{int(x * 100)}%" for x in percentiles]


def _describe_numeric(obj, percentiles):
    # Helper for Series.describe with numerical data.
    data = {
        "count": obj.count(),
        "mean": obj.mean(),
        "std": obj.std(),
        "min": obj.min(),
        **dict(
            zip(
                _format_percentile_names(percentiles),
                obj.quantile(percentiles).to_numpy(na_value=np.nan).tolist(),
            )
        ),
        "max": obj.max(),
    }
    return {k: round(v, 6) for k, v in data.items()}


def _describe_timetype(obj, percentiles, typ):
    # Common helper for Series.describe with timedelta/timestamp data.
    data = {
        "count": str(obj.count()),
        "mean": str(typ(obj.mean())),
        "std": "",
        "min": str(typ(obj.min())),
        **dict(
            zip(
                _format_percentile_names(percentiles),
                obj.quantile(percentiles)
                .astype("str")
                .to_numpy(na_value=np.nan)
                .tolist(),
            )
        ),
        "max": str(typ(obj.max())),
    }

    if typ is pd.Timedelta:
        data["std"] = str(obj.std())
    else:
        data.pop("std")
    return data


def _describe_timedelta(obj, percentiles):
    # Helper for Series.describe with timedelta data.
    return _describe_timetype(obj, percentiles, pd.Timedelta)


def _describe_timestamp(obj, percentiles):
    # Helper for Series.describe with timestamp data.
    return _describe_timetype(obj, percentiles, pd.Timestamp)


def _describe_categorical(obj, percentiles):
    # Helper for Series.describe with categorical data.
    data = {
        "count": obj.count(),
        "unique": len(obj.unique()),
        "top": None,
        "freq": None,
    }
    if data["count"] > 0:
        # In case there's a tie, break the tie by sorting the index
        # and take the top.
        val_counts = obj.value_counts(ascending=False)
        tied_val_counts = val_counts[
            val_counts == val_counts.iloc[0]
        ].sort_index()
        data.update(
            {
                "top": tied_val_counts.index[0],
                "freq": tied_val_counts.iloc[0],
            }
        )
    return data


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

    _frame: cudf.Series

    @_performance_tracking
    def __getitem__(self, arg):
        indexing_spec = indexing_utils.parse_row_iloc_indexer(
            indexing_utils.destructure_series_iloc_indexer(arg, self._frame),
            len(self._frame),
        )
        return self._frame._getitem_preprocessed(indexing_spec)

    @_performance_tracking
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key = list(key)

        # coerce value into a scalar or column
        if is_scalar(value):
            value = to_cudf_compatible_scalar(value)
            if (
                self._frame.dtype.kind not in "mM"
                and cudf.utils.utils._isnat(value)
                and not (
                    self._frame.dtype == "object" and isinstance(value, str)
                )
            ):
                raise MixedTypeError(
                    f"Cannot assign {value=} to non-datetime/non-timedelta "
                    "columns"
                )
            elif (
                not (
                    self._frame.dtype.kind == "f"
                    or (
                        isinstance(self._frame.dtype, cudf.CategoricalDtype)
                        and self._frame.dtype.categories.dtype.kind == "f"
                    )
                )
                and isinstance(value, np.floating)
                and np.isnan(value)
            ):
                raise MixedTypeError(
                    f"Cannot assign {value=} to "
                    f"non-float dtype={self._frame.dtype}"
                )
            elif self._frame.dtype.kind == "b" and not (
                value in {None, cudf.NA}
                or isinstance(value, (np.bool_, bool))
                or (isinstance(value, cudf.Scalar) and value.dtype.kind == "b")
            ):
                raise MixedTypeError(
                    f"Cannot assign {value=} to "
                    f"bool dtype={self._frame.dtype}"
                )
        elif not (
            isinstance(value, (list, dict))
            and isinstance(
                self._frame.dtype, (cudf.ListDtype, cudf.StructDtype)
            )
        ):
            value = as_column(value)

        if (
            (self._frame.dtype.kind in "uifb" or self._frame.dtype == "object")
            and hasattr(value, "dtype")
            and value.dtype.kind in "uifb"
        ):
            # normalize types if necessary:
            # In contrast to Column.__setitem__ (which downcasts the value to
            # the dtype of the column) here we upcast the series to the
            # larger data type mimicking pandas
            to_dtype = np.result_type(value.dtype, self._frame.dtype)
            value = value.astype(to_dtype)
            if to_dtype != self._frame.dtype:
                # Do not remove until pandas-3.0 support is added.
                assert (
                    PANDAS_LT_300
                ), "Need to drop after pandas-3.0 support is added."
                warnings.warn(
                    f"Setting an item of incompatible dtype is deprecated "
                    "and will raise in a future error of pandas. "
                    f"Value '{value}' has dtype incompatible with "
                    f"{self._frame.dtype}, "
                    "please explicitly cast to a compatible dtype first.",
                    FutureWarning,
                )
                self._frame._column._mimic_inplace(
                    self._frame._column.astype(to_dtype), inplace=True
                )

        self._frame._column[key] = value


class _SeriesLocIndexer(_FrameIndexer):
    """
    Label-based selection
    """

    @_performance_tracking
    def __getitem__(self, arg: Any) -> ScalarLike | DataFrameOrSeries:
        if isinstance(arg, pd.MultiIndex):
            arg = cudf.from_pandas(arg)

        if isinstance(self._frame.index, cudf.MultiIndex) and not isinstance(
            arg, cudf.MultiIndex
        ):
            if is_scalar(arg):
                row_arg = (arg,)
            else:
                row_arg = arg
            result = self._frame.index._get_row_major(self._frame, row_arg)
            if (
                isinstance(arg, tuple)
                and len(arg) == self._frame.index.nlevels
                and not any(isinstance(x, slice) for x in arg)
            ):
                result = result.iloc[0]
            return result
        try:
            arg = self._loc_to_iloc(arg)
        except (TypeError, KeyError, IndexError, ValueError) as err:
            raise KeyError(arg) from err

        return self._frame.iloc[arg]

    @_performance_tracking
    def __setitem__(self, key, value):
        try:
            key = self._loc_to_iloc(key)
        except KeyError as e:
            if (
                is_scalar(key)
                and not isinstance(self._frame.index, cudf.MultiIndex)
                and is_scalar(value)
            ):
                idx = self._frame.index
                if isinstance(idx, cudf.RangeIndex):
                    if isinstance(key, int) and (key == idx[-1] + idx.step):
                        idx_copy = cudf.RangeIndex(
                            start=idx.start,
                            stop=idx.stop + idx.step,
                            step=idx.step,
                            name=idx.name,
                        )
                    else:
                        idx_copy = idx._as_int_index()
                        _append_new_row_inplace(idx_copy._column, key)
                else:
                    # TODO: Modifying index in place is bad because
                    # our index are immutable, but columns are not (which
                    # means our index are mutable with internal APIs).
                    # Get rid of the deep copy once columns too are
                    # immutable.
                    idx_copy = idx.copy(deep=True)
                    _append_new_row_inplace(idx_copy._column, key)

                self._frame._index = idx_copy
                _append_new_row_inplace(self._frame._column, value)
                return
            else:
                raise e
        if isinstance(value, (pd.Series, cudf.Series)):
            value = cudf.Series(value)
            value = value._align_to_index(self._frame.index, how="right")
        self._frame.iloc[key] = value

    def _loc_to_iloc(self, arg):
        if isinstance(arg, tuple) and arg and isinstance(arg[0], slice):
            if len(arg) > 1:
                raise IndexError("Too many Indexers")
            arg = arg[0]
        if _is_scalar_or_zero_d_array(arg):
            index_dtype = self._frame.index.dtype
            warn_msg = (
                "Series.__getitem__ treating keys as positions is deprecated. "
                "In a future version, integer keys will always be treated "
                "as labels (consistent with DataFrame behavior). To access "
                "a value by position, use `ser.iloc[pos]`"
            )
            if not _is_non_decimal_numeric_dtype(index_dtype) and not (
                isinstance(index_dtype, cudf.CategoricalDtype)
                and index_dtype.categories.dtype.kind in "iu"
            ):
                # TODO: switch to cudf.utils.dtypes.is_integer(arg)
                if isinstance(arg, cudf.Scalar) and arg.dtype.kind in "iu":
                    # Do not remove until pandas 3.0 support is added.
                    assert (
                        PANDAS_LT_300
                    ), "Need to drop after pandas-3.0 support is added."
                    warnings.warn(warn_msg, FutureWarning)
                    return arg.value
                elif is_integer(arg):
                    # Do not remove until pandas 3.0 support is added.
                    assert (
                        PANDAS_LT_300
                    ), "Need to drop after pandas-3.0 support is added."
                    warnings.warn(warn_msg, FutureWarning)
                    return arg
            try:
                if isinstance(self._frame.index, RangeIndex):
                    indices = self._frame.index._indices_of(arg)
                else:
                    indices = self._frame.index._column.indices_of(arg)
                if (n := len(indices)) == 0:
                    raise KeyError("Label scalar is out of bounds")
                elif n == 1:
                    return indices.element_indexing(0)
                else:
                    return indices
            except (TypeError, KeyError, IndexError, ValueError):
                raise KeyError("Label scalar is out of bounds")

        elif isinstance(arg, slice):
            return _get_label_range_or_mask(
                self._frame.index, arg.start, arg.stop, arg.step
            )
        elif isinstance(arg, (cudf.MultiIndex, pd.MultiIndex)):
            if isinstance(arg, pd.MultiIndex):
                arg = cudf.MultiIndex.from_pandas(arg)

            return _indices_from_labels(self._frame, arg)

        else:
            arg = cudf.core.series.Series._from_column(
                cudf.core.column.as_column(arg)
            )
            if arg.dtype.kind == "b":
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
    values based on their associated index values, they need
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
        default to RangeIndex (0, 1, 2, ..., n) if not provided.
        If both a dict and index sequence are used, the index will
        override the keys found in the dict.

    dtype : str, :class:`numpy.dtype`, or ExtensionDtype, optional
        Data type for the output Series. If not specified,
        this will be inferred from data.

    name : str, optional
        The name to give to the Series.

    copy : bool, default False
        Copy input data. Only affects Series or 1d ndarray input.

    nan_as_null : bool, Default True
        If ``None``/``True``, converts ``np.nan`` values to
        ``null`` values.
        If ``False``, leaves ``np.nan`` values as is.
    """

    _accessors: set[Any] = set()
    _loc_indexer_type = _SeriesLocIndexer
    _iloc_indexer_type = _SeriesIlocIndexer
    _groupby = SeriesGroupBy
    _resampler = SeriesResampler

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
    @_performance_tracking
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
        col = as_column(categorical)
        if codes is not None:
            codes = as_column(codes)

            valid_codes = codes != codes.dtype.type(_DEFAULT_CATEGORICAL_VALUE)

            mask = None
            if not valid_codes.all():
                mask = libcudf.transform.bools_to_mask(valid_codes)
            col = CategoricalColumn(
                data=col.data,
                size=codes.size,
                dtype=col.dtype,
                mask=mask,
                children=(codes,),
            )
        return Series._from_column(col)

    @classmethod
    @_performance_tracking
    def from_arrow(cls, array: pa.Array) -> Self:
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
        >>> cudf.Series.from_arrow(pa.array(["a", "b", None]))
        0       a
        1       b
        2    <NA>
        dtype: object
        """
        return cls._from_column(ColumnBase.from_arrow(array))

    @classmethod
    @_performance_tracking
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
        return cls._from_column(as_column(data).set_mask(mask))

    @_performance_tracking
    def __init__(
        self,
        data=None,
        index=None,
        dtype=None,
        name=None,
        copy=False,
        nan_as_null=no_default,
    ):
        if nan_as_null is no_default:
            nan_as_null = not cudf.get_option("mode.pandas_compatible")
        index_from_data = None
        name_from_data = None
        if data is None:
            data = {}

        if isinstance(data, (pd.Series, pd.Index, BaseIndex, Series)):
            if copy and not isinstance(data, (pd.Series, pd.Index)):
                data = data.copy(deep=True)
            name_from_data = data.name
            column = as_column(data, nan_as_null=nan_as_null, dtype=dtype)
            if isinstance(data, (pd.Series, Series)):
                index_from_data = ensure_index(data.index)
        elif isinstance(data, ColumnAccessor):
            raise TypeError(
                "Use cudf.Series._from_data for constructing a Series from "
                "ColumnAccessor"
            )
        elif isinstance(data, ColumnBase):
            raise TypeError(
                "Use cudf.Series._from_column for constructing a Series from "
                "a ColumnBase"
            )
        elif isinstance(data, dict):
            if not data:
                column = as_column(data, nan_as_null=nan_as_null, dtype=dtype)
                index_from_data = RangeIndex(0)
            else:
                column = as_column(
                    list(data.values()), nan_as_null=nan_as_null, dtype=dtype
                )
                index_from_data = cudf.Index(list(data.keys()))
        else:
            # Using `getattr_static` to check if
            # `data` is on device memory and perform
            # a deep copy later. This is different
            # from `hasattr` because, it doesn't
            # invoke the property we are looking
            # for and the latter actually invokes
            # the property, which in this case could
            # be expensive or mark a buffer as
            # unspillable.
            has_cai = (
                type(
                    inspect.getattr_static(
                        data, "__cuda_array_interface__", None
                    )
                )
                is property
            )
            column = as_column(
                data,
                nan_as_null=nan_as_null,
                dtype=dtype,
                length=len(index) if index is not None else None,
            )
            if copy and has_cai:
                column = column.copy(deep=True)

        assert isinstance(column, ColumnBase)

        if dtype is not None:
            column = column.astype(dtype)

        if name_from_data is not None and name is None:
            name = name_from_data

        if index is not None:
            index = ensure_index(index)

        if index_from_data is not None:
            first_index = index_from_data
            second_index = index
        elif index is None:
            first_index = RangeIndex(len(column))
            second_index = None
        else:
            first_index = index
            second_index = None

        super().__init__({name: column}, index=first_index)
        if second_index is not None:
            # TODO: This there a better way to do this?
            reindexed = self.reindex(index=second_index, copy=False)
            self._data = reindexed._data
            self._index = second_index
        self._check_data_index_length_match()

    @classmethod
    @_performance_tracking
    def _from_column(
        cls,
        column: ColumnBase,
        *,
        name: abc.Hashable = None,
        index: BaseIndex | None = None,
    ) -> Self:
        ca = ColumnAccessor({name: column}, verify=False)
        return cls._from_data(ca, index=index)

    @classmethod
    @_performance_tracking
    def _from_data(
        cls,
        data: MutableMapping,
        index: BaseIndex | None = None,
        name: Any = no_default,
    ) -> Series:
        out = super()._from_data(data=data, index=index)
        if name is not no_default:
            out.name = name
        return out

    @_performance_tracking
    def _from_data_like_self(self, data: MutableMapping):
        out = super()._from_data_like_self(data)
        out.name = self.name
        return out

    @_performance_tracking
    def __contains__(self, item):
        return item in self.index

    @classmethod
    @_performance_tracking
    def from_pandas(cls, s: pd.Series, nan_as_null=no_default):
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
        >>> pds = pd.Series(data, dtype='float64')
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
        if nan_as_null is no_default:
            nan_as_null = (
                False if cudf.get_option("mode.pandas_compatible") else None
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result = cls(s, nan_as_null=nan_as_null)
        return result

    @property  # type: ignore
    @_performance_tracking
    def is_unique(self):
        """Return boolean if values in the object are unique.

        Returns
        -------
        bool
        """
        return self._column.is_unique

    @property  # type: ignore
    @_performance_tracking
    def dt(self):
        """
        Accessor object for datetime-like properties of the Series values.

        Examples
        --------
        >>> s = cudf.Series(cudf.date_range(
        ...   start='2001-02-03 12:00:00',
        ...   end='2001-02-03 14:00:00',
        ...   freq='1H'))
        >>> s.dt.hour
        0    12
        1    13
        2    14
        dtype: int16
        >>> s.dt.second
        0    0
        1    0
        2    0
        dtype: int16
        >>> s.dt.day
        0    3
        1    3
        2    3
        dtype: int16

        Returns
        -------
            A Series indexed like the original Series.

        Raises
        ------
            TypeError if the Series does not contain datetimelike values.
        """
        if self.dtype.kind == "M":
            return DatetimeProperties(self)
        elif self.dtype.kind == "m":
            return TimedeltaProperties(self)
        else:
            raise AttributeError(
                "Can only use .dt accessor with datetimelike values"
            )

    @property  # type: ignore
    @_performance_tracking
    def hasnans(self):
        """
        Return True if there are any NaNs or nulls.

        Returns
        -------
        out : bool
            If Series has at least one NaN or null value, return True,
            if not return False.

        Examples
        --------
        >>> import cudf
        >>> import numpy as np
        >>> series = cudf.Series([1, 2, np.nan, 3, 4], nan_as_null=False)
        >>> series
        0    1.0
        1    2.0
        2    NaN
        3    3.0
        4    4.0
        dtype: float64
        >>> series.hasnans
        True

        `hasnans` returns `True` for the presence of any `NA` values:

        >>> series = cudf.Series([1, 2, 3, None, 4])
        >>> series
        0       1
        1       2
        2       3
        3    <NA>
        4       4
        dtype: int64
        >>> series.hasnans
        True
        """
        return self._column.has_nulls(include_nan=True)

    @_performance_tracking
    def serialize(self):
        header, frames = super().serialize()

        header["index"], index_frames = self.index.serialize()
        header["index_frame_count"] = len(index_frames)
        # For backwards compatibility with older versions of cuDF, index
        # columns are placed before data columns.
        frames = index_frames + frames

        return header, frames

    @classmethod
    @_performance_tracking
    def deserialize(cls, header, frames):
        index_nframes = header["index_frame_count"]
        obj = super().deserialize(
            header, frames[header["index_frame_count"] :]
        )

        idx_typ = pickle.loads(header["index"]["type-serialized"])
        index = idx_typ.deserialize(header["index"], frames[:index_nframes])
        obj.index = index

        return obj

    @_performance_tracking
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
        if axis == 1:
            raise ValueError("No axis named 1 for object type Series")
        # Ignore columns for Series
        if columns is not None:
            columns = []
        return super().drop(
            labels, axis, index, columns, level, inplace, errors
        )

    def tolist(self):  # noqa: D102
        raise TypeError(
            "cuDF does not support conversion to host memory "
            "via the `tolist()` method. Consider using "
            "`.to_arrow().to_pylist()` to construct a Python list."
        )

    to_list = tolist

    @_performance_tracking
    def to_dict(self, into: type[dict] = dict) -> dict:
        """
        Convert Series to {label -> value} dict or dict-like object.

        Parameters
        ----------
        into : class, default dict
            The collections.abc.Mapping subclass to use as the return
            object. Can be the actual class or an empty
            instance of the mapping type you want.  If you want a
            collections.defaultdict, you must pass it initialized.

        Returns
        -------
        collections.abc.Mapping
            Key-value representation of Series.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64
        >>> s.to_dict()
        {0: 1, 1: 2, 2: 3, 3: 4}
        >>> from collections import OrderedDict, defaultdict
        >>> s.to_dict(OrderedDict)  # doctest: +SKIP
        OrderedDict([(0, 1), (1, 2), (2, 3), (3, 4)])
        >>> dd = defaultdict(list)
        >>> s.to_dict(dd)
        defaultdict(<class 'list'>, {0: 1, 1: 2, 2: 3, 3: 4})
        """
        return self.to_pandas().to_dict(into=into)

    @_performance_tracking
    def reindex(
        self,
        index=None,
        *,
        axis=None,
        method: str | None = None,
        copy: bool = True,
        level=None,
        fill_value: ScalarLike | None = None,
        limit: int | None = None,
        tolerance=None,
    ):
        """
        Conform Series to new index.

        Parameters
        ----------
        index : Index, Series-convertible, default None
            New labels / index to conform to,
            should be specified using keywords.
        axis: int, default None
            Unused.
        method: Not Supported
        copy : boolean, default True
        level: Not Supported
        fill_value : Value to use for missing values.
            Defaults to ``NA``, but can be any "compatible" value.
        limit: Not Supported
        tolerance: Not Supported

        Returns
        -------
        Series with changed index.

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

        .. pandas-compat::
            :meth:`pandas.Series.reindex`

            Note: One difference from Pandas is that ``NA`` is used for rows
            that do not match, rather than ``NaN``. One side effect of this is
            that the series retains an integer dtype in cuDF
            where it is cast to float in Pandas.

        """
        if index is None:
            index = self.index
        if fill_value is None:
            fill_value = cudf.NA

        name = self.name or 0
        series = self._reindex(
            deep=copy,
            dtypes={name: self.dtype},
            index=index,
            column_names=[name],
            inplace=False,
            fill_value=fill_value,
            level=level,
            method=method,
            limit=limit,
            tolerance=tolerance,
        )
        series.name = self.name
        return series

    @_performance_tracking
    @docutils.doc_apply(
        doc_reset_index_template.format(
            klass="Series",
            argument="""
        name : object, optional
            The name to use for the column containing the original Series
            values. Uses self.name by default. This argument is ignored when
            ``drop`` is True.""",
            return_type="Series or DataFrame or None",
            return_doc=""" For Series, When drop is False (the default), a DataFrame
            is returned. The newly created columns will come first in the
            DataFrame, followed by the original Series values. When `drop` is
            True, a `Series` is returned. In either case, if ``inplace=True``,
            no value is returned.
""",  # noqa: E501
            example="""
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

        You can also use ``reset_index`` with MultiIndex.

        >>> s2 = cudf.Series(
        ...             range(4), name='foo',
        ...             index=cudf.MultiIndex.from_tuples([
        ...                     ('bar', 'one'), ('bar', 'two'),
        ...                     ('baz', 'one'), ('baz', 'two')],
        ...                     names=['a', 'b']
        ...      ))
        >>> s2
        a    b
        bar  one    0
             two    1
        baz  one    2
             two    3
        Name: foo, dtype: int64
        >>> s2.reset_index(level='a')
               a  foo
        b
        one  bar    0
        two  bar    1
        one  baz    2
        two  baz    3
""",
        )
    )
    def reset_index(
        self,
        level=None,
        drop=False,
        name=no_default,
        inplace=False,
        allow_duplicates=False,
    ):
        if not drop and inplace:
            raise TypeError(
                "Cannot reset_index inplace on a Series "
                "to create a DataFrame"
            )
        data, index = self._reset_index(
            level=level, drop=drop, allow_duplicates=allow_duplicates
        )
        if not drop:
            if name is no_default:
                name = 0 if self.name is None else self.name
            data[name] = data.pop(self.name)
            return self._constructor_expanddim._from_data(data, index)
        # For ``name`` behavior, see:
        # https://github.com/pandas-dev/pandas/issues/44575
        # ``name`` has to be ignored when `drop=True`
        return self._mimic_inplace(
            Series._from_data(data, index, self.name),
            inplace=inplace,
        )

    @_performance_tracking
    def to_frame(self, name: abc.Hashable = no_default) -> cudf.DataFrame:
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
        return self._to_frame(name=name, index=self.index)

    @_performance_tracking
    def memory_usage(self, index=True, deep=False):
        return self._column.memory_usage + (
            self.index.memory_usage() if index else 0
        )

    @_performance_tracking
    def __array_function__(self, func, types, args, kwargs):
        if "out" in kwargs or not all(issubclass(t, Series) for t in types):
            return NotImplemented

        try:
            # Apply a Series method if one exists.
            if cudf_func := getattr(Series, func.__name__, None):
                result = cudf_func(*args, **kwargs)
                if func.__name__ == "unique":
                    # NumPy expects a sorted result for `unique`, which is not
                    # guaranteed by cudf.Series.unique.
                    result = result.sort_values()
                return result

            # Assume that cupy subpackages match numpy and search the
            # corresponding cupy submodule based on the func's __module__.
            numpy_submodule = func.__module__.split(".")[1:]
            cupy_func = cupy
            for name in (*numpy_submodule, func.__name__):
                cupy_func = getattr(cupy_func, name, None)

            # Handle case if cupy does not implement the function or just
            # aliases the numpy function.
            if not cupy_func or cupy_func is func:
                return NotImplemented

            # For now just fail on cases with mismatched indices. There is
            # almost certainly no general solution for all array functions.
            index = args[0].index
            if not all(s.index.equals(index) for s in args):
                return NotImplemented
            out = cupy_func(*(s.values for s in args), **kwargs)

            # Return (host) scalar values immediately.
            if not isinstance(out, cupy.ndarray):
                return out

            # 0D array (scalar)
            if out.ndim == 0:
                return to_cudf_compatible_scalar(out)
            # 1D array
            elif (
                # Only allow 1D arrays
                ((out.ndim == 1) or (out.ndim == 2 and out.shape[1] == 1))
                # If we have an index, it must be the same length as the
                # output for cupy dispatching to be well-defined.
                and len(index) == len(out)
            ):
                return Series(out, index=index)
        except Exception:
            # The rare instance where a "silent" failure is preferable. Except
            # in the (highly unlikely) case that some other library
            # interoperates with cudf objects, the result will be that numpy
            # raises a TypeError indicating that the operation is not
            # implemented, which is much friendlier than an arbitrary internal
            # cudf error.
            pass

        return NotImplemented

    @_performance_tracking
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

        .. pandas-compat::
            :meth:`pandas.Series.map`

            Please note map currently only supports fixed-width numeric
            type functions.
        """
        if isinstance(arg, dict):
            if hasattr(arg, "__missing__"):
                raise NotImplementedError(
                    "default values in dicts are currently not supported."
                )
            lhs = cudf.DataFrame(
                {"x": self, "orig_order": as_column(range(len(self)))}
            )
            rhs = cudf.DataFrame(
                {
                    "x": arg.keys(),
                    "s": arg.values(),
                    "bool": as_column(True, length=len(arg), dtype=self.dtype),
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
            lhs = cudf.DataFrame(
                {"x": self, "orig_order": as_column(range(len(self)))}
            )
            rhs = cudf.DataFrame(
                {
                    "x": arg.keys(),
                    "s": arg,
                    "bool": as_column(True, length=len(arg), dtype=self.dtype),
                }
            )
            res = lhs.merge(rhs, on="x", how="left").sort_values(
                by="orig_order"
            )
            result = res["s"]
            result.name = self.name
            result.index = self.index
        else:
            result = self.apply(arg)
        return result

    def _getitem_preprocessed(
        self,
        spec: indexing_utils.IndexingSpec,
    ) -> Self | ScalarLike:
        """Get subset of entries given structured data

        Parameters
        ----------
        spec
            Indexing specification

        Returns
        -------
        Subsetted Series or else scalar (if a scalar entry is
        requested)

        Notes
        -----
        This function performs no bounds-checking or massaging of the
        inputs.
        """
        if isinstance(spec, indexing_utils.MapIndexer):
            return self._gather(spec.key, keep_index=True)
        elif isinstance(spec, indexing_utils.MaskIndexer):
            return self._apply_boolean_mask(spec.key, keep_index=True)
        elif isinstance(spec, indexing_utils.SliceIndexer):
            return self._slice(spec.key)
        elif isinstance(spec, indexing_utils.ScalarIndexer):
            return self._gather(
                spec.key, keep_index=False
            )._column.element_indexing(0)
        elif isinstance(spec, indexing_utils.EmptyIndexer):
            return self._empty_like(keep_index=True)
        assert_never(spec)

    @_performance_tracking
    def __getitem__(self, arg):
        if isinstance(arg, slice):
            return self.iloc[arg]
        else:
            return self.loc[arg]

    iteritems = SingleColumnFrame.__iter__

    items = SingleColumnFrame.__iter__

    @_performance_tracking
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self.iloc[key] = value
        else:
            self.loc[key] = value

    def __repr__(self):
        _, height = get_terminal_size()
        max_rows = (
            height
            if pd.get_option("display.max_rows") == 0
            else pd.get_option("display.max_rows")
        )
        if max_rows not in (0, None) and len(self) > max_rows:
            top = self.head(int(max_rows / 2 + 1))
            bottom = self.tail(int(max_rows / 2 + 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                preprocess = cudf.concat([top, bottom])
        else:
            preprocess = self.copy()
        preprocess.index = preprocess.index._clean_nulls_from_index()
        if (
            preprocess.nullable
            and not isinstance(
                preprocess.dtype,
                (
                    cudf.CategoricalDtype,
                    cudf.ListDtype,
                    cudf.StructDtype,
                    cudf.core.dtypes.DecimalDtype,
                ),
            )
        ) or preprocess.dtype.kind == "m":
            fill_value = (
                str(cudf.NaT)
                if preprocess.dtype.kind in "mM"
                else str(cudf.NA)
            )
            output = repr(
                preprocess.astype("str").fillna(fill_value).to_pandas()
            )
        elif isinstance(preprocess.dtype, cudf.CategoricalDtype):
            min_rows = (
                height
                if pd.get_option("display.min_rows") == 0
                else pd.get_option("display.min_rows")
            )
            show_dimensions = pd.get_option("display.show_dimensions")
            if preprocess.dtype.categories.dtype.kind == "f":
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
                pd_series = preprocess.to_pandas()
            output = pd_series.to_string(
                name=self.name,
                dtype=self.dtype,
                min_rows=min_rows,
                max_rows=max_rows,
                length=show_dimensions,
                na_rep=str(cudf.NA),
            )
        else:
            output = repr(preprocess.to_pandas())

        lines = output.split("\n")
        if isinstance(preprocess.dtype, cudf.CategoricalDtype):
            category_memory = lines[-1]
            if preprocess.dtype.categories.dtype.kind == "f":
                category_memory = category_memory.replace("'", "").split(": ")
                category_memory = (
                    category_memory[0].replace(
                        "object", preprocess.dtype.categories.dtype.name
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

    def _make_operands_and_index_for_binop(
        self,
        other: Any,
        fn: str,
        fill_value: Any = None,
        reflect: bool = False,
        can_reindex: bool = False,
    ) -> tuple[
        dict[str | None, tuple[ColumnBase, Any, bool, Any]]
        | NotImplementedType,
        BaseIndex | None,
        bool,
    ]:
        # Specialize binops to align indices.
        if isinstance(other, Series):
            if (
                not can_reindex
                and fn in cudf.utils.utils._EQUALITY_OPS
                and not self.index.equals(other.index)
            ):
                raise ValueError(
                    "Can only compare identically-labeled Series objects"
                )
            lhs, other = _align_indices([self, other], allow_non_unique=True)
        else:
            lhs = self

        try:
            can_use_self_column_name = cudf.utils.utils._is_same_name(
                self.name, other.name
            )
        except AttributeError:
            can_use_self_column_name = False

        operands = lhs._make_operands_for_binop(other, fill_value, reflect)
        return operands, lhs.index, can_use_self_column_name

    @copy_docstring(CategoricalAccessor)  # type: ignore
    @property
    @_performance_tracking
    def cat(self):
        return CategoricalAccessor(parent=self)

    @copy_docstring(StringMethods)  # type: ignore
    @property
    @_performance_tracking
    def str(self):
        return StringMethods(parent=self)

    @copy_docstring(ListMethods)  # type: ignore
    @property
    @_performance_tracking
    def list(self):
        return ListMethods(parent=self)

    @copy_docstring(StructMethods)  # type: ignore
    @property
    @_performance_tracking
    def struct(self):
        return StructMethods(parent=self)

    @property  # type: ignore
    @_performance_tracking
    def dtype(self):
        """The dtype of the Series."""
        return self._column.dtype

    @classmethod
    @_performance_tracking
    def _concat(cls, objs, axis=0, index: bool = True):
        # Concatenate index if not provided
        if index is True:
            if isinstance(objs[0].index, cudf.MultiIndex):
                result_index = cudf.MultiIndex._concat([o.index for o in objs])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    result_index = cudf.core.index.Index._concat(
                        [o.index for o in objs]
                    )
        elif index is False:
            result_index = None
        else:
            raise ValueError(f"{index=} must be a bool")

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

        if len(objs):
            col = col._with_type_metadata(objs[0].dtype)

        return cls._from_column(col, name=name, index=result_index)

    @property  # type: ignore
    @_performance_tracking
    def valid_count(self):
        """Number of non-null values"""
        return len(self) - self._column.null_count

    @property  # type: ignore
    @_performance_tracking
    def null_count(self):
        """Number of null values"""
        return self._column.null_count

    @property  # type: ignore
    @_performance_tracking
    def nullable(self):
        """A boolean indicating whether a null-mask is needed"""
        return self._column.nullable

    @property  # type: ignore
    @_performance_tracking
    def has_nulls(self):
        """
        Indicator whether Series contains null values.

        Returns
        -------
        out : bool
            If Series has at least one null value, return True, if not
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
        return self._column.has_nulls()

    @_performance_tracking
    def dropna(
        self, axis=0, inplace=False, how=None, ignore_index: bool = False
    ):
        """
        Return a Series with null values removed.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            There is only one axis to drop values from.
        inplace : bool, default False
            If True, do operation inplace and return None.
        how : str, optional
            Not in use. Kept for compatibility.
        ignore_index : bool, default ``False``
            If ``True``, the resulting axis will be labeled 0, 1, …, n - 1.

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
        2    <NA>
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

        if ignore_index:
            result.index = RangeIndex(len(result))

        return self._mimic_inplace(result, inplace=inplace)

    @_performance_tracking
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

        With the `keep` parameter, the selection behavior of duplicated
        values can be changed. The value 'first' keeps the first
        occurrence for each set of duplicated entries.
        The default value of keep is 'first'. Note that order of
        the rows being returned is not guaranteed
        to be sorted.

        >>> s.drop_duplicates()
        0      lama
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object

        The value 'last' for parameter `keep` keeps the last occurrence
        for each set of duplicated entries.

        >>> s.drop_duplicates(keep='last')
        1       cow
        3    beetle
        4      lama
        5     hippo
        Name: animal, dtype: object

        The value `False` for parameter `keep` discards all sets
        of duplicated entries. Setting the value of 'inplace' to
        `True` performs the operation inplace and returns `None`.

        >>> s.drop_duplicates(keep=False, inplace=True)
        >>> s
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object
        """
        result = super().drop_duplicates(keep=keep, ignore_index=ignore_index)

        return self._mimic_inplace(result, inplace=inplace)

    @_performance_tracking
    def fillna(
        self, value=None, method=None, axis=None, inplace=False, limit=None
    ):
        if isinstance(value, pd.Series):
            value = Series.from_pandas(value)
        elif isinstance(value, abc.Mapping):
            value = Series(value)
        if isinstance(value, cudf.Series):
            if not self.index.equals(value.index):
                value = value.reindex(self.index)
            value = {self.name: value._column}
        return super().fillna(
            value=value, method=method, axis=axis, inplace=inplace, limit=limit
        )

    def between(self, left, right, inclusive="both") -> Series:
        """
        Return boolean Series equivalent to left <= series <= right.

        This function returns a boolean vector containing `True` wherever the
        corresponding Series element is between the boundary values `left` and
        `right`. NA values are treated as `False`.

        Parameters
        ----------
        left : scalar or list-like
            Left boundary.
        right : scalar or list-like
            Right boundary.
        inclusive : {"both", "neither", "left", "right"}
            Include boundaries. Whether to set each bound as closed or open.

        Returns
        -------
        Series
            Series representing whether each element is between left and
            right (inclusive).

        See Also
        --------
        Series.gt : Greater than of series and other.
        Series.lt : Less than of series and other.

        Notes
        -----
        This function is equivalent to ``(left <= ser) & (ser <= right)``

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([2, 0, 4, 8, None])

        Boundary values are included by default:

        >>> s.between(1, 4)
        0     True
        1    False
        2     True
        3    False
        4     <NA>
        dtype: bool

        With `inclusive` set to ``"neither"`` boundary values are excluded:

        >>> s.between(1, 4, inclusive="neither")
        0     True
        1    False
        2    False
        3    False
        4     <NA>
        dtype: bool

        `left` and `right` can be any scalar value:

        >>> s = cudf.Series(['Alice', 'Bob', 'Carol', 'Eve'])
        >>> s.between('Anna', 'Daniel')
        0    False
        1     True
        2     True
        3    False
        dtype: bool
        """
        left_operand = left if is_scalar(left) else as_column(left)
        right_operand = right if is_scalar(right) else as_column(right)

        if inclusive == "both":
            lmask = self._column >= left_operand
            rmask = self._column <= right_operand
        elif inclusive == "left":
            lmask = self._column >= left_operand
            rmask = self._column < right_operand
        elif inclusive == "right":
            lmask = self._column > left_operand
            rmask = self._column <= right_operand
        elif inclusive == "neither":
            lmask = self._column > left_operand
            rmask = self._column < right_operand
        else:
            raise ValueError(
                "Inclusive has to be either string of 'both', "
                "'left', 'right', or 'neither'."
            )
        return self._from_column(
            lmask & rmask, name=self.name, index=self.index
        )

    @_performance_tracking
    def all(self, axis=0, bool_only=None, skipna=True, **kwargs):
        if bool_only not in (None, True):
            raise NotImplementedError(
                "The bool_only parameter is not supported for Series."
            )
        return super().all(axis, skipna, **kwargs)

    @_performance_tracking
    def any(self, axis=0, bool_only=None, skipna=True, **kwargs):
        if bool_only not in (None, True):
            raise NotImplementedError(
                "The bool_only parameter is not supported for Series."
            )
        return super().any(axis, skipna, **kwargs)

    @_performance_tracking
    def to_pandas(
        self,
        *,
        index: bool = True,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Series:
        """
        Convert to a pandas Series.

        Parameters
        ----------
        index : Boolean, Default True
            If ``index`` is ``True``, converts the index of cudf.Series
            and sets it to the pandas.Series. If ``index`` is ``False``,
            no index conversion is performed and pandas.Series will assign
            a default index.
        nullable : Boolean, Default False
            If ``nullable`` is ``True``, the resulting series will be
            having a corresponding nullable Pandas dtype.
            If there is no corresponding nullable Pandas dtype present,
            the resulting dtype will be a regular pandas dtype.
            If ``nullable`` is ``False``, the resulting series will
            either convert null values to ``np.nan`` or ``None``
            depending on the dtype.
        arrow_type : bool, Default False
            Return the Series with a ``pandas.ArrowDtype``

        Returns
        -------
        out : pandas Series

        Notes
        -----
        nullable and arrow_type cannot both be set to ``True``

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

        ``nullable=True`` converts the result to pandas nullable types:

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

        ``arrow_type=True`` converts the result to ``pandas.ArrowDtype``:

        >>> ser.to_pandas(arrow_type=True)
        0      10
        1      20
        2    <NA>
        3      30
        dtype: int64[pyarrow]
        """
        if index is True:
            index = self.index.to_pandas()
        else:
            index = None  # type: ignore[assignment]
        return pd.Series(
            self._column.to_pandas(nullable=nullable, arrow_type=arrow_type),
            index=index,
            name=self.name,
        )

    @property  # type: ignore
    @_performance_tracking
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
        >>> np.array(series.data.memoryview())
        array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
               0, 0, 4, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)
        """  # noqa: E501
        return self._column.data

    @property  # type: ignore
    @_performance_tracking
    def nullmask(self):
        """The gpu buffer for the null-mask"""
        return cudf.Series(self._column.nullmask)

    @_performance_tracking
    def astype(
        self,
        dtype,
        copy: bool = False,
        errors: Literal["raise", "ignore"] = "raise",
    ):
        if is_dict_like(dtype):
            if len(dtype) > 1 or self.name not in dtype:
                raise KeyError(
                    "Only the Series name can be used for the key in Series "
                    "dtype mappings."
                )
        else:
            dtype = {self.name: dtype}
        return super().astype(dtype, copy, errors)

    @_performance_tracking
    def sort_index(
        self,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind=None,
        na_position="last",
        sort_remaining=True,
        ignore_index=False,
        key=None,
    ):
        if axis not in (0, "index"):
            raise ValueError("Only axis=0 is valid for Series.")
        return super().sort_index(
            axis=axis,
            level=level,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
            sort_remaining=sort_remaining,
            ignore_index=ignore_index,
            key=key,
        )

    @_performance_tracking
    def sort_values(
        self,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
        key=None,
    ):
        """Sort by the values along either axis.

        Parameters
        ----------
        ascending : bool or list of bool, default True
            Sort ascending vs. descending. Specify list for multiple sort
            orders. If this is a list of bools, must match the length of the
            by.
        na_position : {'first', 'last'}, default 'last'
            'first' puts nulls at the beginning, 'last' puts nulls at the end
        ignore_index : bool, default False
            If True, index will not be sorted.
        key : callable, optional
            Apply the key function to the values
            before sorting. This is similar to the ``key`` argument in the
            builtin ``sorted`` function, with the notable difference that
            this ``key`` function should be *vectorized*. It should expect a
            ``Series`` and return a Series with the same shape as the input.
            It will be applied to each column in `by` independently.
            Currently not supported.

        Returns
        -------
        Series : Series with sorted values.

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

        .. pandas-compat::
            :meth:`pandas.Series.sort_values`

            * Support axis='index' only.
            * The inplace and kind argument is currently unsupported
        """
        return super().sort_values(
            by=self.name,
            axis=axis,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
            ignore_index=ignore_index,
            key=key,
        )

    @_performance_tracking
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

    @_performance_tracking
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

    @_performance_tracking
    def argsort(
        self,
        axis=0,
        kind="quicksort",
        order=None,
        ascending=True,
        na_position="last",
    ) -> Self:
        col = as_column(
            super().argsort(
                axis=axis,
                kind=kind,
                order=order,
                ascending=ascending,
                na_position=na_position,
            )
        )
        return self._from_data_like_self(
            self._data._from_columns_like_self([col])
        )

    @_performance_tracking
    def replace(
        self,
        to_replace=None,
        value=no_default,
        inplace=False,
        limit=None,
        regex=False,
        method=no_default,
    ):
        if is_dict_like(to_replace) and value not in {None, no_default}:
            raise ValueError(
                "Series.replace cannot use dict-like to_replace and non-None "
                "value"
            )

        return super().replace(
            to_replace,
            value,
            inplace=inplace,
            limit=limit,
            regex=regex,
            method=method,
        )

    @_performance_tracking
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

        >>> s = cudf.Series([1.0, 2.0, 3.0])
        >>> s
        0    1.0
        1    2.0
        2    3.0
        dtype: float64
        >>> s.update(cudf.Series([4.0, np.nan, 6.0], nan_as_null=False))
        >>> s
        0    4.0
        1    2.0
        2    6.0
        dtype: float64

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

    # UDF related
    @_performance_tracking
    def apply(
        self,
        func,
        convert_dtype=True,
        args=(),
        by_row: Literal[False, "compat"] = "compat",
        **kwargs,
    ):
        """
        Apply a scalar function to the values of a Series.
        Similar to ``pandas.Series.apply``.

        ``apply`` relies on Numba to JIT compile ``func``.
        Thus the allowed operations within ``func`` are limited to `those
        supported by the CUDA Python Numba target
        <https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html>`__.
        For more information, see the `cuDF guide to user defined functions
        <https://docs.rapids.ai/api/cudf/stable/user_guide/guide-to-udfs.html>`__.

        Some string functions and methods are supported. Refer to the guide
        to UDFs for details.

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
            Positional arguments passed to func after the series value.
        by_row : False or "compat", default "compat"
            If ``"compat"`` and func is a callable, func will be passed each element of
            the Series, like ``Series.map``. If func is a list or dict of
            callables, will first try to translate each func into pandas methods. If
            that doesn't work, will try call to apply again with ``by_row="compat"``
            and if that fails, will call apply again with ``by_row=False``
            (backward compatible).
            If False, the func will be passed the whole Series at once.

            ``by_row`` has no effect when ``func`` is a string.

            Currently not implemented.
        **kwargs
            Not supported

        Returns
        -------
        result : Series
            The mask and index are preserved.

        Notes
        -----
        UDFs are cached in memory to avoid recompilation. The first
        call to the UDF will incur compilation overhead. `func` may
        call nested functions that are decorated with the decorator
        `numba.cuda.jit(device=True)`, otherwise numba will raise a
        typing error.

        Examples
        --------
        Apply a basic function to a series:

        >>> sr = cudf.Series([1,2,3])
        >>> def f(x):
        ...     return x + 1
        >>> sr.apply(f)
        0    2
        1    3
        2    4
        dtype: int64

        Apply a basic function to a series with nulls:

        >>> sr = cudf.Series([1,cudf.NA,3])
        >>> def f(x):
        ...     return x + 1
        >>> sr.apply(f)
        0       2
        1    <NA>
        2       4
        dtype: int64

        Use a function that does something conditionally,
        based on if the value is or is not null:

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
        dtype:

        >>> sr = cudf.Series([1,cudf.NA,3])
        >>> def f(x):
        ...     return x + 1.5
        >>> sr.apply(f)
        0     2.5
        1    <NA>
        2     4.5
        dtype: float64

        UDFs manipulating string data are allowed, as long as
        they neither modify strings in place nor create new strings.
        For example, the following UDF is allowed:

        >>> def f(st):
        ...     if len(st) == 0:
        ...             return -1
        ...     elif st.startswith('a'):
        ...             return 1
        ...     elif 'example' in st:
        ...             return 2
        ...     else:
        ...             return 3
        ...
        >>> sr = cudf.Series(['', 'abc', 'some_example'])
        >>> sr.apply(f)  # doctest: +SKIP
        0   -1
        1    1
        2    2
        dtype: int64

        However, the following UDF is not allowed since it includes an
        operation that requires the creation of a new string: a call to the
        ``upper`` method. Methods that are not supported in this manner
        will raise an ``AttributeError``.

        >>> def f(st):
        ...     new = st.upper()
        ...     return 'ABC' in new
        ...
        >>> sr.apply(f)  # doctest: +SKIP

        For a complete list of supported functions and methods that may be
        used to manipulate string data, see the UDF guide,
        <https://docs.rapids.ai/api/cudf/stable/user_guide/guide-to-udfs.html>

        """
        if convert_dtype is not True:
            raise ValueError("Series.apply only supports convert_dtype=True")
        elif by_row != "compat":
            raise NotImplementedError("by_row is currently not supported.")

        result = self._apply(func, _get_scalar_kernel, *args, **kwargs)
        result.name = self.name
        return result

    #
    # Stats
    #
    @_performance_tracking
    def count(self):
        """
        Return number of non-NA/null observations in the Series

        Returns
        -------
        int
            Number of non-null values in the Series.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.count()
        5

        .. pandas-compat::
            :meth:`pandas.Series.count`

            Parameters currently not supported is `level`.
        """
        return self.valid_count

    @_performance_tracking
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

        return Series._from_column(
            val_counts.index.sort_values()._column, name=self.name
        )

    @_performance_tracking
    def round(self, decimals=0, how="half_even"):
        if not is_integer(decimals):
            raise ValueError(
                f"decimals must be an int, got {type(decimals).__name__}"
            )
        decimals = int(decimals)
        return super().round(decimals, how)

    @_performance_tracking
    def cov(self, other, min_periods=None, ddof: int | None = None):
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

        Examples
        --------
        >>> import cudf
        >>> ser1 = cudf.Series([0.9, 0.13, 0.62])
        >>> ser2 = cudf.Series([0.12, 0.26, 0.51])
        >>> ser1.cov(ser2)
        -0.015750000000000004

        .. pandas-compat::
            :meth:`pandas.Series.cov`

            `min_periods` parameter is not yet supported.
        """

        if min_periods is not None:
            raise NotImplementedError(
                "min_periods parameter is not implemented yet"
            )
        if ddof is not None:
            raise NotImplementedError("ddof parameter is not implemented yet")

        if self.empty or other.empty:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        lhs = self.nans_to_nulls().dropna()
        rhs = other.nans_to_nulls().dropna()

        lhs, rhs = _align_indices([lhs, rhs], how="inner")

        try:
            return lhs._column.cov(rhs._column)
        except AttributeError:
            raise TypeError(
                f"cannot perform covariance with types {self.dtype}, "
                f"{other.dtype}"
            )

    @_performance_tracking
    def duplicated(self, keep="first"):
        """
        Indicate duplicate Series values.

        Duplicated values are indicated as ``True`` values in the resulting
        Series. Either all duplicates, all except the first or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            Method to handle dropping duplicates:

            - ``'first'`` : Mark duplicates as ``True`` except for the first
              occurrence.
            - ``'last'`` : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        Series[bool]
            Series indicating whether each value has occurred in the
            preceding values.

        See Also
        --------
        Index.duplicated : Equivalent method on cudf.Index.
        DataFrame.duplicated : Equivalent method on cudf.DataFrame.
        Series.drop_duplicates : Remove duplicate values from Series.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set on False and all others on True:

        >>> import cudf
        >>> animals = cudf.Series(['lama', 'cow', 'lama', 'beetle', 'lama'])
        >>> animals.duplicated()
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        which is equivalent to

        >>> animals.duplicated(keep='first')
        0    False
        1    False
        2     True
        3    False
        4     True
        dtype: bool

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> animals.duplicated(keep='last')
        0     True
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        By setting keep on ``False``, all duplicates are True:

        >>> animals.duplicated(keep=False)
        0     True
        1    False
        2     True
        3    False
        4     True
        dtype: bool
        """
        return super().duplicated(keep=keep)

    @_performance_tracking
    def corr(self, other, method="pearson", min_periods=None):
        """Calculates the sample correlation between two Series,
        excluding missing values.

        Parameters
        ----------
        other : Series
            Series with which to compute the correlation.
        method : {'pearson', 'spearman'}, default 'pearson'
            Method used to compute correlation:

            - pearson : Standard correlation coefficient
            - spearman : Spearman rank correlation

        min_periods : int, optional
            Minimum number of observations needed to have a valid result.

        Examples
        --------
        >>> import cudf
        >>> ser1 = cudf.Series([0.9, 0.13, 0.62])
        >>> ser2 = cudf.Series([0.12, 0.26, 0.51])
        >>> ser1.corr(ser2, method="pearson")
        -0.20454263717316126
        >>> ser1.corr(ser2, method="spearman")
        -0.5
        """

        if method not in {"pearson", "spearman"}:
            raise ValueError(f"Unknown method {method}")

        if min_periods is not None:
            raise NotImplementedError("Unsupported argument 'min_periods'")

        if self.empty or other.empty:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        lhs = self.nans_to_nulls().dropna()
        rhs = other.nans_to_nulls().dropna()
        lhs, rhs = _align_indices([lhs, rhs], how="inner")
        if method == "spearman":
            lhs = lhs.rank()
            rhs = rhs.rank()

        try:
            return lhs._column.corr(rhs._column)
        except AttributeError:
            raise TypeError(
                f"cannot perform corr with types {self.dtype}, {other.dtype}"
            )

    @_performance_tracking
    def autocorr(self, lag=1):
        """Compute the lag-N autocorrelation. This method computes the Pearson
        correlation between the Series and its shifted self.

        Parameters
        ----------
        lag : int, default 1
            Number of lags to apply before performing autocorrelation.

        Returns
        -------
        result : float
            The Pearson correlation between self and self.shift(lag).

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([0.25, 0.5, 0.2, -0.05, 0.17])
        >>> s.autocorr()
        0.1438853844...
        >>> s.autocorr(lag=2)
        -0.9647548490...
        """
        return self.corr(self.shift(lag))

    @_performance_tracking
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
        ------
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

        # Even though only list-like objects are supposed to be passed, only
        # scalars throw errors. Other types (like dicts) just transparently
        # return False (see the implementation of ColumnBase.isin).
        if is_scalar(values):
            raise TypeError(
                "only list-like objects are allowed to be passed "
                f"to isin(), you passed a [{type(values).__name__}]"
            )

        return Series._from_column(
            self._column.isin(values), name=self.name, index=self.index
        )

    @_performance_tracking
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
        0       a
        1       b
        2    <NA>
        3       c
        dtype: object
        """
        res = self._column.unique()
        if cudf.get_option("mode.pandas_compatible"):
            return res.values
        return Series._from_column(res, name=self.name)

    @_performance_tracking
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
            only works with numeric data.

        dropna : bool, default True
            Don't include counts of NaN and None.

        Returns
        -------
        result : Series containing counts of unique values.

        See Also
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
        Name: count, dtype: int64

        The order of the counts can be changed by passing ``ascending=True``:

        >>> sr.value_counts(ascending=True)
        1.0    1
        2.0    2
        3.0    3
        Name: count, dtype: int64

        With ``normalize`` set to True, returns the relative frequency
        by dividing all values by the sum of values.

        >>> sr.value_counts(normalize=True)
        3.0    0.500000
        2.0    0.333333
        1.0    0.166667
        Name: proportion, dtype: float64

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
        Name: count, dtype: int64

        >>> s = cudf.Series([3, 1, 2, 3, 4, np.nan])
        >>> s.value_counts(bins=3)
        (2.0, 3.0]      2
        (0.996, 2.0]    2
        (3.0, 4.0]      1
        Name: count, dtype: int64
        """
        if bins is not None:
            series_bins = cudf.cut(self, bins, include_lowest=True)
        result_name = "proportion" if normalize else "count"
        if dropna and self.null_count == len(self):
            return Series(
                [],
                dtype=np.int64,
                name=result_name,
                index=cudf.Index([], dtype=self.dtype, name=self.name),
            )

        if bins is not None:
            res = self.groupby(series_bins, dropna=dropna).count(dropna=dropna)
            res = res[res.index.notna()]
        else:
            res = self.groupby(self, dropna=dropna).count(dropna=dropna)
            if isinstance(self.dtype, cudf.CategoricalDtype) and len(
                res
            ) != len(self.dtype.categories):
                # For categorical dtypes: When there exists
                # categories in dtypes and they are missing in the
                # column, `value_counts` will have to return
                # their occurrences as 0.
                # TODO: Remove this workaround once `observed`
                # parameter support is added to `groupby`
                res = res.reindex(self.dtype.categories).fillna(0)
                res.index = res.index.astype(self.dtype)

        res.index.name = self.name

        if sort:
            res = res.sort_values(ascending=ascending)

        if normalize:
            res = res / float(res._column.sum())

        # Pandas returns an IntervalIndex as the index of res
        # this condition makes sure we do too if bins is given
        if bins is not None and len(res) == len(res.index.categories):
            interval_col = IntervalColumn.from_struct_column(
                res.index._column._get_decategorized_column()
            )
            res.index = cudf.IntervalIndex._from_column(
                interval_col, name=res.index.name
            )
        res.name = result_name
        return res

    @_performance_tracking
    def quantile(
        self, q=0.5, interpolation="linear", exact=True, quant_index=True
    ):
        """
        Return values at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            0 <= q <= 1, the quantile(s) to compute
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points i and j:

                * linear: `i + (j - i) * fraction`, where `fraction` is the
                  fractional part of the index surrounded by `i` and `j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.
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

        return_scalar = is_scalar(q)
        if return_scalar:
            np_array_q = np.asarray([float(q)])
        else:
            try:
                np_array_q = np.asarray(q)
            except TypeError:
                try:
                    np_array_q = cudf.core.column.as_column(q).values_host
                except TypeError:
                    raise TypeError(
                        f"q must be a scalar or array-like, got {type(q)}"
                    )

        result = self._column.quantile(
            np_array_q, interpolation, exact, return_scalar=return_scalar
        )

        if return_scalar:
            return result

        return Series._from_column(
            result,
            name=self.name,
            index=cudf.Index(np_array_q) if quant_index else None,
        )

    @docutils.doc_describe()
    @_performance_tracking
    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
    ):
        """{docstring}"""

        if percentiles is not None:
            if not all(0 <= x <= 1 for x in percentiles):
                raise ValueError(
                    "All percentiles must be between 0 and 1, " "inclusive."
                )

            # describe always includes 50th percentile
            percentiles = list(percentiles)
            if 0.5 not in percentiles:
                percentiles.append(0.5)

            percentiles = np.sort(percentiles)
        else:
            # pandas defaults
            percentiles = np.array([0.25, 0.5, 0.75])

        dtype = "str"
        if self.dtype.kind == "b":
            data = _describe_categorical(self, percentiles)
        elif isinstance(self._column, cudf.core.column.NumericalColumn):
            data = _describe_numeric(self, percentiles)
            dtype = None
        elif isinstance(self._column, TimeDeltaColumn):
            data = _describe_timedelta(self, percentiles)
        elif isinstance(self._column, DatetimeColumn):
            data = _describe_timestamp(self, percentiles)
        else:
            data = _describe_categorical(self, percentiles)

        return Series(
            data=data.values(),
            index=data.keys(),
            dtype=dtype,
            name=self.name,
        )

    @_performance_tracking
    def digitize(self, bins, right=False):
        """Return the indices of the bins to which each value belongs.

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
        return Series._from_column(
            cudf.core.column.numerical.digitize(self._column, bins, right),
            name=self.name,
        )

    @_performance_tracking
    def diff(self, periods=1):
        """First discrete difference of element.

        Calculates the difference of a Series element compared with another
        element in the Series (default is element in previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference,
            accepts negative values.

        Returns
        -------
        Series
            First differences of the Series.

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
        if not is_integer(periods):
            if not (isinstance(periods, float) and periods.is_integer()):
                raise ValueError("periods must be an integer")
            periods = int(periods)

        return self - self.shift(periods=periods)

    @_performance_tracking
    @docutils.doc_apply(
        groupby_doc_template.format(
            ret=textwrap.dedent(
                """
                Returns
                -------
                SeriesGroupBy
                    Returns a SeriesGroupBy object that contains
                    information about the groups.
                """
            )
        )
    )
    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=no_default,
        group_keys=False,
        observed=True,
        dropna=True,
    ):
        return super().groupby(
            by,
            axis,
            level,
            as_index,
            sort,
            group_keys,
            observed,
            dropna,
        )

    @_performance_tracking
    def rename(
        self,
        index=None,
        axis=None,
        copy: bool = True,
        inplace: bool = False,
        level=None,
        errors: Literal["ignore", "raise"] = "ignore",
    ):
        """
        Alter Series name

        Change Series.name with a scalar value

        Parameters
        ----------
        index : Scalar, optional
            Scalar to alter the Series.name attribute
        axis : {0 or 'index'}
            Unused. Parameter needed for compatibility with DataFrame.
        copy : boolean, default True
            Also copy underlying data
        inplace : bool, default False
            Whether to return a new Series. If True the value of copy is ignored.
            Currently not supported.
        level : int or level name, default None
            In case of MultiIndex, only rename labels in the specified level.
            Currently not supported.
        errors : {'ignore', 'raise'}, default 'ignore'
            If 'raise', raise `KeyError` when a `dict-like mapper` or
            `index` contains labels that are not present in the index being transformed.
            If 'ignore', existing keys will be renamed and extra keys will be ignored.
            Currently not supported.

        Returns
        -------
        Series

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

        .. pandas-compat::
            :meth:`pandas.Series.rename`

            - Supports scalar values only for changing name attribute
        """
        if inplace is not False:
            raise NotImplementedError("inplace is currently not supported.")
        if level is not None:
            raise NotImplementedError("level is currently not supported.")
        if errors != "ignore":
            raise NotImplementedError("errors is currently not supported.")
        if not is_scalar(index):
            raise NotImplementedError(
                ".rename does not currently support relabeling the index."
            )
        out_data = self._data.copy(deep=copy)
        return Series._from_data(out_data, self.index, name=index)

    @_performance_tracking
    def add_prefix(self, prefix, axis=None):
        if axis is not None:
            raise NotImplementedError("axis is currently not implemented.")
        return Series._from_data(
            # TODO: Change to deep=False when copy-on-write is default
            data=self._data.copy(deep=True),
            index=prefix + self.index.astype(str),
        )

    @_performance_tracking
    def add_suffix(self, suffix, axis=None):
        if axis is not None:
            raise NotImplementedError("axis is currently not implemented.")
        return Series._from_data(
            # TODO: Change to deep=False when copy-on-write is default
            data=self._data.copy(deep=True),
            index=self.index.astype(str) + suffix,
        )

    @_performance_tracking
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
        RangeIndex(start=0, stop=6, step=1)
        >>> sr = cudf.Series(['a', 'b', 'c'])
        >>> sr
        0    a
        1    b
        2    c
        dtype: object
        >>> sr.keys()
        RangeIndex(start=0, stop=3, step=1)
        >>> sr = cudf.Series([1, 2, 3], index=['a', 'b', 'c'])
        >>> sr
        a    1
        b    2
        c    3
        dtype: int64
        >>> sr.keys()
        Index(['a', 'b', 'c'], dtype='object')
        """
        return self.index

    @_performance_tracking
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
        Series

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
        return super()._explode(self.name, ignore_index)

    @_performance_tracking
    def pct_change(
        self,
        periods=1,
        fill_method=no_default,
        limit=no_default,
        freq=None,
        **kwargs,
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

            .. deprecated:: 24.04
                All options of `fill_method` are deprecated
                except `fill_method=None`.
        limit : int, optional
            The number of consecutive NAs to fill before stopping.
            Not yet implemented.

            .. deprecated:: 24.04
                `limit` is deprecated.
        freq : str, optional
            Increment to use from time series API.
            Not yet implemented.
        **kwargs
            Additional keyword arguments are passed into
            `Series.shift`.

        Returns
        -------
        Series
        """
        if limit is not no_default:
            raise NotImplementedError("limit parameter not supported yet.")
        if freq is not None:
            raise NotImplementedError("freq parameter not supported yet.")
        elif fill_method not in {
            no_default,
            None,
            "ffill",
            "pad",
            "bfill",
            "backfill",
        }:
            raise ValueError(
                "fill_method must be one of None, 'ffill', 'pad', "
                "'bfill', or 'backfill'."
            )
        if fill_method not in (no_default, None) or limit is not no_default:
            # Do not remove until pandas 3.0 support is added.
            assert (
                PANDAS_LT_300
            ), "Need to drop after pandas-3.0 support is added."
            warnings.warn(
                "The 'fill_method' and 'limit' keywords in "
                f"{type(self).__name__}.pct_change are deprecated and will be "
                "removed in a future version. Either fill in any non-leading "
                "NA values prior to calling pct_change or specify "
                "'fill_method=None' to not fill NA values.",
                FutureWarning,
            )

        if fill_method is no_default:
            fill_method = "ffill"
        if limit is no_default:
            limit = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self.fillna(method=fill_method, limit=limit)
        diff = data.diff(periods=periods)
        change = diff / data.shift(periods=periods, freq=freq, **kwargs)
        return change

    @_performance_tracking
    def where(self, cond, other=None, inplace=False, axis=None, level=None):
        if axis is not None:
            raise NotImplementedError("axis is not supported.")
        elif level is not None:
            raise NotImplementedError("level is not supported.")
        result_col = super().where(cond, other, inplace)
        return self._mimic_inplace(
            self._from_data_like_self(
                self._data._from_columns_like_self([result_col])
            ),
            inplace=inplace,
        )


def make_binop_func(op):
    # This function is used to wrap binary operations in Frame with an
    # appropriate API for Series as required for pandas compatibility. The
    # main effect is reordering and error-checking parameters in
    # Series-specific ways.
    wrapped_func = getattr(IndexedFrame, op)

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


class BaseDatelikeProperties:
    """
    Base accessor class for Series values.
    """

    def __init__(self, series: Series):
        self.series = series

    def _return_result_like_self(self, column: ColumnBase) -> Series:
        """Return the method result like self.series"""
        data = ColumnAccessor({self.series.name: column}, verify=False)
        return self.series._from_data_like_self(data)


class DatetimeProperties(BaseDatelikeProperties):
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

    @property  # type: ignore
    @_performance_tracking
    def year(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def month(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def day(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def hour(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def minute(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def second(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def microsecond(self) -> Series:
        """
        The microseconds of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_series = cudf.Series(pd.date_range("2000-01-01",
        ...         periods=3, freq="us"))
        >>> datetime_series
        0    2000-01-01 00:00:00.000000
        1    2000-01-01 00:00:00.000001
        2    2000-01-01 00:00:00.000002
        dtype: datetime64[ns]
        >>> datetime_series.dt.microsecond
        0    0
        1    1
        2    2
        dtype: int32
        """
        micro = self.series._column.get_dt_field("microsecond")
        # Need to manually promote column to int32 because
        # pandas-matching binop behaviour requires that this
        # __mul__ returns an int16 column.
        extra = self.series._column.get_dt_field("millisecond").astype(
            "int32"
        ) * cudf.Scalar(1000, dtype="int32")
        return self._return_result_like_self(micro + extra)

    @property  # type: ignore
    @_performance_tracking
    def nanosecond(self) -> Series:
        """
        The nanoseconds of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_series = cudf.Series(pd.date_range("2000-01-01",
        ...         periods=3, freq="ns"))
        >>> datetime_series
        0    2000-01-01 00:00:00.000000000
        1    2000-01-01 00:00:00.000000001
        2    2000-01-01 00:00:00.000000002
        dtype: datetime64[ns]
        >>> datetime_series.dt.nanosecond
        0    0
        1    1
        2    2
        dtype: int16
        """
        return self._get_dt_field("nanosecond")

    @property  # type: ignore
    @_performance_tracking
    def weekday(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def dayofweek(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def dayofyear(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def day_of_year(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def is_leap_year(self) -> Series:
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

        Examples
        --------
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
        return self._return_result_like_self(res)

    @property  # type: ignore
    @_performance_tracking
    def quarter(self) -> Series:
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
        --------
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
        return self._return_result_like_self(res)

    @_performance_tracking
    def day_name(self, locale: str | None = None) -> Series:
        """
        Return the day names. Currently supports English locale only.

        Examples
        --------
        >>> import cudf
        >>> datetime_series = cudf.Series(cudf.date_range('2016-12-31',
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
        >>> datetime_series.dt.day_name()
        0     Saturday
        1       Sunday
        2       Monday
        3      Tuesday
        4    Wednesday
        5     Thursday
        6       Friday
        7     Saturday
        dtype: object
        """
        return self._return_result_like_self(
            self.series._column.get_day_names(locale)
        )

    @_performance_tracking
    def month_name(self, locale: str | None = None) -> Series:
        """
        Return the month names. Currently supports English locale only.

        Examples
        --------
        >>> import cudf
        >>> datetime_series = cudf.Series(cudf.date_range("2017-12-30", periods=6, freq='W'))
        >>> datetime_series
        0   2017-12-30
        1   2018-01-06
        2   2018-01-13
        3   2018-01-20
        4   2018-01-27
        5   2018-02-03
        dtype: datetime64[ns]
        >>> datetime_series.dt.month_name()
        0    December
        1     January
        2     January
        3     January
        4     January
        5    February
        dtype: object
        """
        return self._return_result_like_self(
            self.series._column.get_month_names(locale)
        )

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
        ca = ColumnAccessor(self.series._column.isocalendar(), verify=False)
        return self.series._constructor_expanddim._from_data(
            ca, index=self.series.index
        )

    @property  # type: ignore
    @_performance_tracking
    def is_month_start(self) -> Series:
        """
        Booleans indicating if dates are the first day of the month.
        """
        return self._return_result_like_self(
            self.series._column.is_month_start
        )

    @property  # type: ignore
    @_performance_tracking
    def days_in_month(self) -> Series:
        """
        Get the total number of days in the month that the date falls on.

        Returns
        -------
        Series
        Integers representing the number of days in month

        Examples
        --------
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
        return self._return_result_like_self(self.series._column.days_in_month)

    @property  # type: ignore
    @_performance_tracking
    def is_month_end(self) -> Series:
        """
        Boolean indicator if the date is the last day of the month.

        Returns
        -------
        Series
        Booleans indicating if dates are the last day of the month.

        Examples
        --------
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
        return self._return_result_like_self(self.series._column.is_month_end)

    @property  # type: ignore
    @_performance_tracking
    def is_quarter_start(self) -> Series:
        """
        Boolean indicator if the date is the first day of a quarter.

        Returns
        -------
        Series
        Booleans indicating if dates are the beginning of a quarter

        Examples
        --------
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
        return self._return_result_like_self(
            self.series._column.is_quarter_start
        )

    @property  # type: ignore
    @_performance_tracking
    def is_quarter_end(self) -> Series:
        """
        Boolean indicator if the date is the last day of a quarter.

        Returns
        -------
        Series
        Booleans indicating if dates are the end of a quarter

        Examples
        --------
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
        return self._return_result_like_self(
            self.series._column.is_quarter_end
        )

    @property  # type: ignore
    @_performance_tracking
    def is_year_start(self) -> Series:
        """
        Boolean indicator if the date is the first day of the year.

        Returns
        -------
        Series
        Booleans indicating if dates are the first day of the year.

        Examples
        --------
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
        return self._return_result_like_self(self.series._column.is_year_start)

    @property  # type: ignore
    @_performance_tracking
    def is_year_end(self) -> Series:
        """
        Boolean indicator if the date is the last day of the year.

        Returns
        -------
        Series
        Booleans indicating if dates are the last day of the year.

        Examples
        --------
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
        return self._return_result_like_self(self.series._column.is_year_end)

    @_performance_tracking
    def _get_dt_field(self, field: str) -> Series:
        return self._return_result_like_self(
            self.series._column.get_dt_field(field)
        )

    @_performance_tracking
    def ceil(self, freq: str) -> Series:
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
        return self._return_result_like_self(self.series._column.ceil(freq))

    @_performance_tracking
    def floor(self, freq: str) -> Series:
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
        return self._return_result_like_self(self.series._column.floor(freq))

    @_performance_tracking
    def round(self, freq: str) -> Series:
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
        Series
            Series with all timestamps rounded to the specified frequency.
            The index is preserved.

        Examples
        --------
        >>> import cudf
        >>> dt_sr = cudf.Series([
        ...     "2001-01-01 00:04:45",
        ...     "2001-01-01 00:04:58",
        ...     "2001-01-01 00:05:04",
        ... ], dtype="datetime64[ns]")
        >>> dt_sr.dt.round("T")
        0   2001-01-01 00:05:00
        1   2001-01-01 00:05:00
        2   2001-01-01 00:05:00
        dtype: datetime64[ns]
        """
        return self._return_result_like_self(self.series._column.round(freq))

    @_performance_tracking
    def strftime(self, date_format: str, *args, **kwargs) -> Series:
        """
        Convert to Series using specified ``date_format``.

        Return a Series of formatted strings specified by ``date_format``,
        which supports the same string format as the python standard library.
        Details of the string format can be found in `python string format doc
        <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior>`_.

        Parameters
        ----------
        date_format : str
            Date format string (e.g. "%Y-%m-%d").

        Returns
        -------
        Series
            Series of formatted strings.

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

        .. pandas-compat::
            :meth:`pandas.DatetimeIndex.strftime`

            The following date format identifiers are not yet
            supported: ``%c``, ``%x``,``%X``
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
        return self._return_result_like_self(
            self.series._column.strftime(format=date_format)
        )

    @copy_docstring(DatetimeIndex.tz_localize)
    def tz_localize(
        self,
        tz: str | None,
        ambiguous: Literal["NaT"] = "NaT",
        nonexistent: Literal["NaT"] = "NaT",
    ) -> Series:
        return self._return_result_like_self(
            self.series._column.tz_localize(tz, ambiguous, nonexistent)
        )

    @copy_docstring(DatetimeIndex.tz_convert)
    def tz_convert(self, tz: str | None) -> Series:
        """
        Parameters
        ----------
        tz : str
            Time zone for time. Corresponding timestamps would be converted
            to this time zone of the Datetime Array/Index.
            A `tz` of None will convert to UTC and remove the
            timezone information.
        """
        return self._return_result_like_self(
            self.series._column.tz_convert(tz)
        )


class TimedeltaProperties(BaseDatelikeProperties):
    """
    Accessor object for timedelta-like properties of the Series values.

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

    @property  # type: ignore
    @_performance_tracking
    def days(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def seconds(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def microseconds(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def nanoseconds(self) -> Series:
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

    @property  # type: ignore
    @_performance_tracking
    def components(self) -> cudf.DataFrame:
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
        ca = ColumnAccessor(self.series._column.components(), verify=False)
        return self.series._constructor_expanddim._from_data(
            ca, index=self.series.index
        )

    @_performance_tracking
    def _get_td_field(self, field: str) -> Series:
        return self._return_result_like_self(
            getattr(self.series._column, field)
        )


@_performance_tracking
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

    combined_index = series_list[0].index
    for sr in series_list[1:]:
        combined_index = (
            cudf.DataFrame(index=sr.index).join(
                cudf.DataFrame(index=combined_index),
                sort=True,
                how=how,
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


@acquire_spill_lock()
@_performance_tracking
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    r"""Returns a boolean array where two arrays are equal within a tolerance.

    Two values in ``a`` and ``b`` are  considered equal when the following
    equation is satisfied.

    .. math::
       |a - b| \le \mathrm{atol} + \mathrm{rtol} |b|

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
        index = cudf.Index(a.index)

    a_col = as_column(a)
    a_array = cupy.asarray(a_col.data_array_view(mode="read"))

    b_col = as_column(b)
    b_array = cupy.asarray(b_col.data_array_view(mode="read"))

    result = cupy.isclose(
        a=a_array, b=b_array, rtol=rtol, atol=atol, equal_nan=equal_nan
    )
    result_col = as_column(result)

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
        return Series._from_column(result_col, index=index)

    result_col[null_values] = False
    if equal_nan is True and a_col.null_count and b_col.null_count:
        result_col[equal_nulls] = True

    return Series._from_column(result_col, index=index)
