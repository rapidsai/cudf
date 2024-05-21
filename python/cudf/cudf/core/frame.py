# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from __future__ import annotations

import copy
import itertools
import operator
import pickle
import types
import warnings
from collections import abc
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

# TODO: The `numpy` import is needed for typing purposes during doc builds
# only, need to figure out why the `np` alias is insufficient then remove.
import cupy
import numpy
import numpy as np
import pyarrow as pa
from typing_extensions import Self

import cudf
from cudf import _lib as libcudf
from cudf._typing import Dtype
from cudf.api.types import is_bool_dtype, is_dtype_equal, is_scalar
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column import (
    ColumnBase,
    as_column,
    build_categorical_column,
    deserialize_columns,
    serialize_columns,
)
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.mixins import BinaryOperand, Scannable
from cudf.utils import ioutils
from cudf.utils.dtypes import find_common_type
from cudf.utils.nvtx_annotation import _cudf_nvtx_annotate
from cudf.utils.utils import _array_ufunc, _warn_no_dask_cudf


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

    @property
    def _dtypes(self):
        return dict(
            zip(self._data.names, (col.dtype for col in self._data.columns))
        )

    @property
    def ndim(self) -> int:
        raise NotImplementedError()

    @_cudf_nvtx_annotate
    def serialize(self):
        # TODO: See if self._data can be serialized outright
        header = {
            "type-serialized": pickle.dumps(type(self)),
            "column_names": pickle.dumps(tuple(self._data.names)),
            "column_rangeindex": pickle.dumps(self._data.rangeindex),
            "column_multiindex": pickle.dumps(self._data.multiindex),
            "column_label_dtype": pickle.dumps(self._data.label_dtype),
            "column_level_names": pickle.dumps(self._data._level_names),
        }
        header["columns"], frames = serialize_columns(self._columns)
        return header, frames

    @classmethod
    @_cudf_nvtx_annotate
    def deserialize(cls, header, frames):
        cls_deserialize = pickle.loads(header["type-serialized"])
        column_names = pickle.loads(header["column_names"])
        columns = deserialize_columns(header["columns"], frames)
        kwargs = {}
        for metadata in [
            "rangeindex",
            "multiindex",
            "label_dtype",
            "level_names",
        ]:
            key = f"column_{metadata}"
            if key in header:
                kwargs[metadata] = pickle.loads(header[key])
        col_accessor = ColumnAccessor(
            data=dict(zip(column_names, columns)), **kwargs
        )
        return cls_deserialize._from_data(col_accessor)

    @classmethod
    @_cudf_nvtx_annotate
    def _from_data(cls, data: MutableMapping) -> Self:
        obj = cls.__new__(cls)
        Frame.__init__(obj, data)
        return obj

    @_cudf_nvtx_annotate
    def _from_data_like_self(self, data: MutableMapping) -> Self:
        return self._from_data(data)

    @_cudf_nvtx_annotate
    def _from_columns_like_self(
        self,
        columns: List[ColumnBase],
        column_names: Optional[abc.Iterable[str]] = None,
        *,
        override_dtypes: Optional[abc.Iterable[Optional[Dtype]]] = None,
    ):
        """Construct a Frame from a list of columns with metadata from self.

        If `column_names` is None, use column names from self.
        """
        if column_names is None:
            column_names = self._column_names
        data = dict(zip(column_names, columns))
        frame = self.__class__._from_data(data)
        return frame._copy_type_metadata(self, override_dtypes=override_dtypes)

    @_cudf_nvtx_annotate
    def _mimic_inplace(
        self, result: Self, inplace: bool = False
    ) -> Optional[Self]:
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
    @_cudf_nvtx_annotate
    def size(self) -> int:
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
        Index([], dtype='float64')
        >>> index.size
        0
        >>> index = cudf.Index([1, 2, 3, 10])
        >>> index
        Index([1, 2, 3, 10], dtype='int64')
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

    @_cudf_nvtx_annotate
    def __len__(self) -> int:
        return self._num_rows

    @_cudf_nvtx_annotate
    def astype(self, dtype, copy: bool = False):
        result_data = {
            col_name: col.astype(dtype.get(col_name, col.dtype), copy=copy)
            for col_name, col in self._data.items()
        }

        return ColumnAccessor(
            data=result_data,
            multiindex=self._data.multiindex,
            level_names=self._data.level_names,
            rangeindex=self._data.rangeindex,
            label_dtype=self._data.label_dtype,
            verify=False,
        )

    @_cudf_nvtx_annotate
    def equals(self, other) -> bool:
        """
        Test whether two objects contain the same elements.

        This function allows two objects to be compared against
        each other to see if they have the same shape and elements. NaNs in
        the same location are considered equal. The column headers do not
        need to have the same type.

        Parameters
        ----------
        other : Index, Series, DataFrame
            The other object to be compared with.

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
    def _get_columns_by_label(self, labels, *, downcast=False) -> Self:
        """
        Returns columns of the Frame specified by `labels`

        """
        return self.__class__._from_data(self._data.select_by_label(labels))

    @property
    @_cudf_nvtx_annotate
    def values(self) -> cupy.ndarray:
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
    @_cudf_nvtx_annotate
    def values_host(self) -> np.ndarray:
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

    @_cudf_nvtx_annotate
    def __array__(self, dtype=None):
        raise TypeError(
            "Implicit conversion to a host NumPy array via __array__ is not "
            "allowed, To explicitly construct a GPU matrix, consider using "
            ".to_cupy()\nTo explicitly construct a host matrix, consider "
            "using .to_numpy()."
        )

    @_cudf_nvtx_annotate
    def __arrow_array__(self, type=None):
        raise TypeError(
            "Implicit conversion to a host PyArrow object via __arrow_array__ "
            "is not allowed. Consider using .to_arrow()"
        )

    @_cudf_nvtx_annotate
    def _to_array(
        self,
        get_array: Callable,
        module: types.ModuleType,
        copy: bool,
        dtype: Union[Dtype, None] = None,
        na_value=None,
    ) -> Union[cupy.ndarray, numpy.ndarray]:
        # Internal function to implement to_cupy and to_numpy, which are nearly
        # identical except for the attribute they access to generate values.

        def to_array(
            col: ColumnBase, dtype: np.dtype
        ) -> Union[cupy.ndarray, numpy.ndarray]:
            if na_value is not None:
                col = col.fillna(na_value)
            array = get_array(col)
            casted_array = module.asarray(array, dtype=dtype)
            if copy and casted_array is array:
                # Don't double copy after asarray
                casted_array = casted_array.copy()
            return casted_array

        ncol = self._num_columns
        if ncol == 0:
            return module.empty(
                shape=(len(self), ncol),
                dtype=numpy.dtype("float64"),
                order="F",
            )

        if dtype is None:
            if ncol == 1:
                dtype = next(iter(self._data.values())).dtype
            else:
                dtype = find_common_type(
                    [col.dtype for col in self._data.values()]
                )

            if not isinstance(dtype, numpy.dtype):
                raise NotImplementedError(
                    f"{dtype} cannot be exposed as an array"
                )

        if self.ndim == 1:
            return to_array(self._data.columns[0], dtype)
        else:
            matrix = module.empty(
                shape=(len(self), ncol), dtype=dtype, order="F"
            )
            for i, col in enumerate(self._data.values()):
                # TODO: col.values may fail if there is nullable data or an
                # unsupported dtype. We may want to catch and provide a more
                # suitable error.
                matrix[:, i] = to_array(col, dtype)
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
            lambda col: col.values,
            cupy,
            copy,
            dtype,
            na_value,
        )

    @_cudf_nvtx_annotate
    def to_numpy(
        self,
        dtype: Union[Dtype, None] = None,
        copy: bool = True,
        na_value=None,
    ) -> numpy.ndarray:
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
            lambda col: col.values_host, numpy, copy, dtype, na_value
        )

    @_cudf_nvtx_annotate
    def where(self, cond, other=None, inplace: bool = False) -> Optional[Self]:
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

        .. pandas-compat::
            **DataFrame.where, Series.where**

            Note that ``where`` treats missing values as falsy,
            in parallel with pandas treatment of nullable data:

            >>> gsr = cudf.Series([1, 2, 3])
            >>> gsr.where([True, False, cudf.NA])
            0       1
            1    <NA>
            2    <NA>
            dtype: int64
            >>> gsr.where([True, False, False])
            0       1
            1    <NA>
            2    <NA>
            dtype: int64
        """
        raise NotImplementedError

    @_cudf_nvtx_annotate
    def fillna(
        self,
        value=None,
        method: Optional[Literal["ffill", "bfill", "pad", "backfill"]] = None,
        axis=None,
        inplace: bool = False,
        limit=None,
    ) -> Optional[Self]:
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

            .. deprecated:: 24.04
                `method` is deprecated.

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
            if not self.index.equals(value.index):  # type: ignore[attr-defined]
                value = value.reindex(self.index)  # type: ignore[attr-defined]
            else:
                value = value
        elif not isinstance(value, abc.Mapping):
            value = {name: copy.deepcopy(value) for name in self._data.names}
        else:
            value = {
                key: value.reindex(self.index)  # type: ignore[attr-defined]
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
                (
                    col_name in value
                    and col.has_nulls(include_nan=True)
                    and not libcudf.scalar._is_null_host_scalar(replace_val)
                )
                or method is not None
                or (
                    isinstance(col, cudf.core.column.CategoricalColumn)
                    and not libcudf.scalar._is_null_host_scalar(replace_val)
                )
            )
            if should_fill:
                filled_data[col_name] = col.fillna(replace_val, method)
            else:
                filled_data[col_name] = col.copy(deep=True)

        return self._mimic_inplace(
            self._from_data(
                data=ColumnAccessor(
                    data=filled_data,
                    multiindex=self._data.multiindex,
                    level_names=self._data.level_names,
                    rangeindex=self._data.rangeindex,
                    label_dtype=self._data.label_dtype,
                    verify=False,
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
    def _quantile_table(
        self,
        q: float,
        interpolation: Literal[
            "LINEAR", "LOWER", "HIGHER", "MIDPOINT", "NEAREST"
        ] = "LINEAR",
        is_sorted: bool = False,
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
            libcudf.quantiles.quantile_table(
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
    def from_arrow(cls, data: pa.Table) -> Self:
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
        pandas_dtypes = {}
        np_dtypes = {}
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
        for name in result:
            if (
                len(result[name]) == 0
                and pandas_dtypes.get(name) == "categorical"
            ):
                # When pandas_dtype is a categorical column and the size
                # of column is 0 (i.e., empty) then we will have an
                # int8 column in result._data[name] returned by libcudf,
                # which needs to be type-casted to 'category' dtype.
                result[name] = result[name].as_categorical_column("category")
            elif (
                pandas_dtypes.get(name) == "empty"
                and np_dtypes.get(name) == "object"
            ):
                # When a string column has all null values, pandas_dtype is
                # is specified as 'empty' and np_dtypes as 'object',
                # hence handling this special case to type-cast the empty
                # float column to str column.
                result[name] = result[name].as_string_column(cudf.dtype("str"))
            elif name in data.column_names and isinstance(
                data[name].type,
                (
                    pa.StructType,
                    pa.ListType,
                    pa.Decimal128Type,
                    pa.TimestampType,
                ),
            ):
                # In case of struct column, libcudf is not aware of names of
                # struct fields, hence renaming the struct fields is
                # necessary by extracting the field names from arrow
                # struct types.

                # In case of decimal column, libcudf is not aware of the
                # decimal precision.

                # In case of list column, there is a possibility of nested
                # list columns to have struct or decimal columns inside them.

                # Datetimes ("timestamps") may need timezone metadata
                # attached to them, as libcudf is timezone-unaware

                # All of these cases are handled by calling the
                # _with_type_metadata method on the column.
                result[name] = result[name]._with_type_metadata(
                    cudf.utils.dtypes.cudf_dtype_from_pa_type(data[name].type)
                )

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
            {str(name): col.to_arrow() for name, col in self._data.items()}
        )

    @_cudf_nvtx_annotate
    def _positions_from_column_names(self, column_names) -> list[int]:
        """Map each column name into their positions in the frame.

        The order of indices returned corresponds to the column order in this
        Frame.
        """
        return [
            i
            for i, name in enumerate(self._column_names)
            if name in set(column_names)
        ]

    @_cudf_nvtx_annotate
    def _copy_type_metadata(
        self,
        other: Self,
        *,
        override_dtypes: Optional[abc.Iterable[Optional[Dtype]]] = None,
    ) -> Self:
        """
        Copy type metadata from each column of `other` to the corresponding
        column of `self`.

        If override_dtypes is provided, any non-None entry
        will be used in preference to the relevant column of other to
        provide the new dtype.

        See `ColumnBase._with_type_metadata` for more information.
        """
        if override_dtypes is None:
            override_dtypes = itertools.repeat(None)
        dtypes = (
            dtype if dtype is not None else col.dtype
            for (dtype, col) in zip(override_dtypes, other._data.values())
        )
        for (name, col), dtype in zip(self._data.items(), dtypes):
            self._data.set_by_label(
                name, col._with_type_metadata(dtype), validate=False
            )

        return self

    @_cudf_nvtx_annotate
    def isna(self):
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
        ``inf`` in case of float are not
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
        >>> df = cudf.DataFrame({'age': [5, 6, np.nan],
        ...                    'born': [pd.NaT, pd.Timestamp('1939-05-27'),
        ...                             pd.Timestamp('1940-04-25')],
        ...                    'name': ['Alfred', 'Batman', ''],
        ...                    'toy': [None, 'Batmobile', 'Joker']})
        >>> df
            age                        born    name        toy
        0     5                        <NA>  Alfred       <NA>
        1     6  1939-05-27 00:00:00.000000  Batman  Batmobile
        2  <NA>  1940-04-25 00:00:00.000000              Joker
        >>> df.isna()
             age   born   name    toy
        0  False   True  False   True
        1  False  False  False  False
        2   True  False  False  False

        Show which entries in a Series are NA.

        >>> ser = cudf.Series([5, 6, np.nan, np.inf, -np.inf])
        >>> ser
        0     5.0
        1     6.0
        2    <NA>
        3     Inf
        4    -Inf
        dtype: float64
        >>> ser.isna()
        0    False
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        Show which entries in an Index are NA.

        >>> idx = cudf.Index([1, 2, None, np.nan, 0.32, np.inf])
        >>> idx
        Index([1.0, 2.0, <NA>, <NA>, 0.32, Inf], dtype='float64')
        >>> idx.isna()
        array([False, False,  True,  True, False, False])
        """
        data_columns = (col.isnull() for col in self._columns)
        return self._from_data_like_self(
            self._data._from_columns_like_self(data_columns)
        )

    # Alias for isna
    isnull = isna

    @_cudf_nvtx_annotate
    def notna(self):
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
        ``inf`` in case of float are not
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
        >>> df = cudf.DataFrame({'age': [5, 6, np.nan],
        ...                    'born': [pd.NaT, pd.Timestamp('1939-05-27'),
        ...                             pd.Timestamp('1940-04-25')],
        ...                    'name': ['Alfred', 'Batman', ''],
        ...                    'toy': [None, 'Batmobile', 'Joker']})
        >>> df
            age                        born    name        toy
        0     5                        <NA>  Alfred       <NA>
        1     6  1939-05-27 00:00:00.000000  Batman  Batmobile
        2  <NA>  1940-04-25 00:00:00.000000              Joker
        >>> df.notna()
             age   born  name    toy
        0   True  False  True  False
        1   True   True  True   True
        2  False   True  True   True

        Show which entries in a Series are NA.

        >>> ser = cudf.Series([5, 6, np.nan, np.inf, -np.inf])
        >>> ser
        0     5.0
        1     6.0
        2    <NA>
        3     Inf
        4    -Inf
        dtype: float64
        >>> ser.notna()
        0     True
        1     True
        2    False
        3     True
        4     True
        dtype: bool

        Show which entries in an Index are NA.

        >>> idx = cudf.Index([1, 2, None, np.nan, 0.32, np.inf])
        >>> idx
        Index([1.0, 2.0, <NA>, <NA>, 0.32, Inf], dtype='float64')
        >>> idx.notna()
        array([ True,  True, False, False,  True,  True])
        """
        data_columns = (col.notnull() for col in self._columns)
        return self._from_data_like_self(
            self._data._from_columns_like_self(data_columns)
        )

    # Alias for notna
    notnull = notna

    @_cudf_nvtx_annotate
    def searchsorted(
        self,
        values,
        side: Literal["left", "right"] = "left",
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ):
        """Find indices where elements should be inserted to maintain order

        Parameters
        ----------
        value : Frame (Shape must be consistent with self)
            Values to be hypothetically inserted into Self
        side : str {'left', 'right'} optional, default 'left'
            If 'left', the index of the first suitable location found is given
            If 'right', return the last such index
        ascending : bool optional, default True
            Sorted Frame is in ascending order (otherwise descending)
        na_position : str {'last', 'first'} optional, default 'last'
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
        # Call libcudf search_sorted primitive

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

        # TODO: Change behavior based on the decision in
        # https://github.com/pandas-dev/pandas/issues/54668
        common_dtype_list = [
            find_common_type([col.dtype, val.dtype])
            for col, val in zip(self._columns, values)
        ]
        sources = [
            col
            if is_dtype_equal(col.dtype, common_dtype)
            else col.astype(common_dtype)
            for col, common_dtype in zip(self._columns, common_dtype_list)
        ]
        values = [
            val
            if is_dtype_equal(val.dtype, common_dtype)
            else val.astype(common_dtype)
            for val, common_dtype in zip(values, common_dtype_list)
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
        result = cupy.asarray(outcol.data_array_view(mode="read"))
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
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs
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

    @_cudf_nvtx_annotate
    def _get_sorted_inds(
        self,
        by=None,
        ascending=True,
        na_position: Literal["first", "last"] = "last",
    ) -> ColumnBase:
        """
        Get the indices required to sort self according to the columns
        specified in by.
        """

        to_sort = [
            *(
                self
                if by is None
                else self._get_columns_by_label(list(by), downcast=False)
            )._columns
        ]

        if is_scalar(ascending):
            ascending_lst = [ascending] * len(to_sort)
        else:
            ascending_lst = list(ascending)

        return libcudf.sort.order_by(
            to_sort,
            ascending_lst,
            na_position,
            stable=True,
        )

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
        return self._from_data_like_self(
            self._data._from_columns_like_self(data_columns)
        )

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
                        with acquire_spill_lock():
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

    @_cudf_nvtx_annotate
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _array_ufunc(self, ufunc, method, inputs, kwargs)

    @_cudf_nvtx_annotate
    @acquire_spill_lock()
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

    # Unary logical operators
    @_cudf_nvtx_annotate
    def __neg__(self):
        """Negate for integral dtypes, logical NOT for bools."""
        return self._from_data_like_self(
            self._data._from_columns_like_self(
                (
                    col.unary_operator("not")
                    if col.dtype.kind == "b"
                    else -1 * col
                    for col in self._data.columns
                )
            )
        )

    @_cudf_nvtx_annotate
    def __pos__(self):
        return self.copy(deep=True)

    @_cudf_nvtx_annotate
    def __abs__(self):
        return self._unaryop("abs")

    # Reductions
    @classmethod
    @_cudf_nvtx_annotate
    def _get_axis_from_axis_arg(cls, axis):
        try:
            return cls._SUPPORT_AXIS_LOOKUP[axis]
        except KeyError:
            raise ValueError(f"No axis named {axis} for object type {cls}")

    @_cudf_nvtx_annotate
    def _reduce(self, *args, **kwargs):
        raise NotImplementedError(
            f"Reductions are not supported for objects of type {type(self)}."
        )

    @_cudf_nvtx_annotate
    def min(
        self,
        axis=0,
        skipna=True,
        numeric_only=False,
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
        numeric_only: bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> min_series = df.min()
        >>> min_series
        a    1
        b    7
        dtype: int64
        >>> min_series.min()
        1

        .. pandas-compat::
            **DataFrame.min, Series.min**

            Parameters currently not supported are `level`, `numeric_only`.
        """
        return self._reduce(
            "min",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def max(
        self,
        axis=0,
        skipna=True,
        numeric_only=False,
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
        numeric_only: bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.max()
        a     4
        b    10
        dtype: int64

        .. pandas-compat::
            **DataFrame.max, Series.max**

            Parameters currently not supported are `level`, `numeric_only`.
        """
        return self._reduce(
            "max",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def all(self, axis=0, skipna=True, **kwargs):
        """
        Return whether all elements are True in DataFrame.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Indicate which axis or axes should be reduced. For `Series`
            this parameter is unused and defaults to `0`.

            - 0 or 'index' : reduce the index, return a Series
                whose index is the original column labels.
            - 1 or 'columns' : reduce the columns, return a Series
                whose index is the original index.
            - None : reduce all axes, return a scalar.
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
        Parameters currently not supported are `bool_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 0, 10, 10]})
        >>> df.all()
        a     True
        b    False
        dtype: bool

        .. pandas-compat::
            **DataFrame.all, Series.all**

            Parameters currently not supported are `axis`, `bool_only`,
            `level`.
        """
        return self._reduce(
            "all",
            axis=axis,
            skipna=skipna,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def any(self, axis=0, skipna=True, **kwargs):
        """
        Return whether any elements is True in DataFrame.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Indicate which axis or axes should be reduced. For `Series`
            this parameter is unused and defaults to `0`.

            - 0 or 'index' : reduce the index, return a Series
                whose index is the original column labels.
            - 1 or 'columns' : reduce the columns, return a Series
                whose index is the original index.
            - None : reduce all axes, return a scalar.
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
        Parameters currently not supported are `bool_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 0, 10, 10]})
        >>> df.any()
        a    True
        b    True
        dtype: bool

        .. pandas-compat::
            **DataFrame.any, Series.any**

            Parameters currently not supported are `axis`, `bool_only`,
            `level`.
        """
        return self._reduce(
            "any",
            axis=axis,
            skipna=skipna,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""

        return cudf.io.dlpack.to_dlpack(self)

    @_cudf_nvtx_annotate
    def __str__(self):
        return repr(self)

    @_cudf_nvtx_annotate
    def __deepcopy__(self, memo):
        return self.copy(deep=True)

    @_cudf_nvtx_annotate
    def __copy__(self):
        return self.copy(deep=False)

    @_cudf_nvtx_annotate
    def __invert__(self):
        """Bitwise invert (~) for integral dtypes, logical NOT for bools."""
        return self._from_data_like_self(
            self._data._from_columns_like_self(
                (_apply_inverse_column(col) for col in self._data.columns)
            )
        )

    @_cudf_nvtx_annotate
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
    @_cudf_nvtx_annotate
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

    @_cudf_nvtx_annotate
    @_warn_no_dask_cudf
    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return [
            type(self),
            str(self._dtypes),
            normalize_token(self.to_pandas()),
        ]


def _apply_inverse_column(col: ColumnBase) -> ColumnBase:
    """Bitwise invert (~) for integral dtypes, logical NOT for bools."""
    if np.issubdtype(col.dtype, np.integer):
        return col.unary_operator("invert")
    elif is_bool_dtype(col.dtype):
        return col.unary_operator("not")
    else:
        raise TypeError(
            f"Operation `~` not supported on {col.dtype.type.__name__}"
        )
