import functools
import itertools
import warnings
from collections import OrderedDict

import cupy
import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal

import cudf
import cudf._lib as libcudf
from cudf._lib.nvtx import annotate
from cudf._lib.scalar import Scalar
from cudf.core import column
from cudf.core.column import as_column, build_categorical_column
from cudf.utils.dtypes import (
    is_categorical_dtype,
    is_datetime_dtype,
    is_numerical_dtype,
    is_scalar,
    is_string_dtype,
    min_scalar_type,
)


class Frame(libcudf.table.Table):
    """
    Frame: A collection of Column objects with an optional index.

    Parameters
    ----------
    data : OrderedColumnDict
        An OrderedColumnDict mapping column names to Columns
    index : Table
        A Frame representing the (optional) index columns.
    """

    @classmethod
    def _from_table(cls, table):
        return cls(table._data, index=table._index)

    @classmethod
    @annotate("CONCAT", color="orange", domain="cudf_python")
    def _concat(cls, objs, axis=0, ignore_index=False):

        # shallow-copy the input DFs in case the same DF instance
        # is concatenated with itself
        objs = [f.copy(deep=False) for f in objs]

        from cudf.core.index import as_index
        from cudf.core.column.column import column_empty
        from cudf.core.column.column import build_categorical_column

        # Create a dictionary of the common, non-null columns
        def get_non_null_cols_and_dtypes(col_idxs, list_of_columns):
            # A mapping of {idx: np.dtype}
            dtypes = dict()
            # A mapping of {idx: [...columns]}, where `[...columns]`
            # is a list of columns with at least one valid value for each
            # column name across all input dataframes
            non_null_columns = dict()
            for idx in col_idxs:
                for cols in list_of_columns:
                    # Skip columns not in this frame
                    if idx >= len(cols) or cols[idx] is None:
                        continue
                    # Store the first dtype we find for a column, even if it's
                    # all-null. This ensures we always have at least one dtype
                    # for each name. This dtype will be overwritten later if a
                    # non-null Column with the same name is found.
                    if idx not in dtypes:
                        dtypes[idx] = cols[idx].dtype
                    if cols[idx].valid_count > 0:
                        if idx not in non_null_columns:
                            non_null_columns[idx] = [cols[idx]]
                        else:
                            non_null_columns[idx].append(cols[idx])
            return non_null_columns, dtypes

        def find_common_dtypes_and_categories(non_null_columns, dtypes):
            # A mapping of {idx: categories}, where `categories` is a
            # column of all the unique categorical values from each
            # categorical column across all input dataframes
            categories = dict()
            for idx, cols in non_null_columns.items():
                # default to the first non-null dtype
                dtypes[idx] = cols[0].dtype
                # If all the non-null dtypes are int/float, find a common dtype
                if all(is_numerical_dtype(col.dtype) for col in cols):
                    dtypes[idx] = np.find_common_type(
                        [col.dtype for col in cols], []
                    )
                # If all categorical dtypes, combine the categories
                elif all(is_categorical_dtype(col.dtype) for col in cols):
                    # Combine and de-dupe the categories
                    categories[idx] = (
                        cudf.concat([col.cat().categories for col in cols])
                        .to_series()
                        .drop_duplicates()
                        ._column
                    )
                    # Set the column dtype to the codes' dtype. The categories
                    # will be re-assigned at the end
                    dtypes[idx] = min_scalar_type(len(categories[idx]))
                # Otherwise raise an error if columns have different dtypes
                elif not all(
                    is_dtype_equal(c.dtype, dtypes[idx]) for c in cols
                ):
                    raise ValueError("All columns must be the same type")
            return categories

        def cast_cols_to_common_dtypes(
            col_idxs, list_of_columns, dtypes, categories
        ):
            # Cast all columns to a common dtype, assign combined categories,
            # and back-fill missing columns with all-null columns
            for idx in col_idxs:
                dtype = dtypes[idx]
                for cols in list_of_columns:
                    # If column not in this df, fill with an all-null column
                    if idx >= len(cols) or cols[idx] is None:
                        n = len(next(filter(lambda x: x is not None, cols)))
                        cols[idx] = column_empty(n, dtype, masked=True)
                    else:
                        # If column is categorical, rebase the codes with the
                        # combined categories, and cast the new codes to the
                        # min-scalar-sized dtype
                        if idx in categories:
                            cols[idx] = (
                                cols[idx]
                                .cat()
                                ._set_categories(
                                    categories[idx], is_unique=True
                                )
                                .codes
                            )
                        cols[idx] = cols[idx].astype(dtype)

        def reassign_categories(categories, cols, col_idxs):
            for name, idx in zip(cols, col_idxs):
                if idx in categories:
                    cols[name] = build_categorical_column(
                        categories=categories[idx],
                        codes=as_column(
                            cols[name].base_data, dtype=cols[name].dtype
                        ),
                        mask=cols[name].base_mask,
                        offset=cols[name].offset,
                        size=cols[name].size,
                    )

        # Get a list of the unique table column names
        names = [name for f in objs for name in f._column_names]
        names = list(OrderedDict.fromkeys(names).keys())

        # Combine the index and table columns for each Frame into a
        # list of [...index_cols, ...table_cols]. If a table is
        # missing a column, that list will have None in the slot instead
        columns = [
            ([] if ignore_index else list(f._index._data.columns))
            + [f._data[name] if name in f._data else None for name in names]
            for i, f in enumerate(objs)
        ]

        # Get a list of the combined index and table column indices
        indices = list(range(functools.reduce(max, map(len, columns))))
        # The position of the first table colum in each
        # combined index + table columns list
        first_data_column_position = len(indices) - len(names)

        # Get the non-null columns and their dtypes
        non_null_cols, dtypes = get_non_null_cols_and_dtypes(indices, columns)

        # Infer common dtypes between numeric columns
        # and combine CategoricalColumn categories
        categories = find_common_dtypes_and_categories(non_null_cols, dtypes)

        # Cast all columns to a common dtype, assign combined categories,
        # and back-fill missing columns with all-null columns
        cast_cols_to_common_dtypes(indices, columns, dtypes, categories)

        # Construct input tables with the index and data columns in the same
        # order. This strips the given index/column names and replaces the
        # names with their integer positions in the `cols` list
        tables = []
        for i, cols in enumerate(columns):
            table_cols = cols[first_data_column_position:]
            table_names = indices[first_data_column_position:]
            table = cls(data=dict(zip(table_names, table_cols)))
            if 1 == first_data_column_position:
                table._index = as_index(cols[0])
            elif first_data_column_position > 1:
                index_cols = cols[:first_data_column_position]
                index_names = indices[:first_data_column_position]
                table._index = cls(data=dict(zip(index_names, index_cols)))
            tables.append(table)

        # Concatenate the Tables
        out = cls._from_table(
            libcudf.concat.concat_tables(tables, ignore_index=ignore_index)
        )

        # Reassign the categories for any categorical table cols
        reassign_categories(
            categories, out._data, indices[first_data_column_position:]
        )

        # Reassign the categories for any categorical index cols
        reassign_categories(
            categories, out._index._data, indices[:first_data_column_position]
        )

        # Reassign index and column names
        if isinstance(objs[0].columns, pd.MultiIndex):
            out.columns = objs[0].columns
        else:
            out.columns = names

        out._index.name = objs[0]._index.name
        out._index.names = objs[0]._index.names

        return out

    def _get_columns_by_label(self, labels, downcast=False):
        """
        Returns columns of the Frame specified by `labels`

        If downcast is True, try and downcast from a DataFrame to a Series
        """
        new_data = self._data.get_by_label(labels)
        if downcast:
            if is_scalar(labels):
                nlevels = 1
            elif isinstance(labels, tuple):
                nlevels = len(labels)
            if self._data.multiindex is False or nlevels == self._data.nlevels:
                return self._constructor_sliced(
                    new_data, name=labels, index=self.index
                )
        return self._constructor(
            new_data, columns=new_data.to_pandas_index(), index=self.index
        )

    def _get_columns_by_index(self, indices):
        """
        Returns columns of the Frame specified by `labels`

        """
        data = self._data.get_by_index(indices)
        return self._constructor(
            data, columns=data.to_pandas_index(), index=self.index
        )

    def _gather(self, gather_map, keep_index=True):
        if not pd.api.types.is_integer_dtype(gather_map.dtype):
            gather_map = gather_map.astype("int32")
        result = self.__class__._from_table(
            libcudf.copying.gather(
                self, as_column(gather_map), keep_index=keep_index
            )
        )
        result._copy_categories(self)
        return result

    def _hash(self, initial_hash_values=None):
        return libcudf.hash.hash(self, initial_hash_values)

    def _hash_partition(
        self, columns_to_hash, num_partitions, keep_index=True
    ):
        output, offsets = libcudf.hash.hash_partition(
            self, columns_to_hash, num_partitions, keep_index
        )
        output = self.__class__._from_table(output)
        output._copy_categories(self, include_index=keep_index)
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

    def _scatter(self, key, value):
        result = self._from_table(libcudf.copying.scatter(value, key, self))

        result._copy_categories(self)
        return result

    def _empty_like(self, keep_index=True):
        result = self._from_table(
            libcudf.copying.table_empty_like(self, keep_index)
        )

        result._copy_categories(self, include_index=keep_index)
        return result

    def _slice(self, arg):
        """
       _slice : slice the frame as per the arg

       Parameters
       ----------
       arg : should always be of type slice and doesn't handle step

       """
        from cudf.core.index import RangeIndex

        num_rows = len(self)
        if num_rows == 0:
            return self
        start, stop, stride = arg.indices(num_rows)

        # This is just to handle RangeIndex type, stop
        # it from materializing unnecessarily
        keep_index = True
        if self.index is not None and isinstance(self.index, RangeIndex):
            keep_index = False

        if start < 0:
            start = start + num_rows
        if stop < 0:
            stop = stop + num_rows

        if start > stop and (stride is None or stride == 1):
            return self._empty_like(keep_index)
        else:
            start = len(self) if start > num_rows else start
            stop = len(self) if stop > num_rows else stop

            if stride is not None and stride != 1:
                return self._gather(
                    cupy.arange(start, stop=stop, step=stride, dtype=np.int32)
                )
            else:
                result = self._from_table(
                    libcudf.copying.table_slice(
                        self, [start, stop], keep_index
                    )[0]
                )

                result._copy_categories(self, include_index=keep_index)
                # Adding index of type RangeIndex back to
                # result
                if keep_index is False and self.index is not None:
                    result.index = self.index[start:stop]
                result.columns = self.columns
                return result

    def _normalize_scalars(self, other):
        """
        Try to normalizes scalar values as per self dtype
        """
        if (
            other is not None
            and (isinstance(other, float) and not np.isnan(other))
        ) and (self.dtype.type(other) != other):
            raise TypeError(
                "Cannot safely cast non-equivalent {} to {}".format(
                    type(other).__name__, self.dtype.name
                )
            )

        return (
            self.dtype.type(other)
            if (
                other is not None
                and (isinstance(other, float) and not np.isnan(other))
            )
            else other
        )

    def _normalize_columns_and_scalars_type(self, other):
        """
        Try to normalize the other's dtypes as per self.

        Parameters
        ----------

        self : Can be a DataFrame or Series or Index
        other : Can be a DataFrame, Series, Index, Array
            like object or a scalar value

            if self is DataFrame, other can be only a
            scalar or array like with size of number of columns
            in DataFrame or a DataFrame with same dimenstion

            if self is Series, other can be only a scalar or
            a series like with same length as self

        Returns:
        --------
        A dataframe/series/list/scalar form of normalized other
        """
        if isinstance(self, cudf.DataFrame) and isinstance(
            other, cudf.DataFrame
        ):
            return [
                other[self_col].astype(self._data[self_col].dtype)._column
                for self_col in self._data.names
            ]

        elif isinstance(self, (cudf.Series, cudf.Index)) and not is_scalar(
            other
        ):
            other = as_column(other)
            return other.astype(self.dtype)

        else:
            # Handles scalar or list/array like scalars
            if isinstance(self, (cudf.Series, cudf.Index)) and is_scalar(
                other
            ):
                return self._normalize_scalars(other)

            elif (self, cudf.DataFrame):
                out = []
                if is_scalar(other):
                    other = [other for i in range(len(self._data.names))]
                out = [
                    self[in_col_name]._normalize_scalars(sclr)
                    for in_col_name, sclr in zip(self._data.names, other)
                ]

                return out
            else:
                raise ValueError(
                    "Inappropriate input {} and other {} combination".format(
                        type(self), type(other)
                    )
                )

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
            dataframe with same dimention as self.

            Series expects only scalar or series like with same length
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        Same type as caller

        Examples:
        ---------
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
        2    null
        3    null
        4    null
        dtype: int64
        """

        if isinstance(self, cudf.DataFrame):
            if hasattr(cond, "__cuda_array_interface__"):
                cond = self.from_gpu_matrix(
                    cond, columns=self._data.names, index=self.index
                )
            elif not isinstance(cond, cudf.DataFrame):
                cond = self.from_pandas(pd.DataFrame(cond))

            common_cols = set(self._data.names).intersection(
                set(cond._data.names)
            )
            if len(common_cols) > 0:
                # If `self` and `cond` are having unequal index,
                # then re-index `cond`.
                if len(self.index) != len(cond.index) or any(
                    self.index != cond.index
                ):
                    cond = cond.reindex(self.index)
            else:
                if cond.shape != self.shape:
                    raise ValueError(
                        """Array conditional must be same shape as self"""
                    )
                # Setting `self` column names to `cond`
                # as `cond` has no column names.
                cond.columns = self.columns

            other = self._normalize_columns_and_scalars_type(other)
            out_df = cudf.DataFrame(index=self.index)
            if len(self._columns) != len(other):
                raise ValueError(
                    """Replacement list length or number of dataframe columns
                    should be equal to Number of columns of dataframe"""
                )

            for column_name, other_column in zip(self._data.names, other):
                input_col = self._data[column_name]
                if column_name in cond._data:
                    if is_categorical_dtype(input_col.dtype):
                        if np.isscalar(other_column):
                            try:
                                other_column = input_col._encode(other_column)
                            except ValueError:
                                # When other is not present in categories,
                                # fill with Null.
                                other_column = None
                        elif hasattr(other_column, "codes"):
                            other_column = other_column.codes
                        input_col = input_col.codes

                    result = libcudf.copying.copy_if_else(
                        input_col, other_column, cond._data[column_name]
                    )

                    if is_categorical_dtype(self._data[column_name].dtype):
                        result = build_categorical_column(
                            categories=self._data[column_name].categories,
                            codes=as_column(
                                result.base_data, dtype=result.dtype
                            ),
                            mask=result.base_mask,
                            size=result.size,
                            offset=result.offset,
                            ordered=self._data[column_name].ordered,
                        )
                else:
                    from cudf._lib.null_mask import MaskState, create_null_mask

                    out_mask = create_null_mask(
                        len(input_col), state=MaskState.ALL_NULL
                    )
                    result = input_col.set_mask(out_mask)
                out_df[column_name] = self[column_name].__class__(result)

            return self._mimic_inplace(out_df, inplace=inplace)

        else:

            if isinstance(other, cudf.DataFrame):
                raise NotImplementedError(
                    "cannot align with a higher dimensional Frame"
                )

            other = self._normalize_columns_and_scalars_type(other)

            cond = as_column(cond)
            if len(cond) != len(self):
                raise ValueError(
                    """Array conditional must be same shape as self"""
                )
            input_col = self._data[self.name]
            if is_categorical_dtype(input_col.dtype):
                if np.isscalar(other):
                    try:
                        other = input_col._encode(other)
                    except ValueError:
                        # When other is not present in categories,
                        # fill with Null.
                        other = None
                elif hasattr(other, "codes"):
                    other = other.codes

                input_col = input_col.codes

            result = libcudf.copying.copy_if_else(input_col, other, cond)

            if is_categorical_dtype(self.dtype):
                result = build_categorical_column(
                    categories=self._data[self.name].categories,
                    codes=as_column(result.base_data, dtype=result.dtype),
                    mask=result.base_mask,
                    size=result.size,
                    offset=result.offset,
                    ordered=self._data[self.name].ordered,
                )

            if isinstance(self, cudf.Index):
                from cudf.core.index import as_index

                result = as_index(result, name=self.name)
            else:
                result = self._copy_construct(data=result)

            return self._mimic_inplace(result, inplace=inplace)

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
            dataframe with same dimention as self.

            Series expects only scalar or series like with same length
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        Same type as caller

        Examples:
        ---------
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
        0    null
        1    null
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

        output_table, output_offsets = libcudf.partitioning.partition(
            self, scatter_map, npartitions, keep_index
        )

        # due to the split limitation mentioned
        # here: https://github.com/rapidsai/cudf/issues/4607
        # we need to remove first & last elements in offsets.
        # TODO: Remove this after the above issue is fixed.
        output_offsets = output_offsets[1:-1]

        result = libcudf.copying.table_split(
            output_table, output_offsets, keep_index=keep_index
        )

        result = [self.__class__._from_table(tbl) for tbl in result]

        for frame in result:
            frame._copy_categories(self, include_index=keep_index)

        if npartitions:
            for i in range(npartitions - len(result)):
                result.append(self._empty_like(keep_index))

        return result

    def dropna(self, axis=0, how="any", subset=None, thresh=None):
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
        subset : list, optional
            List of columns to consider when dropping rows (all columns
            are considered by default). Alternatively, when dropping
            columns, subset is a list of rows to consider.
        thresh: int, optional
            If specified, then drops every row (or column) containing
            less than `thresh` non-null values


        Returns
        -------
        Copy of the DataFrame with rows/columns containing nulls dropped.
        """
        if axis == 0:
            return self._drop_na_rows(how=how, subset=subset, thresh=thresh)
        else:
            return self._drop_na_columns(how=how, subset=subset, thresh=thresh)

    def _drop_na_rows(self, how="any", subset=None, thresh=None):
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
            raise KeyError("columns {!r} do not exist".format(diff))
        subset_cols = [
            name for name, col in self._data.items() if name in subset
        ]
        if len(subset_cols) == 0:
            return self.copy(deep=True)
        result = self.__class__._from_table(
            libcudf.stream_compaction.drop_nulls(
                self, how=how, keys=subset, thresh=thresh
            )
        )
        result._copy_categories(self)
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
            if (len(df[col]) - df[col].null_count) < thresh:
                continue
            out_cols.append(col)

        return self[out_cols]

    def _apply_boolean_mask(self, boolean_mask):
        """
        Applies boolean mask to each row of `self`,
        rows corresponding to `False` is dropped
        """
        result = self.__class__._from_table(
            libcudf.stream_compaction.apply_boolean_mask(
                self, as_column(boolean_mask)
            )
        )
        result._copy_categories(self)
        return result

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

        result = self.__class__._from_table(
            libcudf.quantiles.quantiles(
                self,
                q,
                interpolation,
                is_sorted,
                column_order,
                null_precedence,
            )
        )

        result._copy_categories(self)
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
        axis : {0 or 'index', 1 or 'columns'}, default 0
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
            raise KeyError(na_option)

        # TODO code for selecting numeric columns
        source = self
        if numeric_only:
            warnings.warn("numeric_only=True is not implemented yet")

        out_rank_table = libcudf.sort.rank_columns(
            source, method_enum, na_option, ascending, pct
        )

        return self._from_table(out_rank_table).astype(np.float64)

    def repeat(self, repeats, axis=None):
        """Repeats elements consecutively

        Parameters
        ----------
        repeats : int, array, numpy array, or Column
            the number of times to repeat each element

        Example
        -------
        >>> import cudf as cudf
        >>> s = cudf.Series([0, 2]) # or DataFrame
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
        >>>
        """
        if axis is not None:
            raise NotImplementedError(
                "Only axis=`None` supported at this time."
            )

        return self._repeat(repeats)

    def _repeat(self, count):
        if is_scalar(count):
            count = Scalar(count)
        else:
            count = as_column(count)

        result = self.__class__._from_table(
            libcudf.filling.repeat(self, count)
        )

        result._copy_categories(self)
        return result

    def _fill(self, fill_values, begin, end, inplace):
        col_and_fill = zip(self._columns, fill_values)

        if not inplace:
            data_columns = (c._fill(v, begin, end) for (c, v) in col_and_fill)
            data = zip(self._column_names, data_columns)
            return self.__class__._from_table(Frame(data, self._index))

        for (c, v) in col_and_fill:
            c.fill(v, begin, end, inplace=True)

        return self

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """Shift values by `periods` positions.
        """
        assert axis in (None, 0) and freq is None
        return self._shift(periods)

    def _shift(self, offset, fill_value=None):
        data_columns = (col.shift(offset, fill_value) for col in self._columns)
        data = zip(self._column_names, data_columns)
        return self.__class__._from_table(Frame(data, self._index))

    def drop_duplicates(self, subset=None, keep="first", nulls_are_equal=True):
        """
        Drops rows in frame as per duplicate rows in `subset` columns from
        self.

        subset : list, optional
            List of columns to consider when dropping rows.
        keep : ["first", "last", False] first will keep first of duplicate,
            last will keep last of the duplicate and False drop all
            duplicate
        nulls_are_equal: null elements are considered equal to other null
            elements
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
            raise KeyError("columns {!r} do not exist".format(diff))
        subset_cols = [name for name in self._column_names if name in subset]
        if len(subset_cols) == 0:
            return self.copy(deep=True)

        result = self._from_table(
            libcudf.stream_compaction.drop_duplicates(
                self, keys=subset, keep=keep, nulls_are_equal=nulls_are_equal
            )
        )

        result._copy_categories(self)
        return result

    def replace(self, to_replace, replacement):
        copy_data = self._data.copy()

        for name, col in copy_data.items():
            if not (to_replace is None and replacement is None):
                try:
                    (
                        col_all_nan,
                        col_replacement,
                        col_to_replace,
                    ) = _get_replacement_values(
                        to_replace=to_replace,
                        replacement=replacement,
                        col_name=name,
                        column=col,
                    )

                    copy_data[name] = col.find_and_replace(
                        col_to_replace, col_replacement, col_all_nan
                    )
                except KeyError:
                    # Donot change the copy_data[name]
                    pass

        result = self._from_table(Frame(copy_data, self.index))
        return result

    def _copy_categories(self, other, include_index=True):
        """
        Utility that copies category information from `other`
        to `self`.
        """
        for name, col, other_col in zip(
            self._column_names, self._columns, other._columns
        ):
            if is_categorical_dtype(other_col) and not is_categorical_dtype(
                col
            ):
                self._data[name] = build_categorical_column(
                    categories=other_col.categories,
                    codes=as_column(col.base_data, dtype=col.dtype),
                    mask=col.base_mask,
                    ordered=other_col.ordered,
                    size=col.size,
                    offset=col.offset,
                )
        if include_index:
            from cudf.core.index import RangeIndex

            # include_index will still behave as False
            # incase of self._index being a RangeIndex
            if (self._index is not None) and (
                not isinstance(self._index, RangeIndex)
            ):
                self._index._copy_categories(other._index)
        return self

    def _unaryop(self, op):
        data_columns = (col.unary_operator(op) for col in self._columns)
        data = zip(self._column_names, data_columns)
        return self.__class__._from_table(Frame(data, self._index))

    def isnull(self):
        """Identify missing values.
        """
        data_columns = (col.isnull() for col in self._columns)
        data = zip(self._column_names, data_columns)
        return self.__class__._from_table(Frame(data, self._index))

    def isna(self):
        """Identify missing values. Alias for `isnull`
        """
        return self.isnull()

    def notnull(self):
        """Identify non-missing values.
        """
        data_columns = (col.notnull() for col in self._columns)
        data = zip(self._column_names, data_columns)
        return self.__class__._from_table(Frame(data, self._index))

    def notna(self):
        """Identify non-missing values. Alias for `notnull`.
        """
        return self.notnull()

    def interleave_columns(self):
        """
        Interleave Series columns of a table into a single column.

        Converts the column major table `cols` into a row major column.
        Parameters
        ----------
        cols : input Table containing columns to interleave.

        Example
        -------
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

        Example
        -------
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
        result = self.__class__._from_table(libcudf.reshape.tile(self, count))
        result._copy_categories(self)
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
        """
        # Call libcudf++ search_sorted primitive
        from cudf.utils.dtypes import is_scalar

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

    def _get_sorted_inds(self, ascending=True, na_position="last"):
        """
        Sort by the values.

        Parameters
        ----------
        ascending : bool or list of bool, default True
            If True, sort values in ascending order, otherwise descending.
        na_position : {‘first’ or ‘last’}, default ‘last’
            Argument ‘first’ puts NaNs at the beginning, ‘last’ puts NaNs
            at the end.
        Returns
        -------
        out_column_inds : cuDF Column of indices sorted based on input

        Difference from pandas:
        * Support axis='index' only.
        * Not supporting: inplace, kind
        * Ascending can be a list of bools to control per column
        """

        # This needs to be updated to handle list of bools for ascending
        if ascending is True:
            if na_position == "last":
                na_position = 0
            elif na_position == "first":
                na_position = 1
        elif ascending is False:
            if na_position == "last":
                na_position = 1
            elif na_position == "first":
                na_position = 0
        else:
            warnings.warn(
                "When using a sequence of booleans for `ascending`, "
                "`na_position` flag is not yet supported and defaults to "
                "treating nulls as greater than all numbers"
            )
            na_position = 0

        # If given a scalar need to construct a sequence of length # of columns
        if np.isscalar(ascending):
            ascending = [ascending] * self._num_columns

        return libcudf.sort.order_by(self, ascending, na_position)

    def sin(self):
        return self._unaryop("sin")

    def cos(self):
        return self._unaryop("cos")

    def tan(self):
        return self._unaryop("tan")

    def asin(self):
        return self._unaryop("asin")

    def acos(self):
        return self._unaryop("acos")

    def atan(self):
        return self._unaryop("atan")

    def exp(self):
        return self._unaryop("exp")

    def log(self):
        return self._unaryop("log")

    def sqrt(self):
        return self._unaryop("sqrt")

    @staticmethod
    def _validate_merge_cfg(
        lhs,
        rhs,
        left_on,
        right_on,
        on,
        how,
        left_index=False,
        right_index=False,
        lsuffix=None,
        rsuffix=None,
    ):
        """
        Error for various combinations of merge input parameters
        """
        if isinstance(
            lhs.index, cudf.core.multiindex.MultiIndex
        ) or isinstance(rhs.index, cudf.core.multiindex.MultiIndex):
            raise TypeError("MultiIndex joins not yet supported.")

        len_left_on = len(left_on) if left_on is not None else 0
        len_right_on = len(right_on) if right_on is not None else 0

        # must actually support the requested merge type
        if how not in ["left", "inner", "outer", "leftanti", "leftsemi"]:
            raise NotImplementedError(
                "{!r} merge not supported yet".format(how)
            )

        # Passing 'on' with 'left_on' or 'right_on' is potentially ambiguous
        if on:
            if left_on or right_on:
                raise ValueError(
                    'Can only pass argument "on" OR "left_on" '
                    'and "right_on", not a combination of both.'
                )

        # Require same total number of columns to join on in both operands
        if not (len_left_on + left_index) == (len_right_on + right_index):
            raise ValueError(
                "Merge operands must have same number of join key columns"
            )

        # If nothing specified, must have common cols to use implicitly
        same_named_columns = set(lhs._data.keys()) & set(rhs._data.keys())
        if not (left_index or right_index):
            if not (left_on or right_on):
                if len(same_named_columns) == 0:
                    raise ValueError("No common columns to perform merge on")

        for name in same_named_columns:
            if not (
                name in left_on
                and name in right_on
                and (left_on.index(name) == right_on.index(name))
            ):
                if not (lsuffix or rsuffix):
                    raise ValueError(
                        "there are overlapping columns but "
                        "lsuffix and rsuffix are not defined"
                    )

        if on:
            on_keys = [on] if not isinstance(on, list) else on
            for key in on_keys:
                if not (key in lhs._data.keys() and key in rhs._data.keys()):
                    raise KeyError("Key {} not in both operands".format(on))
        else:
            for key in left_on:
                if key not in lhs._data.keys():
                    raise KeyError('Key "{}" not in left operand'.format(key))
            for key in right_on:
                if key not in rhs._data.keys():
                    raise KeyError('Key "{}" not in right operand'.format(key))

    def _merge(
        self,
        right,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        lsuffix,
        rsuffix,
        how,
        method,
        sort=False,
    ):

        lhs = self
        rhs = right

        if left_on is None:
            left_on = []
        if right_on is None:
            right_on = []

        # Making sure that the "on" arguments are list of column names
        if on:
            on = [on] if isinstance(on, str) else list(on)
        if left_on:
            left_on = [left_on] if isinstance(left_on, str) else list(left_on)
        if right_on:
            right_on = (
                [right_on] if isinstance(right_on, str) else list(right_on)
            )

        self._validate_merge_cfg(
            self,
            right,
            left_on,
            right_on,
            on,
            how,
            left_index=left_index,
            right_index=right_index,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
        )

        if on:
            left_on = right_on = on

        same_named_columns = set(lhs._data.keys()) & set(rhs._data.keys())
        if not (left_on or right_on) and not (left_index and right_index):
            left_on = right_on = list(same_named_columns)

        no_suffix_cols = []
        for name in same_named_columns:
            if left_on is not None and right_on is not None:
                if name in left_on and name in right_on:
                    if left_on.index(name) == right_on.index(name):
                        no_suffix_cols.append(name)

        for name in same_named_columns:
            if name not in no_suffix_cols:
                lhs.rename({name: "%s%s" % (name, lsuffix)}, inplace=True)
                rhs.rename({name: "%s%s" % (name, rsuffix)}, inplace=True)
                if name in left_on:
                    left_on[left_on.index(name)] = "%s%s" % (name, lsuffix)
                if name in right_on:
                    right_on[right_on.index(name)] = "%s%s" % (name, rsuffix)

        categorical_dtypes = {}
        for name, col in itertools.chain(lhs._data.items(), rhs._data.items()):
            if is_categorical_dtype(col):
                categorical_dtypes[name] = col.dtype

        # Save the order of the original column names for preservation later
        org_names = list(itertools.chain(lhs._data.keys(), rhs._data.keys()))

        # If neither left_index or right_index specified, that data won't
        # be carried through the join. We'll get a new RangeIndex afterwards
        lhs_full_view = False
        rhs_full_view = False
        if left_index:
            lhs_full_view = True
        if right_index:
            rhs_full_view = True

        # potentially do an implicit typecast
        (lhs, rhs, to_categorical) = self._typecast_before_merge(
            lhs, rhs, left_on, right_on, left_index, right_index, how
        )

        gdf_result = libcudf.join.join(
            lhs,
            rhs,
            left_on,
            right_on,
            how,
            method,
            left_index=lhs_full_view,
            right_index=rhs_full_view,
        )

        gdf_data = list(gdf_result._data.items())

        result = []
        cat_codes = []
        for org_name in org_names:
            for i in range(len(gdf_data)):
                if gdf_data[i][0] == org_name:
                    result.append(gdf_data.pop(i))
                    break
        for cat_name in to_categorical:
            for i in range(len(gdf_data)):
                if gdf_data[i][0] == cat_name + "_codes":
                    cat_codes.append(gdf_data.pop(i))
        assert len(gdf_data) == 0
        cat_codes = dict(cat_codes)

        # Build a new data frame based on the merged columns from GDF
        to_frame_data = OrderedDict()
        for name, col in result:
            if is_string_dtype(col):
                to_frame_data[name] = col
            elif is_categorical_dtype(categorical_dtypes.get(name, col.dtype)):

                dtype = categorical_dtypes.get(name, col.dtype)
                to_frame_data[name] = column.build_categorical_column(
                    categories=dtype.categories,
                    codes=cat_codes.get(str(name) + "_codes", col),
                    mask=col.base_mask,
                    size=col.size,
                    offset=col.offset,
                    ordered=dtype.ordered,
                )
            else:

                to_frame_data[name] = column.build_column(
                    col.base_data,
                    dtype=categorical_dtypes.get(name, col.dtype),
                    mask=col.base_mask,
                    offset=col.offset,
                    size=col.size,
                )
        gdf_result._data = to_frame_data

        to_return = self.__class__._from_table(gdf_result)

        # If sort=True, Pandas would sort on the key columns in the
        # same order as given in 'on'. If the indices are used as
        # keys, the index will be sorted. If one index is specified,
        # the key column on the other side will be used to sort.
        # If no index is specified, return a new RangeIndex
        if sort:
            to_sort = self.__class__()
            if left_index and right_index:
                by = list(to_return._index._data.columns)
                if left_on and right_on:
                    by += list(to_return[left_on]._data.columns)
            elif left_index:
                by = list(to_return[right_on]._data.columns)
            elif right_index:
                by = list(to_return[left_on]._data.columns)
            else:
                # left_on == right_on, or different names but same columns
                # in both cases we can sort by either
                by = list(to_return[left_on]._data.columns)
            for i, col in enumerate(by):
                to_sort[i] = col
            inds = to_sort.argsort()
            to_return = to_return.take(
                inds, keep_index=(left_index or right_index)
            )
            return to_return
        else:
            return to_return

    def _typecast_before_merge(
        self, lhs, rhs, left_on, right_on, left_index, right_index, how
    ):
        def casting_rules(lhs, rhs, dtype_l, dtype_r, how):
            cast_warn = "can't safely cast column {} from {} with type \
                         {} to {}, upcasting to {}"
            ctgry_err = "can't implicitly cast column {0} to categories \
                         from {1} during {1} join"

            rtn = None
            if pd.api.types.is_dtype_equal(dtype_l, dtype_r):
                rtn = dtype_l
            elif is_categorical_dtype(dtype_l) and is_categorical_dtype(
                dtype_r
            ):
                raise TypeError("Left and right categories must be the same.")
            elif how == "left":

                check_col = rhs._data[rcol].fillna(0)
                if not check_col.can_cast_safely(dtype_l):
                    rtn = casting_rules(lhs, rhs, dtype_l, dtype_r, "inner")
                    warnings.warn(
                        cast_warn.format(rcol, "right", dtype_r, dtype_l, rtn)
                    )
                else:
                    rtn = dtype_l
            elif how == "right":
                check_col = lhs._data[lcol].fillna(0)
                if not check_col.can_cast_safely(dtype_r):
                    rtn = casting_rules(lhs, rhs, dtype_l, dtype_r, "inner")
                    warnings.warn(
                        cast_warn.format(lcol, "left", dtype_l, dtype_r, rtn)
                    )
                else:
                    rtn = dtype_r

            elif is_categorical_dtype(dtype_l):
                if how == "right":
                    raise ValueError(ctgry_err.format(rcol, "right"))

                rtn = lhs[lcol].cat.categories.dtype
                to_categorical.append(lcol)
                lhs[lcol + "_codes"] = lhs[lcol].cat.codes
            elif is_categorical_dtype(dtype_r):
                if how == "left":
                    raise ValueError(ctgry_err.format(lcol, "left"))
                rtn = rhs[rcol].cat.categories.dtype
                to_categorical.append(rcol)
                rhs[rcol + "_codes"] = rhs[rcol].cat.codes
            elif how in ["inner", "outer"]:
                if (np.issubdtype(dtype_l, np.number)) and (
                    np.issubdtype(dtype_r, np.number)
                ):
                    if dtype_l.kind == dtype_r.kind:
                        # both ints or both floats
                        rtn = max(dtype_l, dtype_r)
                    else:
                        rtn = np.find_common_type([], [dtype_l, dtype_r])
                elif is_datetime_dtype(dtype_l) and is_datetime_dtype(dtype_r):
                    rtn = max(dtype_l, dtype_r)
            return rtn

        if left_index or right_index:
            if left_index and right_index:
                to_dtype = casting_rules(
                    lhs.index, rhs.index, lhs.index.dtype, rhs.index.dtype, how
                )
            elif left_index:
                to_dtype = lhs.index.dtype
            elif right_index:
                to_dtype = rhs.index.dtype
            lhs.index = lhs.index.astype(to_dtype)
            rhs.index = rhs.index.astype(to_dtype)
            return lhs, rhs, []

        left_on = sorted(left_on)
        right_on = sorted(right_on)
        to_categorical = []
        for lcol, rcol in zip(left_on, right_on):
            if (lcol not in lhs._data) or (rcol not in rhs._data):
                # probably wrong columns specified, let libcudf error
                continue

            dtype_l = lhs._data[lcol].dtype
            dtype_r = rhs._data[rcol].dtype
            if pd.api.types.is_dtype_equal(dtype_l, dtype_r):
                continue

            to_dtype = casting_rules(lhs, rhs, dtype_l, dtype_r, how)

            if to_dtype is not None:
                lhs[lcol] = lhs[lcol].astype(to_dtype)
                rhs[rcol] = rhs[rcol].astype(to_dtype)

        return lhs, rhs, to_categorical

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


def _get_replacement_values(to_replace, replacement, col_name, column):
    from cudf.utils import utils
    from pandas.api.types import is_dict_like

    all_nan = False

    if is_dict_like(to_replace) and replacement is None:
        replacement = list(to_replace.values())
        to_replace = list(to_replace.keys())
    elif not is_scalar(to_replace):
        if is_scalar(replacement):
            all_nan = replacement is None
            if all_nan:
                replacement = [replacement] * len(to_replace)
            # Do not broadcast numeric dtypes
            elif pd.api.types.is_numeric_dtype(column.dtype):
                if len(to_replace) > 0:
                    replacement = [replacement]
                else:
                    # If to_replace is empty, replacement has to be empty.
                    replacement = []
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

    if is_dict_like(to_replace) and is_dict_like(replacement):
        replacement = replacement[col_name]
        to_replace = to_replace[col_name]

        if is_scalar(replacement):
            replacement = [replacement]
        if is_scalar(to_replace):
            to_replace = [to_replace]

    if isinstance(replacement, list):
        all_nan = replacement.count(None) == len(replacement)
    return all_nan, replacement, to_replace
