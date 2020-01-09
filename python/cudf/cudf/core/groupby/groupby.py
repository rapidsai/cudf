# Copyright (c) 2018, NVIDIA CORPORATION.

import collections
import itertools
import pickle

import cudf
import cudf._lib as libcudf
from cudf import MultiIndex
from cudf.core.column import deserialize_columns, serialize_columns
from cudf.utils.dtypes import is_scalar


def columns_from_dataframe(df):
    cols = list(df._data.values())
    # strip column names
    for col in cols:
        col.name = None
    return cols


def dataframe_from_columns(cols, index_cols=None, index=None, columns=None):
    df = cudf.DataFrame(dict(zip(range(len(cols)), cols)), index=index)
    if columns is not None:
        df.columns = columns
    return df


class _Groupby(object):
    def sum(self):
        return self._apply_aggregation("sum")

    def min(self):
        return self._apply_aggregation("min")

    def max(self):
        return self._apply_aggregation("max")

    def mean(self):
        return self._apply_aggregation("mean")

    def count(self):
        return self._apply_aggregation("count")

    def agg(self, func):
        return self._apply_aggregation(func)

    def size(self):
        from cudf.core.column import column_empty

        nrows = len(self._groupby.obj)
        data = cudf.Series(column_empty(nrows, "int8", masked=False))
        return data.groupby(self._groupby.key_columns).count()

    def serialize(self):
        header, frames = self._groupby.serialize()
        header["type"] = pickle.dumps(type(self))
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        groupby_type = pickle.loads(header["type"])
        _groupby = _GroupbyHelper.deserialize(header, frames)
        by = None
        if _groupby.level is None:
            by = _groupby.by
        return groupby_type(
            _groupby.obj,
            by=by,
            level=_groupby.level,
            sort=_groupby.sort,
            as_index=_groupby.as_index,
        )


class SeriesGroupBy(_Groupby):
    def __init__(
        self,
        sr,
        by=None,
        level=None,
        method="hash",
        sort=True,
        as_index=None,
        dropna=True,
    ):
        self._sr = sr
        if as_index not in (True, None):
            raise TypeError("as_index must be True for SeriesGroupBy")
        self._groupby = _GroupbyHelper(
            obj=self._sr, by=by, level=level, sort=sort, dropna=dropna
        )

    def _apply_aggregation(self, agg):
        return self._groupby.compute_result(agg)

    def quantile(self, q=0.5, interpolation="linear"):
        raise NotImplementedError

    def std(self, ddof=1):
        raise NotImplementedError


class DataFrameGroupBy(_Groupby):
    def __init__(
        self,
        df,
        by=None,
        as_index=True,
        level=None,
        sort=True,
        method="hash",
        dropna=True,
    ):

        if by is not None and level is None:
            df, by = _align_by_and_df(df, by)
        self._df = df
        self._groupby = _GroupbyHelper(
            obj=self._df,
            by=by,
            as_index=as_index,
            level=level,
            sort=sort,
            dropna=dropna,
        )

    def _apply_aggregation(self, agg):
        """
        Applies the aggregation function(s) ``agg`` on all columns
        """
        result = self._groupby.compute_result(agg)
        libcudf.nvtx.nvtx_range_pop()
        return result

    def __getitem__(self, arg):
        if is_scalar(arg):
            return self.__getattr__(arg)
        else:
            arg = list(arg)
            by = None
            if self._groupby.level is None:
                by = self._groupby.key_columns

            return self._df[arg].groupby(
                by=by,
                level=self._groupby.level,
                as_index=self._groupby.as_index,
                sort=self._groupby.sort,
                dropna=self._groupby.dropna,
            )

    def __getattr__(self, key):
        if key == "_df":
            # this guards against RecursionError during pickling/copying
            raise AttributeError()
        if key in self._df.columns:
            by_list = []

            for by_name, by in zip(
                self._groupby.key_names, self._groupby.key_columns
            ):
                by_list.append(cudf.Series(by, name=by_name))
            return self._df[key].groupby(
                by_list,
                as_index=self._groupby.as_index,
                sort=self._groupby.sort,
                dropna=self._groupby.dropna,
            )
        raise AttributeError(
            "'DataFrameGroupBy' object has no attribute " "'{}'".format(key)
        )

    def quantile(self, q=0.5, interpolation="linear"):
        # Get Key_cols from _GroupbyHelper. It's generated at init.
        key_cols = [
            sr._column
            if isinstance(sr, cudf.Series)
            else cudf.Series(sr)._column
            for sr in self._groupby.key_columns
        ]
        # Do the things that'll make it generate the value columns
        self._groupby.normalize_agg("quantile")
        self._groupby.normalize_values()
        # Get the value_columns
        val_cols = self._groupby.value_columns

        out_key_columns, out_value_columns = libcudf.quantile.group_quantile(
            key_cols, val_cols, q, interpolation
        )
        return self._groupby.construct_result(
            out_key_columns, out_value_columns
        )

    def std(self, ddof=1):
        # Get Key_cols from _GroupbyHelper. It's generated at init.
        key_cols = [
            sr._column
            if isinstance(sr, cudf.Series)
            else cudf.Series(sr)._column
            for sr in self._groupby.key_columns
        ]
        # Do the things that'll make it generate the value columns
        self._groupby.normalize_agg("std")
        self._groupby.normalize_values()
        # Get the value_columns
        val_cols = self._groupby.value_columns

        out_key_columns, out_value_columns = libcudf.reduce.group_std(
            key_cols, val_cols, ddof
        )
        return self._groupby.construct_result(
            out_key_columns, out_value_columns
        )


class _GroupbyHelper(object):

    NAMED_AGGS = ("sum", "mean", "min", "max", "count", "quantile", "std")

    def __init__(
        self, obj, by=None, level=None, as_index=True, sort=None, dropna=True
    ):
        """
        Helper class for both SeriesGroupBy and DataFrameGroupBy classes.
        """
        self.obj = obj
        if by is None and level is None:
            raise TypeError("Either 'by' or 'level' must be provided")
        if level is not None:
            if by is not None:
                raise TypeError("Cannot use both 'by' and 'level'")
            by = self.get_by_from_level(level)
        self.level = level
        self.by = by
        self.as_index = as_index
        self.sort = sort
        self.dropna = dropna
        self.normalize_keys()

    def serialize(self):
        header = {}
        frames = []

        header["obj"], obj_frames = self.obj.serialize()
        header["obj_type"] = pickle.dumps(type(self.obj))
        header["obj_frame_count"] = len(obj_frames)
        frames.extend(obj_frames)

        header["key_names"] = self.key_names
        header["level"] = self.level
        header["df_key_names"] = self.df_key_names
        header["as_index"] = self.as_index
        header["sort"] = self.sort

        key_columns = [
            sr._column
            if isinstance(sr, cudf.Series)
            else cudf.Series(sr)._column
            for sr in self.key_columns
        ]
        key_columns_header, key_columns_frames = serialize_columns(key_columns)

        header["key_columns"] = key_columns_header
        frames.extend(key_columns_frames)

        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        obj_frames = frames[: header["obj_frame_count"]]

        obj_type = pickle.loads(header["obj_type"])
        obj = obj_type.deserialize(header["obj"], obj_frames)

        as_index = header["as_index"]
        sort = header["sort"]
        df_key_names = header["df_key_names"]
        level = header["level"]

        key_column_frames = frames[header["obj_frame_count"] :]
        key_columns = deserialize_columns(
            header["key_columns"], key_column_frames
        )

        for col_name, col in zip(header["key_names"], key_columns):
            col.name = col_name

        by = None
        if level is None:
            by = df_key_names
            by.extend(key_columns[len(df_key_names) : :])

        gby = cls(obj, by=by, level=level, as_index=as_index, sort=sort)

        return gby

    def get_by_from_level(self, level):
        """
        Converts the ``level`` argument to a ``by`` argument.
        """
        if not isinstance(level, list):
            level = [level]
        by_list = []
        if isinstance(self.obj.index, cudf.MultiIndex):
            for lev in level:
                by_list.append(self.obj.index.get_level_values(lev))
            return by_list
        else:
            if len(level) > 1 or level[0] != 0:
                raise ValueError("level != 0 only valid with MultiIndex")
            by_list.append(cudf.Series(self.obj.index))
        return by_list

    def normalize_keys(self):
        """
        Sets self.key_names and self.key_columns
        """
        self.df_key_names = []
        if isinstance(self.by, (list, tuple)):
            self.key_names = []
            self.key_columns = []
            for by in self.by:
                name, col = self.key_from_by(by)
                self.key_names.append(name)
                self.key_columns.append(col)
        else:
            # grouping by a single label or Series
            name, col = self.key_from_by(self.by)
            self.key_names = [name]
            self.key_columns = [col]

    def key_from_by(self, by):
        """
        Get (key_name, key_column) pair from a single *by* argument
        """
        if is_scalar(by):
            self.df_key_names.append(by)
            key_name = by
            key_column = self.obj[by]
        else:
            if len(by) != len(self.obj):
                raise NotImplementedError(
                    "cuDF does not support arbitrary series index lengths "
                    "for groupby"
                )
            key_name = by.name
            key_column = by
        return key_name, key_column

    def compute_result(self, agg):
        """
        Computes the groupby result
        """
        self.normalize_agg(agg)
        self.normalize_values()
        aggs_as_list = self.get_aggs_as_list()

        key_columns = [
            sr._column
            if isinstance(sr, cudf.Series)
            else cudf.Series(sr)._column
            for sr in self.key_columns
        ]

        out_key_columns, out_value_columns = _groupby_engine(
            key_columns,
            self.value_columns,
            aggs_as_list,
            self.sort,
            self.dropna,
        )

        return self.construct_result(out_key_columns, out_value_columns)

    def normalize_agg(self, agg):
        """
        Normalize agg to a dictionary with column names
        as keys and lists of aggregations as values.

        For a Series, the dictionary has a single key ``None``
        """
        if isinstance(agg, collections.abc.Mapping):
            for col_name, agg_name in agg.items():
                if not isinstance(agg_name, list):
                    agg[col_name] = [agg_name]
            self.aggs = agg
            return
        if isinstance(agg, str):
            agg = [agg]
        if isinstance(self.obj, cudf.Series):
            value_col_names = [None]
        else:
            value_col_names = []
            # add all non-key columns to value_col_names,
            # dropping "nuisance columns".
            # But don't drop if keys are supplied from
            # some other individual series.
            for col_name in self.obj.columns:
                if col_name not in self.df_key_names:
                    drop = False
                    if isinstance(
                        self.obj[col_name]._column,
                        (
                            cudf.core.column.StringColumn,
                            cudf.core.column.CategoricalColumn,
                        ),
                    ):
                        for agg_name in agg:
                            if agg_name in ("mean", "sum"):
                                drop = True
                    if not drop:
                        value_col_names.append(col_name)
        agg_list = [agg] * len(value_col_names)
        self.aggs = dict(zip(value_col_names, agg_list))
        self.validate_aggs()

    def validate_aggs(self):
        for col_name, agg_list in self.aggs.items():
            for agg in agg_list:
                if agg not in self.NAMED_AGGS:
                    raise ValueError(
                        f"Aggregation function name {agg} not recognized"
                    )

    def normalize_values(self):
        """
        Sets self.value_names and self.value_columns
        """
        if isinstance(self.obj, cudf.Series):
            # SeriesGroupBy
            col = self.obj._column
            agg_list = self.aggs[None]
            if len(agg_list) == 1:
                self.value_columns = [col]
                self.value_names = [self.obj.name]
            else:
                self.value_columns = [col] * len(agg_list)
                self.value_names = agg_list
        else:
            # DataFrameGroupBy
            self.value_columns = []
            self.value_names = []
            for col_name, agg_list in self.aggs.items():
                col = self.obj[col_name]._column
                if len(agg_list) == 1:
                    self.value_columns.append(col)
                    self.value_names.append(col_name)
                else:
                    self.value_columns.extend([col] * len(agg_list))
                    self.value_names.extend([col_name] * len(agg_list))

    def construct_result(self, out_key_columns, out_value_columns):
        if not self.as_index:
            result = cudf.concat(
                [
                    dataframe_from_columns(
                        out_key_columns, columns=self.key_names
                    ),
                    dataframe_from_columns(
                        out_value_columns, columns=self.value_names
                    ),
                ],
                axis=1,
            )
            return result

        result = dataframe_from_columns(
            out_value_columns, columns=self.compute_result_column_index()
        )

        index = self.compute_result_index(out_key_columns, out_value_columns)
        if len(result) == 0 and len(index) != 0:
            # Can't go through the setter in this case
            result._index = index
        else:
            result.index = index

        if isinstance(self.obj, cudf.Series):
            # May need to downcast from DataFrame to Series:
            if len(self.aggs[None]) == 1:
                result = result[result.columns[0]]
                result.name = self.value_names[0]

        return result

    def compute_result_index(self, key_columns, value_columns):
        """
        Computes the index of the result
        """
        key_names = self.key_names
        if (len(key_columns)) == 1:
            return cudf.core.index.as_index(key_columns[0], name=key_names[0])
        else:
            empty_keys = all([len(x) == 0 for x in key_columns])
            if len(value_columns) == 0 and empty_keys:
                return cudf.core.index.GenericIndex(
                    cudf.Series([], dtype="object")
                )
            return MultiIndex(
                source_data=dataframe_from_columns(
                    key_columns, columns=key_names
                ),
                names=key_names,
            )

    def compute_result_column_index(self):
        """
        Computes the column index of the result
        """
        value_names = self.value_names
        aggs_as_list = self.get_aggs_as_list()

        if isinstance(self.obj, cudf.Series):
            if len(aggs_as_list) == 1:
                if self.obj.name is None:
                    return self.obj.name
                else:
                    return [self.obj.name]
            else:
                return aggs_as_list
        else:
            if len(aggs_as_list) == len(self.aggs):
                return value_names
            else:
                return MultiIndex.from_tuples(zip(value_names, aggs_as_list))

    def get_aggs_as_list(self):
        """
        Returns self.aggs as a list of aggs
        """
        aggs_as_list = list(itertools.chain.from_iterable(self.aggs.values()))
        return aggs_as_list


def _groupby_engine(key_columns, value_columns, aggs, sort, dropna):
    """
    Parameters
    ----------
    key_columns : list of Columns
    value_columns : list of Columns
    aggs : list of str
    sort : bool
    dropna : bool

    Returns
    -------
    out_key_columns : list of Columns
    out_value_columns : list of Columns
    """
    out_key_columns, out_value_columns = libcudf.groupby.groupby(
        key_columns, value_columns, aggs, dropna=dropna
    )

    if sort:
        key_names = ["key_" + str(i) for i in range(len(key_columns))]
        value_names = ["value_" + str(i) for i in range(len(value_columns))]
        value_names = _add_prefixes(value_names, aggs)

        # concatenate
        result = cudf.concat(
            [
                dataframe_from_columns(out_key_columns, columns=key_names),
                dataframe_from_columns(out_value_columns, columns=value_names),
            ],
            axis=1,
        )

        # sort values
        result = result.sort_values(key_names)

        # split
        out_key_columns = columns_from_dataframe(result[key_names])
        out_value_columns = columns_from_dataframe(result[value_names])

    return out_key_columns, out_value_columns


def _add_prefixes(names, prefixes):
    """
    Return a copy of ``names`` prefixed with ``prefixes``
    """
    prefixed_names = names.copy()
    if isinstance(prefixes, str):
        prefix = prefixes
        for i, col_name in enumerate(names):
            prefixed_names[i] = f"{prefix}_{col_name}"
    else:
        for i, (prefix, col_name) in enumerate(zip(prefixes, names)):
            prefixed_names[i] = f"{prefix}_{col_name}"
    return prefixed_names


def _align_by_and_df(obj, by, how="inner"):
    """
    Returns a pair of dataframes and a list may be containing
    combination of column names and Series  which are intersected
    as per their indices.

    Examples
    --------
    Dataframe and Series in the 'by' have different indices:

    >>> import cudf
    >>> import cudf.core.groupby.groupby as grp_by

    >>> gdf = cudf.DataFrame(
            {"x": [1.0, 2.0, 3.0], "y": [1, 2, 1]},
            index=[1,2,3]
        )
    >>> gsr = cudf.Series([0.0, 1.0, 2.0], name='a', index=[2,3,4])
    >>> updtd_gdf, updtd_by = grp_by._align_by_and_df(gdf, ['x', gsr])
    >>> print (gdf)
        x     y
    1 	1.0   1
    2 	2.0   2
    3 	3.0   1
    >>> print(updtd_gdf)
        x     y
    2 	2.0   2
    3 	3.0   1
    >>> print(by)
    ['x', 2    0.0
          3    1.0
          4    2.0
          Name: a, dtype: float64]
    >>> print(updtd_by)
    ['x', 2    0.0
          3    1.0
          Name: a, dtype: float64]
    """
    if not isinstance(by, (list, tuple)):
        by = [by]

    series_count = 0
    join_required = False
    series = []
    for by_col in by:
        if not is_scalar(by_col) and not isinstance(by_col, cudf.Index):
            sr = by_col
            if not isinstance(by_col, cudf.Series):
                sr = cudf.Series(by_col)
            if not join_required and not obj.index.equals(sr.index):
                join_required = True
            series.append(sr)

    new_obj = None
    if join_required:
        for sr in series:
            if new_obj is None:
                new_obj = sr.to_frame(series_count)
            else:
                new_obj = new_obj.join(
                    sr.to_frame(series_count), how=how, sort="True"
                )
            series_count += 1

    series_count = 0
    new_by = []
    if new_obj is not None:
        new_obj = new_obj.join(obj, how=how, sort="True")
        columns = new_obj.columns
        for by_col in by:
            if not is_scalar(by_col) and not isinstance(by_col, cudf.Index):
                sr, sr.name = (
                    cudf.Series(new_obj[columns[series_count]]),
                    by_col.name,
                )
                new_by.append(sr)
                series_count += 1
            else:
                new_by.append(by_col)

        new_obj = new_obj[columns[series_count::]]
    else:
        new_obj = obj
        new_by = by

    return new_obj, new_by
