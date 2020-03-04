import collections

import pandas as pd

import cudf
import cudf._libxx.groupby as libgroupby


class GroupBy(object):
    def __init__(self, obj, by=None, level=None, as_index=True):
        self.grouping = _Grouping(obj, by, level)
        self.obj = obj
        self._groupby = libgroupby.GroupBy(self.grouping.keys)
        self.as_index = as_index

    def __iter__(self):
        grouped_keys, grouped_values, offsets = self._groupby.groups(self.obj)

        grouped_keys = cudf.Index._from_table(grouped_keys)
        grouped_values = self.obj.__class__._from_table(grouped_values)
        group_names = grouped_keys.unique()

        for i, name in enumerate(group_names):
            yield name, grouped_values[offsets[i] : offsets[i + 1]]

    def agg(self, aggs):
        normalized_aggs = self._normalize_aggs(aggs)

        result = self._groupby.aggregate(self.obj, normalized_aggs)
        result = self.obj.__class__._from_table(result).sort_index()

        if not _is_multi_agg(aggs):
            # drop the last level
            columns = result.columns.droplevel(-1)
            result.columns = columns

        # set index names to be group key names
        result.index.names = self.grouping.names

        if not self.as_index:
            for col_name in reversed(self.grouping._named_columns):
                result.insert(
                    0,
                    col_name,
                    result.index.get_level_values(col_name)._column,
                )
            result.index = cudf.core.index.RangeIndex(len(result))

        return result

    def _normalize_aggs(self, aggs):
        """
        Normalize agg to a dict mapping column names
        to a list of aggregations.
        """
        if not isinstance(aggs, collections.abc.Mapping):
            # Make col_name->aggs mapping from aggs.
            # Do not include named key columns
            columns = tuple(
                dict.fromkeys(self.obj._column_names, []).keys()
                - dict.fromkeys(self.grouping._named_columns, []).keys()
            )
            out = dict.fromkeys(columns, aggs)
        else:
            out = aggs.copy()

        # Convert all values to list-like:
        for col, agg in out.items():
            if not pd.api.types.is_list_like(agg):
                out[col] = [agg]

        return out


class Grouper(object):
    def __init__(self, key=None, level=None):
        if key is not None and level is not None:
            raise ValueError("Grouper cannot specify both key and level")
        if key is None and level is None:
            raise ValueError("Grouper must specify either key or level")
        self.key = key
        self.level = level


class _Grouping(object):
    def __init__(self, obj, by=None, level=None):
        """
        Parameters
        ----------
        obj : Object on which the GroupBy is performed
        by :
            Any of the following:

            - A Python function called on each value of the object's index
            - A dict or Series that maps index labels to group names
            - A cudf.Index object
            - A str indicating a column name
            - An array of the same length as the object
            - A Grouper object
            - A list of the above
        """
        self.obj = obj
        self._key_columns = []
        self.names = []
        self._named_columns = []

        if level is not None:
            if by is not None:
                raise ValueError("Cannot specify both by and level")
            level_list = level if isinstance(level, list) else [level]
            for level in level_list:
                self._handle_level(level)
        else:
            by_list = by if isinstance(by, list) else [by]

            for by in by_list:
                if callable(by):
                    self._handle_callable(by)
                elif isinstance(by, cudf.Series):
                    self._handle_series(by)
                elif isinstance(by, cudf.Index):
                    self._handle_index(by)
                elif isinstance(by, collections.abc.Mapping):
                    self._handle_mapping(by)
                elif isinstance(by, Grouper):
                    self._handle_grouper(by)
                elif by in self.obj:
                    self._handle_label(by)
                else:
                    self._handle_misc(by)

    @property
    def keys(self):
        nkeys = len(self._key_columns)
        if nkeys > 1:
            return cudf.MultiIndex(
                source_data=cudf.DataFrame(
                    dict(zip(range(nkeys), self._key_columns))
                ),
                names=self.names,
            )
        else:
            return cudf.core.index.as_index(
                self._key_columns[0], name=self.names[0]
            )

    def _handle_callable(self, by):
        by = by(self.obj.index)
        self.__init__(self.obj, by)

    def _handle_series(self, by):
        by = by._align_to_index(self.obj.index, how="right")
        self._key_columns.append(by._column)
        self.names.append(by.name)

    def _handle_index(self, by):
        self._key_columns.extend(by._data.columns)
        self.names.extend(by._data.names)

    def _handle_mapping(self, by):
        by = cudf.Series(by.values(), index=by.keys())
        self._handle_series(by)

    def _handle_label(self, by):
        self._key_columns.append(self.obj._data[by])
        self.names.append(by)
        self._named_columns.append(by)

    def _handle_grouper(self, by):
        if by.key:
            self._handle_label(by.key)
        else:
            self._handle_level(by.level)

    def _handle_level(self, by):
        level_values = self.obj.index.get_level_values(by)
        self._key_columns.append(level_values._column)
        self.names.append(level_values.name)

    def _handle_misc(self, by):
        by = cudf.core.column.as_column(by)
        if len(by) != len(self.obj):
            raise ValueError("Grouper and object must have same length")
        self._key_columns.append(by)
        self.names.append(None)


def _is_multi_agg(aggs):
    if isinstance(aggs, collections.abc.Mapping):
        return any(pd.api.types.is_list_like(agg) for agg in aggs.values())
    if pd.api.types.is_list_like(aggs):
        return True
    return False
