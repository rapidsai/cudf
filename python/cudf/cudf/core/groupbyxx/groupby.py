import collections

import cudf


class _GroupByKeys:
    def __init__(self, obj, by_list):
        """
        Parameters
        ----------

        obj : Object on which the GroupBy is performed
        by_list
            A list of any of the following:

            1. A Python function called on each value of the object's index
            2. A dict or Series that maps index labels to group names
            3. A str indicating a column name
            4. A list or array of the same length as the object
        """
        self.obj = obj
        self.keys = []
        self.names = []

        for by in by_list:
            if callable(by):
                self._handle_callable(by)
            elif isinstance(by, cudf.Series):
                self._handle_series(by)
            elif isinstance(by, collections.abc.Mapping):
                self._handle_mapping(by)
            elif by in self.obj:
                self._handle_label(by)
            else:
                self._handle_misc(by)

    def _handle_callable(self, by):
        self.keys.extend(self.obj.index.values)
        self.names.extend(self.obj.index.names)

    def _handle_series(self, by):
        by = by._align_to_index(self.obj.index, how="right")
        self.keys.append(by._column)
        self.names.append(by.name)

    def _handle_mapping(self, by):
        by = cudf.Series(by.values(), index=by.keys())
        self._handle_series(by)

    def _handle_label(self, by):
        self.keys.append(self.obj._data[by])
        self.names.append(by)

    def _handle_misc(self, by):
        by = cudf.core.column.as_column(by)
        if len(by) != len(self.obj):
            raise ValueError("Grouper and object must have same length")
        self.keys.append(by)
        self.names.append(None)
