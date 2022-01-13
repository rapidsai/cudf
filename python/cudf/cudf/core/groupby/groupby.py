# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import collections
import itertools
import pickle
import warnings

import numpy as np
import pandas as pd
from nvtx import annotate

import cudf
from cudf._lib import groupby as libgroupby
from cudf._typing import DataFrameOrSeries
from cudf.api.types import is_list_like
from cudf.core.abc import Serializable
from cudf.core.column.column import arange, as_column
from cudf.core.multiindex import MultiIndex
from cudf.utils.utils import GetAttrGetItemMixin, cached_property


# The three functions below return the quantiles [25%, 50%, 75%]
# respectively, which are called in the describe() method to output
# the summary stats of a GroupBy object
def _quantile_25(x):
    return x.quantile(0.25)


def _quantile_50(x):
    return x.quantile(0.50)


def _quantile_75(x):
    return x.quantile(0.75)


class GroupBy(Serializable):

    _MAX_GROUPS_BEFORE_WARN = 100

    def __init__(
        self, obj, by=None, level=None, sort=False, as_index=True, dropna=True
    ):
        """
        Group a DataFrame or Series by a set of columns.

        Parameters
        ----------
        by : optional
            Specifies the grouping columns. Can be any of the following:
            - A Python function called on each value of the object's index
            - A dict or Series that maps index labels to group names
            - A cudf.Index object
            - A str indicating a column name
            - An array of the same length as the object
            - A Grouper object
            - A list of the above
        level : int, level_name or list, optional
            For objects with a MultiIndex, `level` can be used to specify
            grouping by one or more levels of the MultiIndex.
        sort : bool, default False
            Sort the result by group keys. Differ from Pandas, cudf defaults
            to False for better performance.
        as_index : bool, optional
            If as_index=True (default), the group names appear
            as the keys of the resulting DataFrame.
            If as_index=False, the groups are returned as ordinary
            columns of the resulting DataFrame, *if they are named columns*.
        dropna : bool, optional
            If True (default), do not include the "null" group.
        """
        self.obj = obj
        self._as_index = as_index
        self._by = by
        self._level = level
        self._sort = sort
        self._dropna = dropna

        if isinstance(by, _Grouping):
            self.grouping = by
        else:
            self.grouping = _Grouping(obj, by, level)

    def __iter__(self):
        group_names, offsets, _, grouped_values = self._grouped()
        if isinstance(group_names, cudf.BaseIndex):
            group_names = group_names.to_pandas()
        for i, name in enumerate(group_names):
            yield name, grouped_values[offsets[i] : offsets[i + 1]]

    @cached_property
    def groups(self):
        """
        Returns a dictionary mapping group keys to row labels.
        """
        group_names, offsets, _, grouped_values = self._grouped()
        grouped_index = grouped_values.index

        if len(group_names) > self._MAX_GROUPS_BEFORE_WARN:
            warnings.warn(
                f"GroupBy.groups() performance scales poorly with "
                f"number of groups. Got {len(group_names)} groups."
            )

        return dict(
            zip(group_names.to_pandas(), grouped_index._split(offsets[1:-1]))
        )

    def get_group(self, name, obj=None):
        """
        Construct DataFrame from group with provided name.

        Parameters
        ----------
        name : object
            The name of the group to get as a DataFrame.
        obj : DataFrame, default None
            The DataFrame to take the DataFrame out of.  If
            it is None, the object groupby was called on will
            be used.

        Returns
        -------
        group : same type as obj

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]})
        >>> df
           X  Y
        0  A  1
        1  B  4
        2  A  3
        3  B  2
        >>> df.groupby("X").get_group("A")
           X  Y
        0  A  1
        2  A  3
        """
        if obj is None:
            obj = self.obj

        return obj.loc[self.groups[name]]

    def size(self):
        """
        Return the size of each group.
        """
        return (
            cudf.Series(
                cudf.core.column.column_empty(
                    len(self.obj), "int8", masked=False
                )
            )
            .groupby(self.grouping, sort=self._sort)
            .agg("size")
        )

    def cumcount(self):
        """
        Return the cumulative count of keys in each group.
        """
        return (
            cudf.Series(
                cudf.core.column.column_empty(
                    len(self.obj), "int8", masked=False
                )
            )
            .groupby(self.grouping, sort=self._sort)
            .agg("cumcount")
            .reset_index(drop=True)
        )

    @cached_property
    def _groupby(self):
        return libgroupby.GroupBy(self.grouping.keys, dropna=self._dropna)

    @annotate("GROUPBY_AGG", domain="cudf_python")
    def agg(self, func):
        """
        Apply aggregation(s) to the groups.

        Parameters
        ----------
        func : str, callable, list or dict

        Returns
        -------
        A Series or DataFrame containing the combined results of the
        aggregation.

        Examples
        --------
        >>> import cudf
        >>> a = cudf.DataFrame(
            {'a': [1, 1, 2], 'b': [1, 2, 3], 'c': [2, 2, 1]})
        >>> a.groupby('a').agg('sum')
           b  c
        a
        2  3  1
        1  3  4

        Specifying a list of aggregations to perform on each column.

        >>> a.groupby('a').agg(['sum', 'min'])
            b       c
          sum min sum min
        a
        2   3   3   1   1
        1   3   1   4   2

        Using a dict to specify aggregations to perform per column.

        >>> a.groupby('a').agg({'a': 'max', 'b': ['min', 'mean']})
            a   b
          max min mean
        a
        2   2   3  3.0
        1   1   1  1.5

        Using lambdas/callables to specify aggregations taking parameters.

        >>> f1 = lambda x: x.quantile(0.5); f1.__name__ = "q0.5"
        >>> f2 = lambda x: x.quantile(0.75); f2.__name__ = "q0.75"
        >>> a.groupby('a').agg([f1, f2])
             b          c
          q0.5 q0.75 q0.5 q0.75
        a
        1  1.5  1.75  2.0   2.0
        2  3.0  3.00  1.0   1.0
        """
        normalized_aggs = self._normalize_aggs(func)

        # Note: When there are no key columns, the below produces
        # a Float64Index, while Pandas returns an Int64Index
        # (GH: 6945)
        result = cudf.DataFrame._from_data(
            *self._groupby.aggregate(self.obj, normalized_aggs)
        )

        if self._sort:
            result = result.sort_index()

        if not _is_multi_agg(func):
            if result._data.nlevels <= 1:  # 0 or 1 levels
                # make sure it's a flat index:
                result._data.multiindex = False

            if result._data.nlevels > 1:
                result._data.droplevel(-1)

                # if, after dropping the last level, the only
                # remaining key is `NaN`, we need to convert to `None`
                # for Pandas compat:
                if result._data.names == (np.nan,):
                    result._data = result._data.rename_levels(
                        {np.nan: None}, level=0
                    )

        if libgroupby._is_all_scan_aggregate(normalized_aggs):
            # Scan aggregations return rows in original index order
            return self._mimic_pandas_order(result)

        # set index names to be group key names
        if len(result):
            result.index.names = self.grouping.names

        # copy categorical information from keys to the result index:
        result.index._copy_type_metadata(self.grouping.keys)
        result._index = cudf.Index(result._index)

        if not self._as_index:
            for col_name in reversed(self.grouping._named_columns):
                result.insert(
                    0,
                    col_name,
                    result.index.get_level_values(col_name)._values,
                )
            result.index = cudf.core.index.RangeIndex(len(result))

        return result

    aggregate = agg

    def nth(self, n):
        """
        Return the nth row from each group.
        """
        result = self.agg(lambda x: x.nth(n)).sort_index()
        sizes = self.size().sort_index()

        return result[sizes > n]

    def serialize(self):
        header = {}
        frames = []

        header["kwargs"] = {
            "sort": self._sort,
            "dropna": self._dropna,
            "as_index": self._as_index,
        }

        obj_header, obj_frames = self.obj.serialize()
        header["obj"] = obj_header
        header["obj_type"] = pickle.dumps(type(self.obj))
        header["num_obj_frames"] = len(obj_frames)
        frames.extend(obj_frames)

        grouping_header, grouping_frames = self.grouping.serialize()
        header["grouping"] = grouping_header
        header["num_grouping_frames"] = len(grouping_frames)
        frames.extend(grouping_frames)

        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        kwargs = header["kwargs"]

        obj_type = pickle.loads(header["obj_type"])
        obj = obj_type.deserialize(
            header["obj"], frames[: header["num_obj_frames"]]
        )
        grouping = _Grouping.deserialize(
            header["grouping"], frames[header["num_obj_frames"] :]
        )
        return cls(obj, grouping, **kwargs)

    def _grouped(self):
        grouped_keys, grouped_values, offsets = self._groupby.groups(self.obj)
        grouped_values = self.obj.__class__._from_data(*grouped_values)
        grouped_values._copy_type_metadata(self.obj)
        group_names = grouped_keys.unique()
        return (group_names, offsets, grouped_keys, grouped_values)

    def _normalize_aggs(self, aggs):
        """
        Normalize aggs to a dict mapping column names
        to a list of aggregations.
        """
        if not isinstance(aggs, collections.abc.Mapping):
            # Make col_name->aggs mapping from aggs.
            # Do not include named key columns

            # Can't do set arithmetic here as sets are
            # not ordered
            if isinstance(self, SeriesGroupBy):
                columns = [self.obj.name]
            else:
                columns = [
                    col_name
                    for col_name in self.obj._data
                    if col_name not in self.grouping._named_columns
                ]
            out = dict.fromkeys(columns, aggs)
        else:
            out = aggs.copy()

        # Convert all values to list-like:
        for col, agg in out.items():
            if not is_list_like(agg):
                out[col] = [agg]
            else:
                out[col] = list(agg)

        return out

    def pipe(self, func, *args, **kwargs):
        """
        Apply a function `func` with arguments to this GroupBy
        object and return the function’s result.

        Parameters
        ----------
        func : function
            Function to apply to this GroupBy object or,
            alternatively, a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the GroupBy object.
        args : iterable, optional
            Positional arguments passed into ``func``.
        kwargs : mapping, optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        See also
        --------
        cudf.Series.pipe
            Apply a function with arguments to a series.

        cudf.DataFrame.pipe
            Apply a function with arguments to a dataframe.

        apply
            Apply function to each group instead of to the full GroupBy object.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'A': ['a', 'b', 'a', 'b'], 'B': [1, 2, 3, 4]})
        >>> df
           A  B
        0  a  1
        1  b  2
        2  a  3
        3  b  4

        To get the difference between each groups maximum and minimum value
        in one pass, you can do

        >>> df.groupby('A').pipe(lambda x: x.max() - x.min())
           B
        A
        a  2
        b  2
        """
        return cudf.core.common.pipe(self, func, *args, **kwargs)

    def apply(self, function):
        """Apply a python transformation function over the grouped chunk.

        Parameters
        ----------
        func : function
          The python transformation function that will be applied
          on the grouped chunk.

        Examples
        --------
        .. code-block:: python

          from cudf import DataFrame
          df = DataFrame()
          df['key'] = [0, 0, 1, 1, 2, 2, 2]
          df['val'] = [0, 1, 2, 3, 4, 5, 6]
          groups = df.groupby(['key'])

          # Define a function to apply to each row in a group
          def mult(df):
            df['out'] = df['key'] * df['val']
            return df

          result = groups.apply(mult)
          print(result)

        Output:

        .. code-block:: python

             key  val  out
          0    0    0    0
          1    0    1    0
          2    1    2    2
          3    1    3    3
          4    2    4    8
          5    2    5   10
          6    2    6   12

        .. pandas-compat::
            **groupby.apply**

            cuDF's ``groupby.apply`` is limited compared to pandas.
            In some situations, Pandas returns the grouped keys as part of
            the index while cudf does not due to redundancy. For example:

            .. code-block::

                >>> df = pd.DataFrame({
                    'a': [1, 1, 2, 2],
                    'b': [1, 2, 1, 2],
                    'c': [1, 2, 3, 4]})
                >>> gdf = cudf.from_pandas(df)
                >>> df.groupby('a').apply(lambda x: x.iloc[[0]])
                        a  b  c
                    a
                    1 0  1  1  1
                    2 2  2  1  3
                >>> gdf.groupby('a').apply(lambda x: x.iloc[[0]])
                        a  b  c
                    0  1  1  1
                    2  2  1  3
        """
        if not callable(function):
            raise TypeError(f"type {type(function)} is not callable")
        group_names, offsets, _, grouped_values = self._grouped()

        ngroups = len(offsets) - 1
        if ngroups > self._MAX_GROUPS_BEFORE_WARN:
            warnings.warn(
                f"GroupBy.apply() performance scales poorly with "
                f"number of groups. Got {ngroups} groups."
            )

        chunks = [
            grouped_values[s:e] for s, e in zip(offsets[:-1], offsets[1:])
        ]
        chunk_results = [function(chk) for chk in chunks]

        if not len(chunk_results):
            return self.obj.head(0)

        if cudf.api.types.is_scalar(chunk_results[0]):
            result = cudf.Series(chunk_results, index=group_names)
            result.index.names = self.grouping.names
        elif isinstance(chunk_results[0], cudf.Series):
            result = cudf.concat(chunk_results, axis=1).T
            result.index.names = self.grouping.names
        else:
            result = cudf.concat(chunk_results)

        if self._sort:
            result = result.sort_index()
        return result

    def apply_grouped(self, function, **kwargs):
        """Apply a transformation function over the grouped chunk.

        This uses numba's CUDA JIT compiler to convert the Python
        transformation function into a CUDA kernel, thus will have a
        compilation overhead during the first run.

        Parameters
        ----------
        func : function
          The transformation function that will be executed on the CUDA GPU.
        incols: list
          A list of names of input columns.
        outcols: list
          A dictionary of output column names and their dtype.
        kwargs : dict
          name-value of extra arguments. These values are passed directly into
          the function.

        Examples
        --------
        .. code-block:: python

            from cudf import DataFrame
            from numba import cuda
            import numpy as np

            df = DataFrame()
            df['key'] = [0, 0, 1, 1, 2, 2, 2]
            df['val'] = [0, 1, 2, 3, 4, 5, 6]
            groups = df.groupby(['key'])

            # Define a function to apply to each group
            def mult_add(key, val, out1, out2):
                for i in range(cuda.threadIdx.x, len(key), cuda.blockDim.x):
                    out1[i] = key[i] * val[i]
                    out2[i] = key[i] + val[i]

            result = groups.apply_grouped(mult_add,
                                          incols=['key', 'val'],
                                          outcols={'out1': np.int32,
                                                   'out2': np.int32},
                                          # threads per block
                                          tpb=8)

            print(result)

        Output:

        .. code-block:: python

               key  val out1 out2
            0    0    0    0    0
            1    0    1    0    1
            2    1    2    2    3
            3    1    3    3    4
            4    2    4    8    6
            5    2    5   10    7
            6    2    6   12    8



        .. code-block:: python

            import cudf
            import numpy as np
            from numba import cuda
            import pandas as pd
            from random import randint


            # Create a random 15 row dataframe with one categorical
            # feature and one random integer valued feature
            df = cudf.DataFrame(
                    {
                        "cat": [1] * 5 + [2] * 5 + [3] * 5,
                        "val": [randint(0, 100) for _ in range(15)],
                    }
                 )

            # Group the dataframe by its categorical feature
            groups = df.groupby("cat")

            # Define a kernel which takes the moving average of a
            # sliding window
            def rolling_avg(val, avg):
                win_size = 3
                for i in range(cuda.threadIdx.x, len(val), cuda.blockDim.x):
                    if i < win_size - 1:
                        # If there is not enough data to fill the window,
                        # take the average to be NaN
                        avg[i] = np.nan
                    else:
                        total = 0
                        for j in range(i - win_size + 1, i + 1):
                            total += val[j]
                        avg[i] = total / win_size

            # Compute moving averages on all groups
            results = groups.apply_grouped(rolling_avg,
                                           incols=['val'],
                                           outcols=dict(avg=np.float64))
            print("Results:", results)

            # Note this gives the same result as its pandas equivalent
            pdf = df.to_pandas()
            pd_results = pdf.groupby('cat')['val'].rolling(3).mean()


        Output:

        .. code-block:: python

            Results:
               cat  val                 avg
            0    1   16
            1    1   45
            2    1   62                41.0
            3    1   45  50.666666666666664
            4    1   26  44.333333333333336
            5    2    5
            6    2   51
            7    2   77  44.333333333333336
            8    2    1                43.0
            9    2   46  41.333333333333336
            [5 more rows]

        This is functionally equivalent to `pandas.DataFrame.Rolling
        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html>`_

        """
        if not callable(function):
            raise TypeError(f"type {type(function)} is not callable")

        _, offsets, _, grouped_values = self._grouped()
        kwargs.update({"chunks": offsets})
        return grouped_values.apply_chunks(function, **kwargs)

    def rolling(self, *args, **kwargs):
        """
        Returns a `RollingGroupby` object that enables rolling window
        calculations on the groups.

        See also
        --------
        cudf.core.window.Rolling
        """
        return cudf.core.window.rolling.RollingGroupby(self, *args, **kwargs)

    def count(self, dropna=True):
        """Compute the number of values in each column.

        Parameters
        ----------
        dropna : bool
            If ``True``, don't include null values in the count.
        """

        def func(x):
            return getattr(x, "count")(dropna=dropna)

        return self.agg(func)

    def describe(self, include=None, exclude=None):
        """
        Generate descriptive statistics that summarizes the central tendency,
        dispersion and shape of a dataset’s distribution, excluding NaN values.

        Analyzes numeric DataFrames only

        Parameters
        ----------
        include: ‘all’, list-like of dtypes or None (default), optional
            list of data types to include in the result.
            Ignored for Series.

        exclude: list-like of dtypes or None (default), optional,
            list of data types to omit from the result.
            Ignored for Series.

        Returns
        -------
        Series or DataFrame
            Summary statistics of the Dataframe provided.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({"Speed": [380.0, 370.0, 24.0, 26.0],
                                  "Score": [50, 30, 90, 80]})
        >>> gdf
        Speed  Score
        0  380.0     50
        1  370.0     30
        2   24.0     90
        3   26.0     80
        >>> gdf.groupby('Score').describe()
             Speed
             count   mean   std    min    25%    50%    75%     max
        Score
        30        1  370.0  <NA>  370.0  370.0  370.0  370.0  370.0
        50        1  380.0  <NA>  380.0  380.0  380.0  380.0  380.0
        80        1   26.0  <NA>   26.0   26.0   26.0   26.0   26.0
        90        1   24.0  <NA>   24.0   24.0   24.0   24.0   24.0

        """
        if exclude is not None and include is not None:
            raise NotImplementedError

        res = self.agg(
            [
                "count",
                "mean",
                "std",
                "min",
                _quantile_25,
                _quantile_50,
                _quantile_75,
                "max",
            ]
        )
        res.rename(
            columns={
                "_quantile_25": "25%",
                "_quantile_50": "50%",
                "_quantile_75": "75%",
            },
            level=1,
            inplace=True,
        )
        return res

    def sum(self):
        """Compute the column-wise sum of the values in each group."""
        return self.agg("sum")

    def prod(self):
        """Compute the column-wise product of the values in each group."""
        return self.agg("prod")

    def idxmin(self):
        """Get the column-wise index of the minimum value in each group."""
        return self.agg("idxmin")

    def idxmax(self):
        """Get the column-wise index of the maximum value in each group."""
        return self.agg("idxmax")

    def min(self):
        """Get the column-wise minimum value in each group."""
        return self.agg("min")

    def max(self):
        """Get the column-wise maximum value in each group."""
        return self.agg("max")

    def mean(self):
        """Compute the column-wise mean of the values in each group."""
        return self.agg("mean")

    def median(self):
        """Get the column-wise median of the values in each group."""
        return self.agg("median")

    def corr(self, method="pearson", min_periods=1):
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method: {"pearson", "kendall", "spearman"} or callable,
            default "pearson". Currently only the pearson correlation
            coefficient is supported.

        min_periods: int, optional
            Minimum number of observations required per pair of columns
            to have a valid result.

        Returns
        ----------
        DataFrame
            Correlation matrix.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({
        ...             "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        ...             "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
        ...             "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
        ...             "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1]})
        >>> gdf
        id  val1  val2  val3
        0  a     5     4     4
        1  a     4     5     5
        2  a     6     6     6
        3  b     4     1     1
        4  b     8     2     2
        5  b     7     9     9
        6  c     4     8     8
        7  c     5     5     5
        8  c     2     1     1
        >>> gdf.groupby("id").corr(method="pearson")
                    val1      val2      val3
        id
        a   val1  1.000000  0.500000  0.500000
            val2  0.500000  1.000000  1.000000
            val3  0.500000  1.000000  1.000000
        b   val1  1.000000  0.385727  0.385727
            val2  0.385727  1.000000  1.000000
            val3  0.385727  1.000000  1.000000
        c   val1  1.000000  0.714575  0.714575
            val2  0.714575  1.000000  1.000000
            val3  0.714575  1.000000  1.000000
        """

        if not method.lower() in ("pearson",):
            raise NotImplementedError(
                "Only pearson correlation is currently supported"
            )

        # create expanded dataframe consisting all combinations of the
        # struct columns-pairs to be correlated
        # i.e (('col1', 'col1'), ('col1', 'col2'), ('col2', 'col2'))
        _cols = self.grouping.values.columns.tolist()
        len_cols = len(_cols)

        new_df_data = {}
        for x, y in itertools.combinations_with_replacement(_cols, 2):
            new_df_data[(x, y)] = cudf.DataFrame._from_data(
                {"x": self.obj._data[x], "y": self.obj._data[y]}
            ).to_struct()
        new_gb = cudf.DataFrame._from_data(new_df_data).groupby(
            by=self.grouping.keys
        )

        try:
            gb_corr = new_gb.agg(lambda x: x.corr(method, min_periods))
        except RuntimeError as e:
            if "Unsupported groupby reduction type-agg combination" in str(e):
                raise TypeError(
                    "Correlation accepts only numerical column-pairs"
                )
            raise

        # ensure that column-pair labels are arranged in ascending order
        cols_list = [
            (y, x) if i > j else (x, y)
            for j, y in enumerate(_cols)
            for i, x in enumerate(_cols)
        ]
        cols_split = [
            cols_list[i : i + len_cols]
            for i in range(0, len(cols_list), len_cols)
        ]

        # interleave: combine the correlation results for each column-pair
        # into a single column
        res = cudf.DataFrame._from_data(
            {
                x: gb_corr.loc[:, i].interleave_columns()
                for i, x in zip(cols_split, _cols)
            }
        )

        # create a multiindex for the groupby correlated dataframe,
        # to match pandas behavior
        unsorted_idx = gb_corr.index.repeat(len_cols)
        idx_sort_order = unsorted_idx._get_sorted_inds()
        sorted_idx = unsorted_idx._gather(idx_sort_order)
        if len(gb_corr):
            # TO-DO: Should the operation below be done on the CPU instead?
            sorted_idx._data[None] = as_column(
                cudf.Series(_cols).tile(len(gb_corr.index))
            )
        res.index = MultiIndex._from_data(sorted_idx._data)

        return res

    def var(self, ddof=1):
        """Compute the column-wise variance of the values in each group.

        Parameters
        ----------
        ddof : int
            The delta degrees of freedom. N - ddof is the divisor used to
            normalize the variance.
        """

        def func(x):
            return getattr(x, "var")(ddof=ddof)

        return self.agg(func)

    def std(self, ddof=1):
        """Compute the column-wise std of the values in each group.

        Parameters
        ----------
        ddof : int
            The delta degrees of freedom. N - ddof is the divisor used to
            normalize the standard deviation.
        """

        def func(x):
            return getattr(x, "std")(ddof=ddof)

        return self.agg(func)

    def quantile(self, q=0.5, interpolation="linear"):
        """Compute the column-wise quantiles of the values in each group.

        Parameters
        ----------
        q : float or array-like
            The quantiles to compute.
        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}
            The interpolation method to use when the desired quantile lies
            between two data points. Defaults to "linear".
        """

        def func(x):
            return getattr(x, "quantile")(q=q, interpolation=interpolation)

        return self.agg(func)

    def nunique(self):
        """Compute the number of unique values in each column in each group."""
        return self.agg("nunique")

    def collect(self):
        """Get a list of all the values for each column in each group."""
        return self.agg("collect")

    def unique(self):
        """Get a list of the unique values for each column in each group."""
        return self.agg("unique")

    def cumsum(self):
        """Compute the column-wise cumulative sum of the values in
        each group."""
        return self.agg("cumsum")

    def cummin(self):
        """Get the column-wise cumulative minimum value in each group."""
        return self.agg("cummin")

    def cummax(self):
        """Get the column-wise cumulative maximum value in each group."""
        return self.agg("cummax")

    def first(self):
        """Get the first non-null value in each group."""
        return self.agg("first")

    def last(self):
        """Get the last non-null value in each group."""
        return self.agg("last")

    def diff(self, periods=1, axis=0):
        """Get the difference between the values in each group.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference,
            accepts negative values.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Take difference over rows (0) or columns (1).
            Only row-wise (0) shift is supported.

        Returns
        -------
        Series or DataFrame
            First differences of the Series or DataFrame.
        """

        if not axis == 0:
            raise NotImplementedError("Only axis=0 is supported.")

        # grouped values
        value_columns = self.grouping.values
        _, (data, index), _ = self._groupby.groups(
            cudf.core.frame.Frame(value_columns._data)
        )
        grouped = self.obj.__class__._from_data(data, index)
        grouped = self._mimic_pandas_order(grouped)

        result = grouped - self.shift(periods=periods)
        return result._copy_type_metadata(value_columns)

    def _scan_fill(self, method: str, limit: int) -> DataFrameOrSeries:
        """Internal implementation for `ffill` and `bfill`"""
        value_columns = self.grouping.values
        result = self.obj.__class__._from_data(
            self._groupby.replace_nulls(
                cudf.core.frame.Frame(value_columns._data), method
            )
        )
        result = self._mimic_pandas_order(result)
        return result._copy_type_metadata(value_columns)

    def pad(self, limit=None):
        """Forward fill NA values.

        Parameters
        ----------
        limit : int, default None
            Unsupported
        """

        if limit is not None:
            raise NotImplementedError("Does not support limit param yet.")

        return self._scan_fill("ffill", limit)

    ffill = pad

    def backfill(self, limit=None):
        """Backward fill NA values.

        Parameters
        ----------
        limit : int, default None
            Unsupported
        """
        if limit is not None:
            raise NotImplementedError("Does not support limit param yet.")

        return self._scan_fill("bfill", limit)

    bfill = backfill

    def fillna(
        self,
        value=None,
        method=None,
        axis=0,
        inplace=False,
        limit=None,
        downcast=None,
    ):
        """Fill NA values using the specified method.

        Parameters
        ----------
        value : scalar, dict
            Value to use to fill the holes. Cannot be specified with method.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series

            - pad/ffill: propagate last valid observation forward to next valid
            - backfill/bfill: use next valid observation to fill gap
        axis : {0 or 'index', 1 or 'columns'}
            Unsupported
        inplace : bool, default False
            If `True`, fill inplace. Note: this will modify other views on this
            object.
        limit : int, default None
            Unsupported
        downcast : dict, default None
            Unsupported

        Returns
        -------
        DataFrame or Series

        .. pandas-compat::
            **groupby.fillna**

            This function may return result in different format to the method
            Pandas supports. For example:

            .. code-block::

                >>> df = pd.DataFrame({'k': [1, 1, 2], 'v': [2, None, 4]})
                >>> gdf = cudf.from_pandas(df)
                >>> df.groupby('k').fillna({'v': 4}) # pandas
                       v
                k
                1 0  2.0
                  1  4.0
                2 2  4.0
                >>> gdf.groupby('k').fillna({'v': 4}) # cudf
                     v
                0  2.0
                1  4.0
                2  4.0
        """
        if inplace:
            raise NotImplementedError("Does not support inplace yet.")
        if limit is not None:
            raise NotImplementedError("Does not support limit param yet.")
        if downcast is not None:
            raise NotImplementedError("Does not support downcast yet.")
        if not axis == 0:
            raise NotImplementedError("Only support axis == 0.")

        if value is None and method is None:
            raise ValueError("Must specify a fill 'value' or 'method'.")
        if value is not None and method is not None:
            raise ValueError("Cannot specify both 'value' and 'method'.")

        if method is not None:
            if method not in {"pad", "ffill", "backfill", "bfill"}:
                raise ValueError(
                    "Method can only be of 'pad', 'ffill',"
                    "'backfill', 'bfill'."
                )
            return getattr(self, method, limit)()

        value_columns = self.grouping.values
        _, (data, index), _ = self._groupby.groups(
            cudf.core.frame.Frame(value_columns._data)
        )

        grouped = self.obj.__class__._from_data(data, index)
        result = grouped.fillna(
            value=value, inplace=inplace, axis=axis, limit=limit
        )
        result = self._mimic_pandas_order(result)
        return result._copy_type_metadata(value_columns)

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """
        Shift each group by ``periods`` positions.

        Parameters
        ----------
        periods : int, default 1
            Number of periods to shift.
        freq : str, unsupported
        axis : 0, axis to shift
            Shift direction. Only row-wise shift is supported
        fill_value : scalar or list of scalars, optional
            The scalar value to use for newly introduced missing values. Can be
            specified with `None`, a single value or multiple values:

            - `None` (default): sets all indeterminable values to null.
            - Single value: fill all shifted columns with this value. Should
              match the data type of all columns.
            - List of values: fill shifted columns with corresponding value in
              the list. The length of the list should match the number of
              columns shifted. Each value should match the data type of the
              column to fill.

        Returns
        -------
        Series or DataFrame
            Object shifted within each group.

        Notes
        -----
        Parameter ``freq`` is unsupported.
        """

        if freq is not None:
            raise NotImplementedError("Parameter freq is unsupported.")

        if not axis == 0:
            raise NotImplementedError("Only axis=0 is supported.")

        value_columns = self.grouping.values
        if is_list_like(fill_value):
            if not len(fill_value) == len(value_columns._data):
                raise ValueError(
                    "Mismatched number of columns and values to fill."
                )
        else:
            fill_value = [fill_value] * len(value_columns._data)

        result = self.obj.__class__._from_data(
            *self._groupby.shift(
                cudf.core.frame.Frame(value_columns._data), periods, fill_value
            )
        )
        result = self._mimic_pandas_order(result)
        return result._copy_type_metadata(value_columns)

    def _mimic_pandas_order(
        self, result: DataFrameOrSeries
    ) -> DataFrameOrSeries:
        """Given a groupby result from libcudf, reconstruct the row orders
        matching that of pandas. This also adds appropriate indices.
        """
        sorted_order_column = arange(0, result._data.nrows)
        _, (order, _), _ = self._groupby.groups(
            cudf.core.frame.Frame({"sorted_order_column": sorted_order_column})
        )
        gather_map = order["sorted_order_column"].argsort()
        result = result.take(gather_map)
        result.index = self.obj.index
        return result


class DataFrameGroupBy(GroupBy, GetAttrGetItemMixin):
    """
    Group DataFrame using a mapper or by a Series of columns.

    A groupby operation involves some combination of splitting the object,
    applying a function, and combining the results. This can be used to
    group large amounts of data and compute operations on these groups.

    Parameters
    ----------
    by : mapping, function, label, or list of labels
        Used to determine the groups for the groupby. If by is a
        function, it’s called on each value of the object’s index.
        If a dict or Series is passed, the Series or dict VALUES will
        be used to determine the groups (the Series’ values are first
        aligned; see .align() method). If a cupy array is passed, the
        values are used as-is determine the groups. A label or list
        of labels may be passed to group by the columns in self.
        Notice that a tuple is interpreted as a (single) key.
    level : int, level name, or sequence of such, default None
        If the axis is a MultiIndex (hierarchical), group by a particular
        level or levels.
    as_index : bool, default True
        For aggregated output, return object with group labels as
        the index. Only relevant for DataFrame input.
        as_index=False is effectively “SQL-style” grouped output.
    sort : bool, default False
        Sort result by group key. Differ from Pandas, cudf defaults to
        ``False`` for better performance. Note this does not influence
        the order of observations within each group. Groupby preserves
        the order of rows within each group.
    dropna : bool, optional
        If True (default), do not include the "null" group.

    Returns
    -------
        DataFrameGroupBy
            Returns a groupby object that contains information
            about the groups.

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> df = cudf.DataFrame({
    ...     'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
    ...     'Max Speed': [380., 370., 24., 26.],
    ... })
    >>> df
       Animal  Max Speed
    0  Falcon      380.0
    1  Falcon      370.0
    2  Parrot       24.0
    3  Parrot       26.0
    >>> df.groupby(['Animal']).mean()
            Max Speed
    Animal
    Falcon      375.0
    Parrot       25.0

    >>> arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
    ...           ['Captive', 'Wild', 'Captive', 'Wild']]
    >>> index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
    >>> df = cudf.DataFrame({'Max Speed': [390., 350., 30., 20.]},
    ...     index=index)
    >>> df
                    Max Speed
    Animal Type
    Falcon Captive      390.0
           Wild         350.0
    Parrot Captive       30.0
           Wild          20.0
    >>> df.groupby(level=0).mean()
            Max Speed
    Animal
    Falcon      370.0
    Parrot       25.0
    >>> df.groupby(level="Type").mean()
            Max Speed
    Type
    Wild         185.0
    Captive      210.0
    """

    _PROTECTED_KEYS = frozenset(("obj",))

    def __getitem__(self, key):
        return self.obj[key].groupby(
            by=self.grouping.keys, dropna=self._dropna, sort=self._sort
        )

    def nunique(self):
        """
        Return the number of unique values per group
        """
        return self.agg("nunique")


class SeriesGroupBy(GroupBy):
    """
    Group Series using a mapper or by a Series of columns.

    A groupby operation involves some combination of splitting the object,
    applying a function, and combining the results. This can be used to
    group large amounts of data and compute operations on these groups.

    Parameters
    ----------
    by : mapping, function, label, or list of labels
        Used to determine the groups for the groupby. If by is a
        function, it’s called on each value of the object’s index.
        If a dict or Series is passed, the Series or dict VALUES will
        be used to determine the groups (the Series’ values are first
        aligned; see .align() method). If an cupy array is passed, the
        values are used as-is determine the groups. A label or list
        of labels may be passed to group by the columns in self.
        Notice that a tuple is interpreted as a (single) key.
    level : int, level name, or sequence of such, default None
        If the axis is a MultiIndex (hierarchical), group by a particular
        level or levels.
    as_index : bool, default True
        For aggregated output, return object with group labels as
        the index. Only relevant for DataFrame input.
        as_index=False is effectively “SQL-style” grouped output.
    sort : bool, default False
        Sort result by group key. Differ from Pandas, cudf defaults to
        ``False`` for better performance. Note this does not influence
        the order of observations within each group. Groupby preserves
        the order of rows within each group.

    Returns
    -------
        SeriesGroupBy
            Returns a groupby object that contains information
            about the groups.

    Examples
    --------
    >>> ser = cudf.Series([390., 350., 30., 20.],
    ...                 index=['Falcon', 'Falcon', 'Parrot', 'Parrot'],
    ...                 name="Max Speed")
    >>> ser
    Falcon    390.0
    Falcon    350.0
    Parrot     30.0
    Parrot     20.0
    Name: Max Speed, dtype: float64
    >>> ser.groupby(level=0).mean()
    Falcon    370.0
    Parrot     25.0
    Name: Max Speed, dtype: float64
    >>> ser.groupby(ser > 100).mean()
    Max Speed
    False     25.0
    True     370.0
    Name: Max Speed, dtype: float64
    """

    def agg(self, func):
        result = super().agg(func)

        # downcast the result to a Series:
        if len(result._data):
            if result.shape[1] == 1 and not is_list_like(func):
                return result.iloc[:, 0]

        # drop the first level if we have a multiindex
        if (
            isinstance(result.columns, pd.MultiIndex)
            and result.columns.nlevels > 1
        ):
            result.columns = result.columns.droplevel(0)

        return result

    def apply(self, func):
        result = super().apply(func)

        # apply Series name to result
        result.name = self.obj.name

        return result


# TODO: should we define this as a dataclass instead?
class Grouper(object):
    def __init__(
        self, key=None, level=None, freq=None, closed=None, label=None
    ):
        if key is not None and level is not None:
            raise ValueError("Grouper cannot specify both key and level")
        if (key, level) == (None, None) and not freq:
            raise ValueError("Grouper must specify either key or level")
        self.key = key
        self.level = level
        self.freq = freq
        self.closed = closed
        self.label = label


class _Grouping(Serializable):
    def __init__(self, obj, by=None, level=None):
        self._obj = obj
        self._key_columns = []
        self.names = []

        # Need to keep track of named key columns
        # to support `as_index=False` correctly
        self._named_columns = []
        self._handle_by_or_level(by, level)

        if len(obj) and not len(self._key_columns):
            raise ValueError("No group keys passed")

    def _handle_by_or_level(self, by=None, level=None):
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
                elif isinstance(by, cudf.BaseIndex):
                    self._handle_index(by)
                elif isinstance(by, collections.abc.Mapping):
                    self._handle_mapping(by)
                elif isinstance(by, Grouper):
                    self._handle_grouper(by)
                else:
                    try:
                        self._handle_label(by)
                    except (KeyError, TypeError):
                        self._handle_misc(by)

    @property
    def keys(self):
        """Return grouping key columns as index"""
        nkeys = len(self._key_columns)

        if nkeys == 0:
            return cudf.core.index.as_index([], name=None)
        elif nkeys > 1:
            return cudf.MultiIndex._from_data(
                dict(zip(range(nkeys), self._key_columns))
            )._set_names(self.names)
        else:
            return cudf.core.index.as_index(
                self._key_columns[0], name=self.names[0]
            )

    @property
    def values(self):
        """Return value columns as a frame.

        Note that in aggregation, value columns can be arbitrarily
        specified. While this method returns all non-key columns from `obj` as
        a frame.

        This is mainly used in transform-like operations.
        """
        # If the key columns are in `obj`, filter them out
        value_column_names = [
            x for x in self._obj._data.names if x not in self._named_columns
        ]
        value_columns = self._obj._data.select_by_label(value_column_names)
        return self._obj.__class__._from_data(value_columns)

    def _handle_callable(self, by):
        by = by(self._obj.index)
        self.__init__(self._obj, by)

    def _handle_series(self, by):
        by = by._align_to_index(self._obj.index, how="right")
        self._key_columns.append(by._column)
        self.names.append(by.name)

    def _handle_index(self, by):
        self._key_columns.extend(by._data.columns)
        self.names.extend(by._data.names)

    def _handle_mapping(self, by):
        by = cudf.Series(by.values(), index=by.keys())
        self._handle_series(by)

    def _handle_label(self, by):
        self._key_columns.append(self._obj._data[by])
        self.names.append(by)
        self._named_columns.append(by)

    def _handle_grouper(self, by):
        if by.freq:
            self._handle_frequency_grouper(by)
        elif by.key:
            self._handle_label(by.key)
        else:
            self._handle_level(by.level)

    def _handle_frequency_grouper(self, by):
        raise NotImplementedError()

    def _handle_level(self, by):
        level_values = self._obj.index.get_level_values(by)
        self._key_columns.append(level_values._values)
        self.names.append(level_values.name)

    def _handle_misc(self, by):
        by = cudf.core.column.as_column(by)
        if len(by) != len(self._obj):
            raise ValueError("Grouper and object must have same length")
        self._key_columns.append(by)
        self.names.append(None)

    def serialize(self):
        header = {}
        frames = []
        header["names"] = pickle.dumps(self.names)
        header["_named_columns"] = pickle.dumps(self._named_columns)
        column_header, column_frames = cudf.core.column.serialize_columns(
            self._key_columns
        )
        header["columns"] = column_header
        frames.extend(column_frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        names = pickle.loads(header["names"])
        _named_columns = pickle.loads(header["_named_columns"])
        key_columns = cudf.core.column.deserialize_columns(
            header["columns"], frames
        )
        out = _Grouping.__new__(_Grouping)
        out.names = names
        out._named_columns = _named_columns
        out._key_columns = key_columns
        return out


def _is_multi_agg(aggs):
    """
    Returns True if more than one aggregation is performed
    on any of the columns as specified in `aggs`.
    """
    if isinstance(aggs, collections.abc.Mapping):
        return any(is_list_like(agg) for agg in aggs.values())
    if is_list_like(aggs):
        return True
    return False
