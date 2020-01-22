# Copyright (c) 2018, NVIDIA CORPORATION.

from collections import OrderedDict, defaultdict, namedtuple
from itertools import chain

import numpy as np
from numba import cuda

import rmm

import cudf
import cudf._lib as libcudf
from cudf.core.series import Series


def _auto_generate_grouper_agg(members):
    def make_fun(f):
        def groupby_agg(self):
            return self.agg(f)

        return groupby_agg

    for k, f in members["_NAMED_FUNCTIONS"].items():
        fn = make_fun(f)
        fn.__name__ = k
        fn.__doc__ = """Compute the {} of each group

Returns
-------

result : DataFrame
""".format(
            k
        )
        members[k] = fn


@cuda.jit
def group_mean(data, segments, output):
    i = cuda.grid(1)
    if i < segments.size:
        s = segments[i]
        e = segments[i + 1] if (i + 1) < segments.size else data.size
        # mean calculation
        carry = 0.0
        n = e - s
        for j in range(s, e):
            carry += data[j]
        output[i] = carry / n


@cuda.jit
def group_max(data, segments, output):
    i = cuda.grid(1)
    if i < segments.size:
        s = segments[i]
        e = segments[i + 1] if (i + 1) < segments.size else data.size

        tmp = data[s]
        for j in range(s + 1, e):
            tmp = max(tmp, data[j])
        output[i] = tmp


@cuda.jit
def group_min(data, segments, output):
    i = cuda.grid(1)
    if i < segments.size:
        s = segments[i]
        e = segments[i + 1] if (i + 1) < segments.size else data.size

        tmp = data[s]
        for j in range(s + 1, e):
            tmp = min(tmp, data[j])
        output[i] = tmp


_dfsegs_pack = namedtuple("_dfsegs_pack", ["df", "segs"])


class Groupby(object):
    """Groupby object returned by cudf.DataFrame.groupby(method="cudf").
    `method=cudf` uses numba kernels to compute aggregations and allows
    custom UDFs via the `apply` and `apply_grouped` methods.

    Notes
    -----
    - `method=cudf` may be deprecated in the future.
    - Grouping and aggregating over columns with null values will
      return incorrect results.
    - Grouping by or aggregating over string columns is currently
      not supported.
    """

    _NAMED_FUNCTIONS = {
        "mean": Series.mean,
        "std": Series.std,
        "var": Series.var,
        "min": Series.min,
        "max": Series.max,
        "count": Series.count,
        "sum": Series.sum,
        "sum_of_squares": Series.sum_of_squares,
    }

    def __init__(self, df, by):
        """
        Parameters
        ----------
        df : DataFrame
        by : str of list of str
            Column(s) that grouping is based on.
            It can be a single or list of column names.
        """
        self._df = df
        self._by = [by] if isinstance(by, str) else list(by)
        self._val_columns = [
            idx for idx in self._df.columns if idx not in self._by
        ]

    def serialize(self, serialize):
        header = {"by": self._by}
        header["df"], frames = serialize(self._df)
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        by = header["by"]
        df = deserialize(header["df"], frames)
        return Groupby(df, by)

    def __iter__(self):
        return self._group_iterator()

    def _group_iterator(self):
        """Group iterator

        Returns each group as a DataFrame.
        """
        grouped = self.as_df()
        segs = grouped.segs.to_array()
        for begin, end in zip(segs, chain(segs[1:], [len(grouped.df)])):
            yield grouped.df[begin:end]

    def as_df(self):
        """Get the intermediate dataframe after shuffling the rows into
        groups.

        Returns
        -------
        (df, segs) : namedtuple
            - df : DataFrame
            - segs : Series
                Beginning offsets of each group.

        Examples
        --------
        .. code-block:: python

          from cudf import DataFrame

          df = DataFrame()
          df['key'] = [0, 0, 1, 1, 2, 2, 2]
          df['val'] = [0, 1, 2, 3, 4, 5, 6]
          groups = df.groupby(['key'], method='cudf')

          df_groups = groups.as_df()

          # DataFrame indexes of group starts
          print(df_groups[1])

          # DataFrame itself
          print(df_groups[0])

        Output:

        .. code-block:: python

          # DataFrame indexes of group starts
          0    0
          1    2
          2    4

          # DataFrame itself
             key  val
          0    0    0
          1    0    1
          2    1    2
          3    1    3
          4    2    4
          5    2    5
          6    2    6

        """
        return self._group_dataframe(self._df, self._by)

    def _agg_groups(self, functors):
        """Aggregate the groups

        Parameters
        ----------
        functors: dict
            Contains key for column names and value for list of functors.

        """
        functors_mapping = OrderedDict()
        # The "value" columns
        for k, vs in functors.items():
            if k not in self._df.columns:
                raise NameError("column {} not found".format(k))
            if len(vs) == 1:
                [functor] = vs
                functors_mapping[k] = {k: functor}
            else:
                functors_mapping[k] = cur_fn_mapping = OrderedDict()
                for functor in vs:
                    newk = "{}_{}".format(k, functor.__name__)
                    cur_fn_mapping[newk] = functor

            del functor
        # Grouping
        grouped_df, sr_segs = self._group_dataframe(self._df, self._by)
        # Grouped values
        outdf = cudf.DataFrame()
        segs = sr_segs.to_array()

        for k in self._by:
            outdf[k] = grouped_df[k].take(sr_segs).reset_index(drop=True)

        size = len(outdf)

        # Append value columns
        for k, infos in functors_mapping.items():
            values = defaultdict(lambda: np.zeros(size, dtype=np.float64))
            begin = segs
            sr = grouped_df[k].reset_index(drop=True)
            for newk, functor in infos.items():
                if functor.__name__ == "mean":
                    dev_begins = rmm.to_device(np.asarray(begin))
                    dev_out = rmm.device_array(size, dtype=np.float64)
                    if size > 0:
                        group_mean.forall(size)(
                            sr._column.data_array_view, dev_begins, dev_out
                        )
                    values[newk] = dev_out

                elif functor.__name__ == "max":
                    dev_begins = rmm.to_device(np.asarray(begin))
                    dev_out = rmm.device_array(size, dtype=sr.dtype)
                    if size > 0:
                        group_max.forall(size)(
                            sr._column.data_array_view, dev_begins, dev_out
                        )
                    values[newk] = dev_out

                elif functor.__name__ == "min":
                    dev_begins = rmm.to_device(np.asarray(begin))
                    dev_out = rmm.device_array(size, dtype=sr.dtype)
                    if size > 0:
                        group_min.forall(size)(
                            sr._column.data_array_view, dev_begins, dev_out
                        )
                    values[newk] = dev_out
                else:
                    end = chain(segs[1:], [len(grouped_df)])
                    for i, (s, e) in enumerate(zip(begin, end)):
                        values[newk][i] = functor(sr[s:e])
            # Store
            for k, buf in values.items():
                outdf[k] = buf

        return outdf

    def _group_dataframe(self, df, levels):
        """Group dataframe.

        The output dataframe has the same number of rows as the input
        dataframe.  The rows are shuffled so that the groups are moved
        together in ascending order based on the multi-level index.

        Parameters
        ----------
        df : DataFrame
        levels : list[str]
            Column names for the multi-level index.

        Returns
        -------
        (df, segs) : namedtuple
            * df : DataFrame
                The grouped dataframe.
            * segs : Series.
                 Group starting index.
        """
        sorted_cols, offsets = libcudf.groupby.groupby_without_aggregations(
            df._columns, df[levels]._columns
        )
        outdf = cudf.DataFrame._from_columns(sorted_cols)
        segs = Series(offsets)
        outdf.columns = df.columns
        return _dfsegs_pack(df=outdf, segs=segs)

    def agg(self, args):
        """Invoke aggregation functions on the groups.

        Parameters
        ----------
        args: dict, list, str, callable
            - str
                The aggregate function name.
            - callable
                The aggregate function.
            - list
                List of *str* or *callable* of the aggregate function.
            - dict
                key-value pairs of source column name and list of
                aggregate functions as *str* or *callable*.

        Returns
        -------
        result : DataFrame
        """

        def _get_function(x):
            if isinstance(x, str):
                return self._NAMED_FUNCTIONS[x]
            else:
                return x

        functors = OrderedDict()
        if isinstance(args, (tuple, list)):
            for k in self._val_columns:
                functors[k] = [_get_function(x) for x in args]

        elif isinstance(args, dict):
            for k, v in args.items():
                functors[k] = (
                    [_get_function(v)]
                    if not isinstance(v, (tuple, list))
                    else [_get_function(x) for x in v]
                )
        else:
            return self.agg([args])
        return self._agg_groups(functors)

    _auto_generate_grouper_agg(locals())

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
          groups = df.groupby(['key'], method='cudf')

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
        """
        if not callable(function):
            raise TypeError("type {!r} is not callable", type(function))

        df, segs = self.as_df()
        ends = chain(segs[1:], [None])
        chunks = [df[s:e] for s, e in zip(segs, ends)]
        return cudf.concat([function(chk) for chk in chunks])

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
            groups = df.groupby(['key'], method='cudf')

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
            groups = df.groupby("cat", method="cudf")

            # Define a kernel which takes the moving average of a
            # sliding window
            def rolling_avg(val, avg):
                win_size = 3
                for row, i in enumerate(range(cuda.threadIdx.x,
                                              len(val), cuda.blockDim.x)):
                    if row < win_size - 1:
                        # If there is not enough data to fill the window,
                        # take the average to be NaN
                        avg[i] = np.nan
                    else:
                        total = 0
                        for j in range(i - win_size + 1, i + 1):
                            total += val[j]
                        avg[i] = total / win_size

            # Compute moving avgs on all groups
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
            raise TypeError("type {!r} is not callable", type(function))

        df, segs = self.as_df()
        kwargs.update({"chunks": segs})
        return df.apply_chunks(function, **kwargs)
