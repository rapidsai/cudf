from collections import OrderedDict, defaultdict
from timeit import default_timer as timer

import numpy as np

from numba import cuda

from .dataframe import DataFrame, Series
from . import concat
from .column import Column
from .buffer import Buffer


TEN_MB = 10 ** 7


class Appender(object):
    """For fast appending of data into a Column.
    """
    def __init__(self, parent, bufsize=TEN_MB):
        # Keep reference to parent Column
        self._parent = parent
        # Get physical dtype
        dtype = np.dtype(parent.data.dtype)
        # Max queue size is buffer size divided by itemsize
        self._max_q_sz = max(bufsize // dtype.itemsize, 1)
        self._queue = []
        # Initialize empty Column
        raw_buf = cuda.device_array(shape=0, dtype=dtype)
        self._result = Column(Buffer.from_empty(raw_buf))

    def append(self, value):
        self._queue.append(value)
        # Flush when queue is full
        if len(self._queue) >= self._max_q_sz:
            self.flush()

    def flush(self):
        # Append to Series
        buf = Buffer(np.asarray(self._queue, dtype=self._result.dtype))
        self._result = self._result.append(Column(buf))
        # Reset queue
        self._queue.clear()

    def get(self):
        self.flush()
        assert self._result is not None
        assert not self._result.has_null_mask
        col = self._result
        return Series(self._parent.replace(data=col.data, mask=None))


def _auto_generate_grouper_agg(members):
    def make_fun(f):
        return lambda self: self.agg(f)

    for k, f in members['_NAMED_FUNCTIONS'].items():
        fn = make_fun(f)
        fn.__name__ = k
        fn.__doc__ = """Compute the {} of each group

Returns
-------

result : DataFrame
""".format(k)
        members[k] = fn


class Grouper(object):
    _NAMED_FUNCTIONS = {'mean': Series.mean,
                        'std': Series.std,
                        'min': Series.min,
                        'max': Series.max,
                        'count': Series.count,
                        }

    def __init__(self, df, by):
        self._df = df
        self._by = [by] if isinstance(by, str) else list(by)
        self._val_columns = [idx for idx in self._df.columns
                             if idx not in self._by]

    def _form_groups(self, functors):
        """
        Parameters
        ----------
        functors: dict
            Contains key for column names and value for list of functors.

        """
        functors_mapping = OrderedDict()
        appenders = OrderedDict()
        # The "value" columns
        for k, vs in functors.items():
            if k not in self._df.columns:
                raise NameError('column {:r} not found'.format(k))
            if len(vs) == 1:
                [functor] = vs
                appenders[k] = Appender(parent=self._df[k]._column)
                functors_mapping[k] = {k: functor}
            else:
                functors_mapping[k] = cur_fn_mapping = OrderedDict()
                for functor in vs:
                    newk = '{}_{}'.format(k, functor.__name__)
                    appenders[newk] = Appender(parent=self._df[k]._column)
                    cur_fn_mapping[newk] = functor
        # Grouping
        grouped_df, segs = self._group_dataframe(self._df, self._by)

        # Grouped values
        outdf = DataFrame()
        sr_segs = Buffer(np.asarray(segs))

        for k in self._by:
            outdf[k] = grouped_df[k].take(sr_segs.to_gpu_array()).reset_index()

        size = len(outdf)

        # Append value columns
        for k, infos in functors_mapping.items():
            values = defaultdict(lambda: np.zeros(size, dtype=np.float64))
            begin = segs
            end = segs[1:] + [len(grouped_df)]
            sr = grouped_df[k].reset_index()
            for i, (s, e) in enumerate(zip(begin, end)):
                for newk, functor in infos.items():
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
        (grouped_df, segs)
            * grouped_df is the grouped DataFrame
            * segs is a list[int] of group starting index.
        """
        # Prepare dataframe
        orig_df = df.copy()
        df = df.loc[:, levels].reset_index()
        rowid_column = '__pygdf.groupby.rowid'
        df[rowid_column] = df.index.as_column()

        # Process first level
        col_stack = list(reversed(levels))
        col = col_stack.pop()
        # first level
        df = df.set_index(col).sort_index()
        segs = df.index.find_segments()

        # Handle the remaining level
        while col_stack:
            col = col_stack.pop()
            newsegs = []
            groups = []
            for s, e in zip(segs, segs[1:] + [len(df)]):
                sliced = df[s:e].copy()
                # the following branch if for optimization on groups
                # that are too small
                if e - s > 1:
                    grouped = sliced.set_index(col).sort_index()
                    grpsegs = np.asarray(grouped.index.find_segments()) + s
                else:
                    # too small
                    grouped = sliced
                    grouped.drop_column(col)
                    grpsegs = [s]
                newsegs.extend(grpsegs)
                groups.append(grouped.reset_index())
            df = concat(groups)   # set new base DF
            segs = newsegs        # set new segments

        # Shuffle
        reordering_indices = df[rowid_column].to_gpu_array()
        out_df = DataFrame()
        for k in orig_df.columns:
            col = orig_df[k].reset_index()
            newcol = col.take(reordering_indices)
            out_df[k] = newcol
        return out_df, segs

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

        Notes
        -----
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
                functors[k] = ([_get_function(v)]
                               if not isinstance(v, (tuple, list))
                               else [_get_function(x) for x in v])
        else:
            return self.agg([args])
        return self._form_groups(functors)

    _auto_generate_grouper_agg(locals())
