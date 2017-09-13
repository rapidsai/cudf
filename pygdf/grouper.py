from collections import OrderedDict, defaultdict
from timeit import default_timer as timer

import numpy as np

from numba import cuda

from .dataframe import DataFrame, Series
from . import concat
from .column import Column
from .buffer import Buffer


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


@cuda.jit
def group_mean(data, segments, output):
    i = cuda.grid(1)
    if i < segments.size:
        s = segments[i]
        e = (segments[i + 1]
             if (i + 1) < segments.size
             else data.size)
        # mean calculation
        carry = 0.0
        n = e - s
        for j in range(s, e):
            carry += data[j]
        output[i] = carry / n


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
        # The "value" columns
        for k, vs in functors.items():
            if k not in self._df.columns:
                raise NameError('column {:r} not found'.format(k))
            if len(vs) == 1:
                [functor] = vs
                functors_mapping[k] = {k: functor}
            else:
                functors_mapping[k] = cur_fn_mapping = OrderedDict()
                for functor in vs:
                    newk = '{}_{}'.format(k, functor.__name__)
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
            if functor.__name__ == 'mean':
                dev_begins = cuda.to_device(np.asarray(begin))
                dev_out = cuda.device_array(size, dtype=np.float64)
                for newk, functor in infos.items():
                    group_mean.forall(size)(sr.to_gpu_array(),
                                            dev_begins,
                                            dev_out)
                    values[newk] = dev_out
            else:
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
