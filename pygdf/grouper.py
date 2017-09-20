from collections import OrderedDict, defaultdict, namedtuple

from itertools import chain
import numpy as np

from numba import cuda

from .dataframe import DataFrame, Series
from . import _gdf, cudautils
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


_dfsegs_pack = namedtuple('_dfsegs_pack', ['df', 'segs'])


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
        """
        return self._group_dataframe(self._df, self._by)

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
        grouped_df, sr_segs = self._group_dataframe(self._df, self._by)
        # Grouped values
        outdf = DataFrame()
        segs = sr_segs.to_array()

        for k in self._by:
            outdf[k] = grouped_df[k].take(sr_segs.to_gpu_array()).reset_index()

        size = len(outdf)

        # Append value columns
        for k, infos in functors_mapping.items():
            values = defaultdict(lambda: np.zeros(size, dtype=np.float64))
            begin = segs
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
                end = chain(segs[1:], [len(grouped_df)])
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
        (df, segs) : namedtuple
            * df : DataFrame
                The grouped dataframe.
            * segs : Series.
                 Group starting index.
        """
        # Prepare dataframe
        orig_df = df.copy()
        df = df.loc[:, levels].reset_index()
        rowid_column = '__pygdf.groupby.rowid'
        df[rowid_column] = df.index.as_column()

        col_order = list(levels)

        # Perform grouping
        df, segs = self._group_first_level(col_order[0], rowid_column, df)
        rowidcol = df[rowid_column]
        sorted_keys = [Series(df.index.as_column())]
        del df

        more_keys, reordering_indices, segs = self._group_inner_levels(
                                            col_order[1:], rowidcol, segs)
        sorted_keys.extend(more_keys)
        valcols = [k for k in orig_df.columns if k not in levels]
        # Prepare output
        # All key columns are already sorted
        out_df = DataFrame()
        for k, sr in zip(levels, sorted_keys):
            out_df[k] = sr
        # Shuffle the value columns
        self._group_shuffle(orig_df.loc[:, valcols],
                            reordering_indices, out_df)
        return _dfsegs_pack(df=out_df, segs=segs)

    def _group_first_level(self, col, rowid_column, df):
        """Group first level *col* of *df*

        Parameters
        ----------
        col : str
            Name of the first group key column.
        df : DataFrame
            The dataframe being grouped.

        Returns
        -------
        (df, segs)
            - df : DataFrame
                Sorted by *col- * index
            - segs : Series
                Group begin offsets
        """
        df = df.loc[:, [col, rowid_column]]
        df = df.set_index(col).sort_index()
        segs = df.index.find_segments()
        return df, Series(segs)

    def _group_inner_levels(self, columns, rowidcol, segs):
        """Group the second and onwards level.

        Parameters
        ----------
        columns : sequence[str]
            Group keys.  The order is important.
        rowid_column : str
            The name of the special column with the original rowid.
            It's internally used to determine the shuffling order.
        df : DataFrame
            The dataframe being grouped.
        segs : Series
            First level group begin offsets.

        Returns
        -------
        (sorted_keys, reordering_indices, segments)
            - sorted_keys : list[Series]
                List of sorted key columns.
                Column order is same as arg *columns*.
            - reordering_indices : device array
                The indices to gather on to shuffle the dataframe
                into the grouped seqence.
            - segments : Series
                Group begin offsets.
        """
        dsegs = segs.astype(dtype=np.uint32).to_gpu_array()
        sorted_keys = []
        for col in columns:
            # Shuffle the key column according to the previous groups
            srkeys = self._df[col].take(rowidcol.to_gpu_array(),
                                        ignore_index=True)
            # Segmented sort on the key
            shuf = Column(Buffer(cudautils.arange(len(srkeys))))
            _gdf.apply_segsort(srkeys._column, shuf, dsegs)
            sorted_keys.append(srkeys)   # keep sorted key cols
            # Determine segments
            dsegs = cudautils.find_segments(srkeys.to_gpu_array(), dsegs)
            # Shuffle
            rowidcol = rowidcol.take(shuf.to_gpu_array(), ignore_index=True)

        reordering_indices = rowidcol.to_gpu_array()
        return sorted_keys, reordering_indices, Series(dsegs)

    def _group_shuffle(self, src_df, reordering_indices, out_df):
        """Shuffle columns in *src_df* with *reordering_indices*
        and store the new columns into *out_df*
        """
        for k in src_df.columns:
            col = src_df[k].reset_index()
            newcol = col.take(reordering_indices, ignore_index=True)
            out_df[k] = newcol
        return out_df

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
