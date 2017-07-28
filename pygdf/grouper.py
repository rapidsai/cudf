import numpy as np

from .dataframe import DataFrame, Series


class Appender(object):
    def __init__(self, dtype):
        self.dtype = dtype
        self.values = []

    def append(self, value):
        # FIXME: inefficient append
        self.values.append(value)

    def get(self):
        return Series(np.asarray(self.values, dtype=self.dtype))


class Grouper(object):
    def __init__(self, df, by):
        self._df = df
        self._by = [by] if isinstance(by, str) else list(by)

    def _form_groups(self, functor):
        appenders = {}
        for k in self._by:
            appenders[k] = Appender(dtype=self._df[k].dtype)
        for k in self._df.columns:
            if k not in appenders:
                appenders[k] = Appender(dtype=self._df[k].dtype)

        for idx, grp in self._group_level(self._df, self._by):
            for k, v in zip(self._by, idx):
                appenders[k].append(v)
            for k in grp.columns:
                appenders[k].append(functor(grp[k]))

        outdf = DataFrame()
        for k in self._df.columns:
            outdf[k] = appenders[k].get()
        return outdf

    def _group_level(self, df, levels, indices=[]):
        """A generator that yields (indices, grouped_df).
        """
        col = levels[0]
        innerlevels = levels[1:]
        df = df.set_index(col).sort_index()
        segs = df.index.find_segments()
        for s, e in zip(segs, segs[1:] + [None]):
            grouped = df[s:e]
            if len(grouped):
                # FIXME numpy.scalar getitem to Index
                index = df.index[int(s)]
                inner_indices = indices + [index]
                if innerlevels:
                    for grp in self._group_level(grouped, innerlevels,
                                                 indices=inner_indices):
                        yield grp
                else:
                    yield inner_indices, grouped

    def mean(self):
        return self._form_groups(lambda sr: sr.mean())



