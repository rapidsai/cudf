import numpy as np

from .dataframe import DataFrame, Series


TEN_MB = 10 ** 7


class Appender(object):
    """For fast appending of data into a Series.
    """
    def __init__(self, dtype, bufsize=TEN_MB):
        dtype = np.dtype(dtype)
        # Max queue size is buffer size divided by itemsize
        self._max_q_sz = max(bufsize // dtype.itemsize, 1)
        self._queue = []
        # Initialize empty Series
        self._result = Series(np.empty(shape=0, dtype=dtype))

    def append(self, value):
        self._queue.append(value)
        # Flush when queue is full
        if len(self._queue) >= self._max_q_sz:
            self.flush()

    def flush(self):
        # Append to Series
        buf = np.asarray(self._queue, dtype=self._result.dtype)
        self._result = self._result.append(buf)
        # Reset queue
        self._queue.clear()

    def get(self):
        self.flush()
        assert self._result is not None
        return self._result


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

