from __future__ import print_function, division

import numpy as np

from numba import cuda

from . import cudautils, utils
from .buffer import Buffer


class Index(object):
    def take(self, indices):
        assert indices.dtype.kind in 'iu'
        index = cudautils.gather(data=self.gpu_values, index=indices)
        index = self.as_series()._copy_construct(buffer=Buffer(index))
        return GenericIndex(index)

    def as_series(self):
        raise NotImplementedError

    def argsort(self, ascending=True):
        return self.as_series().argsort(ascending=ascending)

    @property
    def values(self):
        return np.asarray(self.as_series())

    @property
    def gpu_values(self):
        return self.as_series().to_gpu_array()

    def find_segments(self):
        """Return the beginning index for segments
        """
        segments = cudautils.find_segments(self.gpu_values)
        return list(segments.copy_to_host())


class EmptyIndex(Index):
    """
    A singleton class to represent an empty index when a DataFrame is created
    without any initializer.
    """
    _singleton = None

    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = object.__new__(EmptyIndex)
        return cls._singleton

    def __getitem__(self, index):
        raise IndexError

    def __len__(self):
        return 0

    def as_series(self):
        from .series import Series

        return Series(np.empty(0, dtype=np.int64))


class RangeIndex(Index):
    """Basic start..stop
    """
    def __init__(self, start, stop=None):
        """RangeIndex(size), RangeIndex(start, stop)

        Parameters
        ----------
        size, start, stop: int
        """
        if stop is None:
            start, stop = 0, start
        self._start = start
        self._stop = stop

    def __repr__(self):
        return "{}(start={}, stop={})".format(self.__class__.__name__,
                                              self._start, self._stop)

    def __len__(self):
        return self._stop - self._start

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop = utils.normalize_slice(index, len(self))
            start += self._start
            stop += self._start
            if index.step is None:
                return RangeIndex(start, stop)
            else:
                return index_from_range(start, stop, index.step)
        elif isinstance(index, int):
            index = utils.normalize_index(index, len(self))
            index += self._start
            return index
        else:
            raise ValueError(index)

    def __eq__(self, other):
        if isinstance(other, RangeIndex):
            return (self._start == other._start and self._stop == other._stop)
        else:
            return NotImplemented

    @property
    def dtype(self):
        return np.dtype(np.int64)

    def find_label_range(self, first, last):
        # clip first to range
        if first < self._start:
            begin = self._start
        elif first < self._stop:
            begin = first
        else:
            begin = self._stop
        # clip last to range
        if last < self._start:
            end = begin
        elif last < self._stop:
            end = last + 1
        else:
            end = self._stop
        # shift to index
        return begin - self._start, end - self._start

    def as_series(self):
        from .series import Series
        vals = cudautils.arange(self._start, self._stop, dtype=self.dtype)
        return Series(vals)


def index_from_range(start, stop=None, step=None):
    vals = cudautils.arange(start, stop, step, dtype=np.int64)
    return GenericIndex(vals)


class GenericIndex(Index):
    def __new__(self, values):
        from .series import Series  # cyclic dep on Series

        values = Series(values)
        if len(values) == 0:
            # for empty index, return a EmptyIndex instead
            return EmptyIndex()
        else:
            # Make GenericIndex object
            res = Index.__new__(GenericIndex)
            # Force simple index
            res._values = values.as_index()
            return res

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        vals = [self._values[i] for i in range(min(len(self), 10))]
        return "{}({}, dtype={})".format(self.__class__.__name__,
                                         vals, self._values.dtype)

    def __getitem__(self, index):
        res = self._values[index]
        if not isinstance(index, int):
            return GenericIndex(res)
        else:
            return res

    def __eq__(self, other):
        if isinstance(other, GenericIndex):
            # FIXME inefficient comparison
            booleans = self.as_series() == other.as_series()
            return np.all(booleans.to_array())
        else:
            return NotImplemented

    def as_series(self):
        """Convert the index as a Series.
        """
        from .series import Series

        return self._values

    @property
    def dtype(self):
        return self._values.dtype

    def find_label_range(self, first, last):
        sr = self.as_series()
        begin, end = None, None
        if first is not None:
            begin = sr.find_first_value(first)
        if last is not None:
            end = sr.find_last_value(last)
            end += 1
        return begin, end
