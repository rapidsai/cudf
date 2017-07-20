from __future__ import print_function, division

import numpy as np

from . import cudautils, utils
from .buffer import Buffer


class Index(object):
    def take(self, indices):
        index = cudautils.gather(data=self.gpu_values, index=indices)
        return Int64Index(index)


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
                return Int64Index.make_range(start, stop)[::index.step]
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

    @property
    def values(self):
        return np.arange(self._start, self._stop, dtype=self.dtype)

    @property
    def gpu_values(self):
        return cudautils.arange(self._start, self._stop, dtype=self.dtype)

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


class Int64Index(Index):
    @classmethod
    def make_range(cls, start, stop=None):
        if stop is None:
            return cls(np.arange(start, dtype=np.int64))
        else:
            return cls(np.arange(start, stop, dtype=np.int64))

    def __init__(self, buf):
        if not isinstance(buf, Buffer):
            buf = Buffer(buf)
        self._values = buf

    def __len__(self):
        return self._values.size

    def __repr__(self):
        ar = self._values.to_array()
        return "Int64Index({})".format(ar)

    def __getitem__(self, index):
        res = self._values[index]
        if not isinstance(index, int):
            return Int64Index(res)
        else:
            return res

    def __eq__(self, other):
        if isinstance(other, Int64Index):
            # FIXME inefficient comparison
            return np.all(self._values.to_array() == other._values.to_array())
        else:
            return NotImplemented

    @property
    def values(self):
        return self._values.to_array()

    @property
    def gpu_values(self):
        return self._values.to_gpu_array()

    @property
    def dtype(self):
        return self._values.dtype

    def find_label_range(self, first, last):
        # FIXME inefficient find
        ar = self._values.to_array()
        begin, end = None, None
        if first is not None:
            begin = np.argwhere(ar == first)[0, 0]
        if last is not None:
            end = np.argwhere(ar == last)[-1, 0]
            end += 1
        return begin, end
