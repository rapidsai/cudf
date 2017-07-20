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


class DefaultIndex(Index):
    """Basic 0..size
    """
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if isinstance(index, slice):
            assert index.step is None
            start, stop = utils.normalize_slice(index, len(self))
            return Int64Index.make_range(start, stop)
        elif isinstance(index, int):
            index = utils.normalize_index(index, len(self))
        else:
            raise ValueError(index)
        return index

    def __eq__(self, other):
        if isinstance(other, DefaultIndex):
            return self._size == other._size
        else:
            return NotImplemented

    @property
    def dtype(self):
        return np.dtype(np.int64)

    @property
    def values(self):
        return np.arange(len(self), dtype=self.dtype)

    @property
    def gpu_values(self):
        return cudautils.arange(len(self), dtype=self.dtype)

    def find_label_range(self, first, last):
        begin, end = first, last
        if last is not None:
            end = last + 1
        return begin, end


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
