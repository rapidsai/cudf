"""Provides dummy array for strides calculations.
"""
# Author: Pearu Peterson
# Created: Spetember 2018

import sys
import numpy as np

if sys.platform.startswith('linux'):

    class Array(np.memmap):
        """ Dummy array that has no data storage behind.
        """
        def __new__(cls, shape, strides=None, dtype=np.uint8, order='C'):
            size = np.prod(shape)
            if size == 0:
                return np.ndarray.__new__(cls, shape, dtype=dtype,
                                          buffer=np.array([]),
                                          offset=0, order=order)

            f = open('/dev/zero', 'r')
            obj = np.memmap.__new__(cls, f, mode='r', dtype=dtype,
                                    shape=shape, order=order)
            return np.lib.stride_tricks.as_strided(obj, shape=shape,
                                                   strides=strides,
                                                   writeable=False,
                                                   subok=True)

        @classmethod
        def fromarrayinterface(cls, obj):
            if isinstance(obj, np.ndarray):
                obj = obj.__array_interface__
            return cls(obj['shape'], strides=obj['strides'],
                       dtype=obj['typestr'])

        @property
        def origin(self):
            if isinstance(self.base, type(self)):
                return self.base.origin
            return self.base

        def get_offset(self):
            return (self.__array_interface__['data'][0]
                    - self.origin.__array_interface__['data'][0])

        def __repr__(self):
            d = self.__array_interface__
            d['dtype'] = np.dtype(d['typestr'])
            d['offset'] = self.get_offset()
            return ('Array(shape={shape}, strides={strides},'
                    ' dtype="{dtype}", offset={offset})'.format_map(d))
        __str__ = __repr__

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return False
            return (self.shape, self.strides, self.dtype,
                    self.flags.c_contiguous) \
                == (other.shape, other.strides, other.dtype,
                    other.flags.c_contiguous)

else:
    class Array:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError('Dummy Array on %r' % (sys.platform))
