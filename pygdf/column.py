"""
A column is data + validity-mask.
LibGDF operates on column.
"""
import numpy as np

from . import _gdf
from . import cudautils


class Column(object):
    def __init__(self, data, mask=None, null_count=None):
        # Forces Series content to be contiguous
        if not data.is_contiguous():
            data = data.as_contiguous()

        assert mask is None or mask.is_contiguous()
        self._data = data
        self._mask = mask

        if null_count is None:
            if self._mask is not None:
                nnz = cudautils.count_nonzero_mask(self._mask.mem)
                null_count = len(self) - nnz
            else:
                null_count = 0
        self._null_count = null_count

    def __len__(self):
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def data(self):
        """Data buffer
        """
        return self._data

    @property
    def mask(self):
        """Validity mask buffer
        """
        return self._mask

    @property
    def cffi_view(self):
        """LibGDF CFFI view
        """
        return _gdf.columnview(size=self._data.size,
                               data=self._data,
                               mask=self._mask)

    # XXX: retired this
    _cffi_view = cffi_view

    @property
    def valid_count(self):
        """Number of non-null values"""
        return len(self) - self._null_count

    @property
    def null_count(self):
        """Number of null values"""
        return self._null_count

    @property
    def has_null_mask(self):
        """A boolean indicating whether a null-mask is needed"""
        return self._mask is not None

    @property
    def nullmask(self):
        """The gpu buffer for the null-mask
        """
        if self.has_null_mask:
            return self._mask
        else:
            raise ValueError('Column has no null mask')

    def replace(self, **kwargs):
        params = {
            'data': self.data,
            'mask': self.mask,
            'null_count': self.null_count,
        }
        params.update(kwargs)
        return Column(**params)
