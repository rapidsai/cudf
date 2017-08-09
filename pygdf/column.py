"""
A column is data + validity-mask.
LibGDF operates on column.
"""
from numbers import Number

import numpy as np
from numba import cuda

from . import _gdf
from . import cudautils
from . import utils
from .buffer import Buffer


class Column(object):
    @classmethod
    def _concat(cls, objs):
        head = objs[0]
        for o in objs:
            if not o.is_type_equivalent(head):
                raise ValueError("All series must be of same type")

        newsize = sum(map(len, objs))
        # Concatenate data
        mem = cuda.device_array(shape=newsize, dtype=head.data.dtype)
        data = Buffer.from_empty(mem)
        for o in objs:
            data.extend(o.data.to_gpu_array())

        # Concatenate mask if present
        if all(o.has_null_mask for o in objs):
            # FIXME: Inefficient
            mem = cuda.device_array(shape=newsize, dtype=np.bool)
            mask = Buffer.from_empty(mem)
            null_count = 0
            for o in objs:
                mask.extend(o._get_mask_as_series().to_gpu_array())
                null_count += o._null_count
            mask = Buffer(utils.boolmask_to_bitmask(mask.to_array()))
        else:
            mask = None
            null_count = 0

        col = head.replace(data=data, mask=mask, null_count=null_count)
        return col

    def __init__(self, data, mask=None, null_count=None):
        # Forces Series content to be contiguous
        if not data.is_contiguous():
            data = data.as_contiguous()

        assert mask is None or mask.is_contiguous()
        self._data = data
        self._mask = mask

        if mask is None:
            null_count = 0

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

    def to_gpu_array(self, fillna=None):
        """Get a dense numba device array for the data.

        Parameters
        ----------
        fillna : str or None
            See *fillna* in ``.to_array``.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        return self.to_dense_buffer(fillna=fillna).to_gpu_array()

    def to_array(self, fillna=None):
        """Get a dense numpy array for the data.

        Parameters
        ----------
        fillna : str or None
            Defaults to None, which will skip null values.
            If it equals "pandas", null values are filled with NaNs.
            Non integral dtype is promoted to np.float64.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        return self.to_dense_buffer(fillna=fillna).to_array()

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

    def _replace_defaults(self):
        params = {
            'data': self.data,
            'mask': self.mask,
            'null_count': self.null_count,
        }
        return params

    def replace(self, **kwargs):
        params = self._replace_defaults()
        params.update(kwargs)
        return type(self)(**params)

    def view(self, newcls, **kwargs):
        """View the underlying column data differently.
        """
        params = Column._replace_defaults(self)
        params.update(kwargs)
        return newcls(**params)

    def element_indexing(self, index):
        """Default implementation for indexing to an element

        Raises
        ------
        ``IndexError`` if out-of-bound
        """
        val = self.data[index]  # this can raise IndexError
        valid = (cudautils.mask_get.py_func(self.nullmask, index)
                 if self.has_null_mask else True)
        return val if valid else None

    def __getitem__(self, arg):
        if isinstance(arg, Number):
            arg = int(arg)
            return self.element_indexing(arg)
        elif isinstance(arg, slice):
            # compute mask slice
            start, stop = utils.normalize_slice(arg, len(self))
            if self.null_count > 0:
                if arg.step is not None and arg.step != 1:
                    raise NotImplementedError(arg)

                # slicing
                subdata = self.data[arg]
                submask = self.mask[arg]
                col = self._column.replace(data=Buffer(subdata),
                                           mask=Buffer(submask))
                return col
            else:
                newbuffer = self.data[arg]
                return self.replace(data=newbuffer)
        else:
            raise NotImplementedError(type(arg))

    def fillna(self, value):
        """Fill null values with ``value``.

        Returns a copy with null filled.
        """
        if not self.has_null_mask:
            return self
        out = cudautils.fillna(data=self.data.to_gpu_array(),
                               mask=self.mask.to_gpu_array(),
                               value=value)
        return self.replace(data=Buffer(out), mask=None, null_count=0)

    def to_dense_buffer(self, fillna=None):
        """Get dense (no null values) ``Buffer`` of the data.

        Parameters
        ----------
        fillna : str or None
            See *fillna* in ``.to_array``.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        if fillna not in {None, 'pandas'}:
            raise ValueError('invalid for fillna')

        if self.has_null_mask:
            if fillna == 'pandas':
                # cast non-float types to float64
                col = (self.astype(np.float64)
                       if self.dtype.kind != 'f'
                       else self)
                # fill nan
                return col.fillna(np.nan)
            else:
                return self._copy_to_dense_buffer()
        else:
            return self.data

    def _copy_to_dense_buffer(self):
        data = self.data.to_gpu_array()
        mask = self.mask.to_gpu_array()
        nnz, mem = cudautils.copy_to_dense(data=data, mask=mask)
        return Buffer(mem, size=nnz, capacity=mem.size)

    def find_first_value(self, value):
        """
        Returns offset of first value that matches
        """
        # FIXME: Inefficient find in CPU code
        arr = self.to_array()
        indices = np.argwhere(arr == value)
        if not indices:
            raise ValueError('value not found')
        return indices[0, 0]

    def find_last_value(self, value):
        """
        Returns offset of last value that matches
        """
        arr = self.to_array()
        indices = np.argwhere(arr == value)
        if not indices:
            raise ValueError('value not found')
        return indices[-1, 0]

    def append(self, other):
        """Append another column
        """
        if self.has_null_mask or other.has_null_mask:
            raise NotImplementedError("append masked column is not supported")
        newsize = len(self) + len(other)
        # allocate memory
        mem = cuda.device_array(shape=newsize, dtype=self.data.dtype)
        newbuf = Buffer.from_empty(mem)
        # copy into new memory
        for buf in [self.data, other.data]:
            newbuf.extend(buf.to_gpu_array())
        # return new column
        return self.replace(data=newbuf)

