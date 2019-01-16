# Copyright (c) 2018, NVIDIA CORPORATION.

"""
A column is data + validity-mask.
LibGDF operates on column.
"""
from numbers import Number

import numpy as np
from numba import cuda

from librmm_cffi import librmm as rmm

from cudf import _gdf
from cudf.utils import cudautils, utils
from .buffer import Buffer


class Column(object):
    """An immutable structure for storing data and mask for a column.
    This should be considered as the physical layer that provides
    container operations on the data and mask.

    These operations work on each data element as plain-old-data.
    Any logical operations are implemented in subclasses of *TypedColumnBase*.

    Attributes
    ----------
    _data : Buffer
        The data buffer
    _mask : Buffer
        The validity mask
    _null_count : int
        Number of null values in the mask.

    These attributes are exported in the properties (e.g. *data*, *mask*,
    *null_count*).
    """
    @classmethod
    def _concat(cls, objs):
        if len(objs) == 0:
            return Column(Buffer.null(np.float))
        head = objs[0]
        for o in objs:
            if not o.is_type_equivalent(head):
                raise ValueError("All series must be of same type")
        # Filter out inputs that have 0 length
        objs = [o for o in objs if len(o) > 0]
        nulls = sum(o.null_count for o in objs)
        newsize = sum(map(len, objs))
        mem = rmm.device_array(shape=newsize, dtype=head.data.dtype)
        data = Buffer.from_empty(mem, size=newsize)

        # Allocate output mask only if there's nulls in the input objects
        mask = None
        if nulls:
            mask = Buffer(utils.make_mask(newsize))

        col = head.replace(data=data, mask=mask, null_count=nulls)

        # Performance the actual concatenation
        if newsize > 0:
            col = _gdf._column_concat(objs, col)

        return col

    @staticmethod
    def from_cffi_view(cffi_view):
        """Create a Column object from a cffi struct gdf_column*.
        """
        data_mem, mask_mem = _gdf.cffi_view_to_column_mem(cffi_view)
        data_buf = Buffer(data_mem)

        if mask_mem is not None:
            mask = Buffer(mask_mem)

        return Column(data=data_buf, mask=mask)

    def __init__(self, data, mask=None, null_count=None):
        """
        Parameters
        ----------
        data : Buffer
            The code values
        mask : Buffer; optional
            The validity mask
        null_count : int; optional
            The number of null values in the mask.
        """
        # Forces Column content to be contiguous
        if not data.is_contiguous():
            data = data.as_contiguous()

        assert mask is None or mask.is_contiguous()
        self._data = data
        self._mask = mask

        if mask is None:
            null_count = 0
        else:
            # check that mask length is sufficient
            assert mask.size * utils.mask_bitsize >= len(self)

        assert null_count is None or null_count >= 0
        if null_count is None:
            if self._mask is not None:
                nnz = _gdf.count_nonzero_mask(self._mask.mem,
                                              size=len(self))
                null_count = len(self) - nnz
                if null_count == 0:
                    self._mask = None
            else:
                null_count = 0

        assert 0 <= null_count <= len(self)
        if null_count == 0:
            # Remove mask if null_count is zero
            self._mask = None

        self._null_count = null_count

    def serialize(self, serialize):
        header = {
            'null_count': self._null_count,
        }
        frames = []

        header['data_buffer'], data_frames = serialize(self._data)
        header['data_frame_count'] = len(data_frames)
        frames.extend(data_frames)
        header['mask_buffer'], mask_frames = serialize(self._mask)
        header['mask_frame_count'] = len(mask_frames)
        frames.extend(mask_frames)
        header['frame_count'] = len(frames)
        return header, frames

    @classmethod
    def _deserialize_data_mask(cls, deserialize, header, frames):
        data_nframe = header['data_frame_count']
        mask_nframe = header['mask_frame_count']
        data = deserialize(header['data_buffer'], frames[:data_nframe])
        mask = deserialize(header['mask_buffer'],
                           frames[data_nframe:data_nframe + mask_nframe])
        return data, mask

    def _get_mask_as_column(self):
        from .numerical import NumericalColumn

        data = Buffer(cudautils.ones(len(self), dtype=np.bool_))
        mask = NumericalColumn(data=data, mask=None, null_count=0,
                               dtype=np.bool_)
        if self._mask is not None:
            mask = mask.set_mask(self._mask).fillna(False)
        return mask

    def __sizeof__(self):
        n = self._data.__sizeof__()
        if self._mask:
            n += self._mask.__sizeof__()
        return n

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
                               mask=self._mask,
                               dtype=self.dtype,
                               null_count=self._null_count)

    def set_mask(self, mask, null_count=None):
        """Create new Column by setting the mask

        This will override the existing mask.  The returned Column will
        reference the same data buffer as this Column.

        Parameters
        ----------
        mask : 1D array-like of numpy.uint8
            The null-mask.  Valid values are marked as ``1``; otherwise ``0``.
            The mask bit given the data index ``idx`` is computed as::

                (mask[idx // 8] >> (idx % 8)) & 1
        null_count : int, optional
            The number of null values.
            If None, it is calculated automatically.

        """
        if not isinstance(mask, Buffer):
            mask = Buffer(mask)
        if mask.dtype not in (np.dtype(np.uint8), np.dtype(np.int8)):
            msg = 'mask must be of byte; but got {}'.format(mask.dtype)
            raise ValueError(msg)
        return self.replace(mask=mask, null_count=null_count)

    def allocate_mask(self, all_valid=True):
        """Return a new Column with a newly allocated mask buffer.
        If ``all_valid`` is True, the new mask is set to all valid.
        If ``all_valid`` is False, the new mask is set to all null.
        """
        nelem = len(self)
        mask_sz = utils.calc_chunk_size(nelem, utils.mask_bitsize)
        mask = cuda.device_array(mask_sz, dtype=utils.mask_dtype)
        if nelem > 0:
            cudautils.fill_value(mask, 0xff if all_valid else 0)
        return self.set_mask(mask=mask, null_count=0 if all_valid else nelem)

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

    def copy_data(self):
        """Copy the column with a new allocation of the data but not the mask,
        which is shared by the new column.
        """
        return self.replace(data=self.data.copy())

    def copy(self, deep=True):
        """Columns are immutable, so a deep copy produces a copy of the
        underlying data and mask and a shallow copy creates a new column and
        copies the references of the data and mask.
        """
        if(deep):
            deep = self.copy_data()
            if self.has_null_mask:
                return deep.set_mask(mask=self.mask.copy(),
                                     null_count=self.null_count)
            else:
                return deep.allocate_mask()
        else:
            shallow = Column()
            shallow._data = self._data
            shallow._mask = self._mask
            shallow.has_null_mask = self.has_null_mask
            return shallow

    def replace(self, **kwargs):
        """Replace attributes of the class and return a new Column.

        Valid keywords are valid parameters for ``self.__init__``.
        Any omitted keywords will be defaulted to the corresponding
        attributes in ``self``.
        """
        params = self._replace_defaults()
        params.update(kwargs)
        if 'mask' in kwargs and 'null_count' not in kwargs:
            del params['null_count']
        return type(self)(**params)

    def view(self, newcls, **kwargs):
        """View the underlying column data differently using a subclass of
        *TypedColumnBase*.

        Parameters
        ----------
        newcls : TypedColumnBase
            The logical view to be used
        **kwargs :
            Additional paramters for instantiating instance of *newcls*.
            Valid keywords are valid parameters for ``newcls.__init__``.
            Any omitted keywords will be defaulted to the corresponding
            attributes in ``self``.
        """
        params = Column._replace_defaults(self)
        params.update(kwargs)
        if 'mask' in kwargs and 'null_count' not in kwargs:
            del params['null_count']
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

                # slicing data
                subdata = self.data[arg]
                # slicing mask
                bytemask = cudautils.expand_mask_bits(
                    self.data.size,
                    self.mask.to_gpu_array(),
                    )
                submask = Buffer(cudautils.compact_mask_bytes(bytemask[arg]))
                col = self.replace(data=subdata, mask=submask)
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

        if self.null_count > 0:
            if fillna == 'pandas':
                na_value = self.default_na_value()
                # fill nan
                return self.fillna(na_value)
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
        if not len(indices):
            raise ValueError('value not found')
        return indices[0, 0]

    def find_last_value(self, value):
        """
        Returns offset of last value that matches
        """
        arr = self.to_array()
        indices = np.argwhere(arr == value)
        if not len(indices):
            raise ValueError('value not found')
        return indices[-1, 0]

    def append(self, other):
        """Append another column
        """
        if self.null_count > 0 or other.null_count > 0:
            raise NotImplementedError("Appending columns with nulls is not "
                                      "yet supported")
        newsize = len(self) + len(other)
        # allocate memory
        data_dtype = np.result_type(self.data.dtype, other.data.dtype)
        mem = rmm.device_array(shape=newsize, dtype=data_dtype)
        newbuf = Buffer.from_empty(mem)
        # copy into new memory
        for buf in [self.data, other.data]:
            newbuf.extend(buf.to_gpu_array())
        # return new column
        return self.replace(data=newbuf)

    def quantile(self, q, interpolation, exact):
        if isinstance(q, Number):
            quant = [q]
        elif isinstance(q, list) or isinstance(q, np.ndarray):
            quant = q
        else:
            msg = "`q` must be either a single element, list or numpy array"
            raise TypeError(msg)
        return _gdf.quantile(self, quant, interpolation, exact)

    def take(self, indices, ignore_index=False):
        """Return Column by taking values from the corresponding *indices*.
        """
        indices = Buffer(indices).to_gpu_array()
        # Handle zero size
        if indices.size == 0:
            return self.copy()

        data = cudautils.gather(data=self._data.to_gpu_array(), index=indices)

        if self._mask:
            mask = self._get_mask_as_column().take(indices).as_mask()
            mask = Buffer(mask)
        else:
            mask = None

        return self.replace(data=Buffer(data), mask=mask)

    def as_mask(self):
        """Convert booleans to bitmask

        Returns
        -------
        device array
        """
        return cudautils.compact_mask_bytes(self.to_gpu_array())
