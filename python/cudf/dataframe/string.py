# Copyright (c) 2019, NVIDIA CORPORATION.

import pandas as pd
import numpy as np
import pyarrow as pa
import nvstrings
from numbers import Number
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import warnings

from cudf.dataframe import columnops
from cudf.utils import utils, cudautils
# from cudf.comm.serialize import register_distributed_serializer

from cudf.bindings.cudf_cpp import get_ctype_ptr
from librmm_cffi import librmm as rmm


class StringAccessor(object):
    """
    This mimicks pandas `df.str` interface.
    """
    def __init__(self, parent):
        self._parent = parent

    def __getattr__(self, attr, *args, **kwargs):
        if hasattr(self._parent._data, attr):
            passed_attr = getattr(self._parent._data, attr)
            if callable(passed_attr):
                def wrapper(*args, **kwargs):
                    return getattr(self._parent._data, attr)(*args, **kwargs)
                if isinstance(wrapper, nvstrings.nvstrings):
                    wrapper = columnops.as_column(wrapper)
                return wrapper
            else:
                return passed_attr
        else:
            raise AttributeError(attr)


class StringColumn(columnops.TypedColumnBase):
    """Implements operations for Columns of String type
    """
    def __init__(self, data, null_count=None, **kwargs):
        """
        Parameters
        ----------
        data : nvstrings.nvstrings
            The nvstrings object
        null_count : int; optional
            The number of null values in the mask.
        """
        assert isinstance(data, nvstrings.nvstrings)
        self._data = data

        if null_count is None:
            null_count = data.null_count()
        self._null_count = null_count

    # def serialize(self, serialize):
    #     header, frames = super(StringColumn, self).serialize(serialize)
    #     assert 'dtype' not in header
    #     header['dtype'] = serialize(self._dtype)
    #     header['categories'] = self._categories
    #     header['ordered'] = self._ordered
    #     return header, frames

    # @classmethod
    # def deserialize(cls, deserialize, header, frames):
    #     data, mask = cls._deserialize_data_mask(deserialize, header, frames)
    #     dtype = deserialize(*header['dtype'])
    #     categories = header['categories']
    #     ordered = header['ordered']
    #     col = cls(data=data, mask=mask, null_count=header['null_count'],
    #               dtype=dtype, categories=categories, ordered=ordered)
    #     return col

    def str(self):
        return StringAccessor(self)

    def __len__(self):
        return self._data.size()

    @property
    def dtype(self):
        return np.dtype('str')

    @property
    def data(self):
        """ nvstrings object """
        return self._data

    def element_indexing(self, arg):
        if isinstance(arg, Number):
            arg = int(arg)
            if arg > (len(self) - 1) or arg < 0:
                raise IndexError
            out = self._data[arg]
        elif isinstance(arg, slice):
            out = self._data[arg]
        elif isinstance(arg, list):
            out = self._data[arg]
        elif isinstance(arg, np.ndarray):
            gpu_arr = rmm.to_device(arg)
            return self.element_indexing(gpu_arr)
        elif isinstance(arg, DeviceNDArray):
            # NVStrings gather call expects an array of int32s
            arg = cudautils.astype(arg, np.dtype('int32'))
            if len(arg) > 0:
                gpu_ptr = get_ctype_ptr(arg)
                out = self._data.gather(gpu_ptr, len(arg))
            else:
                out = self._data.gather([])
        else:
            raise NotImplementedError(type(arg))

        if len(out) == 1:
            return out.to_host()[0]
        else:
            return columnops.as_column(out)

    def __getitem__(self, arg):
        return self.element_indexing(arg)

    def astype(self, dtype):
        if self.dtype == dtype:
            return self
        elif dtype in (np.dtype('int8'), np.dtype('int16'), np.dtype('int32'),
                       np.dtype('int64')):
            out_arr = rmm.device_array(shape=len(self), dtype='int32')
            out_ptr = get_ctype_ptr(out_arr)
            self.str().stoi(devptr=out_ptr)
        elif dtype in (np.dtype('float32'), np.dtype('float64')):
            out_arr = rmm.device_array(shape=len(self), dtype='float32')
            out_ptr = get_ctype_ptr(out_arr)
            self.str().stof(devptr=out_ptr)
        out_col = columnops.as_column(out_arr)
        return out_col.astype(dtype)

    def to_arrow(self):
        sbuf = np.empty(self._data.byte_count(), dtype='int8')
        obuf = np.empty(len(self._data) + 1, dtype='int32')

        mask_size = utils.calc_chunk_size(len(self._data), utils.mask_bitsize)
        nbuf = np.empty(mask_size, dtype='int8')

        self.str().to_offsets(sbuf, obuf, nbuf=nbuf)
        sbuf = pa.py_buffer(sbuf)
        obuf = pa.py_buffer(obuf)
        nbuf = pa.py_buffer(nbuf)
        return pa.StringArray.from_buffers(len(self._data), obuf, sbuf, nbuf,
                                           self._data.null_count())

    def to_pandas(self, index=None):
        pd_series = self.to_arrow().to_pandas()
        return pd.Series(pd_series, index=index)

    def to_array(self, fillna=None):
        """Get a dense numpy array for the data.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.

        Raises
        ------
        ``NotImplementedError`` if there are nulls
        """
        if fillna is not None:
            warnings.warn("fillna parameter not supported for string arrays")
        if self.null_count > 0:
            raise NotImplementedError(
                "Converting to NumPy array is not yet supported for columns "
                "with nulls"
            )
        return self.to_arrow().to_pandas()


# register_distributed_serializer(StringColumn)
