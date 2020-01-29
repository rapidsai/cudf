# Copyright (c) 2019, NVIDIA CORPORATION.

from collections import OrderedDict
from collections.abc import Sequence

import numba.cuda.cudadrv.driver
import numpy as np
import pandas as pd
import pyarrow as pa

import rmm

from cudf._lib.arrow._cuda import CudaBuffer
from cudf._lib.gpuarrow import (
    CudaRecordBatchStreamReader as _CudaRecordBatchStreamReader,
)
from cudf.core import Series
from cudf.utils.utils import calc_chunk_size, mask_bitsize, mask_dtype


class CudaRecordBatchStreamReader(_CudaRecordBatchStreamReader):
    """
    Reader for the Arrow streaming binary format

    Parameters
    ----------
    source : pyarrow.cuda.CudaBuffer or numba DeviceNDarray
        Either numba DeviceNDArray, or a pyarrow.cuda.CudaBuffer
    schema : bytes/buffer-like, pyarrow.Schema, pyarrow.NativeFile,
             file-like Python object
        Optional pyarrow.Schema or host-serialized bytes/buffer-like
                 pyarrow.Schema
    """

    def __init__(self, source, schema=None):
        self._open(source, schema)


class GpuArrowReader(Sequence):
    def __init__(self, schema, dev_ary):
        self._table = CudaRecordBatchStreamReader(dev_ary, schema).read_all()

    def __len__(self):
        return self._table.num_columns

    def __getitem__(self, idx):
        return GpuArrowNodeReader(self._table, idx)

    def schema(self):
        return self._table.schema

    def to_dict(self):
        """
        Return a dictionary of Series object
        """
        dc = OrderedDict()
        for node in self:
            dc[node.name] = node.make_series()
        return dc


class GpuArrowNodeReader(object):
    def __init__(self, table, index):
        self._table = table
        self._field = table.schema[index]
        self._series = array_to_series(table.column(index))
        self._series.name = self.name

    def __len__(self):
        return len(self._series)

    @property
    def schema(self):
        return self._table.schema

    @property
    def field_schema(self):
        return self._field

    @property
    def is_dictionary(self):
        return pa.types.is_dictionary(self._field.type)

    @property
    def null_count(self):
        return self._series.null_count

    @property
    def dtype(self):
        return arrow_to_pandas_dtype(self._field.type)

    @property
    def index_dtype(self):
        return self._field.type.index_type.to_pandas_dtype()

    @property
    def name(self):
        return self._field.name

    @property
    def data(self):
        """
        Return the data as the expected dtype
        and with the padding bytes truncated.
        """
        if self.data_raw is not None:
            return self.data_raw.view(
                self.dtype if not self.is_dictionary else self.index_dtype
            )

    @property
    def null(self):
        """
        Return the null mask with the padding bytes truncated.
        """
        if self.null_raw is not None:
            bits = mask_bitsize
            itemsize = mask_dtype.itemsize
            end = ((len(self) + bits - 1) // bits) * itemsize
            return self.null_raw[:end].view(mask_dtype)

    @property
    def data_raw(self):
        "Accessor for the data buffer as a device array"
        return self._series._column.data_array_view

    @property
    def null_raw(self):
        "Accessor for the null buffer as a device array"
        return self._series._column.mask_array_view

    def make_series(self):
        """Make a Series object out of this node
        """
        return self._series.copy(deep=False)

    def _make_dictionary_series(self):
        """Make a dictionary-encoded series from this node
        """
        assert self.is_dictionary
        return self._series.copy(deep=False)


def gpu_view_as(nbytes, buf, dtype, shape=None, strides=None):
    ptr = numba.cuda.cudadrv.driver.device_pointer(buf.to_numba())
    arr = rmm.device_array_from_ptr(ptr, nbytes // dtype.itemsize, dtype=dtype)
    arr.gpu_data._obj = buf
    return arr


def make_device_arrays(array):
    buffers = array.buffers()
    dtypes = [np.dtype(np.int8), None, None]

    if pa.types.is_list(array.type):
        dtypes[1] = np.dtype(np.int32)
    elif pa.types.is_string(array.type) or pa.types.is_binary(array.type):
        dtypes[2] = np.dtype(np.int8)
        dtypes[1] = np.dtype(np.int32)
    elif not pa.types.is_dictionary(array.type):
        dtypes[1] = arrow_to_pandas_dtype(array.type)
    else:
        dtypes[1] = arrow_to_pandas_dtype(array.type.index_type)

    if buffers[0] is not None:
        buf = CudaBuffer.from_buffer(buffers[0])
        nbytes = min(buf.size, calc_chunk_size(len(array), mask_bitsize))
        buffers[0] = gpu_view_as(nbytes, buf, dtypes[0])

    for i in range(1, len(buffers)):
        if buffers[i] is not None:
            buf = CudaBuffer.from_buffer(buffers[i])
            nbytes = min(buf.size, len(array) * dtypes[i].itemsize)
            buffers[i] = gpu_view_as(nbytes, buf, dtypes[i])

    return buffers


def array_to_series(array):
    if isinstance(array, pa.ChunkedArray):
        return Series._concat(
            [array_to_series(chunk) for chunk in array.chunks]
        )

    array_len = len(array)
    null_count = array.null_count
    buffers = make_device_arrays(array)
    mask, data = buffers[0], buffers[1]
    dtype = arrow_to_pandas_dtype(array.type)

    if pa.types.is_dictionary(array.type):
        from cudf.core.column import build_categorical_column
        from cudf.core.buffer import Buffer

        codes = array_to_series(array.indices)
        categories = array_to_series(array.dictionary)
        if mask is not None:
            mask = Buffer(mask)
        data = build_categorical_column(
            categories=categories, codes=codes, mask=mask
        )

    elif pa.types.is_string(array.type):
        import nvstrings

        offs, data = buffers[1], buffers[2]
        offs = offs[array.offset : array.offset + array_len + 1]
        data = None if data is None else data.device_ctypes_pointer.value
        mask = None if mask is None else mask.device_ctypes_pointer.value
        data = nvstrings.from_offsets(
            data,
            offs.device_ctypes_pointer.value,
            array_len,
            mask,
            null_count,
            True,
        )
    elif data is not None:
        data = data[array.offset : array.offset + len(array)]

    series = Series(data, dtype=dtype)

    if null_count > 0 and mask is not None and not series.nullable:
        return series.set_mask(mask, null_count)

    return series


def arrow_to_pandas_dtype(pa_type):
    if pa.types.is_dictionary(pa_type):
        return pd.core.dtypes.dtypes.CategoricalDtype(ordered=pa_type.ordered)
    if pa.types.is_date64(pa_type):
        return np.dtype("datetime64[ms]")
    if pa.types.is_timestamp(pa_type):
        return np.dtype("M8[{}]".format(pa_type.unit))
    return np.dtype(pa_type.to_pandas_dtype())
