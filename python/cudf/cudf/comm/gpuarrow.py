# Copyright (c) 2019-2022, NVIDIA CORPORATION.
from collections import OrderedDict, abc

import numpy as np
import pandas as pd
import pyarrow as pa

from cudf import Series
from cudf._lib.gpuarrow import (
    CudaRecordBatchStreamReader as _CudaRecordBatchStreamReader,
)
from cudf.core import column
from cudf.utils.utils import mask_bitsize, mask_dtype


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


class GpuArrowReader(abc.Sequence):
    def __init__(self, schema, dev_ary):
        self._table = CudaRecordBatchStreamReader(dev_ary, schema).read_all()

    def __len__(self):
        return self._table.num_columns

    def __getitem__(self, idx):
        return GpuArrowNodeReader(self._table, idx)

    def schema(self):
        """
        Return a pyarrow schema
        """
        return self._table.schema

    def to_dict(self):
        """
        Return a dictionary of Series object
        """
        dc = OrderedDict()
        for node in self:
            dc[node.name] = node.make_series()
        return dc


class GpuArrowNodeReader:
    def __init__(self, table, index):
        self._table = table
        self._field = table.schema[index]
        self._series = Series(column.as_column(table.column(index)))
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
        """Accessor for the data buffer as a device array"""
        return self._series._column.data_array_view

    @property
    def null_raw(self):
        """Accessor for the null buffer as a device array"""
        return self._series._column.mask_array_view

    def make_series(self):
        """Make a Series object out of this node"""
        return self._series.copy(deep=False)

    def _make_dictionary_series(self):
        """Make a dictionary-encoded series from this node"""
        assert self.is_dictionary
        return self._series.copy(deep=False)


def arrow_to_pandas_dtype(pa_type):
    if pa.types.is_dictionary(pa_type):
        return pd.core.dtypes.dtypes.CategoricalDtype(ordered=pa_type.ordered)
    if pa.types.is_date64(pa_type):
        return np.dtype("datetime64[ms]")
    if pa.types.is_timestamp(pa_type):
        return np.dtype(f"M8[{pa_type.unit}]")
    return np.dtype(pa_type.to_pandas_dtype())
